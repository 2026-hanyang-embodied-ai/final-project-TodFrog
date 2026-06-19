"""Status-only remote training handoff for the v7 RPS pose branch."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from embodied_rps.real_skeleton_training import load_sweep_config


DEFAULT_REMOTE_HOST = "voice@166.104.167.133"
DEFAULT_REMOTE_WORKSPACE = "/home/voice/workspace/chominkyu/embodied-final"
DEFAULT_V7_CONFIG = Path("configs/real_skeleton_three_class_wait_prediction_v7_rps_pose_tcn_ensemble.yaml")
DEFAULT_V7_DATASET = Path("artifacts/real_guided_three_class_wait_expanded_v7_rps_pose_20260617")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7_remote_training_handoff_20260617")
DEFAULT_EXPECTED_GENERATED_PER_TARGET = 10000


@dataclass(frozen=True)
class V7RemoteTrainingHandoffConfig:
    """Inputs for writing v7 remote training handoff artifacts."""

    training_config_path: Path = DEFAULT_V7_CONFIG
    dataset_root: Path = DEFAULT_V7_DATASET
    output_root: Path = DEFAULT_OUTPUT_ROOT
    remote_host: str = DEFAULT_REMOTE_HOST
    remote_workspace: str = DEFAULT_REMOTE_WORKSPACE
    expected_generated_per_target: int = DEFAULT_EXPECTED_GENERATED_PER_TARGET
    branch_label: str = "v7"
    expected_augmentation_profile: str = "v7_rps_pose"
    expected_profile_metadata_key: str = "v7_rps_pose_profile"


def write_v7_remote_training_handoff(config: V7RemoteTrainingHandoffConfig) -> dict[str, object]:
    """Write remote training handoff artifacts without starting SSH or training."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    training_status = _training_config_status(config.training_config_path, expected_dataset_root=config.dataset_root)
    dataset_status = _dataset_status(
        config.dataset_root,
        expected_generated_per_target=config.expected_generated_per_target,
        expected_augmentation_profile=config.expected_augmentation_profile,
        expected_profile_metadata_key=config.expected_profile_metadata_key,
    )
    status, blocking_stage, next_action = _overall_status(training_status=training_status, dataset_status=dataset_status)
    summary = {
        "status": status,
        "blocking_stage": blocking_stage,
        "next_action": next_action,
        "training_config": config.training_config_path.as_posix(),
        "dataset_root": config.dataset_root.as_posix(),
        "output_root": config.output_root.as_posix(),
        "branch_label": config.branch_label,
        "expected_generated_per_target": config.expected_generated_per_target,
        "expected_augmentation_profile": config.expected_augmentation_profile,
        "expected_profile_metadata_key": config.expected_profile_metadata_key,
        "remote": {"host": config.remote_host, "workspace": config.remote_workspace},
        "training_config_status": training_status,
        "dataset_status": dataset_status,
        "commands": _commands(config),
        "notes": [
            "This handoff does not run SSH, rsync, training, validation, or profile promotion.",
            "Full v7 training is TCN-primary on the confirmed A6000 remote workspace.",
            "GRU is listed only as a smoke/regression command.",
        ],
    }
    _write_outputs(config.output_root, summary)
    return summary


def _training_config_status(path: Path, *, expected_dataset_root: Path) -> dict[str, object]:
    if not path.exists():
        return {"status": "missing", "path": path.as_posix(), "failures": [{"code": "missing_training_config"}]}
    try:
        sweep_config = load_sweep_config(path)
    except Exception as exc:  # pragma: no cover - defensive; load_sweep_config error types vary by parser.
        return {"status": "invalid", "path": path.as_posix(), "failures": [{"code": "config_load_failed", "message": str(exc)}]}
    failures: list[dict[str, object]] = []
    configured_dataset = Path(str(sweep_config.get("dataset_root", "")))
    if configured_dataset != expected_dataset_root:
        failures.append(
            {
                "code": "dataset_root_mismatch",
                "configured_dataset_root": configured_dataset.as_posix(),
                "expected_dataset_root": expected_dataset_root.as_posix(),
            }
        )
    preferred = str(sweep_config.get("preferred_export_model", "gru"))
    if preferred != "tcn":
        failures.append({"code": "preferred_export_model_not_tcn", "preferred_export_model": preferred})
    seeds = [int(seed) for seed in _sequence(sweep_config.get("seeds", []))]
    if seeds != [11, 17, 23]:
        failures.append({"code": "unexpected_tcn_seeds", "seeds": seeds, "expected": [11, 17, 23]})
    models = sweep_config.get("models", {})
    model_names = sorted(str(key) for key in models.keys()) if isinstance(models, Mapping) else []
    if "tcn" not in model_names:
        failures.append({"code": "missing_tcn_model_config", "models": model_names})
    if "gru" not in model_names:
        failures.append({"code": "missing_gru_smoke_config", "models": model_names})
    return {
        "status": "passed" if not failures else "invalid",
        "path": path.as_posix(),
        "configured_dataset_root": configured_dataset.as_posix(),
        "preferred_export_model": preferred,
        "seeds": seeds,
        "models": model_names,
        "best_profile": sweep_config.get("best_profile"),
        "runs_dir": sweep_config.get("runs_dir"),
        "comparison_path": sweep_config.get("comparison_path"),
        "failures": failures,
    }


def _dataset_status(
    dataset_root: Path,
    *,
    expected_generated_per_target: int,
    expected_augmentation_profile: str,
    expected_profile_metadata_key: str,
) -> dict[str, object]:
    validation_path = dataset_root / "validation_summary.json"
    generation_config_path = dataset_root / "generation_config.json"
    metadata_path = dataset_root / "sample_metadata.jsonl"
    if not dataset_root.exists():
        return {
            "status": "missing",
            "dataset_root": dataset_root.as_posix(),
            "validation_summary": validation_path.as_posix(),
            "failures": [{"code": "missing_v7_dataset_root"}],
        }
    if not validation_path.exists():
        return {
            "status": "present_without_validation_summary",
            "dataset_root": dataset_root.as_posix(),
            "validation_summary": validation_path.as_posix(),
            "failures": [{"code": "missing_validation_summary"}],
        }
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "invalid_validation_summary",
            "dataset_root": dataset_root.as_posix(),
            "validation_summary": validation_path.as_posix(),
            "failures": [{"code": "invalid_validation_summary_json", "message": str(exc)}],
        }
    passed = validation.get("status") == "passed" or validation.get("passed") is True
    failures: list[dict[str, object]] = [] if passed else [{"code": "dataset_validation_not_passed", "validation_status": validation.get("status")}]
    failures.extend(
        _v7_dataset_handoff_contract_failures(
            dataset_root=dataset_root,
            validation=validation,
            expected_generated_per_target=expected_generated_per_target,
            expected_augmentation_profile=expected_augmentation_profile,
            expected_profile_metadata_key=expected_profile_metadata_key,
        )
    )
    return {
        "status": "passed" if not failures else "invalid",
        "dataset_root": dataset_root.as_posix(),
        "validation_summary": validation_path.as_posix(),
        "generation_config": generation_config_path.as_posix(),
        "sample_metadata": metadata_path.as_posix(),
        "target_counts": validation.get("target_counts"),
        "failures": failures,
    }


def _v7_dataset_handoff_contract_failures(
    *,
    dataset_root: Path,
    validation: Mapping[str, object],
    expected_generated_per_target: int,
    expected_augmentation_profile: str,
    expected_profile_metadata_key: str,
) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    failures.extend(_v7_shard_presence_failures(dataset_root))
    target_counts = validation.get("target_counts")
    if isinstance(target_counts, Mapping):
        counts = {str(key): int(value) for key, value in target_counts.items()}
        if set(counts) != {"rock", "paper", "scissors"}:
            failures.append({"code": "unexpected_v7_target_counts", "target_counts": counts})
        elif len(set(counts.values())) != 1:
            failures.append({"code": "unbalanced_v7_target_counts", "target_counts": counts})
        elif any(value < expected_generated_per_target for value in counts.values()):
            failures.append(
                {
                    "code": "insufficient_v7_target_counts",
                    "target_counts": counts,
                    "expected_min_per_target": expected_generated_per_target,
                }
            )
    else:
        failures.append({"code": "missing_validation_target_counts"})

    generation_config_path = dataset_root / "generation_config.json"
    generation_config = _read_json_object(generation_config_path)
    if generation_config is None:
        failures.append({"code": "missing_generation_config", "path": generation_config_path.as_posix()})
    else:
        if generation_config.get("augmentation_profile") != expected_augmentation_profile:
            failures.append(
                {
                    "code": "generation_config_not_v7_rps_pose",
                    "augmentation_profile": generation_config.get("augmentation_profile"),
                    "expected_augmentation_profile": expected_augmentation_profile,
                }
            )
        generated_per_target = _optional_int(generation_config.get("generated_per_target"))
        if generated_per_target != expected_generated_per_target:
            failures.append(
                {
                    "code": "unexpected_generated_per_target",
                    "generated_per_target": generated_per_target,
                    "expected_generated_per_target": expected_generated_per_target,
                }
            )
        if not generation_config.get("v7_seed_package_root"):
            failures.append({"code": "missing_v7_seed_package_root"})

    metadata_path = dataset_root / "sample_metadata.jsonl"
    if not metadata_path.exists():
        failures.append({"code": "missing_sample_metadata", "path": metadata_path.as_posix()})
        return failures

    v7_profile_seen = False
    reviewed_seed_seen = False
    metadata_rows = 0
    for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append({"code": "invalid_sample_metadata_json", "line": line_number, "message": str(exc)})
            continue
        if not isinstance(record, Mapping):
            failures.append({"code": "invalid_sample_metadata_record", "line": line_number})
            continue
        metadata_rows += 1
        if record.get(expected_profile_metadata_key) is True or record.get("augmentation_profile") == expected_augmentation_profile:
            v7_profile_seen = True
        if record.get("v7_reviewed_real_seed_anchor") is True or record.get("source_name") == "v7_real_rps_seed":
            reviewed_seed_seen = True
        for key in ("source_path", "seed_package_root", "source_seed_package_root"):
            value = record.get(key)
            if isinstance(value, str) and _contains_heldout_test_component(value):
                failures.append({"code": "heldout_metadata_path", "line": line_number, "field": key, "value": value})
    if metadata_rows == 0:
        failures.append({"code": "empty_sample_metadata", "path": metadata_path.as_posix()})
    if not v7_profile_seen:
        failures.append(
            {
                "code": "missing_v7_rps_pose_metadata",
                "expected_augmentation_profile": expected_augmentation_profile,
                "expected_profile_metadata_key": expected_profile_metadata_key,
            }
        )
    if not reviewed_seed_seen:
        failures.append({"code": "missing_reviewed_v7_seed_metadata"})
    return failures


def _v7_shard_presence_failures(dataset_root: Path) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    shard_index_path = dataset_root / "shard_index.csv"
    if not shard_index_path.exists():
        failures.append({"code": "missing_shard_index", "path": shard_index_path.as_posix()})
    else:
        failures.extend(_indexed_shard_failures(dataset_root=dataset_root, shard_index_path=shard_index_path))
    for split in ("train", "val", "test"):
        split_dir = dataset_root / "shards" / split
        shard_paths = sorted(split_dir.glob("*.npz")) if split_dir.exists() else []
        if not shard_paths:
            failures.append({"code": "missing_v7_shard_split", "split": split, "path": split_dir.as_posix()})
    return failures


def _indexed_shard_failures(*, dataset_root: Path, shard_index_path: Path) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    indexed_splits: set[str] = set()
    try:
        with shard_index_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except csv.Error as exc:
        return [{"code": "invalid_shard_index_csv", "path": shard_index_path.as_posix(), "message": str(exc)}]
    if not rows:
        return [{"code": "empty_shard_index", "path": shard_index_path.as_posix()}]
    for row_number, row in enumerate(rows, start=2):
        split = str(row.get("split", "")).strip()
        path_text = str(row.get("path", "")).strip()
        if split:
            indexed_splits.add(split)
        if split not in {"train", "val", "test"}:
            failures.append({"code": "invalid_shard_index_split", "row": row_number, "split": split})
        if not path_text:
            failures.append({"code": "missing_shard_index_path", "row": row_number, "split": split})
            continue
        shard_path = _resolve_indexed_shard_path(dataset_root=dataset_root, path_text=path_text)
        if shard_path.suffix.lower() != ".npz":
            failures.append({"code": "indexed_shard_not_npz", "row": row_number, "split": split, "path": path_text})
        if not shard_path.exists():
            failures.append({"code": "missing_indexed_v7_shard", "row": row_number, "split": split, "path": path_text})
        if not _path_within(shard_path, dataset_root):
            failures.append(
                {
                    "code": "indexed_v7_shard_outside_dataset_root",
                    "row": row_number,
                    "split": split,
                    "path": path_text,
                    "resolved_path": shard_path.as_posix(),
                    "dataset_root": dataset_root.as_posix(),
                }
            )
    missing_index_splits = [split for split in ("train", "val", "test") if split not in indexed_splits]
    if missing_index_splits:
        failures.append({"code": "missing_shard_index_split_rows", "missing_splits": missing_index_splits})
    return failures


def _resolve_indexed_shard_path(*, dataset_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    dataset_relative = dataset_root / path
    if dataset_relative.exists():
        return dataset_relative
    return Path.cwd() / path


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _contains_heldout_test_component(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    return "/test/" in normalized or normalized.endswith("/test")


def _read_json_object(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _optional_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _overall_status(
    *,
    training_status: Mapping[str, object],
    dataset_status: Mapping[str, object],
) -> tuple[str, str | None, str]:
    if training_status.get("status") != "passed":
        return "invalid_training_config", "training_config", "repair_v7_training_config_before_remote_training"
    if dataset_status.get("status") != "passed":
        return "awaiting_v7_dataset", "dataset", "generate_the_approved_v7_rps_pose_dataset_before_remote_training"
    return "ready_for_remote_tcn_training", None, "sync_or_verify_workspace_then_run_remote_tcn_training"


def _commands(config: V7RemoteTrainingHandoffConfig) -> dict[str, str]:
    remote_prefix = f"ssh {config.remote_host} 'cd {config.remote_workspace} && PYTHONPATH=src python -m"
    use_branch_paths = (
        config.expected_augmentation_profile != "v7_rps_pose"
        or config.expected_profile_metadata_key != "v7_rps_pose_profile"
        or config.branch_label != "v7"
    )
    config_path = config.training_config_path.as_posix() if use_branch_paths else DEFAULT_V7_CONFIG.as_posix()
    dataset_path = config.dataset_root.as_posix() if use_branch_paths else DEFAULT_V7_DATASET.as_posix()
    output_root = config.output_root.as_posix() if use_branch_paths else DEFAULT_OUTPUT_ROOT.as_posix()
    expected_args = ""
    if use_branch_paths:
        expected_args = (
            f" --branch-label {config.branch_label}"
            f" --expected-augmentation-profile {config.expected_augmentation_profile}"
            f" --expected-profile-metadata-key {config.expected_profile_metadata_key}"
        )
    return {
        "local_status": "python -m embodied_rps.tools.write_v7_remote_training_handoff",
        "sync_config_and_dataset": (
            f"rsync -avR {config_path} {dataset_path} "
            f"{config.remote_host}:{config.remote_workspace}/"
        ),
        "remote_preflight_status": (
            f"{remote_prefix} embodied_rps.tools.write_v7_remote_training_handoff "
            f"--training-config {config_path} "
            f"--dataset-root {dataset_path} "
            f"--output-root {output_root}{expected_args}'"
        ),
        "remote_smoke_train_gru": (
            f"{remote_prefix} embodied_rps.tools.train_real_skeleton_predictor "
            f"--config {config_path} --model gru --smoke --max-runs 1'"
        ),
        "remote_full_train_tcn": (
            f"{remote_prefix} embodied_rps.tools.train_real_skeleton_predictor "
            f"--config {config_path} --model tcn'"
        ),
        "remote_training_gate": f"{remote_prefix} embodied_rps.tools.run_v7_training_gate'",
    }


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return []


def _write_outputs(output_root: Path, summary: Mapping[str, object]) -> None:
    json_path = output_root / "v7_remote_training_handoff.json"
    md_path = output_root / "v7_remote_training_handoff.md"
    json_path.write_text(json.dumps(dict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")


def _markdown(summary: Mapping[str, object]) -> str:
    remote = summary.get("remote")
    remote_map = remote if isinstance(remote, Mapping) else {}
    lines = [
        "# V7 Remote Training Handoff",
        "",
        f"- status: `{summary.get('status')}`",
        f"- blocking stage: `{summary.get('blocking_stage')}`",
        f"- next action: `{summary.get('next_action')}`",
        f"- remote host: `{remote_map.get('host')}`",
        f"- remote workspace: `{remote_map.get('workspace')}`",
        "",
        "## Commands",
        "",
    ]
    commands = summary.get("commands")
    if isinstance(commands, Mapping):
        for name, command in commands.items():
            lines.extend([f"### {name}", "", "```powershell", str(command), "```", ""])
    lines.extend(["## Notes", ""])
    notes = summary.get("notes")
    if isinstance(notes, Sequence) and not isinstance(notes, (str, bytes)):
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


__all__ = ["V7RemoteTrainingHandoffConfig", "write_v7_remote_training_handoff"]
