"""Fail-closed remote training preflight for the v7d two-stage branch."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from embodied_rps.real_skeleton_training import load_sweep_config
from embodied_rps.v7d_post_approval_preflight import V7DPostApprovalPreflightConfig

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
REMOTE_HOST = "voice@166.104.167.133"
REMOTE_WORKSPACE = "/home/voice/workspace/chominkyu/embodied-final"
EXPECTED_TCN_SEEDS = [11, 17, 23]
EXPECTED_AUGMENTATION_PROFILE = "v7d_real_seeded_prompt_window_guard"
EXPECTED_GENERATED_PER_TARGET = 10000
EXPECTED_MIN_THREE_CLASS_SAMPLE_COUNT = EXPECTED_GENERATED_PER_TARGET * 3

V7DRemoteTrainingPreflightStatus = Literal[
    "blocked_local_smoke_not_ready",
    "blocked_training_configs_invalid",
    "blocked_two_stage_datasets_invalid",
    "blocked_remote_sync_manifest_invalid",
    "ready_for_remote_tcn_training",
]


@dataclass(frozen=True)
class V7DRemoteTrainingPreflightConfig:
    """Inputs for a status-only v7d remote training preflight."""

    project_root: Path = field(default_factory=Path.cwd)
    output_root: Path = Path("artifacts/real_skeleton_v7d_remote_training_preflight_20260618")
    local_smoke_preflight_root: Path = Path("artifacts/real_skeleton_v7d_local_smoke_preflight_20260618")
    three_class_dataset_root: Path = V7DPostApprovalPreflightConfig.three_class_dataset_root
    stage1_dataset_root: Path = V7DPostApprovalPreflightConfig.stage1_dataset_root
    stage2_dataset_root: Path = V7DPostApprovalPreflightConfig.stage2_dataset_root
    stage1_training_config: Path = V7DPostApprovalPreflightConfig.stage1_training_config
    stage2_training_config: Path = V7DPostApprovalPreflightConfig.stage2_training_config
    remote_host: str = REMOTE_HOST
    remote_workspace: str = REMOTE_WORKSPACE


def write_v7d_remote_training_preflight(config: V7DRemoteTrainingPreflightConfig) -> dict[str, object]:
    """Write a non-training handoff artifact for v7d remote TCN training."""

    project_root = config.project_root.resolve()
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    local_smoke_status = _local_smoke_status(project_root=project_root, config=config)
    stage1_training_status = _training_config_status(
        project_root=project_root,
        path=config.stage1_training_config,
        expected_dataset_root=config.stage1_dataset_root,
        stage_name="stage1_rock_transition",
    )
    stage2_training_status = _training_config_status(
        project_root=project_root,
        path=config.stage2_training_config,
        expected_dataset_root=config.stage2_dataset_root,
        stage_name="stage2_paper_scissors",
    )
    artifact_status = _artifact_status(project_root=project_root, config=config)
    remote_sync_manifest = _remote_sync_manifest(project_root=project_root, config=config)
    status, blocking_stage, next_action = _overall_status(
        local_smoke_status=local_smoke_status,
        stage1_training_status=stage1_training_status,
        stage2_training_status=stage2_training_status,
        artifact_status=artifact_status,
        remote_sync_manifest=remote_sync_manifest,
    )
    summary: dict[str, object] = {
        "status": status,
        "blocking_stage": blocking_stage,
        "next_action": next_action,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "remote": {"host": config.remote_host, "workspace": config.remote_workspace},
        "local_smoke_status": local_smoke_status,
        "three_class_dataset_root": _display_path(
            _resolve_path(project_root, config.three_class_dataset_root),
            base=project_root,
        ),
        "stage1_dataset_root": _display_path(_resolve_path(project_root, config.stage1_dataset_root), base=project_root),
        "stage2_dataset_root": _display_path(_resolve_path(project_root, config.stage2_dataset_root), base=project_root),
        "stage1_training_config": _display_path(_resolve_path(project_root, config.stage1_training_config), base=project_root),
        "stage2_training_config": _display_path(_resolve_path(project_root, config.stage2_training_config), base=project_root),
        "stage1_training_config_status": stage1_training_status,
        "stage2_training_config_status": stage2_training_status,
        "artifact_status": artifact_status,
        "remote_sync_manifest": remote_sync_manifest,
        "planned_commands": _planned_commands(project_root=project_root, config=config, remote_sync_manifest=remote_sync_manifest),
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test MP4s remain validation-only and must not enter seed packages or training metadata",
        "notes": [
            "This preflight writes handoff artifacts only; it does not run rsync, SSH, local training, remote training, validation, or promotion.",
            "Remote TCN training is allowed only after the local v7d GRU/TCN smoke preflight is ready.",
            "The promotion candidate remains the two-stage TCN ensemble; GRU is local smoke/regression only.",
        ],
        "config": _config_summary(project_root=project_root, config=config),
    }
    (output_root / "v7d_remote_training_preflight_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7d_remote_training_preflight_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _local_smoke_status(*, project_root: Path, config: V7DRemoteTrainingPreflightConfig) -> dict[str, object]:
    root = _resolve_path(project_root, config.local_smoke_preflight_root)
    summary_path = root / "v7d_local_smoke_preflight_summary.json"
    if not summary_path.exists():
        return {
            "status": "missing",
            "summary_path": _display_path(summary_path, base=project_root),
            "failures": [{"code": "missing_local_smoke_preflight_summary"}],
        }
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "invalid",
            "summary_path": _display_path(summary_path, base=project_root),
            "failures": [{"code": "invalid_local_smoke_preflight_summary_json", "message": str(exc)}],
        }
    if not isinstance(summary, Mapping):
        return {
            "status": "invalid",
            "summary_path": _display_path(summary_path, base=project_root),
            "failures": [{"code": "invalid_local_smoke_preflight_summary_record"}],
        }
    status = str(summary.get("status", "unknown"))
    failures: list[dict[str, object]] = []
    if status != "ready_for_local_smoke":
        failures.append({"code": "local_smoke_not_ready", "local_smoke_status": status})
    for key in ("training_started", "remote_training_started", "validation_started"):
        if summary.get(key) is True:
            failures.append({"code": f"unexpected_{key}", key: True})
    return {
        "status": status,
        "summary_path": _display_path(summary_path, base=project_root),
        "failures": failures,
    }


def _training_config_status(
    *,
    project_root: Path,
    path: Path,
    expected_dataset_root: Path,
    stage_name: str,
) -> dict[str, object]:
    resolved_path = _resolve_path(project_root, path)
    expected_dataset = _resolve_path(project_root, expected_dataset_root)
    if not resolved_path.exists():
        return {
            "status": "missing",
            "stage_name": stage_name,
            "path": _display_path(resolved_path, base=project_root),
            "failures": [{"code": "missing_training_config"}],
        }
    try:
        sweep_config = load_sweep_config(resolved_path)
    except Exception as exc:  # pragma: no cover - loader error details depend on parser internals.
        return {
            "status": "invalid",
            "stage_name": stage_name,
            "path": _display_path(resolved_path, base=project_root),
            "failures": [{"code": "config_load_failed", "message": str(exc)}],
        }

    failures: list[dict[str, object]] = []
    configured_dataset = _resolve_path(project_root, Path(str(sweep_config.get("dataset_root", ""))))
    if configured_dataset.resolve(strict=False) != expected_dataset.resolve(strict=False):
        failures.append(
            {
                "code": "dataset_root_mismatch",
                "configured_dataset_root": _display_path(configured_dataset, base=project_root),
                "expected_dataset_root": _display_path(expected_dataset, base=project_root),
            }
        )
    preferred = str(sweep_config.get("preferred_export_model", "gru"))
    if preferred != "tcn":
        failures.append({"code": "preferred_export_model_not_tcn", "preferred_export_model": preferred})
    seeds = [int(seed) for seed in _sequence(sweep_config.get("seeds", []))]
    if seeds != EXPECTED_TCN_SEEDS:
        failures.append({"code": "unexpected_tcn_seeds", "seeds": seeds, "expected": EXPECTED_TCN_SEEDS})
    models = sweep_config.get("models", {})
    model_names = sorted(str(key) for key in models.keys()) if isinstance(models, Mapping) else []
    if "tcn" not in model_names:
        failures.append({"code": "missing_tcn_model_config", "models": model_names})
    if "gru" not in model_names:
        failures.append({"code": "missing_gru_smoke_config", "models": model_names})
    return {
        "status": "passed" if not failures else "invalid",
        "stage_name": stage_name,
        "path": _display_path(resolved_path, base=project_root),
        "configured_dataset_root": _display_path(configured_dataset, base=project_root),
        "preferred_export_model": preferred,
        "seeds": seeds,
        "models": model_names,
        "best_profile": sweep_config.get("best_profile"),
        "runs_dir": _display_value(sweep_config.get("runs_dir"), base=project_root),
        "comparison_path": _display_value(sweep_config.get("comparison_path"), base=project_root),
        "failures": failures,
    }


def _artifact_status(*, project_root: Path, config: V7DRemoteTrainingPreflightConfig) -> dict[str, object]:
    three_class_root = _resolve_path(project_root, config.three_class_dataset_root)
    stage1_root = _resolve_path(project_root, config.stage1_dataset_root)
    stage2_root = _resolve_path(project_root, config.stage2_dataset_root)
    root_failures = [
        *_path_contract_failures(project_root=project_root, path=stage1_root, field="stage1_dataset_root"),
        *_path_contract_failures(project_root=project_root, path=stage2_root, field="stage2_dataset_root"),
    ]
    stage1_status = _remap_summary_status(
        project_root=project_root,
        root=stage1_root,
        expected_source_root=three_class_root,
        expected_mode="rock_vs_transition",
        failure_prefix="stage1",
    )
    stage2_status = _remap_summary_status(
        project_root=project_root,
        root=stage2_root,
        expected_source_root=three_class_root,
        expected_mode="paper_vs_scissors",
        failure_prefix="stage2",
    )
    failures = root_failures + list(stage1_status.get("failures", [])) + list(stage2_status.get("failures", []))
    return {
        "three_class_dataset_root_exists": three_class_root.exists(),
        "stage1_dataset_root_exists": stage1_root.exists(),
        "stage1_remap_summary_exists": (stage1_root / "remap_summary.json").exists(),
        "stage1_remap_summary_status": stage1_status.get("status"),
        "stage1_remap_mode": stage1_status.get("mode"),
        "stage1_sample_count": stage1_status.get("sample_count"),
        "stage1_remap_source_root": stage1_status.get("source_root"),
        "stage1_source_generation_summary": stage1_status.get("source_generation_summary"),
        "stage1_source_generated_per_target": stage1_status.get("source_generated_per_target"),
        "stage1_source_augmentation_profile": stage1_status.get("source_augmentation_profile"),
        "stage1_source_sample_count": stage1_status.get("source_sample_count"),
        "stage2_dataset_root_exists": stage2_root.exists(),
        "stage2_remap_summary_exists": (stage2_root / "remap_summary.json").exists(),
        "stage2_remap_summary_status": stage2_status.get("status"),
        "stage2_remap_mode": stage2_status.get("mode"),
        "stage2_sample_count": stage2_status.get("sample_count"),
        "stage2_remap_source_root": stage2_status.get("source_root"),
        "stage2_source_generation_summary": stage2_status.get("source_generation_summary"),
        "stage2_source_generated_per_target": stage2_status.get("source_generated_per_target"),
        "stage2_source_augmentation_profile": stage2_status.get("source_augmentation_profile"),
        "stage2_source_sample_count": stage2_status.get("source_sample_count"),
        "stage1_training_config_exists": _resolve_path(project_root, config.stage1_training_config).exists(),
        "stage2_training_config_exists": _resolve_path(project_root, config.stage2_training_config).exists(),
        "failures": failures,
    }


def _remap_summary_status(
    *,
    project_root: Path,
    root: Path,
    expected_source_root: Path,
    expected_mode: str,
    failure_prefix: str,
) -> dict[str, object]:
    path = root / "remap_summary.json"
    if not path.exists():
        return {
            "status": "missing",
            "mode": None,
            "sample_count": None,
            "source_root": None,
            "source_generation_summary": None,
            "source_generated_per_target": None,
            "source_augmentation_profile": None,
            "source_sample_count": None,
            "failures": [{"code": f"missing_{failure_prefix}_remap_summary", "path": _display_path(path, base=project_root)}],
        }
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "invalid",
            "mode": None,
            "sample_count": None,
            "source_root": None,
            "source_generation_summary": None,
            "source_generated_per_target": None,
            "source_augmentation_profile": None,
            "source_sample_count": None,
            "failures": [{"code": f"invalid_{failure_prefix}_remap_summary_json", "message": str(exc)}],
        }
    if not isinstance(summary, Mapping):
        return {
            "status": "invalid",
            "mode": None,
            "sample_count": None,
            "source_root": None,
            "source_generation_summary": None,
            "source_generated_per_target": None,
            "source_augmentation_profile": None,
            "source_sample_count": None,
            "failures": [{"code": f"invalid_{failure_prefix}_remap_summary_record"}],
        }
    failures: list[dict[str, object]] = []
    mode = summary.get("mode")
    if mode != expected_mode:
        failures.append({"code": f"unexpected_{failure_prefix}_remap_mode", "mode": mode, "expected_mode": expected_mode})
    sample_count = _optional_int(summary.get("sample_count"))
    if sample_count is None or sample_count <= 0:
        failures.append({"code": f"invalid_{failure_prefix}_sample_count", "sample_count": sample_count})
    raw_status = summary.get("status")
    if raw_status not in (None, "passed"):
        failures.append({"code": f"{failure_prefix}_remap_summary_not_passed", "status": raw_status})
    source_root = summary.get("source_root")
    source_root_display: str | None = None
    generation_status: dict[str, object] = {
        "summary_path": None,
        "generated_per_target": None,
        "augmentation_profile": None,
        "sample_count": None,
        "failures": [],
    }
    if isinstance(source_root, str) and _contains_heldout_test_component(source_root):
        failures.append(
            {
                "code": f"heldout_test_path_in_{failure_prefix}_remap_summary",
                "field": "source_root",
                "value": _redact_host_path(source_root),
            }
        )
    if isinstance(source_root, str):
        source_path = _resolve_path(project_root, Path(source_root))
        source_root_display = _display_path(source_path, base=project_root)
        if _contains_sandbox_component(source_root):
            failures.append(
                {
                    "code": f"sandbox_path_in_{failure_prefix}_remap_source_root",
                    "field": "source_root",
                    "value": source_root_display,
                }
            )
        if source_path.resolve(strict=False) != expected_source_root.resolve(strict=False):
            failures.append(
                {
                    "code": f"unexpected_{failure_prefix}_remap_source_root",
                    "source_root": source_root_display,
                    "expected_source_root": _display_path(expected_source_root, base=project_root),
                }
            )
        generation_status = _generation_contract_status(
            project_root=project_root,
            source_root=source_path,
            failure_prefix=f"{failure_prefix}_source",
        )
        failures.extend(_sequence_of_mappings(generation_status.get("failures")))
    else:
        failures.append({"code": f"missing_{failure_prefix}_remap_source_root"})
    return {
        "status": "passed" if not failures else "invalid",
        "mode": mode,
        "sample_count": sample_count,
        "source_root": source_root_display,
        "source_generation_summary": generation_status.get("summary_path"),
        "source_generated_per_target": generation_status.get("generated_per_target"),
        "source_augmentation_profile": generation_status.get("augmentation_profile"),
        "source_sample_count": generation_status.get("sample_count"),
        "failures": failures,
    }


def _generation_contract_status(*, project_root: Path, source_root: Path, failure_prefix: str) -> dict[str, object]:
    summary_path = _first_existing(source_root / "generation_summary.json", source_root / "run_summary.json")
    config_path = source_root / "generation_config.json"
    failures: list[dict[str, object]] = []
    summary: Mapping[str, object] = {}
    generation_config: Mapping[str, object] = {}

    if summary_path is None:
        failures.append(
            {
                "code": f"missing_{failure_prefix}_generation_summary",
                "path": _display_path(source_root / "run_summary.json", base=project_root),
            }
        )
    else:
        parsed_summary, parse_failure = _load_json_mapping(summary_path)
        if parse_failure is not None:
            failures.append({"code": f"invalid_{failure_prefix}_generation_summary", **parse_failure})
        else:
            summary = parsed_summary

    if config_path.exists():
        parsed_config, parse_failure = _load_json_mapping(config_path)
        if parse_failure is not None:
            failures.append({"code": f"invalid_{failure_prefix}_generation_config", **parse_failure})
        else:
            generation_config = parsed_config
    else:
        failures.append(
            {
                "code": f"missing_{failure_prefix}_generation_config",
                "path": _display_path(config_path, base=project_root),
            }
        )

    raw_status = summary.get("status")
    if raw_status not in (None, "passed"):
        failures.append({"code": f"{failure_prefix}_generation_not_passed", "status": raw_status})
    generated_per_target = _optional_int(summary.get("generated_per_target"))
    if generated_per_target is None:
        generated_per_target = _optional_int(generation_config.get("generated_per_target"))
    if generated_per_target != EXPECTED_GENERATED_PER_TARGET:
        failures.append(
            {
                "code": f"unexpected_{failure_prefix}_generated_per_target",
                "generated_per_target": generated_per_target,
                "expected_generated_per_target": EXPECTED_GENERATED_PER_TARGET,
            }
        )
    augmentation_profile = summary.get("augmentation_profile")
    if not isinstance(augmentation_profile, str):
        augmentation_profile = generation_config.get("augmentation_profile")
    if augmentation_profile != EXPECTED_AUGMENTATION_PROFILE:
        failures.append(
            {
                "code": f"unexpected_{failure_prefix}_augmentation_profile",
                "augmentation_profile": augmentation_profile,
                "expected_augmentation_profile": EXPECTED_AUGMENTATION_PROFILE,
            }
        )
    sample_count = _optional_int(summary.get("sample_count"))
    validation = summary.get("validation")
    if sample_count is None and isinstance(validation, Mapping):
        sample_count = _optional_int(validation.get("sample_count"))
    if sample_count is None or sample_count < EXPECTED_MIN_THREE_CLASS_SAMPLE_COUNT:
        failures.append(
            {
                "code": f"insufficient_{failure_prefix}_sample_count",
                "sample_count": sample_count,
                "minimum_sample_count": EXPECTED_MIN_THREE_CLASS_SAMPLE_COUNT,
            }
        )
    return {
        "summary_path": _display_path(summary_path, base=project_root) if summary_path is not None else None,
        "generated_per_target": generated_per_target,
        "augmentation_profile": augmentation_profile,
        "sample_count": sample_count,
        "failures": failures,
    }


def _overall_status(
    *,
    local_smoke_status: Mapping[str, object],
    stage1_training_status: Mapping[str, object],
    stage2_training_status: Mapping[str, object],
    artifact_status: Mapping[str, object],
    remote_sync_manifest: Mapping[str, object],
) -> tuple[V7DRemoteTrainingPreflightStatus, str | None, str]:
    local_failures = local_smoke_status.get("failures")
    if local_smoke_status.get("status") != "ready_for_local_smoke" or _nonempty(local_failures):
        return (
            "blocked_local_smoke_not_ready",
            "local_smoke",
            "run and pass local v7d GRU/TCN smoke before remote sync or training",
        )
    if stage1_training_status.get("status") != "passed" or stage2_training_status.get("status") != "passed":
        return (
            "blocked_training_configs_invalid",
            "training_config",
            "repair v7d two-stage TCN configs before remote sync or training",
        )
    if _nonempty(artifact_status.get("failures")):
        return (
            "blocked_two_stage_datasets_invalid",
            "two_stage_dataset",
            "repair or regenerate v7d two-stage remap datasets before remote sync or training",
        )
    if remote_sync_manifest.get("status") != "passed" or _nonempty(remote_sync_manifest.get("failures")):
        return (
            "blocked_remote_sync_manifest_invalid",
            "remote_sync_manifest",
            "repair the v7d remote sync manifest before rsync or remote training",
        )
    return "ready_for_remote_tcn_training", None, "sync v7d code/configs/datasets to the remote A6000 and train TCN seeds [11, 17, 23]"


def _planned_commands(
    *,
    project_root: Path,
    config: V7DRemoteTrainingPreflightConfig,
    remote_sync_manifest: Mapping[str, object],
) -> dict[str, str]:
    stage1_config = _display_path(_resolve_path(project_root, config.stage1_training_config), base=project_root)
    stage2_config = _display_path(_resolve_path(project_root, config.stage2_training_config), base=project_root)
    stage1_dataset = _display_path(_resolve_path(project_root, config.stage1_dataset_root), base=project_root)
    stage2_dataset = _display_path(_resolve_path(project_root, config.stage2_dataset_root), base=project_root)
    output_root = _display_path(_resolve_path(project_root, config.output_root), base=project_root)
    local_smoke_root = _display_path(_resolve_path(project_root, config.local_smoke_preflight_root), base=project_root)
    remote_prefix = f"ssh {config.remote_host} 'cd {config.remote_workspace} && PYTHONPATH=src python -m"
    sync_paths = _sequence_of_strings(remote_sync_manifest.get("included_paths"))
    return {
        "local_status": "python -m embodied_rps.tools.write_v7d_remote_training_preflight",
        "sync_after_local_gates": (
            f"rsync -avR {' '.join(sync_paths)} "
            f"{config.remote_host}:{config.remote_workspace}/"
        ),
        "remote_preflight": (
            f"{remote_prefix} embodied_rps.tools.write_v7d_remote_training_preflight "
            f"--three-class-dataset-root {_display_path(_resolve_path(project_root, config.three_class_dataset_root), base=project_root)} "
            f"--stage1-dataset-root {stage1_dataset} "
            f"--stage2-dataset-root {stage2_dataset} "
            f"--stage1-training-config {stage1_config} "
            f"--stage2-training-config {stage2_config} "
            f"--local-smoke-preflight-root {local_smoke_root} "
            f"--output-root {output_root}'"
        ),
        "remote_train_tcn_stage1": (
            f"{remote_prefix} embodied_rps.tools.train_real_skeleton_predictor "
            f"--config {stage1_config} --model tcn'"
        ),
        "remote_train_tcn_stage2": (
            f"{remote_prefix} embodied_rps.tools.train_real_skeleton_predictor "
            f"--config {stage2_config} --model tcn'"
        ),
        "copy_back_results": (
            f"rsync -av {config.remote_host}:{config.remote_workspace}/results/ results/"
        ),
    }


def _summary_markdown(summary: Mapping[str, object]) -> str:
    remote = summary.get("remote")
    remote_map = remote if isinstance(remote, Mapping) else {}
    lines = [
        "# V7d Remote Training Preflight",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Blocking stage: `{summary.get('blocking_stage')}`",
        f"- Next action: `{summary.get('next_action')}`",
        f"- Remote host: `{remote_map.get('host')}`",
        f"- Remote workspace: `{remote_map.get('workspace')}`",
        f"- Training started: `{summary.get('training_started')}`",
        f"- Remote training started: `{summary.get('remote_training_started')}`",
        f"- Validation started: `{summary.get('validation_started')}`",
        f"- Promotion eligible: `{summary.get('promotion_eligible')}`",
        f"- Remote sync manifest: `{_mapping_value(summary.get('remote_sync_manifest'), 'status')}`",
        "",
        "## Planned Commands",
        "",
    ]
    commands = summary.get("planned_commands", {})
    if isinstance(commands, Mapping):
        for name, command in commands.items():
            lines.extend([f"### {name}", "", "```text", str(command), "```", ""])
    return "\n".join(lines)


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(base.resolve(strict=False)).as_posix()
    except ValueError:
        return path.name


def _display_value(value: object, *, base: Path) -> object:
    if isinstance(value, str):
        return _display_path(_resolve_path(base, Path(value)), base=base)
    return value


def _remote_sync_manifest(*, project_root: Path, config: V7DRemoteTrainingPreflightConfig) -> dict[str, object]:
    items = [
        ("source_code", Path("src")),
        ("training_configs", Path("configs")),
        ("local_smoke_gate_artifact", config.local_smoke_preflight_root),
        ("stage1_dataset", config.stage1_dataset_root),
        ("stage2_dataset", config.stage2_dataset_root),
    ]
    included_paths: list[str] = []
    entries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for kind, raw_path in items:
        path = _resolve_path(project_root, raw_path)
        display = _display_path(path, base=project_root)
        included_paths.append(display)
        item_failures = _sync_path_failures(project_root=project_root, path=path, display=display, kind=kind)
        entries.append({"kind": kind, "path": display, "exists": path.exists(), "failures": item_failures})
        failures.extend(item_failures)
    return {
        "status": "passed" if not failures else "invalid",
        "included_paths": included_paths,
        "entries": entries,
        "excluded_paths": [
            "proposal.pdf",
            "presentation-slides.pdf",
            "dataset heldout test MP4 roots",
            "sandbox rehearsal roots",
        ],
        "failures": failures,
    }


def _sync_path_failures(*, project_root: Path, path: Path, display: str, kind: str) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    if not path.exists():
        failures.append({"code": "missing_remote_sync_path", "kind": kind, "path": display})
    try:
        path.resolve(strict=False).relative_to(project_root.resolve(strict=False))
    except ValueError:
        failures.append({"code": "remote_sync_path_outside_project_root", "kind": kind, "path": _redact_host_path(path.as_posix())})
    if _contains_sandbox_component(display):
        failures.append({"code": "sandbox_path_in_remote_sync_manifest", "kind": kind, "path": display})
    if _contains_heldout_test_component(display):
        failures.append({"code": "heldout_test_path_in_remote_sync_manifest", "kind": kind, "path": _redact_host_path(display)})
    if Path(display).name in {"proposal.pdf", "presentation-slides.pdf"}:
        failures.append({"code": "protected_pdf_in_remote_sync_manifest", "kind": kind, "path": display})
    return failures


def _path_contract_failures(*, project_root: Path, path: Path, field: str) -> list[dict[str, object]]:
    display = _display_path(path, base=project_root)
    failures: list[dict[str, object]] = []
    if _contains_sandbox_component(display):
        failures.append({"code": f"sandbox_path_in_{field}", "field": field, "path": display})
    if _contains_heldout_test_component(display):
        failures.append({"code": f"heldout_test_path_in_{field}", "field": field, "path": _redact_host_path(display)})
    return failures


def _config_summary(*, project_root: Path, config: V7DRemoteTrainingPreflightConfig) -> dict[str, object]:
    summary = asdict(config)
    summary["project_root"] = "."
    for key, value in list(summary.items()):
        if isinstance(value, Path):
            summary[key] = _display_path(_resolve_path(project_root, value), base=project_root)
    return _json_ready(summary)  # type: ignore[return-value]


def _contains_heldout_test_component(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    return "/test/" in normalized or normalized.endswith("/test")


def _contains_sandbox_component(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    return "/sandbox/" in normalized or normalized.endswith("/sandbox") or normalized.startswith("sandbox/")


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _load_json_mapping(path: Path) -> tuple[Mapping[str, object], dict[str, object] | None]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, {"message": str(exc)}
    if not isinstance(parsed, Mapping):
        return {}, {"message": "expected JSON object"}
    return parsed, None


def _redact_host_path(value: str) -> str:
    if ":" in value or value.startswith("/"):
        return Path(value.replace("\\", "/")).name or "<path>"
    return value


def _optional_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return []


def _sequence_of_mappings(value: object) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for item in _sequence(value):
        if isinstance(item, Mapping):
            result.append({str(key): val for key, val in item.items()})
    return result


def _sequence_of_strings(value: object) -> list[str]:
    return [str(item) for item in _sequence(value)]


def _mapping_value(value: object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def _nonempty(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) > 0


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["V7DRemoteTrainingPreflightConfig", "write_v7d_remote_training_preflight"]
