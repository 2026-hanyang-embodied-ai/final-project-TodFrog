"""Strict training, validation, and promotion gate reporter for v7 RPS pose models."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embodied_rps.real_skeleton_training import load_real_skeleton_dataset, load_sweep_config, real_dataset_summary
from embodied_rps.v7_rps_seed_package import audit_v7_segment_review_readiness


def discover_v7_validation_roots(dataset_search_root: Path = Path("D:/dataset")) -> dict[str, object]:
    """Discover original20 and heldout15 validation roots by globbing under a dataset root."""

    search_root = dataset_search_root.expanduser()
    if not search_root.exists():
        return {
            "status": "missing_dataset_search_root",
            "dataset_search_root": search_root.as_posix(),
            "original20_root": None,
            "heldout15_root": None,
        }

    candidates: list[dict[str, object]] = []
    for test_dir in sorted(path for path in search_root.rglob("test") if path.is_dir()):
        heldout_count = _mp4_count(test_dir)
        original_root = test_dir.parent
        original_count = _mp4_count_excluding(original_root, excluded_root=test_dir)
        label_counts = {
            label: _mp4_count(test_dir / label)
            for label in ("rock", "paper", "scissors")
            if (test_dir / label).exists()
        }
        candidate = {
            "original20_root": original_root.as_posix(),
            "heldout15_root": test_dir.as_posix(),
            "original20_mp4_count": original_count,
            "heldout15_mp4_count": heldout_count,
            "heldout_label_counts": label_counts,
        }
        candidates.append(candidate)
        if original_count >= 20 and heldout_count >= 15:
            return {
                "status": "passed",
                "dataset_search_root": search_root.as_posix(),
                **candidate,
                "candidates": candidates,
            }

    return {
        "status": "validation_roots_not_found",
        "dataset_search_root": search_root.as_posix(),
        "original20_root": None,
        "heldout15_root": None,
        "candidates": candidates,
    }


@dataclass(frozen=True)
class V7TrainingGateConfig:
    """Configuration for the v7 strict gate reporter."""

    seed_package_root: Path
    dataset_root: Path
    training_config_path: Path
    output_root: Path
    profile_json_path: Path
    original20_validation_root: Path
    heldout15_validation_root: Path
    archived_live_replay_root: Path
    approved_segment_replay_root: Path
    fresh_live_retake_root: Path
    event_manifest_path: Path
    original20_root: Path | None = None
    heldout15_root: Path | None = None
    validation_root_discovery: Mapping[str, object] | None = None
    validation_roots_are_discovered: bool = False
    expected_generated_per_target: int = 10000
    branch_label: str = "v7"
    expected_augmentation_profile: str = "v7_rps_pose"
    expected_profile_metadata_key: str = "v7_rps_pose_profile"


def _strict_gate_failure_next_action(config: V7TrainingGateConfig) -> str:
    if config.branch_label == "v7":
        return "keep_v4_live_policy_preserve_diagnostics_and_plan_targeted_v7b_simulation_branch"
    return f"keep_v4_live_policy_preserve_diagnostics_and_plan_next_targeted_branch_after_{config.branch_label}"


def run_v7_training_gate(config: V7TrainingGateConfig) -> dict[str, object]:
    """Report v7 readiness without running training, validation, or promotion."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    completed_stages: list[str] = []
    stage_outputs: dict[str, object] = {}

    seed_status = _seed_package_status(config.seed_package_root)
    stage_outputs["segment_review"] = seed_status.get("segment_review", {})
    stage_outputs["v7_seed_package"] = seed_status
    if seed_status["status"] == "awaiting_manual_segment_approval":
        summary = _summary(
            config=config,
            status="awaiting_manual_segment_approval",
            next_action="review and approve selected v7 segment rows before building the seed package",
            blocking_stage="segment_review",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
        )
        _write_summary(config.output_root, summary)
        return summary
    if seed_status["status"] == "ready_for_seed_package_build":
        summary = _summary(
            config=config,
            status="ready_for_v7_seed_package_build",
            next_action="run_v7_post_review_pipeline_with_execute_dataset_generation_after_review",
            blocking_stage="seed_package",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
        )
        _write_summary(config.output_root, summary)
        return summary
    if seed_status["status"] != "passed":
        summary = _summary(
            config=config,
            status="awaiting_v7_seed_package",
            next_action="build_or_repair_reviewed_v7_seed_package",
            blocking_stage="seed_package",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
        )
        _write_summary(config.output_root, summary)
        return summary
    completed_stages.append("v7_seed_package")

    if not config.training_config_path.exists():
        summary = _summary(
            config=config,
            status="awaiting_training_config",
            next_action="create_or_restore_v7_training_config",
            blocking_stage="training_config",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
        )
        _write_summary(config.output_root, summary)
        return summary

    sweep_config = load_sweep_config(config.training_config_path)
    configured_dataset_root = Path(str(sweep_config.get("dataset_root", config.dataset_root.as_posix())))
    if configured_dataset_root != config.dataset_root:
        stage_outputs["training_config"] = {
            "status": "dataset_root_mismatch",
            "configured_dataset_root": configured_dataset_root.as_posix(),
            "runner_dataset_root": config.dataset_root.as_posix(),
        }
        summary = _summary(
            config=config,
            status="training_config_dataset_mismatch",
            next_action="align_v7_training_config_dataset_root",
            blocking_stage="training_config",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
            sweep_config=sweep_config,
        )
        _write_summary(config.output_root, summary)
        return summary

    dataset_status = _dataset_status(
        config.dataset_root,
        expected_generated_per_target=config.expected_generated_per_target,
        expected_augmentation_profile=config.expected_augmentation_profile,
        expected_profile_metadata_key=config.expected_profile_metadata_key,
    )
    stage_outputs["v7_dataset"] = dataset_status
    if dataset_status["status"] != "passed":
        summary = _summary(
            config=config,
            status="awaiting_v7_dataset",
            next_action="generate_approved_v7_rps_pose_dataset_after_segment_approval",
            blocking_stage="dataset",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
            sweep_config=sweep_config,
        )
        _write_summary(config.output_root, summary)
        return summary
    completed_stages.append("v7_dataset")

    profile_status = _profile_status(config.profile_json_path)
    stage_outputs["v7_profile"] = profile_status
    if profile_status["status"] != "passed":
        summary = _summary(
            config=config,
            status="ready_for_v7_tcn_training",
            next_action="train and export the v7 TCN ensemble; GRU remains smoke/regression only",
            blocking_stage="model_training",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
            sweep_config=sweep_config,
        )
        _write_summary(config.output_root, summary)
        return summary
    completed_stages.append("v7_profile")

    original_status = _validation_status(
        config.original20_validation_root / "validation_summary.json",
        expected_clip_count=20,
        expected_passed_count=20,
    )
    heldout_status = _validation_status(
        config.heldout15_validation_root / "validation_summary.json",
        expected_clip_count=15,
        expected_passed_count=15,
    )
    stage_outputs["original20_strict_validation"] = original_status
    stage_outputs["heldout15_strict_validation"] = heldout_status
    if original_status["status"] == "missing" or heldout_status["status"] == "missing":
        summary = _summary(
            config=config,
            status="ready_for_strict_video_validation",
            next_action="run_original20_then_heldout15_v7_validation_without_training_on_heldout",
            blocking_stage="strict_video_validation",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
            sweep_config=sweep_config,
        )
        _write_summary(config.output_root, summary)
        return summary
    if original_status["status"] != "passed" or heldout_status["status"] != "passed":
        summary = _summary(
            config=config,
            status="v7_strict_gates_failed",
            next_action=_strict_gate_failure_next_action(config),
            blocking_stage="strict_video_validation",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
            sweep_config=sweep_config,
        )
        _write_summary(config.output_root, summary)
        return summary
    completed_stages.extend(["original20_strict_validation", "heldout15_strict_validation"])

    archived_status = _activity_gate_status(
        config.archived_live_replay_root,
        ("replay_summary.json", "validation_summary.json"),
        min_event_count=4,
        required_targets=("rock", "paper", "scissors"),
    )
    stage_outputs["archived_live_replay"] = archived_status
    if archived_status["status"] == "missing":
        summary = _summary(
            config=config,
            status="ready_for_archived_live_replay",
            next_action="replay_archived_live_rock_paper_scissors_runs_with_v7_profile",
            blocking_stage="archived_live_replay",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
            sweep_config=sweep_config,
        )
        _write_summary(config.output_root, summary)
        return summary
    if archived_status["status"] != "passed":
        summary = _summary(
            config=config,
            status="v7_strict_gates_failed",
            next_action=_strict_gate_failure_next_action(config),
            blocking_stage="archived_live_replay",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
            sweep_config=sweep_config,
        )
        _write_summary(config.output_root, summary)
        return summary
    completed_stages.append("archived_live_replay")

    approved_segment_sample_ids = [str(sample_id) for sample_id in _strings_from_sequence(seed_status.get("sample_ids"))]
    approved_segment_min_event_count = max(1, len(approved_segment_sample_ids) or _optional_int(seed_status.get("seed_count")) or 1)
    segment_status = _activity_gate_status(
        config.approved_segment_replay_root,
        ("replay_summary.json", "validation_summary.json"),
        min_event_count=approved_segment_min_event_count,
        required_event_ids=approved_segment_sample_ids,
    )
    stage_outputs["approved_segment_replay"] = segment_status
    if segment_status["status"] == "missing":
        summary = _summary(
            config=config,
            status="ready_for_approved_segment_replay",
            next_action="replay_approved_v7_segments_before_fresh_live_capture",
            blocking_stage="approved_segment_replay",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
            sweep_config=sweep_config,
        )
        _write_summary(config.output_root, summary)
        return summary
    if segment_status["status"] != "passed":
        summary = _summary(
            config=config,
            status="v7_strict_gates_failed",
            next_action=_strict_gate_failure_next_action(config),
            blocking_stage="approved_segment_replay",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
            sweep_config=sweep_config,
        )
        _write_summary(config.output_root, summary)
        return summary
    completed_stages.append("approved_segment_replay")

    fresh_status = _activity_gate_status(
        config.fresh_live_retake_root,
        ("retake_summary.json", "validation_summary.json"),
        min_event_count=3,
        required_targets=("rock", "paper", "scissors"),
    )
    stage_outputs["fresh_live_retakes"] = fresh_status
    if fresh_status["status"] == "missing":
        summary = _summary(
            config=config,
            status="ready_for_fresh_live_retakes",
            next_action="prepare_fresh_ground_truthed_live_retakes_for_rock_paper_scissors",
            blocking_stage="fresh_live_retakes",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
            sweep_config=sweep_config,
        )
        _write_summary(config.output_root, summary)
        return summary
    if fresh_status["status"] != "passed":
        summary = _summary(
            config=config,
            status="v7_strict_gates_failed",
            next_action=_strict_gate_failure_next_action(config),
            blocking_stage="fresh_live_retakes",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
            sweep_config=sweep_config,
        )
        _write_summary(config.output_root, summary)
        return summary
    completed_stages.append("fresh_live_retakes")

    summary = _summary(
        config=config,
        status="v7_promotion_candidate",
        next_action="manually_review_gate_evidence_then_promote_v7_only_if_it_beats_v4_live_behavior",
        blocking_stage=None,
        completed_stages=completed_stages,
        stage_outputs=stage_outputs,
        sweep_config=sweep_config,
    )
    _write_summary(config.output_root, summary)
    return summary


def _seed_package_status(seed_package_root: Path) -> dict[str, object]:
    seed_npz = seed_package_root / "v7_rps_seed_dataset.npz"
    seed_metadata = seed_package_root / "seed_metadata.jsonl"
    seed_quality_summary = seed_package_root / "seed_quality_summary.csv"
    seed_summary = seed_package_root / "seed_package_summary.json"
    required_files = {
        "missing_seed_npz": seed_npz,
        "missing_seed_metadata": seed_metadata,
        "missing_seed_quality_summary": seed_quality_summary,
        "missing_seed_package_summary": seed_summary,
    }
    package_artifact_exists = any(path.exists() for path in required_files.values())
    if package_artifact_exists:
        missing_failures = [
            {"code": code, "path": path.as_posix()}
            for code, path in required_files.items()
            if not path.exists()
        ]
        base_status = {
            "seed_npz": seed_npz.as_posix(),
            "seed_metadata": seed_metadata.as_posix(),
            "seed_quality_summary": seed_quality_summary.as_posix(),
            "summary": seed_summary.as_posix(),
            "required_files": {code: path.as_posix() for code, path in required_files.items()},
        }
        if not seed_summary.exists():
            return {"status": "invalid", **base_status, "failures": missing_failures}
        try:
            summary = json.loads(seed_summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {
                "status": "invalid",
                **base_status,
                "failures": [*missing_failures, {"code": "invalid_seed_package_summary", "message": str(exc)}],
            }
        if summary.get("status") == "passed":
            if missing_failures:
                return {
                    "status": "invalid",
                    **base_status,
                    "seed_package_status": summary.get("status"),
                    "seed_count": summary.get("seed_count"),
                    "target_counts": summary.get("target_counts"),
                    "failures": missing_failures,
                }
            sample_ids, sample_id_failure = _seed_npz_sample_ids(seed_npz)
            if sample_id_failure is not None:
                return {
                    "status": "invalid",
                    **base_status,
                    "seed_package_status": summary.get("status"),
                    "seed_count": summary.get("seed_count"),
                    "target_counts": summary.get("target_counts"),
                    "failures": [sample_id_failure],
                }
            return {
                "status": "passed",
                **base_status,
                "seed_count": summary.get("seed_count"),
                "target_counts": summary.get("target_counts"),
                "sample_ids": sample_ids,
            }
        return {
            "status": "invalid",
            **base_status,
            "seed_package_status": summary.get("status"),
            "failures": missing_failures,
        }
    if (seed_package_root / "proposed_segments.jsonl").exists() or (seed_package_root / "segment_review_manifest.csv").exists():
        review = audit_v7_segment_review_readiness(output_root=seed_package_root)
        return {
            "status": str(review.get("status", "invalid")).replace("ready_for_seed_package_build", "ready_for_seed_package_build"),
            "seed_npz": seed_npz.as_posix(),
            "seed_metadata": seed_metadata.as_posix(),
            "seed_quality_summary": seed_quality_summary.as_posix(),
            "summary": seed_summary.as_posix(),
            "seed_npz_exists": seed_npz.exists(),
            "segment_review": review,
        }
    return {
        "status": "missing",
        "seed_npz": seed_npz.as_posix(),
        "summary": seed_summary.as_posix(),
        "failures": [{"code": "missing_v7_seed_package"}],
    }


def _mp4_count(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*.mp4") if path.is_file())


def _mp4_count_excluding(root: Path, *, excluded_root: Path) -> int:
    if not root.exists():
        return 0
    excluded = excluded_root.resolve(strict=False)
    count = 0
    for path in root.rglob("*.mp4"):
        if not path.is_file():
            continue
        resolved = path.resolve(strict=False)
        if excluded in resolved.parents or resolved == excluded:
            continue
        count += 1
    return count


def _dataset_status(
    dataset_root: Path,
    *,
    expected_generated_per_target: int,
    expected_augmentation_profile: str,
    expected_profile_metadata_key: str,
) -> dict[str, object]:
    if not dataset_root.exists():
        return {"status": "missing", "dataset_root": dataset_root.as_posix(), "failures": [{"code": "missing_dataset_root"}]}
    try:
        dataset = load_real_skeleton_dataset(dataset_root)
        summary = real_dataset_summary(dataset, dataset_root)
        failures = _v7_dataset_contract_failures(
            dataset_root=dataset_root,
            dataset=dataset,
            summary=summary,
            expected_generated_per_target=expected_generated_per_target,
            expected_augmentation_profile=expected_augmentation_profile,
            expected_profile_metadata_key=expected_profile_metadata_key,
        )
        return {
            "status": "passed" if not failures else "invalid",
            "dataset_root": dataset_root.as_posix(),
            "summary": summary,
            "failures": failures,
        }
    except (FileNotFoundError, ValueError) as exc:
        return {"status": "invalid", "dataset_root": dataset_root.as_posix(), "failures": [{"code": "dataset_load_failed", "message": str(exc)}]}


def _v7_dataset_contract_failures(
    *,
    dataset_root: Path,
    dataset: object,
    summary: Mapping[str, object],
    expected_generated_per_target: int,
    expected_augmentation_profile: str,
    expected_profile_metadata_key: str,
) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    label_counts = dict(_mapping(summary.get("label_counts", {})))
    split_counts = dict(_mapping(summary.get("split_counts", {})))
    if set(label_counts) != {"rock", "paper", "scissors"}:
        failures.append({"code": "unexpected_v7_labels", "label_counts": label_counts})
    elif len(set(int(value) for value in label_counts.values())) != 1:
        failures.append({"code": "unbalanced_v7_target_counts", "label_counts": label_counts})
    elif any(int(value) < expected_generated_per_target for value in label_counts.values()):
        failures.append(
            {
                "code": "insufficient_v7_target_counts",
                "label_counts": label_counts,
                "expected_min_per_target": expected_generated_per_target,
            }
        )
    missing_splits = [split for split in ("train", "val", "test") if int(split_counts.get(split, 0)) <= 0]
    if missing_splits:
        failures.append({"code": "missing_v7_splits", "missing_splits": missing_splits, "split_counts": split_counts})

    validation_path = dataset_root / "validation_summary.json"
    validation = _read_json_object(validation_path)
    if validation is None:
        failures.append({"code": "missing_or_invalid_validation_summary", "path": validation_path.as_posix()})
    else:
        if validation.get("status") != "passed":
            failures.append({"code": "validation_summary_not_passed", "status": validation.get("status")})
        validation_counts = dict(_mapping(validation.get("target_counts", {})))
        if validation_counts and validation_counts != label_counts:
            failures.append(
                {
                    "code": "validation_target_counts_mismatch",
                    "validation_target_counts": validation_counts,
                    "loaded_label_counts": label_counts,
                }
            )

    generation_config_path = dataset_root / "generation_config.json"
    generation_config = _read_json_object(generation_config_path)
    if generation_config is None:
        failures.append({"code": "missing_or_invalid_generation_config", "path": generation_config_path.as_posix()})
    else:
        if generation_config.get("augmentation_profile") != expected_augmentation_profile:
            failures.append(
                {
                    "code": "wrong_augmentation_profile",
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

    metadata_path = dataset_root / "sample_metadata.jsonl"
    metadata_status = _v7_metadata_status(
        metadata_path,
        expected_augmentation_profile=expected_augmentation_profile,
        expected_profile_metadata_key=expected_profile_metadata_key,
    )
    if metadata_status["status"] != "passed":
        failures.extend(metadata_status["failures"])
    return failures


def _v7_metadata_status(
    metadata_path: Path,
    *,
    expected_augmentation_profile: str,
    expected_profile_metadata_key: str,
) -> dict[str, object]:
    if not metadata_path.exists():
        return {"status": "failed", "failures": [{"code": "missing_sample_metadata", "path": metadata_path.as_posix()}]}
    failures: list[dict[str, object]] = []
    source_counts: Counter[str] = Counter()
    v7_profile_count = 0
    reviewed_seed_count = 0
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
        source_name = str(record.get("source_name", ""))
        source_counts[source_name] += 1
        if record.get(expected_profile_metadata_key) is True or record.get("augmentation_profile") == expected_augmentation_profile:
            v7_profile_count += 1
        if source_name == "v7_real_rps_seed" or record.get("v7_reviewed_real_seed_anchor") is True:
            reviewed_seed_count += 1
        for key in ("source_path", "seed_package_root", "source_seed_package_root"):
            value = str(record.get(key, ""))
            if _contains_heldout_test_component(value):
                failures.append({"code": "heldout_metadata_path", "line": line_number, "field": key, "value": value})
    if v7_profile_count <= 0:
        failures.append(
            {
                "code": "missing_v7_rps_pose_metadata",
                "expected_augmentation_profile": expected_augmentation_profile,
                "expected_profile_metadata_key": expected_profile_metadata_key,
            }
        )
    if reviewed_seed_count <= 0:
        failures.append({"code": "missing_reviewed_v7_seed_metadata"})
    return {
        "status": "passed" if not failures else "failed",
        "source_counts": dict(sorted(source_counts.items())),
        "v7_profile_count": v7_profile_count,
        "expected_augmentation_profile": expected_augmentation_profile,
        "expected_profile_metadata_key": expected_profile_metadata_key,
        "reviewed_seed_count": reviewed_seed_count,
        "failures": failures,
    }


def _contains_heldout_test_component(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    return "/test/" in normalized or normalized.endswith("/test")


def _read_json_object(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return dict(loaded) if isinstance(loaded, Mapping) else None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _optional_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _seed_npz_sample_ids(seed_npz: Path) -> tuple[list[str], dict[str, object] | None]:
    try:
        with np.load(seed_npz, allow_pickle=False) as data:
            if "sample_ids" not in data.files:
                return [], {"code": "missing_seed_sample_ids", "path": seed_npz.as_posix()}
            sample_ids = [str(item) for item in np.asarray(data["sample_ids"]).reshape(-1).tolist()]
    except (OSError, ValueError) as exc:
        return [], {"code": "invalid_seed_npz", "path": seed_npz.as_posix(), "message": str(exc)}
    if not sample_ids:
        return [], {"code": "empty_seed_sample_ids", "path": seed_npz.as_posix()}
    if len(set(sample_ids)) != len(sample_ids):
        return [], {"code": "duplicate_seed_sample_ids", "path": seed_npz.as_posix()}
    return sample_ids, None


def _profile_status(profile_json_path: Path) -> dict[str, object]:
    profile_pt_path = profile_json_path.with_suffix(".pt")
    if not profile_json_path.exists() or not profile_pt_path.exists():
        return {
            "status": "missing",
            "profile_json": profile_json_path.as_posix(),
            "profile_pt": profile_pt_path.as_posix(),
            "json_exists": profile_json_path.exists(),
            "pt_exists": profile_pt_path.exists(),
        }
    try:
        profile = json.loads(profile_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "invalid", "profile_json": profile_json_path.as_posix(), "failure": str(exc)}
    model = str(profile.get("model", ""))
    label_names = profile.get("label_names")
    if model != "tcn":
        return {
            "status": "wrong_model",
            "profile_json": profile_json_path.as_posix(),
            "profile_pt": profile_pt_path.as_posix(),
            "model": model,
            "label_names": label_names,
        }
    if label_names != ["rock", "paper", "scissors"]:
        return {
            "status": "invalid_labels",
            "profile_json": profile_json_path.as_posix(),
            "profile_pt": profile_pt_path.as_posix(),
            "model": model,
            "label_names": label_names,
        }
    return {
        "status": "passed",
        "profile_json": profile_json_path.as_posix(),
        "profile_pt": profile_pt_path.as_posix(),
        "profile_name": profile.get("profile_name"),
        "model": model,
        "label_names": label_names,
        "selected_accuracy": profile.get("selected_accuracy"),
        "selected_macro_f1": profile.get("selected_macro_f1"),
    }


def _validation_status(summary_path: Path, *, expected_clip_count: int, expected_passed_count: int) -> dict[str, object]:
    if not summary_path.exists():
        return {"status": "missing", "summary_path": summary_path.as_posix()}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "invalid", "summary_path": summary_path.as_posix(), "failure": str(exc), "passed": False}
    passed = bool(summary.get("passed"))
    clip_count = summary.get("clip_count")
    passed_count = summary.get("passed_count")
    strict_passed = passed and clip_count == expected_clip_count and passed_count == expected_passed_count
    return {
        "status": "passed" if strict_passed else "failed",
        "summary_path": summary_path.as_posix(),
        "passed": passed,
        "strict_passed": strict_passed,
        "passed_count": passed_count,
        "failed_count": summary.get("failed_count"),
        "clip_count": clip_count,
        "expected_clip_count": expected_clip_count,
        "expected_passed_count": expected_passed_count,
        "per_class": summary.get("per_class"),
        "event_manifest_written": summary.get("event_manifest_written"),
    }


def _generic_gate_status(root: Path, filenames: Sequence[str]) -> dict[str, object]:
    for filename in filenames:
        path = root / filename
        if path.exists():
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return {"status": "invalid", "summary_path": path.as_posix(), "failure": str(exc), "passed": False}
            passed = bool(summary.get("passed")) or summary.get("status") == "passed"
            return {
                "status": "passed" if passed else "failed",
                "summary_path": path.as_posix(),
                "passed": passed,
                "failed_count": summary.get("failed_count"),
                "details": summary,
            }
    return {"status": "missing", "root": root.as_posix(), "expected_filenames": list(filenames)}


def _activity_gate_status(
    root: Path,
    filenames: Sequence[str],
    *,
    min_event_count: int,
    required_targets: Sequence[str] = (),
    required_event_ids: Sequence[str] = (),
) -> dict[str, object]:
    status = _generic_gate_status(root, filenames)
    if status["status"] == "missing":
        return status
    if status["status"] == "invalid":
        return status

    summary = _mapping(status.get("details"))
    passed = bool(status.get("passed"))
    event_count = _first_int(
        summary,
        (
            "clip_count",
            "entry_count",
            "attempt_count",
            "retake_count",
            "replay_count",
            "segment_count",
            "sample_count",
            "event_count",
        ),
    )
    passed_count = _first_int(summary, ("passed_count", "success_count", "correct_count"))
    failed_count = _optional_int(summary.get("failed_count"))
    observed_targets = _observed_gate_targets(summary)
    observed_event_ids = _observed_gate_event_ids(summary)

    failures: list[dict[str, object]] = []
    if not passed:
        failures.append({"code": "gate_not_passed", "passed": passed})
    if event_count is None:
        failures.append({"code": "missing_event_count", "min_event_count": min_event_count})
    elif event_count < min_event_count:
        failures.append(
            {"code": "insufficient_event_count", "event_count": event_count, "min_event_count": min_event_count}
        )
    if passed_count is not None and event_count is not None and passed_count != event_count:
        failures.append({"code": "passed_count_mismatch", "passed_count": passed_count, "event_count": event_count})
    if failed_count not in (None, 0):
        failures.append({"code": "nonzero_failed_count", "failed_count": failed_count})

    required_target_set = {str(target) for target in required_targets}
    missing_targets = sorted(required_target_set - observed_targets)
    if missing_targets:
        failures.append(
            {
                "code": "missing_required_targets",
                "missing_targets": missing_targets,
                "observed_targets": sorted(observed_targets),
            }
        )
    required_event_id_set = {str(event_id) for event_id in required_event_ids}
    missing_event_ids = sorted(required_event_id_set - observed_event_ids) if event_count is not None else []
    if missing_event_ids:
        failures.append(
            {
                "code": "missing_required_event_ids",
                "missing_event_ids": missing_event_ids,
                "observed_event_ids": sorted(observed_event_ids),
            }
        )

    strict_passed = not failures
    return {
        **status,
        "status": "passed" if strict_passed else "failed",
        "strict_passed": strict_passed,
        "event_count": event_count,
        "min_event_count": min_event_count,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "required_targets": list(required_targets),
        "observed_targets": sorted(observed_targets),
        "required_event_ids": list(required_event_ids),
        "observed_event_ids": sorted(observed_event_ids),
        "failures": failures,
    }


def _first_int(mapping: Mapping[str, object], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = _optional_int(mapping.get(key))
        if value is not None:
            return value
    return None


def _observed_gate_targets(summary: Mapping[str, object]) -> set[str]:
    targets: set[str] = set()
    for key in ("per_class", "target_counts", "gesture_counts", "label_counts"):
        value = summary.get(key)
        if isinstance(value, Mapping):
            for target, evidence in value.items():
                if _has_positive_gate_evidence(evidence):
                    targets.add(str(target))
    for key in ("targets", "gestures", "labels", "target_names", "gesture_names", "label_names"):
        targets.update(_strings_from_sequence(summary.get(key)))
    for key in ("entries", "clips", "attempts", "retakes", "replays", "events", "segments"):
        entries = summary.get(key)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for entry in entries:
            if isinstance(entry, Mapping):
                for target_key in ("target_name", "target", "gesture", "label", "expected_label", "ground_truth"):
                    target = entry.get(target_key)
                    if isinstance(target, str) and target:
                        targets.add(target)
    return targets


def _observed_gate_event_ids(summary: Mapping[str, object]) -> set[str]:
    event_ids: set[str] = set()
    for key in ("sample_ids", "segment_ids", "event_ids", "replay_ids", "source_sample_ids"):
        event_ids.update(_strings_from_sequence(summary.get(key)))
    for key in ("per_sample", "per_segment", "sample_results", "segment_results"):
        value = summary.get(key)
        if isinstance(value, Mapping):
            event_ids.update(str(item) for item in value.keys() if str(item))
    for key in ("entries", "clips", "attempts", "retakes", "replays", "events", "segments", "samples"):
        entries = summary.get(key)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for entry in entries:
            if isinstance(entry, Mapping):
                for event_key in (
                    "sample_id",
                    "segment_id",
                    "event_id",
                    "replay_id",
                    "source_sample_id",
                    "approved_sample_id",
                ):
                    event_id = entry.get(event_key)
                    if isinstance(event_id, str) and event_id:
                        event_ids.add(event_id)
    return event_ids


def _has_positive_gate_evidence(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, Mapping):
        for key in ("clip_count", "entry_count", "attempt_count", "retake_count", "replay_count", "passed_count", "count"):
            count = _optional_int(value.get(key))
            if count is not None:
                return count > 0
        return bool(value)
    return value is not None


def _strings_from_sequence(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    strings: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item or item in seen:
            continue
        strings.append(item)
        seen.add(item)
    return strings


def _summary(
    *,
    config: V7TrainingGateConfig,
    status: str,
    next_action: str,
    blocking_stage: str | None,
    completed_stages: list[str],
    stage_outputs: Mapping[str, object],
    sweep_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "next_action": next_action,
        "blocking_stage": blocking_stage,
        "branch_label": config.branch_label,
        "seed_package_root": config.seed_package_root.as_posix(),
        "dataset_root": config.dataset_root.as_posix(),
        "training_config": config.training_config_path.as_posix(),
        "profile_json": config.profile_json_path.as_posix(),
        "original20_validation_root": config.original20_validation_root.as_posix(),
        "heldout15_validation_root": config.heldout15_validation_root.as_posix(),
        "archived_live_replay_root": config.archived_live_replay_root.as_posix(),
        "approved_segment_replay_root": config.approved_segment_replay_root.as_posix(),
        "fresh_live_retake_root": config.fresh_live_retake_root.as_posix(),
        "event_manifest_path": config.event_manifest_path.as_posix(),
        "expected_generated_per_target": config.expected_generated_per_target,
        "expected_augmentation_profile": config.expected_augmentation_profile,
        "expected_profile_metadata_key": config.expected_profile_metadata_key,
        "validation_root_discovery": dict(config.validation_root_discovery) if config.validation_root_discovery is not None else None,
        "completed_stages": list(completed_stages),
        "stage_outputs": dict(stage_outputs),
        "promotion_decision": {
            "may_promote_v7": status == "v7_promotion_candidate",
            "fallback_policy": "keep_v4_live_policy" if status != "v7_promotion_candidate" else None,
            "promotion_rule": "promote only when original20, heldout15, archived replay, approved segment replay, and fresh live retakes pass",
        },
        "commands": _commands(config, sweep_config=sweep_config),
        "notes": [
            "This runner does not train, validate videos, launch live capture, promote profiles, or start SCHUNK/Isaac packaging.",
            "Held-out 15 MP4s remain validation-only and must not enter v7 seed metadata or training shards.",
            "GRU is smoke/regression only for v7; promotion requires the exported TCN profile.",
        ],
    }


def _commands(config: V7TrainingGateConfig, *, sweep_config: Mapping[str, object] | None) -> dict[str, object]:
    training_config = config.training_config_path.as_posix()
    profile_json = config.profile_json_path.as_posix()
    commands: dict[str, object] = {
        "post_review_execute": [
            "python",
            "-m",
            "embodied_rps.tools.run_v7_post_review_pipeline",
            "--review-root",
            config.seed_package_root.as_posix(),
            "--dataset-output-root",
            config.dataset_root.as_posix(),
            "--execute-dataset-generation",
        ],
        "smoke_train_gru": [
            "python",
            "-m",
            "embodied_rps.tools.train_real_skeleton_predictor",
            "--config",
            training_config,
            "--model",
            "gru",
            "--smoke",
            "--max-runs",
            "1",
        ],
        "full_train_tcn": [
            "python",
            "-m",
            "embodied_rps.tools.train_real_skeleton_predictor",
            "--config",
            training_config,
            "--model",
            "tcn",
        ],
        "validate_original20": _validation_command(
            profile_json=profile_json,
            input_root=None if config.validation_roots_are_discovered else config.original20_root,
            input_root_placeholder="<original20-from-D:/dataset-glob>",
            output_root=config.original20_validation_root,
            event_output=config.event_manifest_path,
            expected_count=20,
            label_mode="transition",
        ),
        "validate_heldout15": _validation_command(
            profile_json=profile_json,
            input_root=None if config.validation_roots_are_discovered else config.heldout15_root,
            input_root_placeholder="<heldout15-from-D:/dataset-glob>",
            output_root=config.heldout15_validation_root,
            event_output=None,
            expected_count=15,
            label_mode="final-label",
        ),
    }
    if sweep_config is not None:
        commands["runs_dir"] = str(sweep_config.get("runs_dir", ""))
        commands["comparison_path"] = str(sweep_config.get("comparison_path", ""))
    return commands


def _validation_command(
    *,
    profile_json: str,
    input_root: Path | None,
    input_root_placeholder: str,
    output_root: Path,
    event_output: Path | None,
    expected_count: int,
    label_mode: str,
) -> list[str]:
    command = [
        "python",
        "-m",
        "embodied_rps.tools.evaluate_real_skeleton_video_predictions",
        "--profile",
        profile_json,
        "--input-root",
        input_root.as_posix() if input_root is not None else input_root_placeholder,
        "--output-root",
        output_root.as_posix(),
        "--expected-count",
        str(expected_count),
        "--label-mode",
        label_mode,
    ]
    if event_output is not None:
        command.extend(["--event-output", event_output.as_posix()])
    return command


def _write_summary(output_root: Path, summary: Mapping[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "v7_training_gate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "v7_training_gate_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")


def _summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V7 Training Gate Summary",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Next action: `{summary.get('next_action')}`",
        f"- Blocking stage: `{summary.get('blocking_stage')}`",
        f"- Seed package root: `{summary.get('seed_package_root')}`",
        f"- Dataset root: `{summary.get('dataset_root')}`",
        f"- Profile JSON: `{summary.get('profile_json')}`",
        "",
        "## Completed Stages",
        "",
    ]
    completed = summary.get("completed_stages")
    if isinstance(completed, list) and completed:
        for stage in completed:
            lines.append(f"- `{stage}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Stage Outputs", ""])
    stage_outputs = summary.get("stage_outputs")
    if isinstance(stage_outputs, Mapping) and stage_outputs:
        for stage, output in stage_outputs.items():
            if isinstance(output, Mapping):
                lines.append(f"- `{stage}`: status=`{output.get('status')}`")
            else:
                lines.append(f"- `{stage}`")
    else:
        lines.append("- None.")
    commands = summary.get("commands")
    if isinstance(commands, Mapping):
        lines.extend(["", "## Commands", ""])
        for name in ("post_review_execute", "smoke_train_gru", "full_train_tcn", "validate_original20", "validate_heldout15"):
            command = commands.get(name)
            if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
                lines.extend([f"### {name}", "", "```powershell", _quote_command([str(part) for part in command]), "```", ""])
    lines.extend(["## Notes", ""])
    notes = summary.get("notes")
    if isinstance(notes, list):
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _quote_command(parts: Sequence[str]) -> str:
    quoted: list[str] = []
    for part in parts:
        if not part or any(char.isspace() for char in part) or any(ord(char) > 127 for char in part) or "/" in part or "\\" in part or ":" in part:
            quoted.append(f"'{part}'")
        else:
            quoted.append(part)
    return " ".join(quoted)


__all__ = ["V7TrainingGateConfig", "discover_v7_validation_roots", "run_v7_training_gate"]
