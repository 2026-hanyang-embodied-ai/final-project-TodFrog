"""Fail-closed handoff for v7e stage1 dataset generation and remap."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from embodied_rps.v7e_seed_package_preflight import DEFAULT_V7E_SEED_PACKAGE_ROOT


BRANCH_LABEL = "v7e_stage1_paper_transition_rescue"
V7E_AUGMENTATION_PROFILE = "v7e_stage1_paper_transition_rescue"
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7e_stage1_dataset_handoff_20260619")
DEFAULT_THREE_CLASS_DATASET_ROOT = Path(
    "artifacts/real_guided_three_class_wait_expanded_v7e_stage1_paper_transition_rescue_20260619"
)
DEFAULT_STAGE1_DATASET_ROOT = Path(
    "artifacts/real_guided_two_stage_rock_transition_v7e_stage1_paper_transition_rescue_20260619"
)
DEFAULT_BASE_DATASET_ROOT = Path("artifacts/real_guided_large_sharded_20260610")
DEFAULT_CALIBRATION_SEED_PACKAGE_ROOT = Path("artifacts/real_skeleton_v4_calibration_seed_package_fewshot_20260615")
DEFAULT_LIVE_ROCK_SEED_PACKAGE_ROOT = Path("artifacts/live_rock_false_trigger_overlay_seed_20260616")
DEFAULT_STAGE1_TRAINING_CONFIG = Path(
    "configs/real_skeleton_two_stage_rock_transition_v7e_stage1_paper_transition_rescue_tcn_ensemble.yaml"
)

V7EStage1DatasetHandoffStatus = Literal[
    "blocked_seed_package_missing",
    "blocked_seed_package_not_passed",
    "ready_for_three_class_dataset_generation",
    "ready_for_stage1_remap",
    "ready_for_local_stage1_smoke",
]


@dataclass(frozen=True)
class V7EStage1DatasetHandoffConfig:
    """Configuration for the v7e stage1 dataset handoff."""

    project_root: Path = field(default_factory=Path.cwd)
    seed_package_root: Path = DEFAULT_V7E_SEED_PACKAGE_ROOT
    three_class_dataset_root: Path = DEFAULT_THREE_CLASS_DATASET_ROOT
    stage1_dataset_root: Path = DEFAULT_STAGE1_DATASET_ROOT
    base_dataset_root: Path = DEFAULT_BASE_DATASET_ROOT
    calibration_seed_package_root: Path = DEFAULT_CALIBRATION_SEED_PACKAGE_ROOT
    live_rock_seed_package_root: Path = DEFAULT_LIVE_ROCK_SEED_PACKAGE_ROOT
    stage1_training_config: Path = DEFAULT_STAGE1_TRAINING_CONFIG
    output_root: Path = DEFAULT_OUTPUT_ROOT
    generated_per_target: int = 10000
    sequence_length: int = 72
    min_length: int = 48
    shard_size: int = 512
    base_rock_stride: int = 2
    seed: int = 20260619


def write_v7e_stage1_dataset_handoff(config: V7EStage1DatasetHandoffConfig) -> dict[str, Any]:
    """Write a non-mutating handoff for v7e stage1 dataset generation."""

    project_root = config.project_root.resolve()
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_status = _artifact_status(project_root=project_root, config=config)
    status = _status(artifact_status)
    summary: dict[str, Any] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "artifact_status": artifact_status,
        "augmentation_profile": V7E_AUGMENTATION_PROFILE,
        "generated_per_target": int(config.generated_per_target),
        "stage1_training_scope": "retrain_rock_vs_transition_only",
        "stage2_policy": "reuse_v7d_paper_scissors_stage2_unless_new_evidence_requires_change",
        "planned_commands": _planned_commands(config=config, status=status),
        "dataset_generated": False,
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "heldout15_started": False,
        "promotion_eligible": False,
        "fallback_policy": "keep_v4_live_demo_fallback",
        "heldout_policy": "heldout */test MP4s remain validation-only and are not copied into v7e training metadata",
        "next_action": _next_action(status),
        "config": _config_summary(project_root=project_root, config=config),
    }
    (output_root / "v7e_stage1_dataset_handoff_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7e_stage1_dataset_handoff_summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _artifact_status(*, project_root: Path, config: V7EStage1DatasetHandoffConfig) -> dict[str, Any]:
    seed_root = _resolve_path(project_root, config.seed_package_root)
    three_class_root = _resolve_path(project_root, config.three_class_dataset_root)
    stage1_root = _resolve_path(project_root, config.stage1_dataset_root)
    config_path = _resolve_path(project_root, config.stage1_training_config)
    seed_summary = _read_json_if_exists(seed_root / "v7e_seed_package_summary.json")
    generation_summary = _read_generation_summary(three_class_root)
    remap_summary = _read_json_if_exists(stage1_root / "remap_summary.json")
    return {
        "seed_package_root_exists": seed_root.exists(),
        "seed_package_summary_exists": bool(seed_summary),
        "seed_package_summary_status": str(seed_summary.get("status", "missing")) if seed_summary else "missing",
        "seed_npz_exists": (seed_root / "v7_rps_seed_dataset.npz").exists(),
        "seed_metadata_exists": (seed_root / "seed_metadata.jsonl").exists(),
        "three_class_dataset_root_exists": three_class_root.exists(),
        "three_class_generation_summary_exists": bool(generation_summary),
        "three_class_generation_status": str(generation_summary.get("status", "missing")) if generation_summary else "missing",
        "stage1_dataset_root_exists": stage1_root.exists(),
        "stage1_remap_summary_exists": bool(remap_summary),
        "stage1_remap_status": str(remap_summary.get("status", "missing")) if remap_summary else "missing",
        "stage1_remap_mode": str(remap_summary.get("mode", "missing")) if remap_summary else "missing",
        "stage1_training_config_exists": config_path.exists(),
    }


def _status(paths: Mapping[str, Any]) -> V7EStage1DatasetHandoffStatus:
    if not paths.get("seed_package_root_exists") or not paths.get("seed_npz_exists") or not paths.get("seed_metadata_exists"):
        return "blocked_seed_package_missing"
    if paths.get("seed_package_summary_status") != "passed":
        return "blocked_seed_package_not_passed"
    if not paths.get("three_class_generation_summary_exists"):
        return "ready_for_three_class_dataset_generation"
    if not paths.get("stage1_remap_summary_exists"):
        return "ready_for_stage1_remap"
    if paths.get("stage1_remap_status") != "passed" or paths.get("stage1_remap_mode") != "rock_vs_transition":
        return "ready_for_stage1_remap"
    return "ready_for_local_stage1_smoke"


def _planned_commands(*, config: V7EStage1DatasetHandoffConfig, status: str) -> dict[str, str]:
    if status.startswith("blocked_seed_package"):
        generate_cmd = "blocked until v7e seed package exists"
    else:
        generate_cmd = (
            "python -m embodied_rps.tools.generate_three_class_wait_skeleton_dataset "
            f"--base-dataset-root {config.base_dataset_root.as_posix()} "
            f"--output-root {config.three_class_dataset_root.as_posix()} "
            f"--generated-per-target {config.generated_per_target} "
            f"--base-rock-stride {config.base_rock_stride} "
            f"--sequence-length {config.sequence_length} "
            f"--min-length {config.min_length} "
            f"--shard-size {config.shard_size} "
            f"--seed {config.seed} "
            f"--augmentation-profile {V7E_AUGMENTATION_PROFILE} "
            f"--calibration-seed-package-root {config.calibration_seed_package_root.as_posix()} "
            f"--live-rock-seed-package-root {config.live_rock_seed_package_root.as_posix()} "
            f"--v7-seed-package-root {config.seed_package_root.as_posix()}"
        )
    return {
        "build_seed_package": "python -m embodied_rps.tools.build_v7e_stage1_paper_transition_rescue_seed_package",
        "generate_three_class_dataset": generate_cmd,
        "remap_stage1_rock_transition": (
            "python -m embodied_rps.tools.remap_real_skeleton_dataset "
            f"--source-root {config.three_class_dataset_root.as_posix()} "
            f"--output-root {config.stage1_dataset_root.as_posix()} "
            "--mode rock_vs_transition"
        ),
        "remap_stage2_paper_scissors": "not planned for v7e unless stage2 diagnostics regress",
        "local_gru_smoke_stage1": (
            "python -m embodied_rps.tools.train_real_skeleton_predictor "
            f"--config {config.stage1_training_config.as_posix()} --model gru --smoke --max-runs 1 --skip-export"
        ),
        "local_tcn_smoke_stage1": (
            "python -m embodied_rps.tools.train_real_skeleton_predictor "
            f"--config {config.stage1_training_config.as_posix()} --model tcn --smoke --max-runs 1 --skip-export"
        ),
        "remote_stage1_tcn": "not planned until local v7e stage1 smoke passes",
        "strict_original20_validation": "not planned until remote stage1 profile returns locally",
        "heldout15_validation": "not planned unless original20 reaches 20/20",
    }


def _next_action(status: str) -> str:
    if status == "blocked_seed_package_missing":
        return "approve v7e paper seeds and build the v7e seed package before dataset generation"
    if status == "blocked_seed_package_not_passed":
        return "fix the v7e seed package summary before dataset generation"
    if status == "ready_for_three_class_dataset_generation":
        return "generate the balanced v7e three-class dataset with the v7e augmentation profile"
    if status == "ready_for_stage1_remap":
        return "remap the v7e three-class dataset into rock_vs_transition stage1"
    return "run local v7e stage1 GRU/TCN smoke before remote sync or training"


def _markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# V7e Stage1 Dataset Handoff",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Branch: `{summary.get('branch_label')}`",
            f"- Augmentation profile: `{summary.get('augmentation_profile')}`",
            f"- Stage1 scope: `{summary.get('stage1_training_scope')}`",
            f"- Stage2 policy: `{summary.get('stage2_policy')}`",
            "- Dataset generated: `False`",
            "- Training started: `False`",
            "- Heldout15 started: `False`",
            "- Promotion eligible: `False`",
            f"- Next action: {summary.get('next_action')}",
            "",
        ]
    )


def _config_summary(*, project_root: Path, config: V7EStage1DatasetHandoffConfig) -> dict[str, Any]:
    return {
        "seed_package_root": _display_path(_resolve_path(project_root, config.seed_package_root), base=project_root),
        "three_class_dataset_root": _display_path(_resolve_path(project_root, config.three_class_dataset_root), base=project_root),
        "stage1_dataset_root": _display_path(_resolve_path(project_root, config.stage1_dataset_root), base=project_root),
        "stage1_training_config": _display_path(_resolve_path(project_root, config.stage1_training_config), base=project_root),
        "generated_per_target": int(config.generated_per_target),
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object in {path}")
    return dict(value)


def _read_generation_summary(root: Path) -> dict[str, Any]:
    for filename in ("generation_summary.json", "run_summary.json"):
        summary = _read_json_if_exists(root / filename)
        if summary:
            return summary
    return {}


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "BRANCH_LABEL",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_STAGE1_DATASET_ROOT",
    "DEFAULT_THREE_CLASS_DATASET_ROOT",
    "V7E_AUGMENTATION_PROFILE",
    "V7EStage1DatasetHandoffConfig",
    "write_v7e_stage1_dataset_handoff",
]
