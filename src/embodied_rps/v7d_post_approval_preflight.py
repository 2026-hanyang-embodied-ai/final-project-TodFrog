"""Fail-closed v7d post-approval preflight and handoff commands."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from embodied_rps.v7d_seed_package import V7DSeedReadinessConfig, check_v7d_prompt_pose_seed_readiness

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
APPLY_CONFIRMATION_PHRASE = "reviewed_temporal_evidence_for_v7d"
REMOTE_WORKSPACE = "/home/voice/workspace/chominkyu/embodied-final"
REMOTE_HOST = "voice@166.104.167.133"

V7DPostApprovalPreflightStatus = Literal[
    "blocked_manual_approval_required",
    "ready_for_seed_package_build",
    "ready_for_three_class_dataset_generation",
    "ready_for_two_stage_remap",
    "ready_for_local_smoke_training",
]


@dataclass(frozen=True)
class V7DPostApprovalPreflightConfig:
    """Configuration for the v7d post-approval preflight."""

    project_root: Path = field(default_factory=Path.cwd)
    review_root: Path = Path("artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618")
    shortlist_root: Path = Path("artifacts/real_skeleton_v7d_manual_review_shortlist_20260618")
    selection_root: Path = Path("artifacts/real_skeleton_v7d_manual_approval_selection_20260618")
    selection_decision_materialization_root: Path = Path(
        "artifacts/real_skeleton_v7d_selection_decision_materialization_20260618"
    )
    readiness_root: Path = Path("artifacts/real_skeleton_v7d_seed_readiness_20260618")
    seed_package_root: Path = Path("artifacts/real_skeleton_v7d_prompt_pose_seed_package_20260618")
    three_class_dataset_root: Path = Path(
        "artifacts/real_guided_three_class_wait_expanded_v7d_real_seeded_prompt_window_guard_20260618"
    )
    stage1_dataset_root: Path = Path(
        "artifacts/real_guided_two_stage_rock_transition_v7d_real_seeded_prompt_window_guard_20260618"
    )
    stage2_dataset_root: Path = Path(
        "artifacts/real_guided_two_stage_paper_scissors_v7d_real_seeded_prompt_window_guard_20260618"
    )
    base_dataset_root: Path = Path("artifacts/real_guided_large_sharded_20260610")
    calibration_seed_package_root: Path = Path("artifacts/real_skeleton_v4_calibration_seed_package_fewshot_20260615")
    live_rock_seed_package_root: Path = Path("artifacts/live_rock_false_trigger_overlay_seed_20260616")
    stage1_training_config: Path = Path(
        "configs/real_skeleton_two_stage_rock_transition_v7d_real_seeded_prompt_window_guard_tcn_ensemble.yaml"
    )
    stage2_training_config: Path = Path(
        "configs/real_skeleton_two_stage_paper_scissors_v7d_real_seeded_prompt_window_guard_tcn_ensemble.yaml"
    )
    output_root: Path = Path("artifacts/real_skeleton_v7d_post_approval_preflight_20260618")
    post_approval_pipeline_root: Path = Path("artifacts/real_skeleton_v7d_post_approval_pipeline_20260618")
    review_decision_validation_root: Path = Path("artifacts/real_skeleton_v7d_review_decision_validation_20260618")
    local_smoke_preflight_root: Path = Path("artifacts/real_skeleton_v7d_local_smoke_preflight_20260618")
    remote_training_preflight_root: Path = Path("artifacts/real_skeleton_v7d_remote_training_preflight_20260618")
    strict_validation_preflight_root: Path = Path("artifacts/real_skeleton_v7d_strict_validation_preflight_20260618")
    stage1_profile_json: Path = Path(
        "results/model_profiles/real_skeleton_two_stage_rock_transition_v7d_real_seeded_prompt_window_guard_tcn_ensemble.json"
    )
    stage2_profile_json: Path = Path(
        "results/model_profiles/real_skeleton_two_stage_paper_scissors_v7d_real_seeded_prompt_window_guard_tcn_ensemble.json"
    )
    generated_per_target: int = 10000
    sequence_length: int = 72
    min_length: int = 48
    shard_size: int = 512
    base_rock_stride: int = 2
    seed: int = 20260618


def write_v7d_post_approval_preflight(config: V7DPostApprovalPreflightConfig) -> dict[str, object]:
    """Write a status-only preflight for the v7d post-approval path."""

    project_root = config.project_root.resolve()
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    readiness = check_v7d_prompt_pose_seed_readiness(
        V7DSeedReadinessConfig(
            project_root=project_root,
            review_root=config.review_root,
            output_root=config.readiness_root,
        )
    )
    paths = _artifact_status(project_root, config)
    commands = _planned_commands(config)
    status = _preflight_status(readiness=readiness, paths=paths)
    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "review_root": _display_path(_resolve_path(project_root, config.review_root), base=project_root),
        "readiness_status": readiness.get("status"),
        "missing_required_approved_roles": readiness.get("missing_required_approved_roles", []),
        "approved_counts_by_role": readiness.get("approved_counts_by_role", {}),
        "quality_pass_counts_by_role": readiness.get("quality_pass_counts_by_role", {}),
        "artifact_status": paths,
        "planned_commands": commands,
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test MP4s remain validation-only and must not enter seed packages or training metadata",
        "next_action": _next_action(status),
        "config": _config_summary(project_root=project_root, config=config),
    }
    (output_root / "v7d_post_approval_preflight_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7d_post_approval_preflight_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _artifact_status(project_root: Path, config: V7DPostApprovalPreflightConfig) -> dict[str, object]:
    seed_root = _resolve_path(project_root, config.seed_package_root)
    three_class_root = _resolve_path(project_root, config.three_class_dataset_root)
    stage1_root = _resolve_path(project_root, config.stage1_dataset_root)
    stage2_root = _resolve_path(project_root, config.stage2_dataset_root)
    shortlist_root = _resolve_path(project_root, config.shortlist_root)
    validation_root = _resolve_path(project_root, config.review_decision_validation_root)
    selection_root = _resolve_path(project_root, config.selection_root)
    materialization_root = _resolve_path(project_root, config.selection_decision_materialization_root)
    return {
        "seed_required_decision_template_exists": (shortlist_root / "seed_required_decision_template.csv").exists(),
        "approval_selection_template_exists": (selection_root / "approval_selection_template.csv").exists(),
        "approval_selection_options_exists": (selection_root / "approval_selection_options.csv").exists(),
        "selection_decision_materialization_root_exists": materialization_root.exists(),
        "selection_decision_materialization_summary_exists": (
            materialization_root / "selection_decision_materialization_summary.json"
        ).exists(),
        "materialized_decisions_csv_exists": (
            materialization_root / "seed_required_decision_template_from_selection.csv"
        ).exists(),
        "review_decision_validation_root_exists": validation_root.exists(),
        "review_decision_validation_summary_exists": (
            validation_root / "v7d_review_decision_validation_summary.json"
        ).exists(),
        "seed_package_root_exists": seed_root.exists(),
        "seed_npz_exists": (seed_root / "v7_rps_seed_dataset.npz").exists(),
        "seed_metadata_exists": (seed_root / "seed_metadata.jsonl").exists(),
        "three_class_dataset_root_exists": three_class_root.exists(),
        "three_class_generation_summary_exists": (three_class_root / "generation_summary.json").exists(),
        "stage1_dataset_root_exists": stage1_root.exists(),
        "stage1_remap_summary_exists": (stage1_root / "remap_summary.json").exists(),
        "stage2_dataset_root_exists": stage2_root.exists(),
        "stage2_remap_summary_exists": (stage2_root / "remap_summary.json").exists(),
        "stage1_training_config_exists": _resolve_path(project_root, config.stage1_training_config).exists(),
        "stage2_training_config_exists": _resolve_path(project_root, config.stage2_training_config).exists(),
    }


def _preflight_status(
    *,
    readiness: Mapping[str, object],
    paths: Mapping[str, object],
) -> V7DPostApprovalPreflightStatus:
    if readiness.get("status") != "ready_for_v7d_seed_package":
        return "blocked_manual_approval_required"
    if not paths.get("seed_npz_exists") or not paths.get("seed_metadata_exists"):
        return "ready_for_seed_package_build"
    if not paths.get("three_class_generation_summary_exists"):
        return "ready_for_three_class_dataset_generation"
    if not paths.get("stage1_remap_summary_exists") or not paths.get("stage2_remap_summary_exists"):
        return "ready_for_two_stage_remap"
    return "ready_for_local_smoke_training"


def _planned_commands(config: V7DPostApprovalPreflightConfig) -> dict[str, str]:
    decision_csv = config.selection_decision_materialization_root / "seed_required_decision_template_from_selection.csv"
    materialize_cmd = (
        "python -m embodied_rps.tools.write_v7d_selection_decision_materialization "
        f"--review-root {config.review_root.as_posix()} "
        f"--selection-root {config.selection_root.as_posix()} "
        f"--shortlist-root {config.shortlist_root.as_posix()} "
        f"--output-root {config.selection_decision_materialization_root.as_posix()}"
    )
    validate_cmd = (
        "python -m embodied_rps.tools.validate_v7d_review_decisions "
        f"--review-root {config.review_root.as_posix()} "
        f"--decisions-csv {decision_csv.as_posix()} "
        f"--output-root {config.review_decision_validation_root.as_posix()}"
    )
    pipeline_base = (
        "python -m embodied_rps.tools.run_v7d_post_approval_pipeline "
        f"--review-root {config.review_root.as_posix()} "
        f"--shortlist-root {config.shortlist_root.as_posix()} "
        f"--output-root {config.post_approval_pipeline_root.as_posix()} "
        f"--preflight-output-root {config.output_root.as_posix()} "
        f"--decisions-csv {decision_csv.as_posix()}"
    )
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
        "--augmentation-profile v7d_real_seeded_prompt_window_guard "
        f"--calibration-seed-package-root {config.calibration_seed_package_root.as_posix()} "
        f"--live-rock-seed-package-root {config.live_rock_seed_package_root.as_posix()} "
        f"--v7-seed-package-root {config.seed_package_root.as_posix()}"
    )
    sync_paths = (
        "src configs "
        f"{config.local_smoke_preflight_root.as_posix()} "
        f"{config.stage1_dataset_root.as_posix()} "
        f"{config.stage2_dataset_root.as_posix()}"
    )
    return {
        "materialize_selection_decisions": materialize_cmd,
        "validate_review_decisions": validate_cmd,
        "apply_decisions_dry_run": pipeline_base + " --review-decision-mode dry-run",
        "apply_decisions": (
            pipeline_base
            + f" --review-decision-mode apply --apply-confirmation {APPLY_CONFIRMATION_PHRASE}"
        ),
        "check_seed_readiness": "python -m embodied_rps.tools.check_v7d_prompt_pose_seed_readiness",
        "build_seed_package": "python -m embodied_rps.tools.build_v7d_prompt_pose_seed_package",
        "generate_three_class_dataset": generate_cmd,
        "remap_stage1_rock_transition": (
            "python -m embodied_rps.tools.remap_real_skeleton_dataset "
            f"--source-root {config.three_class_dataset_root.as_posix()} "
            f"--output-root {config.stage1_dataset_root.as_posix()} "
            "--mode rock_vs_transition"
        ),
        "remap_stage2_paper_scissors": (
            "python -m embodied_rps.tools.remap_real_skeleton_dataset "
            f"--source-root {config.three_class_dataset_root.as_posix()} "
            f"--output-root {config.stage2_dataset_root.as_posix()} "
            "--mode paper_vs_scissors"
        ),
        "local_smoke_preflight": (
            "python -m embodied_rps.tools.write_v7d_local_smoke_preflight "
            f"--stage1-dataset-root {config.stage1_dataset_root.as_posix()} "
            f"--stage2-dataset-root {config.stage2_dataset_root.as_posix()} "
            f"--stage1-training-config {config.stage1_training_config.as_posix()} "
            f"--stage2-training-config {config.stage2_training_config.as_posix()}"
        ),
        "local_gru_smoke_stage1": (
            "python -m embodied_rps.tools.train_real_skeleton_predictor "
            f"--config {config.stage1_training_config.as_posix()} --model gru --smoke --max-runs 1 --skip-export"
        ),
        "local_tcn_smoke_stage1": (
            "python -m embodied_rps.tools.train_real_skeleton_predictor "
            f"--config {config.stage1_training_config.as_posix()} --model tcn --smoke --max-runs 1 --skip-export"
        ),
        "local_gru_smoke_stage2": (
            "python -m embodied_rps.tools.train_real_skeleton_predictor "
            f"--config {config.stage2_training_config.as_posix()} --model gru --smoke --max-runs 1 --skip-export"
        ),
        "local_tcn_smoke_stage2": (
            "python -m embodied_rps.tools.train_real_skeleton_predictor "
            f"--config {config.stage2_training_config.as_posix()} --model tcn --smoke --max-runs 1 --skip-export"
        ),
        "remote_training_preflight": (
            "python -m embodied_rps.tools.write_v7d_remote_training_preflight "
            f"--three-class-dataset-root {config.three_class_dataset_root.as_posix()} "
            f"--stage1-dataset-root {config.stage1_dataset_root.as_posix()} "
            f"--stage2-dataset-root {config.stage2_dataset_root.as_posix()} "
            f"--stage1-training-config {config.stage1_training_config.as_posix()} "
            f"--stage2-training-config {config.stage2_training_config.as_posix()} "
            f"--local-smoke-preflight-root {config.local_smoke_preflight_root.as_posix()} "
            f"--output-root {config.remote_training_preflight_root.as_posix()}"
        ),
        "remote_sync_after_local_gates": f"rsync -avR {sync_paths} {REMOTE_HOST}:{REMOTE_WORKSPACE}/",
        "remote_tcn_stage1": (
            f"ssh {REMOTE_HOST} 'cd {REMOTE_WORKSPACE} && "
            f"PYTHONPATH=src python -m embodied_rps.tools.train_real_skeleton_predictor "
            f"--config {config.stage1_training_config.as_posix()} --model tcn'"
        ),
        "remote_tcn_stage2": (
            f"ssh {REMOTE_HOST} 'cd {REMOTE_WORKSPACE} && "
            f"PYTHONPATH=src python -m embodied_rps.tools.train_real_skeleton_predictor "
            f"--config {config.stage2_training_config.as_posix()} --model tcn'"
        ),
        "strict_validation_preflight": (
            "python -m embodied_rps.tools.write_v7d_strict_validation_preflight "
            f"--remote-training-preflight-root {config.remote_training_preflight_root.as_posix()} "
            f"--stage1-profile-json {config.stage1_profile_json.as_posix()} "
            f"--stage2-profile-json {config.stage2_profile_json.as_posix()} "
            f"--output-root {config.strict_validation_preflight_root.as_posix()}"
        ),
    }


def _next_action(status: V7DPostApprovalPreflightStatus) -> str:
    if status == "blocked_manual_approval_required":
        return (
            "fill approval_selection_template.csv, run write_v7d_selection_decision_materialization, validate the derived "
            "decision CSV, then run run_v7d_post_approval_pipeline --review-decision-mode dry-run; do not build seed "
            "package or train yet"
        )
    if status == "ready_for_seed_package_build":
        return "build the v7d prompt-pose seed package"
    if status == "ready_for_three_class_dataset_generation":
        return "generate the balanced v7d three-class dataset"
    if status == "ready_for_two_stage_remap":
        return "remap the v7d three-class dataset into stage1 and stage2 datasets"
    return "run local GRU/TCN smoke before remote TCN training"


def _summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V7d Post-Approval Preflight",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Readiness status: `{summary.get('readiness_status')}`",
        f"- Missing approved roles: `{summary.get('missing_required_approved_roles')}`",
        f"- Training started: `{summary.get('training_started')}`",
        f"- Remote training started: `{summary.get('remote_training_started')}`",
        f"- Validation started: `{summary.get('validation_started')}`",
        f"- Next action: `{summary.get('next_action')}`",
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
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _config_summary(*, project_root: Path, config: V7DPostApprovalPreflightConfig) -> dict[str, object]:
    summary = asdict(config)
    summary["project_root"] = "."
    for key, value in list(summary.items()):
        if isinstance(value, Path):
            summary[key] = _display_path(_resolve_path(project_root, value), base=project_root)
    return _json_ready(summary)  # type: ignore[return-value]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["V7DPostApprovalPreflightConfig", "write_v7d_post_approval_preflight"]
