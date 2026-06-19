"""Fail-closed local v7d post-approval dataset pipeline."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from embodied_rps.artifact_path_sanitizer import relativize_project_root_text_paths
from embodied_rps.real_skeleton_dataset_remap import remap_real_skeleton_dataset
from embodied_rps.three_class_wait_skeletons import ThreeClassWaitExpansionConfig, generate_three_class_wait_dataset
from embodied_rps.v7_rps_seed_package import apply_v7_segment_review_decisions
from embodied_rps.v7d_local_smoke_preflight import (
    V7DLocalSmokePreflightConfig,
    write_v7d_local_smoke_preflight,
)
from embodied_rps.v7d_post_approval_preflight import V7DPostApprovalPreflightConfig, write_v7d_post_approval_preflight
from embodied_rps.v7d_review_decision_validator import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_REVIEW_DECISION_VALIDATION_OUTPUT_ROOT,
    V7DReviewDecisionValidatorConfig,
    validate_v7d_review_decisions,
)
from embodied_rps.v7d_seed_package import REQUIRED_APPROVED_ROLES, V7DSeedPackageConfig, build_v7d_prompt_pose_seed_package
from embodied_rps.v7d_selection_decision_materializer import (
    V7DSelectionDecisionMaterializerConfig,
    write_v7d_selection_decision_materialization,
)

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
APPLY_CONFIRMATION_PHRASE = "reviewed_temporal_evidence_for_v7d"
ReviewDecisionMode = Literal["none", "dry-run", "apply"]

V7DPostApprovalPipelineStatus = Literal[
    "blocked_manual_approval_required",
    "ready_for_review_decision_apply",
    "review_decision_role_coverage_incomplete",
    "ready_for_local_pipeline_execution",
    "local_v7d_datasets_ready",
    "review_decision_validation_failed",
    "selection_decision_materialization_blocked",
    "apply_confirmation_required",
    "official_full_scale_required",
    "seed_package_build_failed",
    "three_class_dataset_generation_failed",
    "two_stage_remap_failed",
]


@dataclass(frozen=True)
class V7DPostApprovalPipelineConfig:
    """Configuration for the local v7d seed/dataset/remap pipeline."""

    project_root: Path = field(default_factory=Path.cwd)
    review_root: Path = V7DPostApprovalPreflightConfig.review_root
    shortlist_root: Path = V7DPostApprovalPreflightConfig.shortlist_root
    readiness_root: Path = V7DPostApprovalPreflightConfig.readiness_root
    seed_package_root: Path = V7DPostApprovalPreflightConfig.seed_package_root
    three_class_dataset_root: Path = V7DPostApprovalPreflightConfig.three_class_dataset_root
    stage1_dataset_root: Path = V7DPostApprovalPreflightConfig.stage1_dataset_root
    stage2_dataset_root: Path = V7DPostApprovalPreflightConfig.stage2_dataset_root
    base_dataset_root: Path = V7DPostApprovalPreflightConfig.base_dataset_root
    calibration_seed_package_root: Path = V7DPostApprovalPreflightConfig.calibration_seed_package_root
    live_rock_seed_package_root: Path = V7DPostApprovalPreflightConfig.live_rock_seed_package_root
    stage1_training_config: Path = V7DPostApprovalPreflightConfig.stage1_training_config
    stage2_training_config: Path = V7DPostApprovalPreflightConfig.stage2_training_config
    preflight_output_root: Path = V7DPostApprovalPreflightConfig.output_root
    output_root: Path = Path("artifacts/real_skeleton_v7d_post_approval_pipeline_20260618")
    local_smoke_preflight_root: Path = Path("artifacts/real_skeleton_v7d_local_smoke_preflight_20260618")
    selection_root: Path = V7DPostApprovalPreflightConfig.selection_root
    selection_decision_materialization_root: Path = V7DPostApprovalPreflightConfig.selection_decision_materialization_root
    decisions_csv: Path = Path("artifacts/real_skeleton_v7d_manual_review_shortlist_20260618/seed_required_decision_template.csv")
    review_decision_validation_output_root: Path = DEFAULT_REVIEW_DECISION_VALIDATION_OUTPUT_ROOT
    generated_per_target: int = V7DPostApprovalPreflightConfig.generated_per_target
    sequence_length: int = V7DPostApprovalPreflightConfig.sequence_length
    min_length: int = V7DPostApprovalPreflightConfig.min_length
    shard_size: int = V7DPostApprovalPreflightConfig.shard_size
    base_rock_stride: int = V7DPostApprovalPreflightConfig.base_rock_stride
    seed: int = V7DPostApprovalPreflightConfig.seed
    review_decision_mode: ReviewDecisionMode = "none"
    apply_confirmation: str = ""
    materialize_selection_decisions: bool = False
    execute_local: bool = False
    overwrite_outputs: bool = False


def run_v7d_post_approval_pipeline(config: V7DPostApprovalPipelineConfig) -> dict[str, object]:
    """Run the local v7d seed, generation, and remap path after approval gates pass."""

    project_root = config.project_root.resolve()
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    completed_stages: list[str] = []
    effective_config = _materialized_config(config)
    selection_decision_materialization_summary: dict[str, object] | None = None
    if config.materialize_selection_decisions:
        selection_decision_materialization_summary = write_v7d_selection_decision_materialization(
            V7DSelectionDecisionMaterializerConfig(
                project_root=project_root,
                review_root=config.review_root,
                selection_root=config.selection_root,
                shortlist_root=config.shortlist_root,
                output_root=config.selection_decision_materialization_root,
            )
        )
        completed_stages.append("selection_decision_materialization")
        if selection_decision_materialization_summary.get("status") != "ready_for_review_decision_apply":
            preflight = write_v7d_post_approval_preflight(_preflight_config(config))
            completed_stages.append("post_approval_preflight")
            summary = _base_summary(
                project_root=project_root,
                output_root=output_root,
                config=effective_config,
                preflight=preflight,
                completed_stages=completed_stages,
                review_decision_summary=None,
                review_decision_role_preview=None,
                review_decision_validation_summary=None,
                selection_decision_materialization_summary=selection_decision_materialization_summary,
            )
            summary.update(
                {
                    "status": "selection_decision_materialization_blocked",
                    "next_action": "fill the compact selection template before dry-run, apply, seed-package, or dataset steps",
                }
            )
            _write_summary(output_root, summary)
            return summary

    review_decision_role_precheck: dict[str, object] | None = None
    review_decision_validation_summary: dict[str, object] | None = None
    if effective_config.review_decision_mode == "apply":
        apply_confirmation_guard = _apply_confirmation_guard(effective_config)
        if not apply_confirmation_guard["confirmation_valid"]:
            completed_stages.append("apply_confirmation_guard")
            preflight = write_v7d_post_approval_preflight(_preflight_config(config))
            completed_stages.append("post_approval_preflight")
            summary = _base_summary(
                project_root=project_root,
                output_root=output_root,
                config=effective_config,
                preflight=preflight,
                completed_stages=completed_stages,
                review_decision_summary=None,
                review_decision_role_preview=None,
                review_decision_validation_summary=None,
                selection_decision_materialization_summary=selection_decision_materialization_summary,
            )
            summary.update(
                {
                    "status": "apply_confirmation_required",
                    "apply_confirmation_guard": apply_confirmation_guard,
                    "next_action": (
                        "rerun apply only after temporal evidence review with "
                        f"--apply-confirmation {APPLY_CONFIRMATION_PHRASE}"
                    ),
                }
            )
            _write_summary(output_root, summary)
            return summary
        review_decision_role_precheck = _review_decision_role_preview(project_root=project_root, config=effective_config)
        if review_decision_role_precheck.get("missing_required_approved_roles"):
            completed_stages.append("review_decision_role_precheck")
            preflight = write_v7d_post_approval_preflight(_preflight_config(config))
            completed_stages.append("post_approval_preflight")
            summary = _base_summary(
                project_root=project_root,
                output_root=output_root,
                config=effective_config,
                preflight=preflight,
                completed_stages=completed_stages,
                review_decision_summary=None,
                review_decision_role_preview=review_decision_role_precheck,
                review_decision_validation_summary=None,
                selection_decision_materialization_summary=selection_decision_materialization_summary,
            )
            summary.update(
                {
                    "status": "review_decision_role_coverage_incomplete",
                    "next_action": "add explicit approvals for every required v7d prompt-window role before applying decisions",
                }
            )
            _write_summary(output_root, summary)
            return summary
        review_decision_validation_summary = validate_v7d_review_decisions(
            V7DReviewDecisionValidatorConfig(
                project_root=project_root,
                review_root=effective_config.review_root,
                decisions_csv=effective_config.decisions_csv,
                output_root=effective_config.review_decision_validation_output_root,
            )
        )
        if review_decision_validation_summary.get("status") != "ready_for_apply":
            completed_stages.append("review_decision_validation")
            preflight = write_v7d_post_approval_preflight(_preflight_config(config))
            completed_stages.append("post_approval_preflight")
            summary = _base_summary(
                project_root=project_root,
                output_root=output_root,
                config=effective_config,
                preflight=preflight,
                completed_stages=completed_stages,
                review_decision_summary=None,
                review_decision_role_preview=review_decision_role_precheck,
                review_decision_validation_summary=review_decision_validation_summary,
                selection_decision_materialization_summary=selection_decision_materialization_summary,
            )
            summary.update(
                {
                    "status": "review_decision_validation_failed",
                    "next_action": "fix the explicit review decision CSV before applying approvals or building datasets",
                }
            )
            _write_summary(output_root, summary)
            return summary

    review_decision_summary = _run_review_decision_step(project_root=project_root, config=effective_config)
    if review_decision_summary is not None:
        completed_stages.append(
            "review_decision_apply" if effective_config.review_decision_mode == "apply" else "review_decision_dry_run"
        )

    preflight = write_v7d_post_approval_preflight(_preflight_config(config))
    completed_stages.append("post_approval_preflight")
    preflight_status = str(preflight.get("status", ""))

    summary = _base_summary(
        project_root=project_root,
        output_root=output_root,
        config=effective_config,
        preflight=preflight,
        completed_stages=completed_stages,
        review_decision_summary=review_decision_summary,
        review_decision_role_preview=review_decision_role_precheck,
        review_decision_validation_summary=review_decision_validation_summary,
        selection_decision_materialization_summary=selection_decision_materialization_summary,
    )
    if review_decision_summary is not None and review_decision_summary.get("status") == "invalid_decisions":
        summary.update(
            {
                "status": "review_decision_validation_failed",
                "next_action": "fix the explicit review decision CSV before applying approvals or building datasets",
            }
        )
        _write_summary(output_root, summary)
        return summary

    role_preview = summary.get("review_decision_role_preview")
    if (
        review_decision_summary is not None
        and review_decision_summary.get("status") == "dry_run_ready"
        and isinstance(role_preview, Mapping)
        and role_preview.get("missing_required_approved_roles")
    ):
        summary.update(
            {
                "status": "review_decision_role_coverage_incomplete",
                "next_action": "add explicit approvals for every required v7d prompt-window role before applying decisions",
            }
        )
        _write_summary(output_root, summary)
        return summary

    if review_decision_summary is not None and review_decision_summary.get("status") == "dry_run_ready":
        validation = validate_v7d_review_decisions(
            V7DReviewDecisionValidatorConfig(
                project_root=project_root,
                review_root=effective_config.review_root,
                decisions_csv=effective_config.decisions_csv,
                output_root=effective_config.review_decision_validation_output_root,
            )
        )
        summary["review_decision_validation_summary"] = _relativize_paths(validation, project_root=project_root)
        if validation.get("status") != "ready_for_apply":
            summary.update(
                {
                    "status": "review_decision_validation_failed",
                    "next_action": "fix the explicit review decision CSV before applying approvals or building datasets",
                }
            )
            _write_summary(output_root, summary)
            return summary

    if review_decision_summary is not None and review_decision_summary.get("status") == "dry_run_ready":
        summary.update(
            {
                "status": "ready_for_review_decision_apply",
                "next_action": "rerun with --review-decision-mode apply after confirming the dry-run approval set",
            }
        )
        _write_summary(output_root, summary)
        return summary

    if preflight_status == "blocked_manual_approval_required":
        summary.update(
            {
                "status": "blocked_manual_approval_required",
                "next_action": "complete manual segment approval before building seed packages or datasets",
            }
        )
        _write_summary(output_root, summary)
        return summary

    full_scale_guard = _official_full_scale_guard(project_root=project_root, config=effective_config)
    if config.execute_local and full_scale_guard["official_output_roots"] and full_scale_guard["actual_generated_per_target"] != full_scale_guard["expected_generated_per_target"]:
        completed_stages.append("official_full_scale_guard")
        summary.update(
            {
                "status": "official_full_scale_required",
                "completed_stages": completed_stages,
                "generated_per_target_guard": full_scale_guard,
                "next_action": "rerun official v7d local execution with generated_per_target = 10000 or use sandbox roots for smoke-scale rehearsal",
            }
        )
        _write_summary(output_root, summary)
        return summary

    if not config.execute_local:
        summary.update(
            {
                "status": "ready_for_local_pipeline_execution",
                "next_action": "rerun with --execute-local after confirming approvals and local artifact targets",
            }
        )
        _write_summary(output_root, summary)
        return summary

    seed_summary: dict[str, object] | None = None
    three_class_summary: dict[str, object] | None = None
    stage1_summary: dict[str, object] | None = None
    stage2_summary: dict[str, object] | None = None
    local_smoke_preflight_summary: dict[str, object] | None = None

    if preflight_status == "ready_for_seed_package_build":
        try:
            seed_summary = build_v7d_prompt_pose_seed_package(
                V7DSeedPackageConfig(
                    project_root=project_root,
                    review_root=effective_config.review_root,
                    output_root=effective_config.seed_package_root,
                    sequence_length=effective_config.sequence_length,
                    overwrite=effective_config.overwrite_outputs,
                )
            )
            completed_stages.append("seed_package_build")
        except Exception as exc:  # pragma: no cover - exercised via failure status contract.
            summary.update(
                _failure_payload(
                    status="seed_package_build_failed",
                    failed_stage="seed_package_build",
                    exc=exc,
                )
            )
            _write_summary(output_root, summary)
            return summary
    else:
        completed_stages.append("seed_package_existing")

    if preflight_status in {"ready_for_seed_package_build", "ready_for_three_class_dataset_generation"}:
        try:
            three_class_summary = generate_three_class_wait_dataset(
                ThreeClassWaitExpansionConfig(
                    output_root=_resolve_path(project_root, config.three_class_dataset_root),
                    base_dataset_root=_resolve_path(project_root, config.base_dataset_root),
                    generated_per_target=config.generated_per_target,
                    sequence_length=config.sequence_length,
                    min_length=config.min_length,
                    shard_size=config.shard_size,
                    seed=config.seed,
                    base_rock_stride=config.base_rock_stride,
                    augmentation_profile="v7d_real_seeded_prompt_window_guard",
                    calibration_seed_package_root=_resolve_path(project_root, config.calibration_seed_package_root),
                    live_rock_seed_package_root=_resolve_path(project_root, config.live_rock_seed_package_root),
                    v7_seed_package_root=_resolve_path(project_root, config.seed_package_root),
                    overwrite=config.overwrite_outputs,
                )
            )
            completed_stages.append("three_class_dataset_generation")
        except Exception as exc:  # pragma: no cover - exercised via failure status contract.
            summary.update(
                _failure_payload(
                    status="three_class_dataset_generation_failed",
                    failed_stage="three_class_dataset_generation",
                    exc=exc,
                )
            )
            summary["seed_package_summary"] = _relativize_paths(seed_summary, project_root=project_root)
            _write_summary(output_root, summary)
            return summary
    else:
        completed_stages.append("three_class_dataset_existing")

    if preflight_status in {
        "ready_for_seed_package_build",
        "ready_for_three_class_dataset_generation",
        "ready_for_two_stage_remap",
    }:
        try:
            stage1_summary = remap_real_skeleton_dataset(
                _resolve_path(project_root, config.three_class_dataset_root),
                _resolve_path(project_root, config.stage1_dataset_root),
                mode="rock_vs_transition",
            )
            completed_stages.append("stage1_rock_transition_remap")
            stage2_summary = remap_real_skeleton_dataset(
                _resolve_path(project_root, config.three_class_dataset_root),
                _resolve_path(project_root, config.stage2_dataset_root),
                mode="paper_vs_scissors",
            )
            completed_stages.append("stage2_paper_scissors_remap")
        except Exception as exc:  # pragma: no cover - exercised via failure status contract.
            summary.update(
                _failure_payload(
                    status="two_stage_remap_failed",
                    failed_stage="two_stage_remap",
                    exc=exc,
                )
            )
            summary["seed_package_summary"] = _relativize_paths(seed_summary, project_root=project_root)
            summary["three_class_summary"] = _relativize_paths(three_class_summary, project_root=project_root)
            summary["stage1_remap_summary"] = _relativize_paths(stage1_summary, project_root=project_root)
            _write_summary(output_root, summary)
            return summary
    else:
        completed_stages.append("two_stage_datasets_existing")

    local_smoke_preflight_summary = write_v7d_local_smoke_preflight(
        V7DLocalSmokePreflightConfig(
            project_root=project_root,
            output_root=config.local_smoke_preflight_root,
            stage1_dataset_root=config.stage1_dataset_root,
            stage2_dataset_root=config.stage2_dataset_root,
            stage1_training_config=config.stage1_training_config,
            stage2_training_config=config.stage2_training_config,
        )
    )
    completed_stages.append("local_smoke_preflight")
    path_sanitization = relativize_project_root_text_paths(
        project_root=project_root,
        roots=[
            _resolve_path(project_root, config.seed_package_root),
            _resolve_path(project_root, config.three_class_dataset_root),
            _resolve_path(project_root, config.stage1_dataset_root),
            _resolve_path(project_root, config.stage2_dataset_root),
            _resolve_path(project_root, config.local_smoke_preflight_root),
        ],
        external_roots={"dataset": Path("D:/dataset")},
    )

    summary.update(
        {
            "status": "local_v7d_datasets_ready",
            "completed_stages": completed_stages,
            "seed_package_summary": _relativize_paths(seed_summary, project_root=project_root),
            "three_class_summary": _relativize_paths(three_class_summary, project_root=project_root),
            "stage1_remap_summary": _relativize_paths(stage1_summary, project_root=project_root),
            "stage2_remap_summary": _relativize_paths(stage2_summary, project_root=project_root),
            "local_smoke_preflight_summary": _relativize_paths(
                local_smoke_preflight_summary,
                project_root=project_root,
            ),
            "path_sanitization": path_sanitization,
            "next_action": "run local GRU/TCN smoke for both two-stage configs before remote sync or training",
        }
    )
    _write_summary(output_root, summary)
    return summary


def _preflight_config(config: V7DPostApprovalPipelineConfig) -> V7DPostApprovalPreflightConfig:
    return V7DPostApprovalPreflightConfig(
        project_root=config.project_root,
        review_root=config.review_root,
        shortlist_root=config.shortlist_root,
        selection_root=config.selection_root,
        selection_decision_materialization_root=config.selection_decision_materialization_root,
        readiness_root=config.readiness_root,
        seed_package_root=config.seed_package_root,
        three_class_dataset_root=config.three_class_dataset_root,
        stage1_dataset_root=config.stage1_dataset_root,
        stage2_dataset_root=config.stage2_dataset_root,
        base_dataset_root=config.base_dataset_root,
        calibration_seed_package_root=config.calibration_seed_package_root,
        live_rock_seed_package_root=config.live_rock_seed_package_root,
        stage1_training_config=config.stage1_training_config,
        stage2_training_config=config.stage2_training_config,
        output_root=config.preflight_output_root,
        generated_per_target=config.generated_per_target,
        sequence_length=config.sequence_length,
        min_length=config.min_length,
        shard_size=config.shard_size,
        base_rock_stride=config.base_rock_stride,
        seed=config.seed,
    )


def _materialized_config(config: V7DPostApprovalPipelineConfig) -> V7DPostApprovalPipelineConfig:
    if not config.materialize_selection_decisions:
        return config
    return V7DPostApprovalPipelineConfig(
        **{
            **asdict(config),
            "decisions_csv": config.selection_decision_materialization_root / "seed_required_decision_template_from_selection.csv",
        }
    )


def _base_summary(
    *,
    project_root: Path,
    output_root: Path,
    config: V7DPostApprovalPipelineConfig,
    preflight: Mapping[str, object],
    completed_stages: list[str],
    review_decision_summary: Mapping[str, object] | None,
    review_decision_role_preview: Mapping[str, object] | None = None,
    review_decision_validation_summary: Mapping[str, object] | None = None,
    selection_decision_materialization_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    role_preview = review_decision_role_preview or (
        _review_decision_role_preview(project_root=project_root, config=config)
        if review_decision_summary is not None
        else None
    )
    return {
        "status": None,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "review_decision_mode": config.review_decision_mode,
        "materialize_selection_decisions": config.materialize_selection_decisions,
        "decisions_csv": _display_path(_resolve_path(project_root, config.decisions_csv), base=project_root),
        "selection_decision_materialization_summary": _relativize_paths(
            selection_decision_materialization_summary,
            project_root=project_root,
        ),
        "review_decision_apply_summary": _relativize_paths(review_decision_summary, project_root=project_root),
        "review_decision_validation_summary": _relativize_paths(
            review_decision_validation_summary,
            project_root=project_root,
        ),
        "review_decision_role_preview": _relativize_paths(role_preview, project_root=project_root),
        "execute_local": config.execute_local,
        "overwrite_outputs": config.overwrite_outputs,
        "preflight_status": preflight.get("status"),
        "readiness_status": preflight.get("readiness_status"),
        "missing_required_approved_roles": preflight.get("missing_required_approved_roles", []),
        "artifact_status": preflight.get("artifact_status", {}),
        "completed_stages": completed_stages,
        "failed_stage": None,
        "error": None,
        "seed_package_summary": None,
        "three_class_summary": None,
        "stage1_remap_summary": None,
        "stage2_remap_summary": None,
        "local_smoke_preflight_summary": None,
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test MP4s remain validation-only and are not used by this pipeline",
        "config": _config_summary(project_root=project_root, config=config),
    }


def _run_review_decision_step(
    *,
    project_root: Path,
    config: V7DPostApprovalPipelineConfig,
) -> dict[str, object] | None:
    if config.review_decision_mode == "none":
        return None
    summary = apply_v7_segment_review_decisions(
        output_root=_resolve_path(project_root, config.review_root),
        decisions_csv=_resolve_path(project_root, config.decisions_csv),
        apply=config.review_decision_mode == "apply",
    )
    relative_summary = _relativize_paths(summary, project_root=project_root)
    _rewrite_review_decision_apply_json(
        project_root=project_root,
        review_root=_resolve_path(project_root, config.review_root),
        summary=relative_summary,
    )
    return relative_summary


def _review_decision_role_preview(
    *,
    project_root: Path,
    config: V7DPostApprovalPipelineConfig,
) -> dict[str, object]:
    review_root = _resolve_path(project_root, config.review_root)
    proposed_by_id = {
        str(row.get("segment_id", "")).strip(): row
        for row in _read_jsonl(review_root / "proposed_segments.jsonl")
        if str(row.get("segment_id", "")).strip()
    }
    approved_by_role: dict[str, set[str]] = {role: set() for role in REQUIRED_APPROVED_ROLES}

    for row in _read_csv(review_root / "segment_review_manifest.csv"):
        segment_id = str(row.get("segment_id", "")).strip()
        role = str(proposed_by_id.get(segment_id, {}).get("proposal_role", "")).strip()
        if role in approved_by_role and _manifest_row_approved(row):
            approved_by_role[role].add(segment_id)

    for row in _read_csv(_resolve_path(project_root, config.decisions_csv)):
        segment_id = str(row.get("segment_id", "")).strip()
        decision = _normalize_review_decision(str(row.get("decision", "")))
        if not segment_id or decision is None:
            continue
        role = str(proposed_by_id.get(segment_id, {}).get("proposal_role", "")).strip()
        if role not in approved_by_role:
            continue
        if decision == "approve":
            approved_by_role[role].add(segment_id)
        else:
            approved_by_role[role].discard(segment_id)

    approved_counts = {role: len(approved_by_role[role]) for role in REQUIRED_APPROVED_ROLES}
    missing_required = [role for role in REQUIRED_APPROVED_ROLES if approved_counts[role] <= 0]
    return {
        "required_approved_roles": list(REQUIRED_APPROVED_ROLES),
        "approved_counts_by_role": approved_counts,
        "approved_segment_ids_by_role": {role: sorted(approved_by_role[role]) for role in REQUIRED_APPROVED_ROLES},
        "missing_required_approved_roles": missing_required,
        "ready_for_v7d_seed_package_after_apply": not missing_required,
    }


def _apply_confirmation_guard(config: V7DPostApprovalPipelineConfig) -> dict[str, object]:
    actual = config.apply_confirmation.strip()
    return {
        "required_confirmation": APPLY_CONFIRMATION_PHRASE,
        "confirmation_present": bool(actual),
        "confirmation_valid": actual == APPLY_CONFIRMATION_PHRASE,
    }


def _official_full_scale_guard(*, project_root: Path, config: V7DPostApprovalPipelineConfig) -> dict[str, object]:
    expected_roots = {
        "seed_package_root": V7DPostApprovalPipelineConfig.seed_package_root.as_posix(),
        "three_class_dataset_root": V7DPostApprovalPipelineConfig.three_class_dataset_root.as_posix(),
        "stage1_dataset_root": V7DPostApprovalPipelineConfig.stage1_dataset_root.as_posix(),
        "stage2_dataset_root": V7DPostApprovalPipelineConfig.stage2_dataset_root.as_posix(),
    }
    actual_roots = {
        key: _display_path(_resolve_path(project_root, getattr(config, key)), base=project_root)
        for key in expected_roots
    }
    root_matches = {key: actual_roots[key] == expected for key, expected in expected_roots.items()}
    return {
        "official_output_roots": all(root_matches.values()),
        "root_matches": root_matches,
        "expected_roots": expected_roots,
        "actual_roots": actual_roots,
        "expected_generated_per_target": V7DPostApprovalPipelineConfig.generated_per_target,
        "actual_generated_per_target": config.generated_per_target,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if isinstance(value, Mapping):
                rows.append(dict(value))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _manifest_row_approved(row: Mapping[str, object]) -> bool:
    return _truthy(row.get("approved_for_training")) and str(row.get("review_status", "")).strip().lower() == "approved"


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "approved"}


def _normalize_review_decision(value: str) -> str | None:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return None
    if normalized in {"approve", "approved", "accept", "accepted"}:
        return "approve"
    if normalized in {"reject", "rejected", "deny", "denied"}:
        return "reject"
    if normalized in {"needs_review", "pending", "pending_manual_review"}:
        return "needs_review"
    return None


def _rewrite_review_decision_apply_json(
    *,
    project_root: Path,
    review_root: Path,
    summary: Mapping[str, object],
) -> None:
    review_root.resolve().relative_to(project_root)
    (review_root / "segment_review_decision_apply_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _failure_payload(*, status: V7DPostApprovalPipelineStatus, failed_stage: str, exc: Exception) -> dict[str, object]:
    return {
        "status": status,
        "failed_stage": failed_stage,
        "error": f"{type(exc).__name__}: {exc}",
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "next_action": f"fix {failed_stage} before running local smoke, remote training, or validation",
    }


def _write_summary(output_root: Path, summary: Mapping[str, object]) -> None:
    payload = _json_ready(summary)
    (output_root / "v7d_post_approval_pipeline_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7d_post_approval_pipeline_summary.md").write_text(
        _summary_markdown(payload),
        encoding="utf-8",
    )


def _summary_markdown(summary: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# V7d Post-Approval Pipeline",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Preflight status: `{summary.get('preflight_status')}`",
            f"- Review decision mode: `{summary.get('review_decision_mode')}`",
            f"- Execute local: `{summary.get('execute_local')}`",
            f"- Completed stages: `{summary.get('completed_stages')}`",
            f"- Failed stage: `{summary.get('failed_stage')}`",
            f"- Training started: `{summary.get('training_started')}`",
            f"- Remote training started: `{summary.get('remote_training_started')}`",
            f"- Validation started: `{summary.get('validation_started')}`",
            f"- Promotion eligible: `{summary.get('promotion_eligible')}`",
            f"- Next action: `{summary.get('next_action')}`",
            "",
        ]
    )


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _config_summary(*, project_root: Path, config: V7DPostApprovalPipelineConfig) -> dict[str, object]:
    summary = asdict(config)
    summary["project_root"] = "."
    for key, value in list(summary.items()):
        if isinstance(value, Path):
            summary[key] = _display_path(_resolve_path(project_root, value), base=project_root)
    return _json_ready(summary)  # type: ignore[return-value]


def _relativize_paths(value: Any, *, project_root: Path) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _relativize_paths(item, project_root=project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_relativize_paths(item, project_root=project_root) for item in value]
    if isinstance(value, tuple):
        return tuple(_relativize_paths(item, project_root=project_root) for item in value)
    if isinstance(value, str) and value:
        path = Path(value)
        if path.is_absolute():
            return _display_path(path, base=project_root)
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["V7DPostApprovalPipelineConfig", "run_v7d_post_approval_pipeline"]
