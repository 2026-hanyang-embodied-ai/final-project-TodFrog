"""Sandbox rehearsal for the v7d local post-approval pipeline."""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from embodied_rps.artifact_path_sanitizer import relativize_project_root_text_paths
from embodied_rps.v7d_post_approval_pipeline import (
    APPLY_CONFIRMATION_PHRASE,
    V7DPostApprovalPipelineConfig,
    run_v7d_post_approval_pipeline,
)
from embodied_rps.v7d_prefill_pipeline_simulation import (
    _copy_selected_evidence_links,
    _display_path,
    _json_ready,
    _relative_path,
    _relativize_paths,
    _resolve_path,
    _source_hashes,
    _write_sandbox_selection,
)
from embodied_rps.v7d_selection_decision_materializer import (
    V7DSelectionDecisionMaterializerConfig,
    write_v7d_selection_decision_materialization,
)

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_REVIEW_ROOT = Path("artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618")
DEFAULT_SHORTLIST_ROOT = Path("artifacts/real_skeleton_v7d_manual_review_shortlist_20260618")
DEFAULT_SELECTION_ROOT = Path("artifacts/real_skeleton_v7d_manual_approval_selection_20260618")
DEFAULT_PREFILL_ROOT = Path("artifacts/real_skeleton_v7d_selection_prefill_draft_20260618")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_local_pipeline_rehearsal_20260618")
OFFICIAL_SEED_PACKAGE_ROOT = Path("artifacts/real_skeleton_v7d_prompt_pose_seed_package_20260618")
OFFICIAL_THREE_CLASS_DATASET_ROOT = Path(
    "artifacts/real_guided_three_class_wait_expanded_v7d_real_seeded_prompt_window_guard_20260618"
)
OFFICIAL_STAGE1_DATASET_ROOT = Path(
    "artifacts/real_guided_two_stage_rock_transition_v7d_real_seeded_prompt_window_guard_20260618"
)
OFFICIAL_STAGE2_DATASET_ROOT = Path(
    "artifacts/real_guided_two_stage_paper_scissors_v7d_real_seeded_prompt_window_guard_20260618"
)


@dataclass(frozen=True)
class V7DLocalPipelineRehearsalConfig:
    """Inputs for rehearsing the v7d local pipeline in a sandbox."""

    project_root: Path = field(default_factory=Path.cwd)
    review_root: Path = DEFAULT_REVIEW_ROOT
    shortlist_root: Path = DEFAULT_SHORTLIST_ROOT
    selection_root: Path = DEFAULT_SELECTION_ROOT
    prefill_root: Path = DEFAULT_PREFILL_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    base_dataset_root: Path = V7DPostApprovalPipelineConfig.base_dataset_root
    calibration_seed_package_root: Path = V7DPostApprovalPipelineConfig.calibration_seed_package_root
    live_rock_seed_package_root: Path = V7DPostApprovalPipelineConfig.live_rock_seed_package_root
    stage1_training_config: Path = V7DPostApprovalPipelineConfig.stage1_training_config
    stage2_training_config: Path = V7DPostApprovalPipelineConfig.stage2_training_config
    generated_per_target: int = 3
    sequence_length: int = V7DPostApprovalPipelineConfig.sequence_length
    min_length: int = V7DPostApprovalPipelineConfig.min_length
    shard_size: int = 16
    base_rock_stride: int = V7DPostApprovalPipelineConfig.base_rock_stride
    seed: int = V7DPostApprovalPipelineConfig.seed


def write_v7d_local_pipeline_rehearsal(config: V7DLocalPipelineRehearsalConfig) -> dict[str, object]:
    """Apply copied approvals and execute local v7d seed/dataset/remap steps in a sandbox only."""

    project_root = config.project_root.resolve()
    review_root = _resolve_path(project_root, config.review_root)
    shortlist_root = _resolve_path(project_root, config.shortlist_root)
    selection_root = _resolve_path(project_root, config.selection_root)
    prefill_root = _resolve_path(project_root, config.prefill_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    required_inputs = [
        review_root / "proposed_segments.jsonl",
        review_root / "segment_review_manifest.csv",
        shortlist_root / "seed_required_decision_template.csv",
        selection_root / "approval_selection_template.csv",
        selection_root / "approval_selection_options.csv",
        prefill_root / "approval_selection_prefill_draft.csv",
    ]
    for path in required_inputs:
        if not path.exists():
            raise FileNotFoundError(f"Missing v7d local rehearsal input: {path}")

    before_hashes = _source_hashes(
        review_root=review_root,
        shortlist_root=shortlist_root,
        selection_root=selection_root,
        prefill_root=prefill_root,
    )
    sandbox_root = output_root / "sandbox"
    _reset_sandbox(project_root=project_root, sandbox_root=sandbox_root)
    sandbox_review = sandbox_root / "review"
    sandbox_shortlist = sandbox_root / "shortlist"
    sandbox_selection = sandbox_root / "selection"
    sandbox_materialization = sandbox_root / "materialization"
    sandbox_validation = sandbox_root / "review_decision_validation"
    sandbox_pipeline = sandbox_root / "pipeline"
    sandbox_preflight = sandbox_root / "preflight"
    sandbox_seed = sandbox_root / "seed_package"
    sandbox_three_class = sandbox_root / "three_class"
    sandbox_stage1 = sandbox_root / "stage1"
    sandbox_stage2 = sandbox_root / "stage2"
    sandbox_local_smoke = sandbox_root / "local_smoke_preflight"

    shutil.copytree(review_root, sandbox_review)
    sandbox_shortlist.mkdir(parents=True, exist_ok=True)
    sandbox_selection.mkdir(parents=True, exist_ok=True)
    shutil.copy2(shortlist_root / "seed_required_decision_template.csv", sandbox_shortlist / "seed_required_decision_template.csv")
    shutil.copy2(selection_root / "approval_selection_options.csv", sandbox_selection / "approval_selection_options.csv")
    selected_ids = _write_sandbox_selection(
        source_selection_csv=selection_root / "approval_selection_template.csv",
        prefill_csv=prefill_root / "approval_selection_prefill_draft.csv",
        output_csv=sandbox_selection / "approval_selection_template.csv",
    )
    _copy_selected_evidence_links(
        source_selection_root=selection_root,
        sandbox_selection_root=sandbox_selection,
        selected_ids=selected_ids,
    )
    materialization = write_v7d_selection_decision_materialization(
        V7DSelectionDecisionMaterializerConfig(
            project_root=project_root,
            review_root=_relative_path(sandbox_review, project_root=project_root),
            selection_root=_relative_path(sandbox_selection, project_root=project_root),
            shortlist_root=_relative_path(sandbox_shortlist, project_root=project_root),
            output_root=_relative_path(sandbox_materialization, project_root=project_root),
        )
    )
    pipeline_summary: dict[str, object] | None = None
    if materialization.get("status") == "ready_for_review_decision_apply":
        pipeline_summary = run_v7d_post_approval_pipeline(
            V7DPostApprovalPipelineConfig(
                project_root=project_root,
                review_root=_relative_path(sandbox_review, project_root=project_root),
                shortlist_root=_relative_path(sandbox_shortlist, project_root=project_root),
                selection_root=_relative_path(sandbox_selection, project_root=project_root),
                selection_decision_materialization_root=_relative_path(sandbox_materialization, project_root=project_root),
                decisions_csv=_relative_path(
                    sandbox_materialization / "seed_required_decision_template_from_selection.csv",
                    project_root=project_root,
                ),
                review_decision_validation_output_root=_relative_path(sandbox_validation, project_root=project_root),
                output_root=_relative_path(sandbox_pipeline, project_root=project_root),
                preflight_output_root=_relative_path(sandbox_preflight, project_root=project_root),
                readiness_root=_relative_path(sandbox_root / "readiness", project_root=project_root),
                seed_package_root=_relative_path(sandbox_seed, project_root=project_root),
                three_class_dataset_root=_relative_path(sandbox_three_class, project_root=project_root),
                stage1_dataset_root=_relative_path(sandbox_stage1, project_root=project_root),
                stage2_dataset_root=_relative_path(sandbox_stage2, project_root=project_root),
                base_dataset_root=config.base_dataset_root,
                calibration_seed_package_root=config.calibration_seed_package_root,
                live_rock_seed_package_root=config.live_rock_seed_package_root,
                stage1_training_config=config.stage1_training_config,
                stage2_training_config=config.stage2_training_config,
                local_smoke_preflight_root=_relative_path(sandbox_local_smoke, project_root=project_root),
                generated_per_target=config.generated_per_target,
                sequence_length=config.sequence_length,
                min_length=config.min_length,
                shard_size=config.shard_size,
                base_rock_stride=config.base_rock_stride,
                seed=config.seed,
                review_decision_mode="apply",
                apply_confirmation=APPLY_CONFIRMATION_PHRASE,
                execute_local=True,
                overwrite_outputs=True,
            )
        )
    path_sanitization = relativize_project_root_text_paths(
        project_root=project_root,
        roots=[sandbox_root],
        external_roots={"dataset": Path("D:/dataset")},
    )

    after_hashes = _source_hashes(
        review_root=review_root,
        shortlist_root=shortlist_root,
        selection_root=selection_root,
        prefill_root=prefill_root,
    )
    source_artifacts_unchanged = before_hashes == after_hashes
    official_status = _official_artifact_status(project_root)
    pipeline_status = str(pipeline_summary.get("status", "")) if pipeline_summary else ""
    status = (
        "sandbox_local_v7d_datasets_ready"
        if materialization.get("status") == "ready_for_review_decision_apply"
        and pipeline_status == "local_v7d_datasets_ready"
        and source_artifacts_unchanged
        and not any(bool(value) for value in official_status.values())
        else "sandbox_local_pipeline_rehearsal_blocked"
    )
    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "sandbox_root": _display_path(sandbox_root, base=project_root),
        "selected_segment_ids_by_role": selected_ids,
        "generated_per_target": config.generated_per_target,
        "source_hashes_before": before_hashes,
        "source_hashes_after": after_hashes,
        "source_artifacts_unchanged": source_artifacts_unchanged,
        "sandbox_only": True,
        "path_sanitization": path_sanitization,
        "materialization_summary": _relativize_paths(materialization, project_root=project_root),
        "pipeline_summary": _compact_pipeline_summary(pipeline_summary, project_root=project_root),
        "official_artifact_status": official_status,
        "seed_package_created": False,
        "dataset_generated": False,
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test paths are rejected from v7d local pipeline rehearsal metadata",
        "next_action": (
            "after real human approval, run the guarded real post-approval pipeline; this rehearsal does not replace "
            "full generated_per_target=10000 training data"
        ),
        "config": _config_summary(project_root=project_root, config=config),
    }
    (output_root / "local_pipeline_rehearsal_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "local_pipeline_rehearsal_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _reset_sandbox(*, project_root: Path, sandbox_root: Path) -> None:
    if sandbox_root.exists():
        sandbox_root.resolve(strict=False).relative_to(project_root.resolve(strict=False))
        if sandbox_root.name != "sandbox":
            raise ValueError(f"Refusing to reset unexpected v7d local rehearsal path: {sandbox_root}")
        shutil.rmtree(sandbox_root, onerror=_remove_readonly)
    sandbox_root.mkdir(parents=True, exist_ok=True)


def _remove_readonly(function: object, path: str, _exc_info: object) -> None:
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    if callable(function):
        function(path)


def _official_artifact_status(project_root: Path) -> dict[str, bool]:
    return {
        "seed_package_root_exists": _resolve_path(project_root, OFFICIAL_SEED_PACKAGE_ROOT).exists(),
        "three_class_dataset_root_exists": _resolve_path(project_root, OFFICIAL_THREE_CLASS_DATASET_ROOT).exists(),
        "stage1_dataset_root_exists": _resolve_path(project_root, OFFICIAL_STAGE1_DATASET_ROOT).exists(),
        "stage2_dataset_root_exists": _resolve_path(project_root, OFFICIAL_STAGE2_DATASET_ROOT).exists(),
    }


def _compact_pipeline_summary(value: Mapping[str, object] | None, *, project_root: Path) -> dict[str, object] | None:
    if value is None:
        return None
    compact: dict[str, object] = {
        "status": value.get("status"),
        "completed_stages": value.get("completed_stages", []),
        "preflight_status": value.get("preflight_status"),
        "readiness_status": value.get("readiness_status"),
        "failed_stage": value.get("failed_stage"),
        "error": value.get("error"),
        "execute_local": value.get("execute_local"),
        "training_started": value.get("training_started"),
        "remote_training_started": value.get("remote_training_started"),
        "validation_started": value.get("validation_started"),
        "promotion_eligible": value.get("promotion_eligible"),
        "seed_package_summary": _compact_seed_summary(value.get("seed_package_summary"), project_root=project_root),
        "three_class_summary": _compact_dataset_summary(value.get("three_class_summary"), project_root=project_root),
        "stage1_remap_summary": _compact_remap_summary(value.get("stage1_remap_summary"), project_root=project_root),
        "stage2_remap_summary": _compact_remap_summary(value.get("stage2_remap_summary"), project_root=project_root),
        "local_smoke_preflight_summary": _compact_smoke_summary(
            value.get("local_smoke_preflight_summary"),
            project_root=project_root,
        ),
    }
    return _relativize_paths(compact, project_root=project_root)


def _compact_seed_summary(value: object, *, project_root: Path) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    builder = value.get("builder_summary")
    readiness = value.get("readiness_summary")
    return {
        "status": value.get("status"),
        "output_root": value.get("output_root"),
        "approved_segment_count": _mapping_get(builder, "approved_segment_count"),
        "target_counts": _mapping_get(builder, "target_counts"),
        "readiness_status": _mapping_get(readiness, "status"),
        "approved_counts_by_role": _mapping_get(readiness, "approved_counts_by_role"),
    }


def _compact_dataset_summary(value: object, *, project_root: Path) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    validation = value.get("validation")
    return {
        "status": value.get("status"),
        "output_root": value.get("output_root"),
        "sample_count": value.get("sample_count"),
        "generated_per_target": value.get("generated_per_target"),
        "target_counts": _mapping_get(validation, "target_counts"),
        "split_counts": _mapping_get(validation, "split_counts"),
        "source_counts": _mapping_get(validation, "source_counts"),
    }


def _compact_remap_summary(value: object, *, project_root: Path) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        "mode": value.get("mode"),
        "source_root": value.get("source_root"),
        "output_root": value.get("output_root"),
        "sample_count": value.get("sample_count"),
        "target_counts": value.get("target_counts"),
    }


def _compact_smoke_summary(value: object, *, project_root: Path) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        "status": value.get("status"),
        "output_root": value.get("output_root"),
        "artifact_status": value.get("artifact_status"),
        "training_started": value.get("training_started"),
        "remote_training_started": value.get("remote_training_started"),
        "validation_started": value.get("validation_started"),
        "promotion_eligible": value.get("promotion_eligible"),
    }


def _mapping_get(value: object, key: str) -> object:
    return value.get(key) if isinstance(value, Mapping) else None


def _summary_markdown(summary: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# V7d Local Pipeline Rehearsal",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Sandbox root: `{summary.get('sandbox_root')}`",
            f"- Generated per target: `{summary.get('generated_per_target')}`",
            f"- Source artifacts unchanged: `{summary.get('source_artifacts_unchanged')}`",
            "- This rehearsal runs only inside the sandbox and does not create official v7d seed/dataset roots.",
            "- It does not start local training, remote training, validation, replay, live retakes, or promotion.",
            f"- Next action: `{summary.get('next_action')}`",
            "",
        ]
    )


def _config_summary(*, project_root: Path, config: V7DLocalPipelineRehearsalConfig) -> dict[str, object]:
    summary = asdict(config)
    summary["project_root"] = "."
    for key, value in list(summary.items()):
        if isinstance(value, Path):
            summary[key] = _display_path(_resolve_path(project_root, value), base=project_root)
    return _json_ready(summary)  # type: ignore[return-value]


__all__ = ["V7DLocalPipelineRehearsalConfig", "write_v7d_local_pipeline_rehearsal"]
