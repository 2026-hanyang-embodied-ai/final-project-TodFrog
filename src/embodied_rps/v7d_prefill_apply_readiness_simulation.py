"""Sandbox apply-readiness rehearsal for v7d prefilled approvals."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from embodied_rps.v7d_post_approval_pipeline import (
    APPLY_CONFIRMATION_PHRASE,
    V7DPostApprovalPipelineConfig,
    run_v7d_post_approval_pipeline,
)
from embodied_rps.v7d_prefill_pipeline_simulation import (
    _copy_selected_evidence_links,
    _copy_selected_segment_npz,
    _display_path,
    _json_ready,
    _prepare_sandbox_dirs,
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
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_prefill_apply_readiness_simulation_20260618")


@dataclass(frozen=True)
class V7DPrefillApplyReadinessSimulationConfig:
    """Inputs for rehearsing v7d apply-to-readiness in a sandbox."""

    project_root: Path = field(default_factory=Path.cwd)
    review_root: Path = DEFAULT_REVIEW_ROOT
    shortlist_root: Path = DEFAULT_SHORTLIST_ROOT
    selection_root: Path = DEFAULT_SELECTION_ROOT
    prefill_root: Path = DEFAULT_PREFILL_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


def write_v7d_prefill_apply_readiness_simulation(
    config: V7DPrefillApplyReadinessSimulationConfig,
) -> dict[str, object]:
    """Apply copied prefill decisions only in a sandbox and report seed-package readiness."""

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
            raise FileNotFoundError(f"Missing v7d apply-readiness simulation input: {path}")

    before_hashes = _source_hashes(
        review_root=review_root,
        shortlist_root=shortlist_root,
        selection_root=selection_root,
        prefill_root=prefill_root,
    )
    sandbox_root = output_root / "sandbox"
    sandbox_review = sandbox_root / "review"
    sandbox_shortlist = sandbox_root / "shortlist"
    sandbox_selection = sandbox_root / "selection"
    sandbox_materialization = sandbox_root / "materialization"
    sandbox_pipeline = sandbox_root / "pipeline"
    sandbox_preflight = sandbox_root / "preflight"
    sandbox_validation = sandbox_root / "review_decision_validation"
    _prepare_sandbox_dirs(
        sandbox_review,
        sandbox_shortlist,
        sandbox_selection,
        sandbox_materialization,
        sandbox_pipeline,
        sandbox_preflight,
        sandbox_validation,
    )

    selected_ids = _write_sandbox_selection(
        source_selection_csv=selection_root / "approval_selection_template.csv",
        prefill_csv=prefill_root / "approval_selection_prefill_draft.csv",
        output_csv=sandbox_selection / "approval_selection_template.csv",
    )
    _copy_file(selection_root / "approval_selection_options.csv", sandbox_selection / "approval_selection_options.csv")
    _copy_file(shortlist_root / "seed_required_decision_template.csv", sandbox_shortlist / "seed_required_decision_template.csv")
    _copy_file(review_root / "proposed_segments.jsonl", sandbox_review / "proposed_segments.jsonl")
    _copy_file(review_root / "segment_review_manifest.csv", sandbox_review / "segment_review_manifest.csv")
    _copy_selected_segment_npz(review_root=review_root, sandbox_review=sandbox_review, selected_ids=selected_ids)
    _copy_selected_evidence_links(
        source_selection_root=selection_root,
        sandbox_selection_root=sandbox_selection,
        selected_ids=selected_ids,
    )

    sandbox_manifest_before = _file_text(sandbox_review / "segment_review_manifest.csv")
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
                seed_package_root=_relative_path(sandbox_root / "no_seed_package_created", project_root=project_root),
                three_class_dataset_root=_relative_path(
                    sandbox_root / "no_three_class_dataset_created",
                    project_root=project_root,
                ),
                stage1_dataset_root=_relative_path(sandbox_root / "no_stage1_dataset_created", project_root=project_root),
                stage2_dataset_root=_relative_path(sandbox_root / "no_stage2_dataset_created", project_root=project_root),
                local_smoke_preflight_root=_relative_path(
                    sandbox_root / "no_local_smoke_started",
                    project_root=project_root,
                ),
                review_decision_mode="apply",
                apply_confirmation=APPLY_CONFIRMATION_PHRASE,
                materialize_selection_decisions=False,
                execute_local=False,
            )
        )

    sandbox_manifest_after = _file_text(sandbox_review / "segment_review_manifest.csv")
    after_hashes = _source_hashes(
        review_root=review_root,
        shortlist_root=shortlist_root,
        selection_root=selection_root,
        prefill_root=prefill_root,
    )
    source_artifacts_unchanged = before_hashes == after_hashes
    pipeline_status = str(pipeline_summary.get("status", "")) if pipeline_summary else ""
    preflight_status = str(pipeline_summary.get("preflight_status", "")) if pipeline_summary else ""
    readiness_status = str(pipeline_summary.get("readiness_status", "")) if pipeline_summary else ""
    seed_package_created = (sandbox_root / "no_seed_package_created").exists()
    status = (
        "sandbox_ready_for_seed_package_build"
        if materialization.get("status") == "ready_for_review_decision_apply"
        and pipeline_status == "ready_for_local_pipeline_execution"
        and preflight_status == "ready_for_seed_package_build"
        and readiness_status == "ready_for_v7d_seed_package"
        and source_artifacts_unchanged
        and not seed_package_created
        else "sandbox_apply_readiness_blocked"
    )

    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "sandbox_root": _display_path(sandbox_root, base=project_root),
        "selected_segment_ids_by_role": selected_ids,
        "source_hashes_before": before_hashes,
        "source_hashes_after": after_hashes,
        "source_artifacts_unchanged": source_artifacts_unchanged,
        "sandbox_only": True,
        "sandbox_review_manifest_modified": sandbox_manifest_before != sandbox_manifest_after,
        "materialization_summary": _relativize_paths(materialization, project_root=project_root),
        "pipeline_summary": _relativize_paths(pipeline_summary, project_root=project_root),
        "review_manifest_modified": False,
        "seed_package_created": seed_package_created,
        "dataset_generated": False,
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test paths are rejected from apply-readiness simulation metadata and remain validation-only",
        "next_action": (
            "after human temporal review, apply the guarded copy to the real selection sheet, run materialize/dry-run, "
            "then apply real decisions only after confirming the dry-run summary"
        ),
        "config": _config_summary(project_root=project_root, config=config),
    }
    (output_root / "prefill_apply_readiness_simulation_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "prefill_apply_readiness_simulation_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _summary_markdown(summary: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# V7d Prefill Apply Readiness Simulation",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Sandbox root: `{summary.get('sandbox_root')}`",
            f"- Source artifacts unchanged: `{summary.get('source_artifacts_unchanged')}`",
            f"- Sandbox review manifest modified: `{summary.get('sandbox_review_manifest_modified')}`",
            f"- Seed package created: `{summary.get('seed_package_created')}`",
            "- This rehearsal applies decisions only inside the sandbox.",
            "- It does not approve real review rows, build seed packages, generate datasets, train, validate, or promote.",
            f"- Next action: `{summary.get('next_action')}`",
            "",
        ]
    )


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(source, target)


def _file_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _config_summary(*, project_root: Path, config: V7DPrefillApplyReadinessSimulationConfig) -> dict[str, object]:
    summary = asdict(config)
    summary["project_root"] = "."
    for key, value in list(summary.items()):
        if isinstance(value, Path):
            summary[key] = _display_path(_resolve_path(project_root, value), base=project_root)
    return _json_ready(summary)  # type: ignore[return-value]


__all__ = ["V7DPrefillApplyReadinessSimulationConfig", "write_v7d_prefill_apply_readiness_simulation"]
