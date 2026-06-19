"""Sandbox simulation for copied v7d prefill selections."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from embodied_rps.v7d_post_approval_pipeline import V7DPostApprovalPipelineConfig, run_v7d_post_approval_pipeline
from embodied_rps.v7d_selection_decision_materializer import (
    V7DSelectionDecisionMaterializerConfig,
    write_v7d_selection_decision_materialization,
)

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_REVIEW_ROOT = Path("artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618")
DEFAULT_SHORTLIST_ROOT = Path("artifacts/real_skeleton_v7d_manual_review_shortlist_20260618")
DEFAULT_SELECTION_ROOT = Path("artifacts/real_skeleton_v7d_manual_approval_selection_20260618")
DEFAULT_PREFILL_ROOT = Path("artifacts/real_skeleton_v7d_selection_prefill_draft_20260618")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_prefill_pipeline_simulation_20260618")
REQUIRED_ROLES: tuple[str, ...] = (
    "hard_paper_prompt_window",
    "rock_wait_prompt_window",
    "scissors_boundary_control",
)
SELECTION_FIELDS: tuple[str, ...] = (
    "proposal_role",
    "minimum_required_approved_count",
    "selected_segment_id",
    "decision",
    "review_notes",
    "available_segment_ids",
    "first_candidate_segment_id",
    "first_candidate_suggested_review_note",
    "decision_template_csv",
    "instruction",
)
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = (
    "decision_template_csv",
    "skeleton_npz",
    "source_path",
    "preview_image",
    "temporal_strip",
)


@dataclass(frozen=True)
class V7DPrefillPipelineSimulationConfig:
    """Inputs for simulating the post-review path in an isolated sandbox."""

    project_root: Path = field(default_factory=Path.cwd)
    review_root: Path = DEFAULT_REVIEW_ROOT
    shortlist_root: Path = DEFAULT_SHORTLIST_ROOT
    selection_root: Path = DEFAULT_SELECTION_ROOT
    prefill_root: Path = DEFAULT_PREFILL_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


def write_v7d_prefill_pipeline_simulation(config: V7DPrefillPipelineSimulationConfig) -> dict[str, object]:
    """Simulate materialize plus dry-run after copying prefill rows, without touching live review state."""

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
            raise FileNotFoundError(f"Missing v7d prefill simulation input: {path}")

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
    shutil.copy2(selection_root / "approval_selection_options.csv", sandbox_selection / "approval_selection_options.csv")
    shutil.copy2(shortlist_root / "seed_required_decision_template.csv", sandbox_shortlist / "seed_required_decision_template.csv")
    shutil.copy2(review_root / "proposed_segments.jsonl", sandbox_review / "proposed_segments.jsonl")
    shutil.copy2(review_root / "segment_review_manifest.csv", sandbox_review / "segment_review_manifest.csv")
    _copy_selected_segment_npz(review_root=review_root, sandbox_review=sandbox_review, selected_ids=selected_ids)
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
                review_decision_mode="dry-run",
                materialize_selection_decisions=False,
                execute_local=False,
            )
        )

    after_hashes = _source_hashes(
        review_root=review_root,
        shortlist_root=shortlist_root,
        selection_root=selection_root,
        prefill_root=prefill_root,
    )
    source_artifacts_unchanged = before_hashes == after_hashes
    pipeline_status = str(pipeline_summary.get("status", "")) if pipeline_summary else ""
    status = (
        "sandbox_ready_for_review_decision_apply"
        if materialization.get("status") == "ready_for_review_decision_apply"
        and pipeline_status == "ready_for_review_decision_apply"
        and source_artifacts_unchanged
        else "sandbox_simulation_blocked"
    )
    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "sandbox_root": _display_path(sandbox_root, base=project_root),
        "selected_segment_ids_by_role": {
            role: selected_ids[role] for role in REQUIRED_ROLES if role in selected_ids
        },
        "source_hashes_before": before_hashes,
        "source_hashes_after": after_hashes,
        "source_artifacts_unchanged": source_artifacts_unchanged,
        "sandbox_only": True,
        "materialization_summary": _relativize_paths(materialization, project_root=project_root),
        "pipeline_summary": _relativize_paths(pipeline_summary, project_root=project_root),
        "decisions_applied": False,
        "review_manifest_modified": False,
        "seed_package_created": False,
        "dataset_generated": False,
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test paths are rejected from prefill simulation metadata and remain validation-only",
        "next_action": (
            "if the reviewer accepts these rows, copy them into the real approval_selection_template.csv, "
            "then rerun materialize/dry-run against the real roots"
        ),
        "config": _config_summary(project_root=project_root, config=config),
    }
    (output_root / "prefill_pipeline_simulation_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "prefill_pipeline_simulation_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _write_sandbox_selection(*, source_selection_csv: Path, prefill_csv: Path, output_csv: Path) -> dict[str, str]:
    source_rows, fieldnames = _read_csv_with_fieldnames(source_selection_csv)
    prefill_rows, _ = _read_csv_with_fieldnames(prefill_csv)
    if not set(SELECTION_FIELDS).issubset(set(fieldnames)):
        raise ValueError(f"{source_selection_csv} is missing required selection columns")
    prefill_by_role: dict[str, Mapping[str, str]] = {}
    for row in prefill_rows:
        _reject_heldout_metadata(row, context=prefill_csv)
        role = str(row.get("proposal_role", "")).strip()
        if role in REQUIRED_ROLES:
            prefill_by_role[role] = row

    selected_ids: dict[str, str] = {}
    output_rows: list[dict[str, object]] = []
    for source in source_rows:
        _reject_heldout_metadata(source, context=source_selection_csv)
        role = str(source.get("proposal_role", "")).strip()
        row = dict(source)
        prefill = prefill_by_role.get(role)
        if prefill is not None:
            selected_id = str(prefill.get("selected_segment_id", "")).strip()
            decision = str(prefill.get("decision", "")).strip().lower()
            review_notes = str(prefill.get("review_notes", "")).strip()
            if not selected_id or decision != "approve" or not review_notes:
                raise ValueError(f"{prefill_csv} has incomplete prefill row for role {role}")
            row["selected_segment_id"] = selected_id
            row["decision"] = "approve"
            row["review_notes"] = review_notes
            selected_ids[role] = selected_id
        output_rows.append(row)

    missing = [role for role in REQUIRED_ROLES if role not in selected_ids]
    if missing:
        raise ValueError(f"{prefill_csv} is missing required prefill roles: {missing}")
    _write_csv(output_csv, fieldnames, output_rows)
    return selected_ids


def _copy_selected_segment_npz(*, review_root: Path, sandbox_review: Path, selected_ids: Mapping[str, str]) -> None:
    proposed = {
        str(row.get("segment_id", "")).strip(): row
        for row in _read_jsonl(review_root / "proposed_segments.jsonl")
        if str(row.get("segment_id", "")).strip()
    }
    for segment_id in selected_ids.values():
        proposal = proposed.get(segment_id)
        if proposal is None:
            raise ValueError(f"Selected segment missing from proposed segments: {segment_id}")
        _reject_heldout_metadata(proposal, context=review_root / "proposed_segments.jsonl")
        skeleton_rel = Path(str(proposal.get("skeleton_npz", "")).strip())
        source = review_root / skeleton_rel
        if not source.exists():
            raise FileNotFoundError(f"Selected segment skeleton is missing: {source}")
        target = sandbox_review / skeleton_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _copy_selected_evidence_links(
    *,
    source_selection_root: Path,
    sandbox_selection_root: Path,
    selected_ids: Mapping[str, str],
) -> None:
    option_rows, _ = _read_csv_with_fieldnames(source_selection_root / "approval_selection_options.csv")
    selected_segment_ids = set(selected_ids.values())
    for row in option_rows:
        _reject_heldout_metadata(row, context=source_selection_root / "approval_selection_options.csv")
        segment_id = str(row.get("segment_id", "")).strip()
        if segment_id not in selected_segment_ids:
            continue
        for field_name in ("temporal_strip", "preview_image", "skeleton_npz"):
            value = str(row.get(field_name, "")).strip()
            if not value:
                raise ValueError(f"Selected v7d option is missing evidence field {field_name}: {segment_id}")
            source = _resolve_selection_link(source_selection_root, value)
            if not source.exists():
                raise FileNotFoundError(f"Selected v7d evidence file is missing: {source}")
            target = _resolve_selection_link(sandbox_selection_root, value)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _resolve_selection_link(selection_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else selection_root / path


def _prepare_sandbox_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _source_hashes(
    *,
    review_root: Path,
    shortlist_root: Path,
    selection_root: Path,
    prefill_root: Path,
) -> dict[str, str]:
    return {
        "proposed_segments": _sha256(review_root / "proposed_segments.jsonl"),
        "review_manifest": _sha256(review_root / "segment_review_manifest.csv"),
        "decision_template": _sha256(shortlist_root / "seed_required_decision_template.csv"),
        "selection_template": _sha256(selection_root / "approval_selection_template.csv"),
        "selection_options": _sha256(selection_root / "approval_selection_options.csv"),
        "prefill_draft": _sha256(prefill_root / "approval_selection_prefill_draft.csv"),
    }


def _summary_markdown(summary: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# V7d Prefill Pipeline Simulation",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Sandbox root: `{summary.get('sandbox_root')}`",
            f"- Source artifacts unchanged: `{summary.get('source_artifacts_unchanged')}`",
            f"- Selected segment IDs by role: `{summary.get('selected_segment_ids_by_role')}`",
            "- This simulation runs against copied sandbox artifacts only.",
            "- It does not approve real review rows, build seed packages, generate datasets, train, validate, or promote.",
            f"- Next action: `{summary.get('next_action')}`",
            "",
        ]
    )


def _read_csv_with_fieldnames(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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


def _reject_heldout_metadata(row: Mapping[str, object], *, context: Path) -> None:
    for field_name in PATH_FIELDS_TO_AUDIT:
        value = str(row.get(field_name, "")).strip()
        if value and _is_heldout_test_path(value):
            raise ValueError(f"{context} contains held-out test path in {field_name}: {value}")


def _is_heldout_test_path(value: str) -> bool:
    normalized = value.replace("\\", "/").lower().rstrip("/")
    return "/test/" in normalized or normalized.endswith("/test")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(base.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _relative_path(path: Path, *, project_root: Path) -> Path:
    return Path(_display_path(path, base=project_root))


def _config_summary(*, project_root: Path, config: V7DPrefillPipelineSimulationConfig) -> dict[str, object]:
    summary = asdict(config)
    summary["project_root"] = "."
    for key, value in list(summary.items()):
        if isinstance(value, Path):
            summary[key] = _display_path(_resolve_path(project_root, value), base=project_root)
    return _json_ready(summary)  # type: ignore[return-value]


def _relativize_paths(value: Any, *, project_root: Path) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return _display_path(value, base=project_root)
    if isinstance(value, Mapping):
        return {str(key): _relativize_paths(item, project_root=project_root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_relativize_paths(item, project_root=project_root) for item in value]
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["V7DPrefillPipelineSimulationConfig", "write_v7d_prefill_pipeline_simulation"]
