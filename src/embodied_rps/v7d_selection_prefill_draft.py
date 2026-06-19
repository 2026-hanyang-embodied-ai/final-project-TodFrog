"""Non-mutating v7d approval-selection prefill draft."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_SELECTION_ROOT = Path("artifacts/real_skeleton_v7d_manual_approval_selection_20260618")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_selection_prefill_draft_20260618")
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
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = ("decision_template_csv",)


@dataclass(frozen=True)
class V7DSelectionPrefillDraftConfig:
    """Inputs for writing a reviewer-only prefilled selection draft."""

    project_root: Path = field(default_factory=Path.cwd)
    selection_root: Path = DEFAULT_SELECTION_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


def write_v7d_selection_prefill_draft(config: V7DSelectionPrefillDraftConfig) -> dict[str, object]:
    """Write copy-ready reviewer suggestions without editing the real selection sheet."""

    project_root = config.project_root.resolve()
    selection_root = _resolve_path(project_root, config.selection_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selection_csv = selection_root / "approval_selection_template.csv"
    if not selection_csv.exists():
        raise FileNotFoundError(f"Missing v7d approval selection template: {selection_csv}")

    source_rows = _read_csv(selection_csv)
    source_by_role = {str(row.get("proposal_role", "")).strip(): row for row in source_rows}
    draft_rows: list[dict[str, object]] = []
    missing_candidate_roles: list[str] = []
    for role in REQUIRED_ROLES:
        source_row = source_by_role.get(role, {"proposal_role": role, "minimum_required_approved_count": "1"})
        _reject_heldout_metadata(source_row, context=selection_csv)
        draft_row = _draft_row(source_row)
        if not draft_row["selected_segment_id"] or not draft_row["review_notes"]:
            missing_candidate_roles.append(role)
        draft_rows.append(draft_row)

    filled_count = sum(
        1
        for row in draft_rows
        if str(row.get("selected_segment_id", "")).strip()
        and str(row.get("decision", "")).strip() == "approve"
        and str(row.get("review_notes", "")).strip()
    )
    status = "review_required_before_copy" if not missing_candidate_roles else "missing_candidate_roles"
    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "selection_root": _display_path(selection_root, base=project_root),
        "source_selection_template_csv": _display_path(selection_csv, base=project_root),
        "draft_csv": _display_path(output_root / "approval_selection_prefill_draft.csv", base=project_root),
        "summary_json": _display_path(output_root / "approval_selection_prefill_draft_summary.json", base=project_root),
        "summary_md": _display_path(output_root / "approval_selection_prefill_draft.md", base=project_root),
        "source_selection_template_sha256": _sha256(selection_csv),
        "draft_row_count": len(draft_rows),
        "filled_draft_count": filled_count,
        "required_roles": list(REQUIRED_ROLES),
        "missing_candidate_roles": missing_candidate_roles,
        "pipeline_selection_root_compatible": False,
        "decisions_applied": False,
        "review_manifest_modified": False,
        "seed_package_created": False,
        "dataset_generated": False,
        "training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test paths are rejected from selection prefill draft metadata",
        "next_action": (
            "inspect temporal evidence, then copy the reviewed values into approval_selection_template.csv "
            "before materialization; do not point the pipeline at this draft root"
        ),
        "config": _config_summary(project_root=project_root, config=config),
    }

    _write_csv(output_root / "approval_selection_prefill_draft.csv", SELECTION_FIELDS, draft_rows)
    (output_root / "approval_selection_prefill_draft_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "approval_selection_prefill_draft.md").write_text(
        _summary_markdown(summary=summary, rows=draft_rows),
        encoding="utf-8",
    )
    return summary


def _draft_row(source_row: Mapping[str, object]) -> dict[str, object]:
    selected_segment_id = str(source_row.get("first_candidate_segment_id", "")).strip()
    review_notes = str(source_row.get("first_candidate_suggested_review_note", "")).strip()
    row = {field: str(source_row.get(field, "")).strip() for field in SELECTION_FIELDS}
    row["selected_segment_id"] = selected_segment_id
    row["decision"] = "approve" if selected_segment_id and review_notes else ""
    row["review_notes"] = review_notes
    row["instruction"] = (
        "DRAFT ONLY: inspect the temporal strip and preview first, then copy selected_segment_id, "
        "decision, and review_notes into approval_selection_template.csv only if the row is manually approved. "
        "This draft is not a pipeline selection root and does not apply decisions."
    )
    return row


def _summary_markdown(*, summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# V7d Selection Prefill Draft",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Draft rows: `{summary.get('draft_row_count')}`",
        f"- Filled draft rows: `{summary.get('filled_draft_count')}`",
        f"- Source selection template: `{summary.get('source_selection_template_csv')}`",
        f"- Source selection SHA-256: `{summary.get('source_selection_template_sha256')}`",
        "- This artifact is not a pipeline selection root and does not include `approval_selection_options.csv`.",
        "- It does not approve segments, edit manifests, build seed packages, train, validate, or promote.",
        "- After visual temporal review, copy the reviewed values into `approval_selection_template.csv` before materialization.",
        "",
        "| Role | Draft Segment | Draft Decision | Draft Review Notes |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('proposal_role', '')}`",
                    f"`{row.get('selected_segment_id', '')}`",
                    f"`{row.get('decision', '')}`",
                    str(row.get("review_notes", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Next Command After Manual Copy",
            "",
            "```powershell",
            "$env:PYTHONPATH='src'",
            "python -m embodied_rps.tools.run_v7d_post_approval_pipeline --materialize-selection-decisions --review-decision-mode dry-run",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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


def _config_summary(*, project_root: Path, config: V7DSelectionPrefillDraftConfig) -> dict[str, object]:
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


__all__ = ["V7DSelectionPrefillDraftConfig", "write_v7d_selection_prefill_draft"]
