"""Guarded v7d prefill-to-selection copy plan."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_SELECTION_ROOT = Path("artifacts/real_skeleton_v7d_manual_approval_selection_20260618")
DEFAULT_PREFILL_ROOT = Path("artifacts/real_skeleton_v7d_selection_prefill_draft_20260618")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_prefill_selection_copy_plan_20260618")
CONFIRMATION_PHRASE = "reviewed_temporal_evidence_for_v7d"
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
PLAN_FIELDS: tuple[str, ...] = (
    "proposal_role",
    "before_selected_segment_id",
    "after_selected_segment_id",
    "before_decision",
    "after_decision",
    "before_review_notes",
    "after_review_notes",
    "available_segment_ids",
    "would_update",
    "instruction",
)
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = ("decision_template_csv",)


@dataclass(frozen=True)
class V7DPrefillSelectionCopyConfig:
    """Inputs for planning or applying a prefill-to-selection copy."""

    project_root: Path = field(default_factory=Path.cwd)
    selection_root: Path = DEFAULT_SELECTION_ROOT
    prefill_root: Path = DEFAULT_PREFILL_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    apply: bool = False
    reviewer_confirmation: str = ""


def write_v7d_prefill_selection_copy_plan(config: V7DPrefillSelectionCopyConfig) -> dict[str, object]:
    """Write a guarded copy plan; mutate the selection sheet only with explicit confirmation."""

    project_root = config.project_root.resolve()
    selection_root = _resolve_path(project_root, config.selection_root)
    prefill_root = _resolve_path(project_root, config.prefill_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selection_csv = selection_root / "approval_selection_template.csv"
    prefill_csv = prefill_root / "approval_selection_prefill_draft.csv"
    for path in (selection_csv, prefill_csv):
        if not path.exists():
            raise FileNotFoundError(f"Missing v7d prefill selection copy input: {path}")

    selection_rows, fieldnames = _read_csv_with_fieldnames(selection_csv)
    if not set(SELECTION_FIELDS).issubset(set(fieldnames)):
        raise ValueError(f"{selection_csv} is missing required selection columns")
    prefill_rows, _ = _read_csv_with_fieldnames(prefill_csv)
    before_hash = _sha256(selection_csv)
    updated_rows, plan_rows, failures = _planned_rows(selection_rows=selection_rows, prefill_rows=prefill_rows)
    planned_update_count = sum(1 for row in plan_rows if row.get("would_update") == "true")

    _write_csv(output_root / "approval_selection_copy_plan.csv", PLAN_FIELDS, plan_rows)
    _write_csv(output_root / "approval_selection_template_after_copy_preview.csv", fieldnames, updated_rows)

    backup_path: Path | None = None
    selection_template_modified = False
    status = "ready_for_manual_copy_or_apply"
    if failures:
        status = "invalid_prefill_copy_plan"
    elif config.apply:
        if config.reviewer_confirmation != CONFIRMATION_PHRASE:
            status = "apply_confirmation_required"
        else:
            backup_path = output_root / "approval_selection_template.csv.bak"
            shutil.copy2(selection_csv, backup_path)
            _write_csv(selection_csv, fieldnames, updated_rows)
            selection_template_modified = True
            status = "applied_to_selection_template"

    after_hash = _sha256(selection_csv)
    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "selection_root": _display_path(selection_root, base=project_root),
        "prefill_root": _display_path(prefill_root, base=project_root),
        "selection_template_csv": _display_path(selection_csv, base=project_root),
        "prefill_csv": _display_path(prefill_csv, base=project_root),
        "copy_plan_csv": _display_path(output_root / "approval_selection_copy_plan.csv", base=project_root),
        "after_copy_preview_csv": _display_path(
            output_root / "approval_selection_template_after_copy_preview.csv",
            base=project_root,
        ),
        "backup_selection_template_csv": _display_path(backup_path, base=project_root) if backup_path else "",
        "source_selection_template_sha256_before": before_hash,
        "source_selection_template_sha256_after": after_hash,
        "source_selection_template_unchanged": before_hash == after_hash,
        "planned_update_count": planned_update_count,
        "required_roles": list(REQUIRED_ROLES),
        "failures": failures,
        "apply": bool(config.apply),
        "confirmation_phrase_required_for_apply": CONFIRMATION_PHRASE,
        "selection_template_modified": selection_template_modified,
        "decisions_applied": selection_template_modified,
        "review_manifest_modified": False,
        "seed_package_created": False,
        "dataset_generated": False,
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test paths are rejected from prefill copy metadata and remain validation-only",
        "next_action": _next_action(status),
        "config": _config_summary(project_root=project_root, config=config),
    }
    (output_root / "approval_selection_copy_plan_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "approval_selection_copy_plan.md").write_text(
        _summary_markdown(summary=summary, rows=plan_rows),
        encoding="utf-8",
    )
    return summary


def _planned_rows(
    *,
    selection_rows: Sequence[Mapping[str, object]],
    prefill_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    prefill_by_role: dict[str, Mapping[str, object]] = {}
    failures: list[dict[str, object]] = []
    for row in prefill_rows:
        _reject_heldout_metadata(row, context="prefill")
        role = str(row.get("proposal_role", "")).strip()
        if role in REQUIRED_ROLES:
            prefill_by_role[role] = row

    updated_rows: list[dict[str, object]] = []
    plan_rows: list[dict[str, object]] = []
    seen_roles: set[str] = set()
    for row in selection_rows:
        _reject_heldout_metadata(row, context="selection")
        role = str(row.get("proposal_role", "")).strip()
        updated = dict(row)
        seen_roles.add(role)
        if role not in REQUIRED_ROLES:
            updated_rows.append(updated)
            continue
        prefill = prefill_by_role.get(role)
        if prefill is None:
            failures.append({"code": "missing_prefill_role", "proposal_role": role})
            updated_rows.append(updated)
            continue
        selected_id = str(prefill.get("selected_segment_id", "")).strip()
        decision = str(prefill.get("decision", "")).strip().lower()
        review_notes = str(prefill.get("review_notes", "")).strip()
        available_ids = _split_available_ids(str(row.get("available_segment_ids", "")).strip())
        if not selected_id or decision != "approve" or not review_notes:
            failures.append({"code": "incomplete_prefill_row", "proposal_role": role})
        if selected_id not in available_ids:
            failures.append({"code": "selected_segment_not_available", "proposal_role": role, "segment_id": selected_id})
        updated["selected_segment_id"] = selected_id
        updated["decision"] = "approve" if selected_id and review_notes and decision == "approve" else ""
        updated["review_notes"] = review_notes
        would_update = (
            str(row.get("selected_segment_id", "")).strip() != str(updated.get("selected_segment_id", "")).strip()
            or str(row.get("decision", "")).strip() != str(updated.get("decision", "")).strip()
            or str(row.get("review_notes", "")).strip() != str(updated.get("review_notes", "")).strip()
        )
        plan_rows.append(
            {
                "proposal_role": role,
                "before_selected_segment_id": str(row.get("selected_segment_id", "")).strip(),
                "after_selected_segment_id": str(updated.get("selected_segment_id", "")).strip(),
                "before_decision": str(row.get("decision", "")).strip(),
                "after_decision": str(updated.get("decision", "")).strip(),
                "before_review_notes": str(row.get("review_notes", "")).strip(),
                "after_review_notes": str(updated.get("review_notes", "")).strip(),
                "available_segment_ids": str(row.get("available_segment_ids", "")).strip(),
                "would_update": str(would_update).lower(),
                "instruction": (
                    "Review temporal evidence before accepting this copy. Apply requires confirmation phrase "
                    f"{CONFIRMATION_PHRASE}."
                ),
            }
        )
        updated_rows.append(updated)

    for role in REQUIRED_ROLES:
        if role not in seen_roles:
            failures.append({"code": "missing_selection_role", "proposal_role": role})
    return updated_rows, plan_rows, failures


def _next_action(status: str) -> str:
    if status == "ready_for_manual_copy_or_apply":
        return (
            "inspect temporal evidence and the after-copy preview; if accepted, rerun with --apply and the "
            "required confirmation phrase, then run materialize/dry-run"
        )
    if status == "apply_confirmation_required":
        return "rerun apply only after human temporal review using the exact confirmation phrase"
    if status == "applied_to_selection_template":
        return "run the integrated materialize/dry-run command against the real selection root"
    return "fix invalid prefill rows before any copy or apply step"


def _summary_markdown(*, summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# V7d Prefill Selection Copy Plan",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Apply: `{summary.get('apply')}`",
        f"- Planned updates: `{summary.get('planned_update_count')}`",
        f"- Selection template modified: `{summary.get('selection_template_modified')}`",
        f"- Source unchanged: `{summary.get('source_selection_template_unchanged')}`",
        f"- Confirmation phrase for apply: `{summary.get('confirmation_phrase_required_for_apply')}`",
        "- This artifact does not build seed packages, generate datasets, train, validate, or promote.",
        "",
        "| Role | Before Segment | After Segment | After Decision | Would Update |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('proposal_role', '')}`",
                    f"`{row.get('before_selected_segment_id', '')}`",
                    f"`{row.get('after_selected_segment_id', '')}`",
                    f"`{row.get('after_decision', '')}`",
                    f"`{row.get('would_update', '')}`",
                ]
            )
            + " |"
        )
    lines.extend(["", f"Next action: `{summary.get('next_action')}`", ""])
    return "\n".join(lines)


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


def _reject_heldout_metadata(row: Mapping[str, object], *, context: str) -> None:
    for field_name in PATH_FIELDS_TO_AUDIT:
        value = str(row.get(field_name, "")).strip()
        if value and _is_heldout_test_path(value):
            raise ValueError(f"{context} contains held-out test path in {field_name}: {value}")


def _is_heldout_test_path(value: str) -> bool:
    normalized = value.replace("\\", "/").lower().rstrip("/")
    return "/test/" in normalized or normalized.endswith("/test")


def _split_available_ids(value: str) -> set[str]:
    return {part.strip() for part in value.split(";") if part.strip()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path | None, *, base: Path) -> str:
    if path is None:
        return ""
    try:
        return path.resolve(strict=False).relative_to(base.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _config_summary(*, project_root: Path, config: V7DPrefillSelectionCopyConfig) -> dict[str, object]:
    summary = asdict(config)
    summary["project_root"] = "."
    summary["reviewer_confirmation"] = "[redacted]" if summary.get("reviewer_confirmation") else ""
    for key, value in list(summary.items()):
        if isinstance(value, Path):
            summary[key] = _display_path(_resolve_path(project_root, value), base=project_root)
    return _json_ready(summary)  # type: ignore[return-value]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["CONFIRMATION_PHRASE", "V7DPrefillSelectionCopyConfig", "write_v7d_prefill_selection_copy_plan"]
