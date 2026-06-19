"""Non-mutating validator for v7d manual review decision CSVs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_REVIEW_ROOT = Path("artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618")
DEFAULT_DECISIONS_CSV = Path("artifacts/real_skeleton_v7d_manual_review_shortlist_20260618/seed_required_decision_template.csv")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_review_decision_validation_20260618")
REQUIRED_ROLES: tuple[str, ...] = (
    "hard_paper_prompt_window",
    "rock_wait_prompt_window",
    "scissors_boundary_control",
)
REQUIRED_DECISION_COLUMNS: frozenset[str] = frozenset(
    {"segment_id", "decision", "proposed_segments_sha256", "review_manifest_sha256"}
)


@dataclass(frozen=True)
class V7DReviewDecisionValidatorConfig:
    """Inputs for validating v7d manual review decisions without mutation."""

    project_root: Path = field(default_factory=Path.cwd)
    review_root: Path = DEFAULT_REVIEW_ROOT
    decisions_csv: Path = DEFAULT_DECISIONS_CSV
    output_root: Path = DEFAULT_OUTPUT_ROOT


def validate_v7d_review_decisions(config: V7DReviewDecisionValidatorConfig) -> dict[str, object]:
    """Validate a v7d review decision CSV and write an isolated summary artifact."""

    project_root = config.project_root.resolve()
    review_root = _resolve_path(project_root, config.review_root)
    decisions_csv = _resolve_path(project_root, config.decisions_csv)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    proposed_path = review_root / "proposed_segments.jsonl"
    manifest_path = review_root / "segment_review_manifest.csv"
    if not proposed_path.exists():
        raise FileNotFoundError(f"Missing proposed segments: {proposed_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing review manifest: {manifest_path}")
    if not decisions_csv.exists():
        raise FileNotFoundError(f"Missing decision CSV: {decisions_csv}")

    proposed_sha256 = _file_sha256(proposed_path)
    manifest_sha256 = _file_sha256(manifest_path)
    proposed_by_id = {
        str(row.get("segment_id", "")).strip(): row
        for row in _read_jsonl(proposed_path)
        if str(row.get("segment_id", "")).strip()
    }
    manifest_rows, manifest_fieldnames = _read_csv_with_fieldnames(manifest_path)
    if not {"segment_id", "review_status", "approved_for_training"}.issubset(set(manifest_fieldnames)):
        raise ValueError(f"{manifest_path} is missing required review columns")
    manifest_by_id = {
        str(row.get("segment_id", "")).strip(): row
        for row in manifest_rows
        if str(row.get("segment_id", "")).strip()
    }
    decision_rows, decision_fieldnames = _read_csv_with_fieldnames(decisions_csv)

    decisions: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    missing_columns = sorted(REQUIRED_DECISION_COLUMNS.difference(set(decision_fieldnames)))
    if missing_columns:
        failures.append({"code": "decision_csv_missing_required_columns", "columns": missing_columns})

    seen_segments: set[str] = set()
    for row_number, row in enumerate(decision_rows, start=2):
        segment_id = str(row.get("segment_id", "")).strip()
        raw_decision = str(row.get("decision", "")).strip()
        if not raw_decision:
            continue
        if segment_id in seen_segments:
            failures.append({"code": "duplicate_review_decision_segment", "segment_id": segment_id, "row": row_number})
            continue
        if segment_id:
            seen_segments.add(segment_id)
        if str(row.get("proposed_segments_sha256", "")).strip() != proposed_sha256:
            failures.append({"code": "stale_proposed_segments_hash", "segment_id": segment_id, "row": row_number})
            continue
        if str(row.get("review_manifest_sha256", "")).strip() != manifest_sha256:
            failures.append({"code": "stale_review_manifest_hash", "segment_id": segment_id, "row": row_number})
            continue
        normalized = _normalize_review_decision(raw_decision)
        if not segment_id:
            failures.append({"code": "missing_segment_id", "row": row_number})
            continue
        proposal = proposed_by_id.get(segment_id)
        if proposal is None:
            failures.append({"code": "decision_segment_missing_from_proposals", "segment_id": segment_id, "row": row_number})
            continue
        if segment_id not in manifest_by_id:
            failures.append({"code": "decision_segment_missing_from_review_manifest", "segment_id": segment_id, "row": row_number})
            continue
        if normalized is None:
            failures.append(
                {"code": "unsupported_review_decision", "segment_id": segment_id, "decision": raw_decision, "row": row_number}
            )
            continue
        if normalized == "approve":
            if str(proposal.get("quality_status", "")).strip() != "auto_quality_pass":
                failures.append({"code": "cannot_approve_failed_auto_quality", "segment_id": segment_id, "row": row_number})
            if _is_heldout_test_path(str(proposal.get("source_path", ""))):
                failures.append({"code": "cannot_approve_heldout_source_path", "segment_id": segment_id, "row": row_number})
            skeleton_npz = _resolve_path(review_root, Path(str(proposal.get("skeleton_npz", "")).strip()))
            if not skeleton_npz.exists():
                failures.append({"code": "cannot_approve_missing_skeleton_npz", "segment_id": segment_id, "row": row_number})
        decisions.append(
            {
                "segment_id": segment_id,
                "decision": normalized,
                "target_name": str(proposal.get("target_name", "")).strip(),
                "proposal_role": str(proposal.get("proposal_role", "")).strip(),
                "review_notes": str(row.get("review_notes", "")).strip(),
                "row": row_number,
            }
        )

    approved_segment_ids_by_role = _predicted_approved_segments_by_role(
        proposed_by_id=proposed_by_id,
        manifest_rows=manifest_rows,
        decisions=decisions,
    )
    approved_counts_by_role = {role: len(approved_segment_ids_by_role[role]) for role in REQUIRED_ROLES}
    missing_required = [role for role in REQUIRED_ROLES if approved_counts_by_role[role] <= 0]
    approval_decisions_missing_review_notes = sorted(
        str(decision.get("segment_id", ""))
        for decision in decisions
        if decision.get("decision") == "approve" and not str(decision.get("review_notes", "")).strip()
    )

    decision_count = len(decisions)
    approve_count = sum(1 for decision in decisions if decision.get("decision") == "approve")
    reject_count = sum(1 for decision in decisions if decision.get("decision") == "reject")
    needs_review_count = sum(1 for decision in decisions if decision.get("decision") == "needs_review")
    status = _status(
        failures=failures,
        decision_count=decision_count,
        missing_required=missing_required,
        approval_decisions_missing_review_notes=approval_decisions_missing_review_notes,
    )
    decisions_apply_safe = status == "ready_for_apply"
    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "review_root": _display_path(review_root, base=project_root),
        "decisions_csv": _display_path(decisions_csv, base=project_root),
        "output_root": _display_path(output_root, base=project_root),
        "review_manifest": _display_path(manifest_path, base=project_root),
        "current_proposed_segments_sha256": proposed_sha256,
        "current_review_manifest_sha256": manifest_sha256,
        "decision_count": decision_count,
        "approve_count": approve_count,
        "reject_count": reject_count,
        "needs_review_count": needs_review_count,
        "decision_target_counts": _count_by(decisions, "target_name"),
        "approve_target_counts": _count_by(
            [decision for decision in decisions if decision.get("decision") == "approve"],
            "target_name",
        ),
        "required_approved_roles": list(REQUIRED_ROLES),
        "approved_counts_by_role": approved_counts_by_role,
        "approved_segment_ids_by_role": {
            role: sorted(approved_segment_ids_by_role[role])
            for role in REQUIRED_ROLES
        },
        "missing_required_approved_roles": missing_required,
        "approval_decisions_missing_review_notes": approval_decisions_missing_review_notes,
        "failures": failures,
        "decisions_apply_safe": decisions_apply_safe,
        "review_manifest_modified": False,
        "seed_package_created": False,
        "dataset_generated": False,
        "training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test paths are rejected from approved v7d decision metadata",
        "next_action": _next_action(status),
    }
    (output_root / "v7d_review_decision_validation_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7d_review_decision_validation_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _predicted_approved_segments_by_role(
    *,
    proposed_by_id: Mapping[str, Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]],
) -> dict[str, set[str]]:
    approved: dict[str, set[str]] = {role: set() for role in REQUIRED_ROLES}
    for row in manifest_rows:
        segment_id = str(row.get("segment_id", "")).strip()
        role = str(proposed_by_id.get(segment_id, {}).get("proposal_role", "")).strip()
        if role in approved and _manifest_row_approved(row):
            approved[role].add(segment_id)
    for decision in decisions:
        segment_id = str(decision.get("segment_id", "")).strip()
        role = str(decision.get("proposal_role", "")).strip()
        if role not in approved or not segment_id:
            continue
        if decision.get("decision") == "approve":
            approved[role].add(segment_id)
        elif decision.get("decision") in {"reject", "needs_review"}:
            approved[role].discard(segment_id)
    return approved


def _status(
    *,
    failures: Sequence[Mapping[str, object]],
    decision_count: int,
    missing_required: Sequence[str],
    approval_decisions_missing_review_notes: Sequence[str],
) -> str:
    if failures:
        return "invalid_decisions"
    if decision_count <= 0:
        return "no_review_decisions"
    if missing_required:
        return "missing_required_roles"
    if approval_decisions_missing_review_notes:
        return "approval_notes_missing"
    return "ready_for_apply"


def _next_action(status: str) -> str:
    if status == "ready_for_apply":
        return "run the v7d post-approval pipeline with --review-decision-mode apply"
    if status == "missing_required_roles":
        return "add explicit approvals for every required v7d prompt-window role"
    if status == "approval_notes_missing":
        return "add review_notes for every approval decision before applying"
    if status == "invalid_decisions":
        return "fix the decision CSV hashes, syntax, notes, or segment references before apply"
    return "fill explicit decisions after visual temporal review"


def _summary_markdown(summary: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# V7d Review Decision Validation",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Decision count: `{summary.get('decision_count')}`",
            f"- Approve count: `{summary.get('approve_count')}`",
            f"- Approved counts by role: `{summary.get('approved_counts_by_role')}`",
            f"- Missing required roles: `{summary.get('missing_required_approved_roles')}`",
            f"- Apply safe: `{summary.get('decisions_apply_safe')}`",
            f"- Review manifest modified: `{summary.get('review_manifest_modified')}`",
            f"- Next action: `{summary.get('next_action')}`",
            "",
        ]
    )


def _count_by(rows: Sequence[Mapping[str, object]], field_name: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        key = str(row.get(field_name, "")).strip()
        if key:
            counts[key] += 1
    return dict(sorted(counts.items()))


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


def _read_csv_with_fieldnames(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_heldout_test_path(value: str) -> bool:
    normalized = value.replace("\\", "/").lower().rstrip("/")
    return "/test/" in normalized or normalized.endswith("/test")


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
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["V7DReviewDecisionValidatorConfig", "validate_v7d_review_decisions"]
