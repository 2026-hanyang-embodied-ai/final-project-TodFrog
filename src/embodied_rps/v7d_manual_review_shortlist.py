"""Manual-review shortlist for v7d prompt-pose blocker candidates."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_REVIEW_ROOT = Path("artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_manual_review_shortlist_20260618")
BLOCKER_ROLES: tuple[str, ...] = ("hard_paper_prompt_window", "rock_wait_prompt_window")
SEED_REQUIRED_ROLES: tuple[str, ...] = (*BLOCKER_ROLES, "scissors_boundary_control")
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = (
    "candidate_root",
    "preview_image",
    "skeleton_npz",
    "source_path",
    "source_overlay_video",
    "source_skeleton_npz",
    "source_frame_log",
)
SHORTLIST_FIELDS: tuple[str, ...] = (
    "shortlist_rank",
    "target_name",
    "proposal_role",
    "segment_id",
    "suggested_review_tier",
    "detection_coverage",
    "severe_landmark_jump_count",
    "frame_count",
    "start_s",
    "end_s",
    "preview_image",
    "skeleton_npz",
    "decision",
    "reviewer_notes",
    "instruction",
)
REQUIRED_ROLE_CHECKLIST_FIELDS: tuple[str, ...] = (
    "proposal_role",
    "minimum_required_approved_count",
    "current_approved_count",
    "candidate_count",
    "role_state",
    "first_candidate_segment_id",
    "first_candidate_preview_image",
    "first_candidate_skeleton_npz",
    "next_action",
)


@dataclass(frozen=True)
class V7DManualReviewShortlistConfig:
    """Inputs for writing the v7d blocker manual-review shortlist."""

    project_root: Path = field(default_factory=Path.cwd)
    review_root: Path = DEFAULT_REVIEW_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


def write_v7d_manual_review_shortlist(config: V7DManualReviewShortlistConfig) -> dict[str, object]:
    """Write a non-mutating apply-compatible decision shortlist for v7d blocker roles."""

    project_root = config.project_root.resolve()
    review_root = _resolve_path(project_root, config.review_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    worklist_path = review_root / "segment_review_worklist.csv"
    decision_template_path = review_root / "segment_review_decision_template.csv"
    manifest_path = review_root / "segment_review_manifest.csv"
    gallery_path = review_root / "segment_review_gallery.html"
    if not worklist_path.exists():
        raise FileNotFoundError(f"Missing v7d segment review worklist: {worklist_path}")
    if not decision_template_path.exists():
        raise FileNotFoundError(f"Missing v7d segment review decision template: {decision_template_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing v7d segment review manifest: {manifest_path}")

    worklist_rows = _read_csv(worklist_path)
    decision_rows, decision_fieldnames = _read_csv_with_fieldnames(decision_template_path)
    decision_by_id = {str(row.get("segment_id", "")).strip(): row for row in decision_rows}

    blocker_rows = [_shortlist_row(row) for row in worklist_rows if _is_candidate_for_roles(row, BLOCKER_ROLES)]
    seed_required_rows = [_shortlist_row(row) for row in worklist_rows if _is_candidate_for_roles(row, SEED_REQUIRED_ROLES)]
    for row in seed_required_rows:
        _reject_heldout_metadata(row, context=worklist_path)
    blocker_rows.sort(key=_shortlist_sort_key)
    seed_required_rows.sort(key=_shortlist_sort_key)
    ranked_rows = [_ranked_row(index=index, row=row) for index, row in enumerate(blocker_rows, start=1)]
    seed_ranked_rows = [_ranked_row(index=index, row=row) for index, row in enumerate(seed_required_rows, start=1)]
    manifest_rows = _read_csv(manifest_path)
    required_role_checklist = _required_role_checklist(
        worklist_rows=worklist_rows,
        seed_required_rows=seed_ranked_rows,
        manifest_rows=manifest_rows,
    )
    missing_required_roles = [
        str(row["proposal_role"])
        for row in required_role_checklist
        if str(row.get("role_state")) != "approved"
    ]
    required_role_checklist_status = (
        "ready_after_manual_approval" if not missing_required_roles else "awaiting_required_role_approvals"
    )

    missing_from_template = [
        str(row.get("segment_id", "")).strip()
        for row in seed_ranked_rows
        if str(row.get("segment_id", "")).strip() not in decision_by_id
    ]
    if missing_from_template:
        raise ValueError(
            "Shortlist rows are missing from the apply-compatible decision template: "
            + ", ".join(missing_from_template)
        )
    blocker_decisions = [_blank_decision_row(decision_by_id[str(row["segment_id"])]) for row in ranked_rows]
    seed_required_decisions = [
        _blank_decision_row(decision_by_id[str(row["segment_id"])]) for row in seed_ranked_rows
    ]
    role_counts = dict(Counter(str(row.get("proposal_role", "")) for row in ranked_rows))
    seed_required_role_counts = dict(Counter(str(row.get("proposal_role", "")) for row in seed_ranked_rows))

    summary: dict[str, object] = {
        "status": "awaiting_manual_decisions",
        "branch_label": BRANCH_LABEL,
        "review_root": _display_path(review_root, base=project_root),
        "output_root": _display_path(output_root, base=project_root),
        "source_worklist_csv": _display_path(worklist_path, base=project_root),
        "source_decision_template_csv": _display_path(decision_template_path, base=project_root),
        "source_review_manifest_csv": _display_path(manifest_path, base=project_root),
        "source_gallery_html": _display_path(gallery_path, base=project_root) if gallery_path.exists() else "",
        "shortlist_csv": _display_path(output_root / "blocker_shortlist.csv", base=project_root),
        "shortlist_md": _display_path(output_root / "blocker_shortlist.md", base=project_root),
        "blocker_decision_template_csv": _display_path(output_root / "blocker_decision_template.csv", base=project_root),
        "seed_required_shortlist_csv": _display_path(output_root / "seed_required_shortlist.csv", base=project_root),
        "seed_required_decision_template_csv": _display_path(
            output_root / "seed_required_decision_template.csv",
            base=project_root,
        ),
        "required_role_checklist_csv": _display_path(
            output_root / "required_role_approval_checklist.csv",
            base=project_root,
        ),
        "required_role_checklist_md": _display_path(
            output_root / "required_role_approval_checklist.md",
            base=project_root,
        ),
        "summary_json": _display_path(output_root / "manual_review_shortlist_summary.json", base=project_root),
        "summary_md": _display_path(output_root / "manual_review_shortlist_summary.md", base=project_root),
        "shortlist_count": len(ranked_rows),
        "role_counts": role_counts,
        "seed_required_count": len(seed_ranked_rows),
        "seed_required_role_counts": seed_required_role_counts,
        "required_role_checklist_status": required_role_checklist_status,
        "missing_required_approved_roles": missing_required_roles,
        "decision_rows_populated": 0,
        "review_manifest_modified": False,
        "seed_package_created": False,
        "training_started": False,
        "approval_policy": (
            "this artifact only narrows the manual-review pool; decisions must be filled explicitly and validated "
            "with validate_v7d_review_decisions before dry-run/apply; every approve decision requires nonblank "
            "review_notes before any seed package can be built"
        ),
        "heldout_policy": "heldout */test MP4s remain validation-only and are rejected from shortlist metadata",
        "next_actions": [
            "inspect the gallery and blocker_shortlist.csv as temporal prompt-window evidence",
            "fill decision and review_notes values in seed_required_decision_template.csv after visual review when preparing a full v7d seed package",
            "run validate_v7d_review_decisions with the selected decision template",
            "dry-run run_v7d_post_approval_pipeline with --review-decision-mode dry-run",
            "rerun run_v7d_post_approval_pipeline with --review-decision-mode apply only after validation and dry-run report valid decisions",
            "build the v7d prompt-pose seed package only after hard-paper, rock/wait, and scissors boundary approvals exist",
        ],
    }
    _write_csv(output_root / "blocker_shortlist.csv", SHORTLIST_FIELDS, ranked_rows)
    _write_csv(output_root / "blocker_decision_template.csv", decision_fieldnames, blocker_decisions)
    _write_csv(output_root / "seed_required_shortlist.csv", SHORTLIST_FIELDS, seed_ranked_rows)
    _write_csv(output_root / "seed_required_decision_template.csv", decision_fieldnames, seed_required_decisions)
    _write_csv(output_root / "required_role_approval_checklist.csv", REQUIRED_ROLE_CHECKLIST_FIELDS, required_role_checklist)
    (output_root / "manual_review_shortlist_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "manual_review_shortlist_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    (output_root / "blocker_shortlist.md").write_text(
        _shortlist_markdown(summary=summary, rows=ranked_rows),
        encoding="utf-8",
    )
    (output_root / "required_role_approval_checklist.md").write_text(
        _required_role_checklist_markdown(summary=summary, rows=required_role_checklist),
        encoding="utf-8",
    )
    return summary


def _required_role_checklist(
    *,
    worklist_rows: Sequence[Mapping[str, object]],
    seed_required_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    role_by_segment = {
        str(row.get("segment_id", "")).strip(): str(row.get("proposal_role", "")).strip()
        for row in worklist_rows
    }
    approved_segment_ids = {
        str(row.get("segment_id", "")).strip()
        for row in manifest_rows
        if _manifest_row_approved(row)
    }
    approved_counts: dict[str, int] = {role: 0 for role in SEED_REQUIRED_ROLES}
    for segment_id in approved_segment_ids:
        role = role_by_segment.get(segment_id, "")
        if role in approved_counts:
            approved_counts[role] += 1

    rows: list[dict[str, object]] = []
    for role in SEED_REQUIRED_ROLES:
        candidates = [
            row for row in seed_required_rows
            if str(row.get("proposal_role", "")).strip() == role
            and str(row.get("segment_id", "")).strip() not in approved_segment_ids
        ]
        approved_count = approved_counts[role]
        first_candidate = candidates[0] if candidates else {}
        if approved_count > 0:
            role_state = "approved"
            next_action = "no additional approval required for this role before seed-readiness check"
        elif candidates:
            role_state = "missing_approval"
            next_action = (
                "review at least one candidate after temporal visual inspection, then fill decision and "
                "review_notes and run validate_v7d_review_decisions before dry-run/apply"
            )
        else:
            role_state = "no_quality_pass_candidate"
            next_action = "collect or extract additional nonheldout prompt-window evidence for this role"
        rows.append(
            {
                "proposal_role": role,
                "minimum_required_approved_count": 1,
                "current_approved_count": approved_count,
                "candidate_count": len(candidates),
                "role_state": role_state,
                "first_candidate_segment_id": first_candidate.get("segment_id", ""),
                "first_candidate_preview_image": first_candidate.get("preview_image", ""),
                "first_candidate_skeleton_npz": first_candidate.get("skeleton_npz", ""),
                "next_action": next_action,
            }
        )
    return rows


def _is_candidate_for_roles(row: Mapping[str, object], roles: Sequence[str]) -> bool:
    role = str(row.get("proposal_role", "")).strip()
    return (
        role in roles
        and str(row.get("eligible_for_manual_approval", "")).strip().lower() == "true"
        and str(row.get("quality_status", "")).strip() == "auto_quality_pass"
        and str(row.get("currently_approved_for_training", "")).strip().lower() != "true"
    )


def _shortlist_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "target_name": str(row.get("target_name", "")).strip(),
        "proposal_role": str(row.get("proposal_role", "")).strip(),
        "segment_id": str(row.get("segment_id", "")).strip(),
        "suggested_review_tier": _review_tier(str(row.get("proposal_role", "")).strip()),
        "detection_coverage": _format_float(row.get("detection_coverage")),
        "severe_landmark_jump_count": str(row.get("severe_landmark_jump_count", "")).strip(),
        "frame_count": str(row.get("frame_count", "")).strip(),
        "start_s": _format_float(row.get("start_s")),
        "end_s": _format_float(row.get("end_s")),
        "preview_image": str(row.get("preview_image", "")).strip(),
        "skeleton_npz": str(row.get("skeleton_npz", "")).strip(),
        "decision": "",
        "reviewer_notes": "",
        "instruction": "inspect_prompt_window_temporal_segment_then_fill_explicit_decision",
    }


def _ranked_row(*, index: int, row: Mapping[str, object]) -> dict[str, object]:
    ranked = dict(row)
    ranked["shortlist_rank"] = index
    return ranked


def _blank_decision_row(row: Mapping[str, object]) -> dict[str, object]:
    blank = dict(row)
    blank["decision"] = ""
    if "review_notes" in blank:
        blank["review_notes"] = ""
    if "instruction" in blank:
        blank["instruction"] = (
            "after temporal visual inspection, set decision to approve/reject/needs_review; "
            "approval requires nonblank review_notes and validate_v7d_review_decisions before apply"
        )
    return blank


def _shortlist_sort_key(row: Mapping[str, object]) -> tuple[int, int, float, float, str]:
    role = str(row.get("proposal_role", "")).strip()
    role_priority = {
        "hard_paper_prompt_window": 0,
        "rock_wait_prompt_window": 1,
        "scissors_boundary_control": 2,
    }.get(role, 99)
    jumps = _int_value(row.get("severe_landmark_jump_count"))
    coverage = _float_value(row.get("detection_coverage"))
    start_s = _float_value(row.get("start_s"))
    segment_id = str(row.get("segment_id", "")).strip()
    return (role_priority, jumps, -coverage, start_s, segment_id)


def _review_tier(role: str) -> str:
    if role == "hard_paper_prompt_window":
        return "review_first_hard_paper_prompt_window"
    if role == "rock_wait_prompt_window":
        return "review_first_rock_wait_prompt_window"
    if role == "scissors_boundary_control":
        return "review_scissors_boundary_control_before_full_seed_package"
    return "not_a_v7d_blocker_role"


def _summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V7d Manual Review Shortlist",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Shortlist count: `{summary.get('shortlist_count')}`",
        f"- Seed-required count: `{summary.get('seed_required_count')}`",
        f"- Review root: `{summary.get('review_root')}`",
        f"- Decision template: `{summary.get('blocker_decision_template_csv')}`",
        "- This artifact does not approve rows, edit `segment_review_manifest.csv`, build a seed package, or start training.",
        "",
        "## Role Counts",
        "",
    ]
    role_counts = summary.get("role_counts")
    if isinstance(role_counts, Mapping):
        for role, count in sorted(role_counts.items()):
            lines.append(f"- `{role}`: `{count}`")
    lines.extend(["", "## Commands After Manual Review", ""])
    lines.extend(
        [
            "```powershell",
            "$env:PYTHONPATH='src'",
            "python -m embodied_rps.tools.validate_v7d_review_decisions --review-root artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618 --decisions-csv artifacts/real_skeleton_v7d_manual_review_shortlist_20260618/seed_required_decision_template.csv --output-root artifacts/real_skeleton_v7d_review_decision_validation_20260618",
            "python -m embodied_rps.tools.run_v7d_post_approval_pipeline --review-decision-mode dry-run",
            "python -m embodied_rps.tools.run_v7d_post_approval_pipeline --review-decision-mode apply",
            "python -m embodied_rps.tools.check_v7d_prompt_pose_seed_readiness",
            "python -m embodied_rps.tools.build_v7d_prompt_pose_seed_package",
            "```",
            "",
            "Every `approve` decision requires nonblank `review_notes`. Run the validator before dry-run/apply.",
            "",
        ]
    )
    return "\n".join(lines)


def _shortlist_markdown(*, summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# V7d Blocker Segment Shortlist",
        "",
        f"- Source gallery: `{summary.get('source_gallery_html')}`",
        "- Review these as prompt-conditioned temporal segments, not as independent final-label thumbnails.",
        "- Fill decisions in the apply-compatible `blocker_decision_template.csv` only after visual review.",
        "",
        "| Rank | Segment | Target | Role | Coverage | Jumps | Time | Preview |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("shortlist_rank", "")),
                    f"`{row.get('segment_id', '')}`",
                    str(row.get("target_name", "")),
                    str(row.get("proposal_role", "")),
                    str(row.get("detection_coverage", "")),
                    str(row.get("severe_landmark_jump_count", "")),
                    f"{row.get('start_s', '')}-{row.get('end_s', '')}",
                    str(row.get("preview_image", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _required_role_checklist_markdown(*, summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# V7d Required Role Approval Checklist",
        "",
        f"- Status: `{summary.get('required_role_checklist_status')}`",
        f"- Missing required roles: `{summary.get('missing_required_approved_roles')}`",
        "- This file is a review checklist only. It does not approve rows or edit the review manifest.",
        "",
        "| Role | Approved | Candidates | State | First Candidate | Next Action |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('proposal_role', '')}`",
                    str(row.get("current_approved_count", "")),
                    str(row.get("candidate_count", "")),
                    str(row.get("role_state", "")),
                    f"`{row.get('first_candidate_segment_id', '')}`",
                    str(row.get("next_action", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    rows, _ = _read_csv_with_fieldnames(path)
    return rows


def _read_csv_with_fieldnames(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return [dict(row) for row in reader], fieldnames


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _manifest_row_approved(row: Mapping[str, object]) -> bool:
    return _truthy(row.get("approved_for_training")) and str(row.get("review_status", "")).strip().lower() == "approved"


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "approved"}


def _reject_heldout_metadata(row: Mapping[str, object], *, context: Path) -> None:
    for field_name in PATH_FIELDS_TO_AUDIT:
        value = str(row.get(field_name, "")).strip()
        if value and _is_heldout_test_path(value):
            raise ValueError(f"{context} contains held-out test candidate path in {field_name}: {value}")


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


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.6f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""


def _float_value(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _int_value(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 999999


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["V7DManualReviewShortlistConfig", "write_v7d_manual_review_shortlist"]
