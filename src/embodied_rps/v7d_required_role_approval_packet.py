"""Non-mutating v7d required-role approval review packet."""

from __future__ import annotations

import csv
import html
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_SHORTLIST_ROOT = Path("artifacts/real_skeleton_v7d_manual_review_shortlist_20260618")
DEFAULT_TEMPORAL_REVIEW_ROOT = Path("artifacts/real_skeleton_v7d_temporal_review_20260618")
DEFAULT_REVIEW_ROOT = Path("artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_required_role_approval_packet_20260618")
REQUIRED_ROLES: tuple[str, ...] = (
    "hard_paper_prompt_window",
    "rock_wait_prompt_window",
    "scissors_boundary_control",
)
PACKET_FIELDS: tuple[str, ...] = (
    "proposal_role",
    "role_state",
    "current_approved_count",
    "candidate_count",
    "first_candidate_segment_id",
    "target_name",
    "temporal_strip",
    "preview_image",
    "skeleton_npz",
    "decision_template_csv",
    "suggested_decision",
    "instruction",
)
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = (
    "temporal_strip",
    "preview_image",
    "skeleton_npz",
    "decision_template_csv",
)


@dataclass(frozen=True)
class V7DRequiredRoleApprovalPacketConfig:
    """Inputs for writing the v7d required-role approval packet."""

    project_root: Path = field(default_factory=Path.cwd)
    shortlist_root: Path = DEFAULT_SHORTLIST_ROOT
    temporal_review_root: Path = DEFAULT_TEMPORAL_REVIEW_ROOT
    review_root: Path = DEFAULT_REVIEW_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


def write_v7d_required_role_approval_packet(
    config: V7DRequiredRoleApprovalPacketConfig,
) -> dict[str, object]:
    """Write a compact review packet for the required v7d seed roles."""

    project_root = config.project_root.resolve()
    shortlist_root = _resolve_path(project_root, config.shortlist_root)
    temporal_root = _resolve_path(project_root, config.temporal_review_root)
    review_root = _resolve_path(project_root, config.review_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    checklist_path = shortlist_root / "required_role_approval_checklist.csv"
    temporal_manifest_path = temporal_root / "temporal_review_manifest.csv"
    decision_template_path = shortlist_root / "seed_required_decision_template.csv"
    for path in (checklist_path, temporal_manifest_path, decision_template_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing v7d approval packet input: {path}")

    checklist_rows = _read_csv(checklist_path)
    temporal_by_segment = {
        str(row.get("segment_id", "")).strip(): row
        for row in _read_csv(temporal_manifest_path)
        if str(row.get("segment_id", "")).strip()
    }
    packet_rows = [
        _packet_row(
            checklist_row=row,
            temporal_by_segment=temporal_by_segment,
            output_root=output_root,
            shortlist_root=shortlist_root,
            temporal_root=temporal_root,
            review_root=review_root,
        )
        for row in checklist_rows
        if str(row.get("proposal_role", "")).strip() in REQUIRED_ROLES
    ]
    packet_rows.sort(key=lambda row: REQUIRED_ROLES.index(str(row["proposal_role"])))
    for row in packet_rows:
        _reject_heldout_metadata(row, context=checklist_path)

    missing_roles = [
        str(row["proposal_role"])
        for row in packet_rows
        if _int_value(row.get("current_approved_count")) < 1
    ]
    summary: dict[str, object] = {
        "status": "awaiting_manual_approval" if missing_roles else "ready_after_manual_approval",
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "shortlist_root": _display_path(shortlist_root, base=project_root),
        "temporal_review_root": _display_path(temporal_root, base=project_root),
        "review_root": _display_path(review_root, base=project_root),
        "packet_csv": _display_path(output_root / "required_role_approval_packet.csv", base=project_root),
        "packet_html": _display_path(output_root / "required_role_approval_packet.html", base=project_root),
        "packet_summary_json": _display_path(output_root / "required_role_approval_packet_summary.json", base=project_root),
        "packet_summary_md": _display_path(output_root / "required_role_approval_packet_summary.md", base=project_root),
        "decision_template_csv": _display_path(decision_template_path, base=project_root),
        "packet_row_count": len(packet_rows),
        "required_roles": list(REQUIRED_ROLES),
        "missing_required_approved_roles": missing_roles,
        "decisions_applied": False,
        "review_manifest_modified": False,
        "seed_package_created": False,
        "dataset_generated": False,
        "training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test paths are rejected from the approval packet metadata",
        "next_action": (
            "inspect temporal strips and previews, then fill decision and review_notes in "
            "seed_required_decision_template.csv and run validate_v7d_review_decisions"
        ),
    }
    _write_csv(output_root / "required_role_approval_packet.csv", PACKET_FIELDS, packet_rows)
    (output_root / "required_role_approval_packet.html").write_text(
        _packet_html(summary=summary, rows=packet_rows),
        encoding="utf-8",
    )
    (output_root / "required_role_approval_packet_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "required_role_approval_packet_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _packet_row(
    *,
    checklist_row: Mapping[str, object],
    temporal_by_segment: Mapping[str, Mapping[str, object]],
    output_root: Path,
    shortlist_root: Path,
    temporal_root: Path,
    review_root: Path,
) -> dict[str, object]:
    segment_id = str(checklist_row.get("first_candidate_segment_id", "")).strip()
    temporal = temporal_by_segment.get(segment_id, {})
    temporal_strip = str(temporal.get("temporal_strip", "")).strip()
    preview_image = str(checklist_row.get("first_candidate_preview_image", "")).strip()
    skeleton_npz = str(checklist_row.get("first_candidate_skeleton_npz", "")).strip()
    return {
        "proposal_role": str(checklist_row.get("proposal_role", "")).strip(),
        "role_state": str(checklist_row.get("role_state", "")).strip(),
        "current_approved_count": str(checklist_row.get("current_approved_count", "")).strip(),
        "candidate_count": str(checklist_row.get("candidate_count", "")).strip(),
        "first_candidate_segment_id": segment_id,
        "target_name": str(temporal.get("target_name", "")).strip(),
        "temporal_strip": _relative_link(output_root, temporal_root / temporal_strip) if temporal_strip else "",
        "preview_image": _relative_link(output_root, review_root / preview_image) if preview_image else "",
        "skeleton_npz": _relative_link(output_root, review_root / skeleton_npz) if skeleton_npz else "",
        "decision_template_csv": _relative_link(output_root, shortlist_root / "seed_required_decision_template.csv"),
        "suggested_decision": "",
        "instruction": (
            "inspect temporal strip and preview; if approving, fill decision=approve and nonblank "
            "review_notes, then run validate_v7d_review_decisions before dry-run/apply"
        ),
    }


def _packet_html(*, summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    cards: list[str] = []
    for row in rows:
        strip = html.escape(str(row.get("temporal_strip", "")))
        preview = html.escape(str(row.get("preview_image", "")))
        segment_id = html.escape(str(row.get("first_candidate_segment_id", "")))
        role = html.escape(str(row.get("proposal_role", "")))
        state = html.escape(str(row.get("role_state", "")))
        cards.append(
            "\n".join(
                [
                    '<section class="role-card">',
                    f"<h2>{role}</h2>",
                    f"<p>state: <code>{state}</code> segment: <code>{segment_id}</code></p>",
                    f'<img src="{strip}" alt="temporal strip for {segment_id}">',
                    f'<p><a href="{preview}">Preview image</a> | <a href="{html.escape(str(row.get("decision_template_csv", "")))}">Decision template</a></p>',
                    "</section>",
                ]
            )
        )
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>V7d Required Role Approval Packet</title>",
            "<style>body{font-family:Arial,sans-serif;margin:24px;max-width:1100px} .role-card{border:1px solid #ccc;padding:12px;margin:16px 0} img{max-width:100%;height:auto;display:block}</style>",
            "</head>",
            "<body>",
            "<h1>V7d Required Role Approval Packet</h1>",
            f"<p>Status: <code>{html.escape(str(summary.get('status', '')))}</code></p>",
            "<p>This packet is evidence only. It does not approve rows, train models, validate MP4s, or promote v7d.</p>",
            "<p>Approval requires a nonblank <code>review_notes</code> value and a passing <code>validate_v7d_review_decisions</code> run before dry-run/apply.</p>",
            *cards,
            "</body>",
            "</html>",
            "",
        ]
    )


def _summary_markdown(summary: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# V7d Required Role Approval Packet",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Packet rows: `{summary.get('packet_row_count')}`",
            f"- Missing required roles: `{summary.get('missing_required_approved_roles')}`",
            f"- Decision template: `{summary.get('decision_template_csv')}`",
            "- This packet is evidence only and does not apply approval decisions.",
            "- Approval requires nonblank `review_notes` and `validate_v7d_review_decisions` before dry-run/apply.",
            f"- Next action: `{summary.get('next_action')}`",
            "",
        ]
    )


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


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _relative_link(output_root: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=output_root.resolve()).replace("\\", "/")


def _int_value(value: object) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["V7DRequiredRoleApprovalPacketConfig", "write_v7d_required_role_approval_packet"]
