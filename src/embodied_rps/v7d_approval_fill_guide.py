"""Non-mutating v7d manual approval fill guide."""

from __future__ import annotations

import csv
import html
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_SHORTLIST_ROOT = Path("artifacts/real_skeleton_v7d_manual_review_shortlist_20260618")
DEFAULT_TEMPORAL_REVIEW_ROOT = Path("artifacts/real_skeleton_v7d_temporal_review_20260618")
DEFAULT_REVIEW_ROOT = Path("artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_approval_fill_guide_20260618")
REQUIRED_ROLES: tuple[str, ...] = (
    "hard_paper_prompt_window",
    "rock_wait_prompt_window",
    "scissors_boundary_control",
)
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = (
    "preview_image",
    "skeleton_npz",
    "source_path",
    "temporal_strip",
    "decision_template_csv",
)
GUIDE_FIELDS: tuple[str, ...] = (
    "proposal_role",
    "rank_in_role",
    "segment_id",
    "target_name",
    "detection_coverage",
    "severe_landmark_jump_count",
    "frame_count",
    "start_s",
    "end_s",
    "temporal_strip",
    "preview_image",
    "skeleton_npz",
    "decision_template_csv",
    "suggested_review_note",
    "decision",
    "review_notes",
    "instruction",
)


@dataclass(frozen=True)
class V7DApprovalFillGuideConfig:
    """Inputs for the v7d approval fill guide."""

    project_root: Path = field(default_factory=Path.cwd)
    shortlist_root: Path = DEFAULT_SHORTLIST_ROOT
    temporal_review_root: Path = DEFAULT_TEMPORAL_REVIEW_ROOT
    review_root: Path = DEFAULT_REVIEW_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    candidates_per_role: int = 3


def write_v7d_approval_fill_guide(config: V7DApprovalFillGuideConfig) -> dict[str, object]:
    """Write a ranked reviewer aid without applying decisions."""

    project_root = config.project_root.resolve()
    shortlist_root = _resolve_path(project_root, config.shortlist_root)
    temporal_root = _resolve_path(project_root, config.temporal_review_root)
    review_root = _resolve_path(project_root, config.review_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    shortlist_path = shortlist_root / "seed_required_shortlist.csv"
    temporal_path = temporal_root / "temporal_review_manifest.csv"
    decision_template = shortlist_root / "seed_required_decision_template.csv"
    for path in (shortlist_path, temporal_path, decision_template):
        if not path.exists():
            raise FileNotFoundError(f"Missing v7d approval fill-guide input: {path}")

    temporal_by_segment = {
        str(row.get("segment_id", "")).strip(): row
        for row in _read_csv(temporal_path)
        if str(row.get("segment_id", "")).strip()
    }
    rows_by_role: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(shortlist_path):
        role = str(row.get("proposal_role", "")).strip()
        if role in REQUIRED_ROLES:
            _reject_heldout_metadata(row, context=shortlist_path)
            rows_by_role[role].append(row)

    guide_rows: list[dict[str, object]] = []
    for role in REQUIRED_ROLES:
        ranked = sorted(rows_by_role.get(role, []), key=_rank_key)
        for rank, row in enumerate(ranked[: max(0, int(config.candidates_per_role))], start=1):
            guide_row = _guide_row(
                row=row,
                rank_in_role=rank,
                temporal_by_segment=temporal_by_segment,
                output_root=output_root,
                review_root=review_root,
                temporal_root=temporal_root,
                shortlist_root=shortlist_root,
            )
            _reject_heldout_metadata(guide_row, context=shortlist_path)
            guide_rows.append(guide_row)

    missing_roles = [role for role in REQUIRED_ROLES if role not in rows_by_role or not rows_by_role[role]]
    summary: dict[str, object] = {
        "status": "awaiting_manual_approval",
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "shortlist_root": _display_path(shortlist_root, base=project_root),
        "temporal_review_root": _display_path(temporal_root, base=project_root),
        "review_root": _display_path(review_root, base=project_root),
        "guide_csv": _display_path(output_root / "approval_fill_guide.csv", base=project_root),
        "guide_html": _display_path(output_root / "approval_fill_guide.html", base=project_root),
        "guide_md": _display_path(output_root / "approval_fill_guide.md", base=project_root),
        "summary_json": _display_path(output_root / "approval_fill_guide_summary.json", base=project_root),
        "decision_template_csv": _display_path(decision_template, base=project_root),
        "guide_row_count": len(guide_rows),
        "candidates_per_role": int(config.candidates_per_role),
        "required_roles": list(REQUIRED_ROLES),
        "missing_candidate_roles": missing_roles,
        "decisions_applied": False,
        "review_manifest_modified": False,
        "seed_package_created": False,
        "dataset_generated": False,
        "training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test paths are rejected from fill-guide metadata",
        "next_action": (
            "open approval_fill_guide.html and temporal evidence, then manually copy selected segment decisions "
            "and nonblank review_notes into seed_required_decision_template.csv before validation"
        ),
        "config": _config_summary(project_root=project_root, config=config),
    }
    _write_csv(output_root / "approval_fill_guide.csv", GUIDE_FIELDS, guide_rows)
    (output_root / "approval_fill_guide.html").write_text(
        _guide_html(summary=summary, rows=guide_rows),
        encoding="utf-8",
    )
    (output_root / "approval_fill_guide.md").write_text(
        _guide_markdown(summary=summary, rows=guide_rows),
        encoding="utf-8",
    )
    (output_root / "approval_fill_guide_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _guide_row(
    *,
    row: Mapping[str, object],
    rank_in_role: int,
    temporal_by_segment: Mapping[str, Mapping[str, object]],
    output_root: Path,
    review_root: Path,
    temporal_root: Path,
    shortlist_root: Path,
) -> dict[str, object]:
    segment_id = str(row.get("segment_id", "")).strip()
    role = str(row.get("proposal_role", "")).strip()
    temporal = temporal_by_segment.get(segment_id, {})
    temporal_strip = str(temporal.get("temporal_strip", "")).strip()
    preview = str(row.get("preview_image", "")).strip()
    skeleton = str(row.get("skeleton_npz", "")).strip()
    return {
        "proposal_role": role,
        "rank_in_role": int(rank_in_role),
        "segment_id": segment_id,
        "target_name": str(row.get("target_name", "")).strip(),
        "detection_coverage": str(row.get("detection_coverage", "")).strip(),
        "severe_landmark_jump_count": str(row.get("severe_landmark_jump_count", "")).strip(),
        "frame_count": str(row.get("frame_count", "")).strip(),
        "start_s": str(row.get("start_s", "")).strip(),
        "end_s": str(row.get("end_s", "")).strip(),
        "temporal_strip": _relative_link(output_root, temporal_root / temporal_strip) if temporal_strip else "",
        "preview_image": _relative_link(output_root, review_root / preview) if preview else "",
        "skeleton_npz": _relative_link(output_root, review_root / skeleton) if skeleton else "",
        "decision_template_csv": _relative_link(output_root, shortlist_root / "seed_required_decision_template.csv"),
        "suggested_review_note": _suggested_note(role=role, segment_id=segment_id),
        "decision": "",
        "review_notes": "",
        "instruction": (
            "decision intentionally blank; after temporal inspection, copy decision=approve and a nonblank "
            "review_notes value into seed_required_decision_template.csv, then run validate_v7d_review_decisions"
        ),
    }


def _suggested_note(*, role: str, segment_id: str) -> str:
    role_text = {
        "hard_paper_prompt_window": "Approved after temporal review as prompt-window paper evidence resolving inside PROMPT SCISSORS.",
        "rock_wait_prompt_window": "Approved after temporal review as prompt-window rock/wait hard-negative evidence during PROMPT SCISSORS.",
        "scissors_boundary_control": "Approved after temporal review as prompt-window scissors boundary-control evidence during PROMPT SCISSORS.",
    }.get(role, "Approved after temporal prompt-window review.")
    return f"{role_text} Segment {segment_id}."


def _rank_key(row: Mapping[str, object]) -> tuple[int, float, float, str]:
    return (
        _int_value(row.get("severe_landmark_jump_count")),
        -_float_value(row.get("detection_coverage")),
        _float_value(row.get("start_s")),
        str(row.get("segment_id", "")).strip(),
    )


def _guide_html(*, summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    cards: list[str] = []
    for row in rows:
        role = html.escape(str(row.get("proposal_role", "")))
        segment = html.escape(str(row.get("segment_id", "")))
        strip = html.escape(str(row.get("temporal_strip", "")))
        preview = html.escape(str(row.get("preview_image", "")))
        note = html.escape(str(row.get("suggested_review_note", "")))
        cards.append(
            "\n".join(
                [
                    '<section class="candidate">',
                    f"<h2>{role} #{html.escape(str(row.get('rank_in_role', '')))}</h2>",
                    f"<p>segment: <code>{segment}</code></p>",
                    f'<img src="{strip}" alt="temporal strip for {segment}">',
                    f'<p><a href="{preview}">Preview image</a></p>',
                    f"<p>Suggested review note: <code>{note}</code></p>",
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
            "<title>V7d Approval Fill Guide</title>",
            "<style>body{font-family:Arial,sans-serif;margin:24px;max-width:1120px}.candidate{border:1px solid #ccc;margin:16px 0;padding:12px}img{max-width:100%;height:auto;display:block}</style>",
            "</head>",
            "<body>",
            "<h1>V7d Approval Fill Guide</h1>",
            f"<p>Status: <code>{html.escape(str(summary.get('status', '')))}</code></p>",
            "<p>This guide does not approve rows, edit manifests, build seed packages, train, validate, or promote.</p>",
            "<p>Use it to choose rows, then manually fill <code>seed_required_decision_template.csv</code> and run <code>validate_v7d_review_decisions</code>.</p>",
            *cards,
            "</body>",
            "</html>",
            "",
        ]
    )


def _guide_markdown(*, summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# V7d Approval Fill Guide",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Guide rows: `{summary.get('guide_row_count')}`",
        f"- Decision template: `{summary.get('decision_template_csv')}`",
        "- This guide is evidence only; it does not approve rows or modify `segment_review_manifest.csv`.",
        "- Every approval still requires manual `decision=approve`, nonblank `review_notes`, and `validate_v7d_review_decisions` before dry-run/apply.",
        "",
        "| Role | Rank | Segment | Target | Coverage | Jumps | Suggested Review Note |",
        "| --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('proposal_role', '')}`",
                    str(row.get("rank_in_role", "")),
                    f"`{row.get('segment_id', '')}`",
                    str(row.get("target_name", "")),
                    str(row.get("detection_coverage", "")),
                    str(row.get("severe_landmark_jump_count", "")),
                    str(row.get("suggested_review_note", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Validation Command",
            "",
            "```powershell",
            "$env:PYTHONPATH='src'",
            "python -m embodied_rps.tools.validate_v7d_review_decisions --review-root artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618 --decisions-csv artifacts/real_skeleton_v7d_manual_review_shortlist_20260618/seed_required_decision_template.csv --output-root artifacts/real_skeleton_v7d_review_decision_validation_20260618",
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


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(base.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _relative_link(output_root: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=output_root.resolve()).replace("\\", "/")


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


def _config_summary(*, project_root: Path, config: V7DApprovalFillGuideConfig) -> dict[str, object]:
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


__all__ = ["V7DApprovalFillGuideConfig", "write_v7d_approval_fill_guide"]
