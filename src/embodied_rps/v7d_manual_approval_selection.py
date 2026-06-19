"""Non-mutating v7d one-row-per-role manual approval selection aid."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_FILL_GUIDE_ROOT = Path("artifacts/real_skeleton_v7d_approval_fill_guide_20260618")
DEFAULT_SHORTLIST_ROOT = Path("artifacts/real_skeleton_v7d_manual_review_shortlist_20260618")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_manual_approval_selection_20260618")
REQUIRED_ROLES: tuple[str, ...] = (
    "hard_paper_prompt_window",
    "rock_wait_prompt_window",
    "scissors_boundary_control",
)
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = (
    "temporal_strip",
    "preview_image",
    "skeleton_npz",
    "decision_template_csv",
    "source_path",
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
OPTION_FIELDS: tuple[str, ...] = (
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
    "instruction",
)


@dataclass(frozen=True)
class V7DManualApprovalSelectionConfig:
    """Inputs for the v7d manual approval selection template."""

    project_root: Path = field(default_factory=Path.cwd)
    fill_guide_root: Path = DEFAULT_FILL_GUIDE_ROOT
    shortlist_root: Path = DEFAULT_SHORTLIST_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


def write_v7d_manual_approval_selection(config: V7DManualApprovalSelectionConfig) -> dict[str, object]:
    """Write a three-row selection aid without applying review decisions."""

    project_root = config.project_root.resolve()
    fill_guide_root = _resolve_path(project_root, config.fill_guide_root)
    shortlist_root = _resolve_path(project_root, config.shortlist_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    fill_guide_csv = fill_guide_root / "approval_fill_guide.csv"
    decision_template = shortlist_root / "seed_required_decision_template.csv"
    for path in (fill_guide_csv, decision_template):
        if not path.exists():
            raise FileNotFoundError(f"Missing v7d manual approval selection input: {path}")

    rows_by_role: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(fill_guide_csv):
        _reject_heldout_metadata(row, context=fill_guide_csv)
        role = str(row.get("proposal_role", "")).strip()
        if role in REQUIRED_ROLES:
            rows_by_role[role].append(row)

    selection_rows: list[dict[str, object]] = []
    option_rows: list[dict[str, object]] = []
    missing_roles: list[str] = []
    for role in REQUIRED_ROLES:
        candidates = sorted(rows_by_role.get(role, []), key=_rank_key)
        if not candidates:
            missing_roles.append(role)
        selection_rows.append(
            _selection_row(
                role=role,
                candidates=candidates,
                decision_template=decision_template,
                project_root=project_root,
            )
        )
        for candidate in candidates:
            option_row = {field: str(candidate.get(field, "")).strip() for field in OPTION_FIELDS}
            option_row["instruction"] = (
                "Candidate only; select at most one segment per required role in "
                "approval_selection_template.csv, then manually transfer approved rows to "
                "seed_required_decision_template.csv and run validate_v7d_review_decisions."
            )
            _reject_heldout_metadata(option_row, context=fill_guide_csv)
            option_rows.append(option_row)

    summary: dict[str, object] = {
        "status": "awaiting_manual_selection",
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "fill_guide_root": _display_path(fill_guide_root, base=project_root),
        "shortlist_root": _display_path(shortlist_root, base=project_root),
        "selection_template_csv": _display_path(output_root / "approval_selection_template.csv", base=project_root),
        "selection_options_csv": _display_path(output_root / "approval_selection_options.csv", base=project_root),
        "summary_json": _display_path(output_root / "approval_selection_summary.json", base=project_root),
        "summary_md": _display_path(output_root / "approval_selection_summary.md", base=project_root),
        "decision_template_csv": _display_path(decision_template, base=project_root),
        "selection_row_count": len(selection_rows),
        "option_row_count": len(option_rows),
        "required_roles": list(REQUIRED_ROLES),
        "missing_candidate_roles": missing_roles,
        "decisions_applied": False,
        "review_manifest_modified": False,
        "seed_package_created": False,
        "dataset_generated": False,
        "training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test paths are rejected from manual approval selection metadata",
        "next_action": (
            "choose one segment_id per required role, fill decision=approve and nonblank review_notes in the "
            "real seed_required_decision_template.csv, then run validate_v7d_review_decisions"
        ),
        "config": _config_summary(project_root=project_root, config=config),
    }

    _write_csv(output_root / "approval_selection_template.csv", SELECTION_FIELDS, selection_rows)
    _write_csv(output_root / "approval_selection_options.csv", OPTION_FIELDS, option_rows)
    (output_root / "approval_selection_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "approval_selection_summary.md").write_text(
        _summary_markdown(summary=summary, rows=selection_rows),
        encoding="utf-8",
    )
    return summary


def _selection_row(
    *,
    role: str,
    candidates: Sequence[Mapping[str, object]],
    decision_template: Path,
    project_root: Path,
) -> dict[str, object]:
    segment_ids = [str(row.get("segment_id", "")).strip() for row in candidates if str(row.get("segment_id", "")).strip()]
    first = candidates[0] if candidates else {}
    first_segment = str(first.get("segment_id", "")).strip()
    first_note = str(first.get("suggested_review_note", "")).strip()
    return {
        "proposal_role": role,
        "minimum_required_approved_count": 1,
        "selected_segment_id": "",
        "decision": "",
        "review_notes": "",
        "available_segment_ids": ";".join(segment_ids),
        "first_candidate_segment_id": first_segment,
        "first_candidate_suggested_review_note": first_note,
        "decision_template_csv": _display_path(decision_template, base=project_root),
        "instruction": (
            "Pick one available segment_id only after temporal evidence review; keep this sheet as guidance, "
            "then fill seed_required_decision_template.csv with decision=approve and nonblank review_notes. "
            "Run validate_v7d_review_decisions before any seed package, dataset, training, validation, or promotion step."
        ),
    }


def _summary_markdown(*, summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# V7d Manual Approval Selection",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Selection rows: `{summary.get('selection_row_count')}`",
        f"- Option rows: `{summary.get('option_row_count')}`",
        f"- Decision template: `{summary.get('decision_template_csv')}`",
        "- This artifact does not approve segments, modify review manifests, build seed packages, train, validate, or promote.",
        "- Manual approval still requires editing `seed_required_decision_template.csv` and running `validate_v7d_review_decisions`.",
        "",
        "| Role | Available Segment IDs | First Candidate |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('proposal_role', '')}`",
                    f"`{row.get('available_segment_ids', '')}`",
                    f"`{row.get('first_candidate_segment_id', '')}`",
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


def _rank_key(row: Mapping[str, object]) -> tuple[int, str]:
    return (_int_value(row.get("rank_in_role")), str(row.get("segment_id", "")).strip())


def _int_value(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 999999


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(base.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _config_summary(*, project_root: Path, config: V7DManualApprovalSelectionConfig) -> dict[str, object]:
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


__all__ = ["V7DManualApprovalSelectionConfig", "write_v7d_manual_approval_selection"]
