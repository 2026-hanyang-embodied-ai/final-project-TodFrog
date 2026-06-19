"""V7d real-seed review packet for prompt-window blocker evidence."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_PREPARATION_ROOT = Path("artifacts/real_skeleton_v7d_prompt_window_guard_preparation_20260618")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_real_seed_review_20260618")
BLOCKER_ROLES: tuple[str, ...] = ("hard_paper_prompt_window", "rock_wait_prompt_window")
ALL_REVIEW_ROLES: tuple[str, ...] = (*BLOCKER_ROLES, "scissors_boundary_control")
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = (
    "source_path",
    "source_overlay_video",
    "source_skeleton_npz",
    "source_frame_log",
    "skeleton_npz",
    "preview_image",
)
WORKLIST_FIELDS: tuple[str, ...] = (
    "review_priority",
    "segment_id",
    "target_name",
    "v7d_seed_role",
    "candidate_status",
    "eligible_for_manual_approval",
    "quality_status",
    "current_review_status",
    "currently_approved_for_training",
    "source_run_id",
    "proposal_role",
    "frame_count",
    "detection_coverage",
    "start_s",
    "end_s",
    "candidate_root",
    "preview_image",
    "skeleton_npz",
    "review_instruction",
)
DECISION_TEMPLATE_FIELDS: tuple[str, ...] = (
    "segment_id",
    "target_name",
    "v7d_seed_role",
    "candidate_root",
    "source_run_id",
    "proposal_role",
    "decision",
    "reviewer_notes",
    "preview_image",
    "skeleton_npz",
)


@dataclass(frozen=True)
class V7DRealSeedReviewConfig:
    """Inputs for writing the v7d blocker-focused real-seed review packet."""

    project_root: Path = field(default_factory=Path.cwd)
    preparation_root: Path = DEFAULT_PREPARATION_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


def write_v7d_real_seed_review_packet(config: V7DRealSeedReviewConfig) -> dict[str, object]:
    """Write a status-only review packet for v7d hard paper and rock/wait blockers."""

    project_root = config.project_root.resolve()
    preparation_root = _resolve_path(project_root, config.preparation_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_jsonl(preparation_root / "v7d_seed_candidate_manifest.jsonl")
    enriched_rows = [_enriched_candidate_row(project_root=project_root, row=row) for row in manifest_rows]
    for row in enriched_rows:
        _reject_heldout_metadata(row, context=preparation_root / "v7d_seed_candidate_manifest.jsonl")
    worklist_rows = [_worklist_row(row) for row in enriched_rows if str(row.get("v7d_seed_role", "")) in BLOCKER_ROLES]
    worklist_rows.sort(key=lambda row: (int(row["review_priority"]), str(row["target_name"]), str(row["segment_id"])))
    decision_rows = [_decision_template_row(row) for row in worklist_rows if row["eligible_for_manual_approval"] == "true"]
    role_counts = _role_counts(worklist_rows)
    status = _packet_status(role_counts)
    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "preparation_root": _display_path(preparation_root, base=project_root),
        "worklist_csv": _display_path(output_root / "v7d_real_seed_review_worklist.csv", base=project_root),
        "worklist_md": _display_path(output_root / "v7d_real_seed_review_worklist.md", base=project_root),
        "decision_template_csv": _display_path(output_root / "v7d_real_seed_review_decision_template.csv", base=project_root),
        "summary_json": _display_path(output_root / "v7d_real_seed_review_summary.json", base=project_root),
        "summary_md": _display_path(output_root / "v7d_real_seed_review_summary.md", base=project_root),
        "candidate_count": len(worklist_rows),
        "decision_template_candidate_count": len(decision_rows),
        "review_role_counts": role_counts,
        "training_started": False,
        "seed_package_created": False,
        "dataset_generated": False,
        "approval_policy": (
            "this packet does not approve rows; it isolates v7d blocker evidence for manual review and replacement collection"
        ),
        "heldout_policy": "heldout */test MP4s remain validation-only and are rejected from v7d review metadata",
        "next_actions": _next_actions(role_counts),
    }
    _write_outputs(output_root=output_root, summary=summary, worklist_rows=worklist_rows, decision_rows=decision_rows)
    return summary


def _enriched_candidate_row(*, project_root: Path, row: Mapping[str, object]) -> dict[str, object]:
    enriched = dict(row)
    candidate_root_value = str(enriched.get("candidate_root", "")).strip()
    if not candidate_root_value:
        return enriched
    candidate_root = _resolve_path(project_root, Path(candidate_root_value))
    source_rows = _candidate_source_rows(candidate_root)
    segment_id = str(enriched.get("segment_id", "")).strip()
    source = source_rows.get(segment_id, {})
    if source:
        merged = dict(source)
        merged.update(enriched)
        for key in (
            "preview_image",
            "skeleton_npz",
            "source_run_id",
            "frame_count",
            "detection_coverage",
            "start_s",
            "end_s",
            "proposal_role",
            "target_name",
            "quality_status",
        ):
            if key in source and not str(enriched.get(key, "")).strip():
                merged[key] = source[key]
        enriched = merged
    enriched["candidate_root"] = _display_path(candidate_root, base=project_root)
    return enriched


def _candidate_source_rows(candidate_root: Path) -> dict[str, dict[str, object]]:
    proposed_rows = _read_jsonl_if_exists(candidate_root / "proposed_segments.jsonl")
    review_rows = _read_review_rows(candidate_root / "segment_review_manifest.csv")
    review_by_id = {str(row.get("segment_id", "")).strip(): row for row in review_rows}
    rows: dict[str, dict[str, object]] = {}
    for proposed in proposed_rows:
        segment_id = str(proposed.get("segment_id", "")).strip()
        merged = dict(proposed)
        review = review_by_id.get(segment_id, {})
        for key in ("review_status", "approved_for_training", "review_notes", "quality_status"):
            if key in review:
                merged[key] = review[key]
        rows[segment_id] = merged
    return rows


def _worklist_row(row: Mapping[str, object]) -> dict[str, object]:
    role = str(row.get("v7d_seed_role", "")).strip()
    quality_status = str(row.get("quality_status", "")).strip()
    approved = _truthy(row.get("approved_for_training")) and str(row.get("review_status", "")).strip().lower() == "approved"
    eligible = quality_status == "auto_quality_pass" and not approved
    candidate_status = _candidate_status(role=role, quality_status=quality_status, approved=approved)
    return {
        "review_priority": _review_priority(role=role, eligible=eligible),
        "segment_id": str(row.get("segment_id", "")).strip(),
        "target_name": str(row.get("target_name", "")).strip(),
        "v7d_seed_role": role,
        "candidate_status": candidate_status,
        "eligible_for_manual_approval": _bool_text(eligible),
        "quality_status": quality_status,
        "current_review_status": str(row.get("review_status", "")).strip(),
        "currently_approved_for_training": _bool_text(approved),
        "source_run_id": str(row.get("source_run_id", "")).strip(),
        "proposal_role": str(row.get("proposal_role", "")).strip(),
        "frame_count": str(row.get("frame_count", "")).strip(),
        "detection_coverage": _format_optional_float(row.get("detection_coverage")),
        "start_s": _format_optional_float(row.get("start_s")),
        "end_s": _format_optional_float(row.get("end_s")),
        "candidate_root": str(row.get("candidate_root", "")).strip(),
        "preview_image": str(row.get("preview_image", "")).strip(),
        "skeleton_npz": str(row.get("skeleton_npz", "")).strip(),
        "review_instruction": _review_instruction(role=role, eligible=eligible, quality_status=quality_status),
    }


def _candidate_status(*, role: str, quality_status: str, approved: bool) -> str:
    if approved:
        return "already_approved"
    if quality_status == "auto_quality_pass":
        return "manual_review_candidate"
    if role == "rock_wait_prompt_window":
        return "collect_replacement_nonheldout_rock_wait"
    if role == "hard_paper_prompt_window":
        return "collect_replacement_nonheldout_hard_paper"
    return "not_eligible"


def _review_instruction(*, role: str, eligible: bool, quality_status: str) -> str:
    if eligible:
        return "inspect_temporal_segment_then_record_explicit_manual_decision"
    if role == "rock_wait_prompt_window" and quality_status != "auto_quality_pass":
        return "do_not_approve_collect_replacement_nonheldout_rock_wait_evidence"
    if role == "hard_paper_prompt_window" and quality_status != "auto_quality_pass":
        return "do_not_approve_collect_replacement_nonheldout_hard_paper_evidence"
    return "no_manual_approval_action"


def _review_priority(*, role: str, eligible: bool) -> int:
    if role == "hard_paper_prompt_window" and eligible:
        return 10
    if role == "rock_wait_prompt_window" and eligible:
        return 20
    if role == "rock_wait_prompt_window":
        return 70
    return 80


def _decision_template_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "segment_id": row.get("segment_id", ""),
        "target_name": row.get("target_name", ""),
        "v7d_seed_role": row.get("v7d_seed_role", ""),
        "candidate_root": row.get("candidate_root", ""),
        "source_run_id": row.get("source_run_id", ""),
        "proposal_role": row.get("proposal_role", ""),
        "decision": "",
        "reviewer_notes": "",
        "preview_image": row.get("preview_image", ""),
        "skeleton_npz": row.get("skeleton_npz", ""),
    }


def _role_counts(worklist_rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        role: {"candidate_count": 0, "eligible_count": 0, "approved_count": 0, "quality_failed_count": 0}
        for role in ALL_REVIEW_ROLES
    }
    for row in worklist_rows:
        role = str(row.get("v7d_seed_role", "")).strip()
        if role not in counts:
            continue
        counts[role]["candidate_count"] += 1
        if str(row.get("eligible_for_manual_approval", "")) == "true":
            counts[role]["eligible_count"] += 1
        if str(row.get("currently_approved_for_training", "")) == "true":
            counts[role]["approved_count"] += 1
        if str(row.get("quality_status", "")) != "auto_quality_pass":
            counts[role]["quality_failed_count"] += 1
    return counts


def _packet_status(role_counts: Mapping[str, Mapping[str, int]]) -> str:
    paper = role_counts.get("hard_paper_prompt_window", {})
    rock = role_counts.get("rock_wait_prompt_window", {})
    paper_approved = int(paper.get("approved_count", 0))
    rock_approved = int(rock.get("approved_count", 0))
    paper_eligible = int(paper.get("eligible_count", 0))
    rock_eligible = int(rock.get("eligible_count", 0))
    if paper_approved > 0 and rock_approved > 0:
        return "ready_for_v7d_seed_package_design"
    if paper_eligible > 0 and rock_eligible == 0:
        return "awaiting_manual_hard_paper_review_and_rock_wait_collection"
    if paper_eligible > 0 or rock_eligible > 0:
        return "awaiting_manual_v7d_seed_review"
    return "awaiting_new_nonheldout_v7d_evidence_collection"


def _next_actions(role_counts: Mapping[str, Mapping[str, int]]) -> list[str]:
    actions: list[str] = []
    paper = role_counts.get("hard_paper_prompt_window", {})
    rock = role_counts.get("rock_wait_prompt_window", {})
    if int(paper.get("eligible_count", 0)) > 0 and int(paper.get("approved_count", 0)) == 0:
        actions.append("visually review the eligible hard paper prompt-window row before any approval")
    if int(rock.get("eligible_count", 0)) == 0 and int(rock.get("approved_count", 0)) == 0:
        actions.append("collect or extract new non-heldout rock/wait prompt-window evidence; current rock candidates are not approval-eligible")
    actions.append("do not train v7d until hard paper and rock/wait roles both have approved non-heldout evidence")
    return actions


def _write_outputs(
    *,
    output_root: Path,
    summary: Mapping[str, object],
    worklist_rows: Sequence[Mapping[str, object]],
    decision_rows: Sequence[Mapping[str, object]],
) -> None:
    _write_csv(output_root / "v7d_real_seed_review_worklist.csv", WORKLIST_FIELDS, worklist_rows)
    _write_csv(output_root / "v7d_real_seed_review_decision_template.csv", DECISION_TEMPLATE_FIELDS, decision_rows)
    (output_root / "v7d_real_seed_review_summary.json").write_text(
        json.dumps(_json_ready(dict(summary)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "v7d_real_seed_review_worklist.md").write_text(
        _worklist_markdown(summary=summary, rows=worklist_rows),
        encoding="utf-8",
    )
    (output_root / "v7d_real_seed_review_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _worklist_markdown(*, summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# V7d Real-Seed Review Worklist",
        "",
        f"- Status: `{summary.get('status')}`",
        "- Scope: hard paper prompt-window and rock/wait prompt-window blocker evidence only.",
        "- This packet does not approve rows or edit the source review manifest.",
        "",
        "| Priority | Segment | Target | Role | Eligible | Quality | Preview | Instruction |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        preview = str(row.get("preview_image", "")).strip() or "not recorded"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("review_priority", "")),
                    f"`{row.get('segment_id', '')}`",
                    str(row.get("target_name", "")),
                    str(row.get("v7d_seed_role", "")),
                    str(row.get("eligible_for_manual_approval", "")),
                    str(row.get("quality_status", "")),
                    preview,
                    str(row.get("review_instruction", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V7d Real-Seed Review Summary",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Decision-template candidates: `{summary.get('decision_template_candidate_count')}`",
        f"- Training started: `{summary.get('training_started')}`",
        f"- Seed package created: `{summary.get('seed_package_created')}`",
        "- Heldout policy: validation-only `*/test` MP4s are rejected from review metadata.",
        "",
        "## Role Counts",
        "",
    ]
    role_counts = summary.get("review_role_counts")
    if isinstance(role_counts, Mapping):
        for role, counts in role_counts.items():
            if isinstance(counts, Mapping):
                lines.append(
                    f"- `{role}`: candidates={counts.get('candidate_count')}, "
                    f"eligible={counts.get('eligible_count')}, approved={counts.get('approved_count')}, "
                    f"quality_failed={counts.get('quality_failed_count')}"
                )
    lines.extend(["", "## Next Actions", ""])
    actions = summary.get("next_actions")
    if isinstance(actions, Sequence):
        for action in actions:
            lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing v7d seed candidate manifest: {path}")
    return _read_jsonl_if_exists(path)


def _read_jsonl_if_exists(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, Mapping):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(dict(value))
    return rows


def _read_review_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "approved"}


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _format_optional_float(value: object) -> str:
    try:
        return f"{float(value):.6f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["V7DRealSeedReviewConfig", "write_v7d_real_seed_review_packet"]
