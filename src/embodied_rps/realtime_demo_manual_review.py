"""Manual visual review recorder for archived realtime demo runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_REVIEW_STATUSES = {"approved", "rejected_by_manual_review"}


@dataclass(frozen=True)
class RealtimeDemoManualReviewConfig:
    """Input/output paths and review decision for one archived run."""

    archive_index: Path = Path("artifacts/realtime_demo_run_archive_20260616/run_archive_index.json")
    manual_review_decisions: Path = Path("artifacts/realtime_demo_manual_review_20260616/manual_review_decisions.json")
    output_root: Path = Path("artifacts/realtime_demo_manual_review_20260616")
    run_id: str | None = None
    status: str = "approved"
    notes: str = ""


def record_realtime_demo_manual_review(config: RealtimeDemoManualReviewConfig) -> dict[str, object]:
    """Record an operator manual review decision and merge it into the decisions JSON."""

    archive_index = _read_json_if_exists(config.archive_index) or {}
    run = _select_run(archive_index=archive_index, run_id=config.run_id)
    status = _normalize_status(config.status)
    if status == "approved" and run.get("ground_truth_passed") is not True:
        raise ValueError(
            f"Cannot approve run {run.get('run_id')}: ground_truth_passed is not true in the archive index."
        )

    config.output_root.mkdir(parents=True, exist_ok=True)
    config.manual_review_decisions.parent.mkdir(parents=True, exist_ok=True)
    decisions = _read_json_if_exists(config.manual_review_decisions) or {}
    reviews = _dict_value(decisions, "run_reviews")
    run_id = str(run.get("run_id") or "")
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    review_record = {
        "manual_review_status": status,
        "manual_review_notes": str(config.notes or "").strip(),
        "reviewed_at_utc": now,
        "expected_actual_gesture": run.get("expected_actual_gesture"),
        "ground_truth_passed": run.get("ground_truth_passed"),
        "ground_truth_match": run.get("ground_truth_match"),
        "robot_action_match": run.get("robot_action_match"),
        "reviewed_artifacts": {
            "primary_video": run.get("primary_video"),
            "live_overlay_video": run.get("live_overlay_video"),
            "live_response_decision_frame": run.get("live_response_decision_frame"),
            "archive_dir": run.get("archive_dir"),
        },
    }
    reviews[run_id] = review_record
    merged = {
        **decisions,
        "review_status": "has_manual_reviews",
        "updated_at_utc": now,
        "run_reviews": reviews,
    }
    config.manual_review_decisions.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    output_json = config.output_root / "manual_review_summary.json"
    output_md = config.output_root / "manual_review_summary.md"
    summary: dict[str, object] = {
        "review_status": "recorded",
        "run_id": run_id,
        "manual_review_status": status,
        "manual_review_notes": review_record["manual_review_notes"],
        "expected_actual_gesture": run.get("expected_actual_gesture"),
        "ground_truth_passed": run.get("ground_truth_passed"),
        "manual_review_decisions": config.manual_review_decisions.as_posix(),
        "next_actions": [
            "refresh_run_archive_index",
            "select_final_demo_candidate",
            "refresh_submission_packet",
        ],
        "outputs": {
            "summary_json": output_json.as_posix(),
            "summary_md": output_md.as_posix(),
        },
        "claim_scope": "records operator visual review metadata; does not change model inference or live capture artifacts",
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _select_run(*, archive_index: dict[str, Any], run_id: str | None) -> dict[str, Any]:
    runs = archive_index.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Archive index contains no runs to review.")
    if run_id:
        for raw_run in runs:
            if isinstance(raw_run, dict) and str(raw_run.get("run_id") or "") == run_id:
                return raw_run
        raise ValueError(f"Run ID not found in archive index: {run_id}")
    for raw_run in reversed(runs):
        if not isinstance(raw_run, dict):
            continue
        if raw_run.get("primary_video") and raw_run.get("live_response_decision_frame"):
            return raw_run
    raise ValueError("Archive index has no complete live run with primary video and response decision frame.")


def _normalize_status(status: str) -> str:
    parsed = str(status or "").strip().lower()
    if parsed not in VALID_REVIEW_STATUSES:
        raise ValueError(f"Manual review status must be one of {sorted(VALID_REVIEW_STATUSES)}.")
    return parsed


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Realtime Demo Manual Review",
        "",
        f"- Review status: `{summary.get('review_status')}`",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Manual review status: `{summary.get('manual_review_status')}`",
        f"- Expected actual gesture: `{summary.get('expected_actual_gesture')}`",
        f"- Ground truth passed: `{summary.get('ground_truth_passed')}`",
        f"- Notes: {summary.get('manual_review_notes')}",
        "",
        "## Next Actions",
        "",
    ]
    actions = summary.get("next_actions", [])
    if isinstance(actions, list):
        lines.extend(f"- `{action}`" for action in actions)
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoManualReviewConfig", "record_realtime_demo_manual_review"]
