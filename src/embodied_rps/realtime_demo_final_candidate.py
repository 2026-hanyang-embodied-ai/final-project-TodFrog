"""Select the final realtime demo video candidate from archived runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeDemoFinalCandidateConfig:
    """Input and output paths for final demo candidate selection."""

    output_root: Path = Path("artifacts/realtime_demo_final_candidate_20260616")
    archive_index: Path = Path("artifacts/realtime_demo_run_archive_20260616/run_archive_index.json")


def select_realtime_demo_final_candidate(config: RealtimeDemoFinalCandidateConfig) -> dict[str, object]:
    """Write a packaging-facing final demo candidate manifest from the archive index."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    output_json = config.output_root / "final_demo_candidate.json"
    output_md = config.output_root / "final_demo_candidate.md"

    archive_index = _read_json_if_exists(config.archive_index) or {}
    candidate = _dict_value(archive_index, "latest_final_video_candidate") or _latest_complete_run(archive_index)
    primary_video = str(candidate.get("primary_video") or "") if candidate else ""

    if candidate and primary_video and _candidate_is_approved(candidate):
        summary = _selected_summary(
            archive_index=archive_index,
            candidate=candidate,
            primary_video=primary_video,
            output_json=output_json,
            output_md=output_md,
            config=config,
        )
    else:
        summary = _awaiting_summary(
            archive_index=archive_index,
            candidate=candidate,
            output_json=output_json,
            output_md=output_md,
            config=config,
        )

    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(_candidate_markdown(summary), encoding="utf-8")
    return summary


def _selected_summary(
    *,
    archive_index: dict[str, Any],
    candidate: dict[str, Any],
    primary_video: str,
    output_json: Path,
    output_md: Path,
    config: RealtimeDemoFinalCandidateConfig,
) -> dict[str, object]:
    return {
        "candidate_status": "selected_final_demo_candidate",
        "ready_for_video_packaging": True,
        "selected_run_id": candidate.get("run_id"),
        "primary_video": primary_video,
        "live_overlay_video": candidate.get("live_overlay_video"),
        "live_response_decision_frame": candidate.get("live_response_decision_frame"),
        "expected_actual_gesture": candidate.get("expected_actual_gesture"),
        "ground_truth_passed": candidate.get("ground_truth_passed"),
        "ground_truth_match": candidate.get("ground_truth_match"),
        "robot_action_match": candidate.get("robot_action_match"),
        "manual_review_status": candidate.get("manual_review_status"),
        "live_rock_retake_gate_status": candidate.get("live_rock_retake_gate_status"),
        "live_rock_retake_gate_passed": candidate.get("live_rock_retake_gate_passed"),
        "live_rock_retake_binary_decision_frame_count": candidate.get(
            "live_rock_retake_binary_decision_frame_count"
        ),
        "live_rock_retake_confirmed_binary_decision_count": candidate.get(
            "live_rock_retake_confirmed_binary_decision_count"
        ),
        "archive_dir": candidate.get("archive_dir"),
        "archive_status": candidate.get("archive_status"),
        "operator_state": candidate.get("operator_state"),
        "operator_recommended_exit_code": candidate.get("operator_recommended_exit_code"),
        "archive_manifest_path": candidate.get("manifest_path"),
        "archive_index_status": archive_index.get("index_status"),
        "archive_run_count": archive_index.get("run_count", 0),
        "next_action": "Inspect the selected composite video, then use it as the final demo-video source if visual review passes.",
        "outputs": {
            "candidate_json": output_json.as_posix(),
            "candidate_md": output_md.as_posix(),
        },
        "inputs": {
            "archive_index": config.archive_index.as_posix(),
        },
        "claim_scope": "final demo candidate selection over an archive index; does not copy, transcode, upload, or verify video content",
    }


def _awaiting_summary(
    *,
    archive_index: dict[str, Any],
    candidate: dict[str, Any],
    output_json: Path,
    output_md: Path,
    config: RealtimeDemoFinalCandidateConfig,
) -> dict[str, object]:
    latest_run_id = archive_index.get("latest_run_id")
    missing_reason = "no_final_video_candidate"
    blocked_unapproved_run = _latest_complete_unapproved_run(archive_index)
    blocked_run = blocked_unapproved_run or _latest_composite_without_decision_frame(archive_index)
    if not archive_index:
        missing_reason = "archive_index_missing_or_unreadable"
    elif candidate and not candidate.get("primary_video"):
        missing_reason = "candidate_missing_primary_video"
    elif candidate and candidate.get("primary_video"):
        missing_reason = _candidate_missing_reason(candidate)
    elif blocked_unapproved_run:
        missing_reason = _candidate_missing_reason(blocked_unapproved_run)
    elif blocked_run:
        missing_reason = "composite_without_response_decision_frame"
    next_action = "Run the live demo pipeline until the archive index contains a final-video candidate."
    if missing_reason == "composite_without_response_decision_frame":
        next_action = (
            "Rerun post-capture decision-frame extraction or rerun the live demo pipeline so the archived composite "
            "also includes a live response-decision frame."
        )
    elif missing_reason == "manual_review_rejected":
        next_action = "Retake the live demo with operator ground truth; the current complete media was rejected by manual review."
    elif missing_reason in {"manual_review_required", "ground_truth_validation_missing_or_failed"}:
        next_action = "Run or repair operator-ground-truth validation and manual review before selecting final demo media."
    elif missing_reason == "live_rock_retake_gate_missing_or_failed":
        next_action = (
            "Rerun the expected-rock live retake through the strict wrapper so the live-rock retake gate passes "
            "before final demo media can be selected."
        )
    blocked_missing = _blocked_missing_artifacts(candidate=candidate, blocked_run=blocked_run)
    return {
        "candidate_status": "awaiting_final_video_candidate",
        "ready_for_video_packaging": False,
        "selected_run_id": None,
        "primary_video": None,
        "latest_run_id": latest_run_id,
        "blocked_run_id": candidate.get("run_id") if candidate and candidate.get("primary_video") else (blocked_run.get("run_id") if blocked_run else None),
        "blocked_primary_video": candidate.get("primary_video") if candidate and candidate.get("primary_video") else (blocked_run.get("primary_video") if blocked_run else None),
        "blocked_missing_artifacts": blocked_missing,
        "archive_index_status": archive_index.get("index_status"),
        "archive_run_count": archive_index.get("run_count", 0),
        "missing_reason": missing_reason,
        "next_action": next_action,
        "outputs": {
            "candidate_json": output_json.as_posix(),
            "candidate_md": output_md.as_posix(),
        },
        "inputs": {
            "archive_index": config.archive_index.as_posix(),
        },
        "claim_scope": "final demo candidate selection over an archive index; does not run camera capture, model inference, rendering, or video packaging",
    }


def _latest_composite_without_decision_frame(archive_index: dict[str, Any]) -> dict[str, Any] | None:
    runs = archive_index.get("runs")
    if not isinstance(runs, list):
        return None
    for raw_run in reversed(runs):
        if not isinstance(raw_run, dict):
            continue
        if raw_run.get("primary_video") and not raw_run.get("live_response_decision_frame"):
            return raw_run
    return None


def _latest_complete_run(archive_index: dict[str, Any]) -> dict[str, Any]:
    runs = archive_index.get("runs")
    if not isinstance(runs, list):
        return {}
    for raw_run in reversed(runs):
        if not isinstance(raw_run, dict):
            continue
        if raw_run.get("primary_video") and raw_run.get("live_response_decision_frame") and _candidate_is_approved(raw_run):
            return raw_run
    return {}


def _latest_complete_unapproved_run(archive_index: dict[str, Any]) -> dict[str, Any] | None:
    runs = archive_index.get("runs")
    if not isinstance(runs, list):
        return None
    for raw_run in reversed(runs):
        if not isinstance(raw_run, dict):
            continue
        if raw_run.get("primary_video") and raw_run.get("live_response_decision_frame") and not _candidate_is_approved(raw_run):
            return raw_run
    return None


def _candidate_is_approved(candidate: dict[str, Any]) -> bool:
    return (
        candidate.get("ground_truth_passed") is True
        and candidate.get("manual_review_status") == "approved"
        and _live_rock_gate_allows_candidate(candidate)
    )


def _candidate_missing_reason(candidate: dict[str, Any]) -> str:
    if candidate.get("manual_review_status") == "rejected_by_manual_review":
        return "manual_review_rejected"
    if candidate.get("ground_truth_passed") is not True:
        return "ground_truth_validation_missing_or_failed"
    if not _live_rock_gate_allows_candidate(candidate):
        return "live_rock_retake_gate_missing_or_failed"
    return "manual_review_required"


def _candidate_blocked_missing_artifacts(candidate: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if candidate.get("ground_truth_passed") is not True:
        missing.append("ground_truth_passed")
    if candidate.get("manual_review_status") != "approved":
        missing.append("manual_review_approval")
    if not _live_rock_gate_allows_candidate(candidate):
        missing.append("live_rock_retake_gate_passed")
    return missing


def _live_rock_gate_allows_candidate(candidate: dict[str, Any]) -> bool:
    if candidate.get("expected_actual_gesture") != "rock":
        return True
    return candidate.get("live_rock_retake_gate_passed") is True


def _blocked_missing_artifacts(*, candidate: dict[str, Any], blocked_run: dict[str, Any] | None) -> list[str]:
    if candidate and candidate.get("primary_video"):
        return _candidate_blocked_missing_artifacts(candidate)
    if not blocked_run:
        return []
    if blocked_run.get("live_response_decision_frame"):
        return _candidate_blocked_missing_artifacts(blocked_run)
    return ["live_response_decision_frame"]


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _candidate_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Realtime Demo Final Candidate",
        "",
        f"- Candidate status: `{summary.get('candidate_status')}`",
        f"- Ready for video packaging: `{summary.get('ready_for_video_packaging')}`",
        f"- Selected run ID: `{summary.get('selected_run_id')}`",
        f"- Primary video: `{summary.get('primary_video')}`",
        f"- Blocked run ID: `{summary.get('blocked_run_id')}`",
        f"- Blocked primary video: `{summary.get('blocked_primary_video')}`",
        f"- Operator state: `{summary.get('operator_state')}`",
        f"- Archive index status: `{summary.get('archive_index_status')}`",
        f"- Archive run count: `{summary.get('archive_run_count')}`",
        f"- Missing reason: `{summary.get('missing_reason')}`",
        f"- Next action: {summary.get('next_action')}",
        "",
        "## Blocked Missing Artifacts",
        "",
    ]
    blocked_missing = summary.get("blocked_missing_artifacts", [])
    if isinstance(blocked_missing, list) and blocked_missing:
        lines.extend(f"- `{item}`" for item in blocked_missing)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Inputs",
            "",
        ]
    )
    inputs = summary.get("inputs", {})
    if isinstance(inputs, dict):
        for key in sorted(inputs):
            lines.append(f"- `{key}`: `{inputs[key]}`")
    lines.extend(["", "## Outputs", ""])
    outputs = summary.get("outputs", {})
    if isinstance(outputs, dict):
        for key in sorted(outputs):
            lines.append(f"- `{key}`: `{outputs[key]}`")
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoFinalCandidateConfig", "select_realtime_demo_final_candidate"]
