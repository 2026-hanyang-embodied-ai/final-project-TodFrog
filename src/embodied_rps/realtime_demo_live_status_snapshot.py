"""One-screen status snapshot for the realtime demo live run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeDemoLiveStatusSnapshotConfig:
    """Input and output paths for live-demo status snapshot generation."""

    output_root: Path = Path("artifacts/realtime_demo_live_status_snapshot_20260616")
    goal_audit: Path = Path("artifacts/realtime_demo_goal_progress_audit_20260616/goal_progress_audit.json")
    final_run_card: Path = Path("artifacts/realtime_demo_final_run_card_20260616/final_run_card.json")
    learning_queue: Path = Path("artifacts/realtime_demo_learning_queue_20260616/learning_queue.json")
    readiness_summary: Path = Path("artifacts/realtime_demo_readiness_20260616/readiness_summary.json")
    evidence_bundle: Path = Path("artifacts/realtime_demo_evidence_bundle_20260616/demo_evidence_bundle.json")
    live_overlay: Path = Path("artifacts/realtime_demo_rehearsal_20260616/live_camera_overlay.mp4")
    live_frame_log: Path = Path("artifacts/realtime_demo_rehearsal_20260616/live_camera_frames.jsonl")
    live_composite: Path = Path("artifacts/realtime_schunk_live_demo_composite_20260616/realtime_schunk_demo_composite.mp4")


def build_realtime_demo_live_status_snapshot(config: RealtimeDemoLiveStatusSnapshotConfig) -> dict[str, object]:
    """Write a compact status snapshot over live-demo evidence and next actions."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    output_json = config.output_root / "live_status_snapshot.json"
    output_md = config.output_root / "live_status_snapshot.md"
    goal_audit = _read_json_if_exists(config.goal_audit) or {}
    final_card = _read_json_if_exists(config.final_run_card) or {}
    learning_queue = _read_json_if_exists(config.learning_queue) or {}
    readiness = _read_json_if_exists(config.readiness_summary) or {}
    evidence_bundle = _read_json_if_exists(config.evidence_bundle) or {}

    summary = _snapshot_summary(
        goal_audit=goal_audit,
        final_card=final_card,
        learning_queue=learning_queue,
        readiness=readiness,
        evidence_bundle=evidence_bundle,
        config=config,
        output_json=output_json,
        output_md=output_md,
    )
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(_snapshot_markdown(summary), encoding="utf-8")
    return summary


def _snapshot_summary(
    *,
    goal_audit: dict[str, Any],
    final_card: dict[str, Any],
    learning_queue: dict[str, Any],
    readiness: dict[str, Any],
    evidence_bundle: dict[str, Any],
    config: RealtimeDemoLiveStatusSnapshotConfig,
    output_json: Path,
    output_md: Path,
) -> dict[str, object]:
    live_artifacts = {
        "live_overlay": config.live_overlay.as_posix(),
        "live_overlay_exists": config.live_overlay.exists(),
        "live_frame_log": config.live_frame_log.as_posix(),
        "live_frame_log_exists": config.live_frame_log.exists(),
        "live_composite": config.live_composite.as_posix(),
        "live_composite_exists": config.live_composite.exists(),
    }
    evidence = _dict_value(evidence_bundle, "evidence")
    live_evidence = _dict_value(evidence, "live")
    live_artifact_freshness = _dict_value(evidence, "live_artifact_freshness")
    response_decision_frame = _first_present(
        live_artifact_freshness.get("response_decision_frame_path"),
        live_evidence.get("response_decision_frame_png"),
    )
    live_artifacts["live_response_decision_frame"] = str(response_decision_frame) if response_decision_frame else None
    live_artifacts["live_response_decision_frame_exists"] = (
        Path(str(response_decision_frame)).exists() if response_decision_frame else False
    )
    response_prompt_diagnostic_frame = live_evidence.get("response_prompt_diagnostic_frame_png")
    live_artifacts["live_response_prompt_diagnostic_frame"] = (
        str(response_prompt_diagnostic_frame) if response_prompt_diagnostic_frame else None
    )
    live_artifacts["live_response_prompt_diagnostic_frame_exists"] = (
        Path(str(response_prompt_diagnostic_frame)).exists() if response_prompt_diagnostic_frame else False
    )
    live_artifacts["live_response_prompt_diagnostic_frame_reason"] = live_evidence.get(
        "response_prompt_diagnostic_frame_reason"
    )
    goal_complete = goal_audit.get("goal_complete") is True
    completion_candidate = (
        goal_complete
        and final_card.get("ready_for_submission_linking") is True
        and learning_queue.get("queue_status") == "no_research_needed"
    )
    snapshot_status = _snapshot_status(
        goal_audit=goal_audit,
        final_card=final_card,
        learning_queue=learning_queue,
        completion_candidate=completion_candidate,
    )
    recommended_command = None if completion_candidate else _recommended_command(
        snapshot_status=snapshot_status,
        goal_audit=goal_audit,
        final_card=final_card,
    )
    raw_recommended_command_absolute = None if completion_candidate else final_card.get("primary_command_absolute")
    recommended_command_absolute = _ascii_safe_command(
        str(raw_recommended_command_absolute) if raw_recommended_command_absolute else None
    )
    return {
        "snapshot_status": snapshot_status,
        "goal_status": goal_audit.get("goal_status"),
        "goal_complete": goal_complete,
        "completion_candidate": completion_candidate,
        "recommended_track": "live_capture" if snapshot_status == "awaiting_live_capture" else learning_queue.get("recommended_track"),
        "recommended_command": str(recommended_command) if recommended_command else None,
        "recommended_command_absolute": recommended_command_absolute,
        "recommended_command_absolute_note": _absolute_command_note(
            raw_command=str(raw_recommended_command_absolute) if raw_recommended_command_absolute else None,
            displayed_command=recommended_command_absolute,
        ),
        "missing_requirements": _list_value(goal_audit, "missing_requirements"),
        "live_artifacts": live_artifacts,
        "status_inputs": {
            "readiness_status": readiness.get("status"),
            "evidence_bundle_status": evidence_bundle.get("status"),
            "final_card_status": final_card.get("card_status"),
            "learning_queue_status": learning_queue.get("queue_status"),
        },
        "inputs": {
            "goal_audit": config.goal_audit.as_posix(),
            "final_run_card": config.final_run_card.as_posix(),
            "learning_queue": config.learning_queue.as_posix(),
            "readiness_summary": config.readiness_summary.as_posix(),
            "evidence_bundle": config.evidence_bundle.as_posix(),
        },
        "outputs": {
            "snapshot_json": output_json.as_posix(),
            "snapshot_md": output_md.as_posix(),
        },
        "claim_scope": "status snapshot over existing artifacts; does not run camera capture, inference, rendering, training, upload, or repository edits",
    }


def _snapshot_status(
    *,
    goal_audit: dict[str, Any],
    final_card: dict[str, Any],
    learning_queue: dict[str, Any],
    completion_candidate: bool,
) -> str:
    if completion_candidate:
        return "complete_evidence_ready"
    card_status = str(final_card.get("card_status") or "")
    if card_status in {"ready_to_record_live_demo", "retake_live_capture"}:
        return "awaiting_live_capture"
    goal_status = str(goal_audit.get("goal_status") or "")
    if goal_status == "incomplete_awaiting_live_capture":
        return "awaiting_live_capture"
    if goal_status == "incomplete_postprocess_repair_needed":
        return "postprocess_repair_required"
    queue_status = str(learning_queue.get("queue_status") or "")
    if queue_status == "postprocess_repair_required":
        return "postprocess_repair_required"
    if queue_status == "simulation_first_research_queued":
        return "research_iteration_needed"
    if card_status == "setup_fix_needed":
        return "setup_fix_needed"
    return goal_status or queue_status or card_status or "manual_review_needed"


def _recommended_command(
    *,
    snapshot_status: str,
    goal_audit: dict[str, Any],
    final_card: dict[str, Any],
) -> object:
    if snapshot_status == "awaiting_live_capture":
        return final_card.get("primary_command") or goal_audit.get("next_action")
    return goal_audit.get("next_action") or final_card.get("primary_command")


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _list_value(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _first_present(*values: object) -> object:
    for value in values:
        if value is not None:
            return value
    return None


def _ascii_safe_command(command: str | None) -> str | None:
    if not command:
        return None
    try:
        command.encode("ascii")
    except UnicodeEncodeError:
        return None
    return command


def _absolute_command_note(*, raw_command: str | None, displayed_command: str | None) -> str | None:
    if raw_command and displayed_command is None:
        return (
            "Absolute command omitted because the workspace path contains non-ASCII characters; "
            "run the recommended command from the project root."
        )
    if displayed_command:
        return "Absolute command is ASCII-safe and can be used from any PowerShell location."
    return None


def _snapshot_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Realtime Demo Live Status Snapshot",
        "",
        f"- Snapshot status: `{summary.get('snapshot_status')}`",
        f"- Goal status: `{summary.get('goal_status')}`",
        f"- Goal complete: `{summary.get('goal_complete')}`",
        f"- Completion candidate: `{summary.get('completion_candidate')}`",
        f"- Recommended track: `{summary.get('recommended_track')}`",
        f"- Recommended command: `{summary.get('recommended_command')}`",
        f"- Recommended command absolute: `{summary.get('recommended_command_absolute')}`",
        f"- Recommended command absolute note: `{summary.get('recommended_command_absolute_note')}`",
        "",
        "## Live Artifacts",
        "",
    ]
    live_artifacts = summary.get("live_artifacts", {})
    if isinstance(live_artifacts, dict):
        for key in sorted(live_artifacts):
            lines.append(f"- `{key}`: `{live_artifacts[key]}`")
    lines.extend(["", "## Missing Requirements", ""])
    missing = summary.get("missing_requirements", [])
    if isinstance(missing, list) and missing:
        lines.extend(f"- `{item}`" for item in missing)
    else:
        lines.append("- None")
    lines.extend(["", "## Status Inputs", ""])
    status_inputs = summary.get("status_inputs", {})
    if isinstance(status_inputs, dict):
        for key in sorted(status_inputs):
            lines.append(f"- `{key}`: `{status_inputs[key]}`")
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoLiveStatusSnapshotConfig", "build_realtime_demo_live_status_snapshot"]
