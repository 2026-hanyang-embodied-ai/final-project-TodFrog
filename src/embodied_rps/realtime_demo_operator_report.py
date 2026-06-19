"""Operator-facing outcome report for the realtime RPS demo workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_rps.realtime_demo_acceptance_report import LIVE_PIPELINE_COMMAND


OPERATOR_EXIT_CODES: dict[str, tuple[int, str]] = {
    "ready_for_final_video": (0, "ready_for_final_video"),
    "ready_to_record": (10, "live_capture_missing"),
    "retake_capture": (20, "retake_capture"),
    "postprocess_or_repair_capture": (30, "postprocess_or_repair_capture"),
    "research_iteration_needed": (40, "research_iteration_needed"),
    "setup_fix_needed": (50, "setup_fix_needed"),
    "manual_review_needed": (60, "manual_review_needed"),
}


@dataclass(frozen=True)
class RealtimeDemoOperatorReportConfig:
    """Artifact paths used to build the post-run operator outcome report."""

    output_root: Path
    acceptance_report: Path = Path("artifacts/realtime_demo_acceptance_report_20260616/acceptance_report.json")
    triage_summary: Path = Path("artifacts/realtime_demo_triage_20260616/triage_summary.json")
    review_packet_manifest: Path = Path("artifacts/realtime_demo_review_packet_20260616/review_packet_manifest.json")
    readiness_summary: Path = Path("artifacts/realtime_demo_readiness_20260616/readiness_summary.json")


def build_realtime_demo_operator_report(config: RealtimeDemoOperatorReportConfig) -> dict[str, object]:
    """Write a concise operator decision report from existing demo artifacts."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    output_json = config.output_root / "operator_outcome.json"
    output_md = config.output_root / "operator_outcome.md"

    acceptance = _read_json_if_exists(config.acceptance_report) or {}
    triage = _read_json_if_exists(config.triage_summary) or {}
    review_packet = _read_json_if_exists(config.review_packet_manifest) or {}
    readiness = _read_json_if_exists(config.readiness_summary) or {}
    triage_evidence = _dict_value(triage, "evidence")

    decision = _operator_decision(acceptance=acceptance, triage=triage, review_packet=review_packet)
    exit_code, exit_code_meaning = OPERATOR_EXIT_CODES.get(
        str(decision.get("operator_state")),
        OPERATOR_EXIT_CODES["manual_review_needed"],
    )
    summary: dict[str, object] = {
        **decision,
        "recommended_exit_code": exit_code,
        "exit_code_meaning": exit_code_meaning,
        "status_inputs": {
            "acceptance_status": acceptance.get("status"),
            "ready_for_youtube_demo": acceptance.get("ready_for_youtube_demo"),
            "triage_status": triage.get("status"),
            "triage_failure_category": triage.get("failure_category"),
            "readiness_status": readiness.get("status"),
            "review_packet_status": review_packet.get("status"),
            "live_rock_retake_gate_status": triage_evidence.get("live_rock_retake_gate_status"),
            "live_rock_retake_binary_decision_frame_count": triage_evidence.get(
                "live_rock_retake_binary_decision_frame_count"
            ),
            "live_rock_retake_confirmed_binary_decision_count": triage_evidence.get(
                "live_rock_retake_confirmed_binary_decision_count"
            ),
            "live_rock_retake_first_binary_decision": triage_evidence.get(
                "live_rock_retake_first_binary_decision"
            ),
            "live_rock_retake_first_binary_robot_action": triage_evidence.get(
                "live_rock_retake_first_binary_robot_action"
            ),
        },
        "blocking_requirements": _list_value(acceptance, "blocking_requirements"),
        "artifact_paths": {
            "acceptance_report": config.acceptance_report.as_posix(),
            "triage_summary": config.triage_summary.as_posix(),
            "review_packet_manifest": config.review_packet_manifest.as_posix(),
            "readiness_summary": config.readiness_summary.as_posix(),
            "review_packet_readme": _dict_value(review_packet, "outputs").get("readme_md"),
        },
        "outputs": {
            "summary_json": output_json.as_posix(),
            "summary_md": output_md.as_posix(),
        },
        "claim_scope": "operator outcome report over existing artifacts; does not run camera capture, model inference, or rendering",
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _operator_decision(
    *,
    acceptance: dict[str, Any],
    triage: dict[str, Any],
    review_packet: dict[str, Any],
) -> dict[str, object]:
    triage_status = triage.get("status")
    failure_category = triage.get("failure_category")
    triage_action = triage.get("recommended_next_action")
    acceptance_status = acceptance.get("status")
    acceptance_command = acceptance.get("next_operator_command")
    live_composite = _dict_value(review_packet, "referenced_videos").get("live_composite_mp4")

    if triage_status == "needs_live_capture" or failure_category == "live_capture_missing":
        command = acceptance_command if isinstance(acceptance_command, str) else LIVE_PIPELINE_COMMAND
        if not ("24_run_live_demo_operator_confirmed_strict.ps1" in command):
            command = LIVE_PIPELINE_COMMAND
        return {
            "operator_state": "ready_to_record",
            "primary_action_type": "run_live_pipeline",
            "primary_action": "Run the one-command live pipeline and perform the gesture during PROMPT SCISSORS.",
            "primary_command": command,
            "primary_video": live_composite,
            "failure_category": failure_category,
        }

    if acceptance.get("ready_for_youtube_demo") is True or acceptance_status == "ready_for_youtube_demo":
        return {
            "operator_state": "ready_for_final_video",
            "primary_action_type": "inspect_and_package_demo",
            "primary_action": "Inspect the live composite and use it as the final demo-video source if the visual review passes.",
            "primary_command": None,
            "primary_video": live_composite,
            "failure_category": None,
        }

    if triage_status == "needs_research_iteration":
        return {
            "operator_state": "research_iteration_needed",
            "primary_action_type": "simulation_first_research",
            "primary_action": str(triage_action or "Inspect the frame log and add targeted simulated 3D skeleton augmentation."),
            "primary_command": None,
            "primary_video": live_composite,
            "failure_category": failure_category,
        }

    if triage_status == "needs_retake":
        return {
            "operator_state": "retake_capture",
            "primary_action_type": "retake_live_capture",
            "primary_action": str(triage_action or "Adjust capture setup and rerun the live demo pipeline."),
            "primary_command": LIVE_PIPELINE_COMMAND,
            "primary_video": live_composite,
            "failure_category": failure_category,
        }

    if triage_status == "needs_setup_fix":
        return {
            "operator_state": "setup_fix_needed",
            "primary_action_type": "fix_local_setup",
            "primary_action": str(triage_action or "Fix local config, environment, model profile, or camera setup before recording."),
            "primary_command": None,
            "primary_video": live_composite,
            "failure_category": failure_category,
        }

    if triage_status == "needs_postprocessing" or acceptance_status == "live_demo_needs_repair":
        return {
            "operator_state": "postprocess_or_repair_capture",
            "primary_action_type": "rerun_postprocessing",
            "primary_action": str(triage_action or acceptance_command or "Rerun live verification and SCHUNK composite generation."),
            "primary_command": str(acceptance_command) if isinstance(acceptance_command, str) else None,
            "primary_video": live_composite,
            "failure_category": failure_category,
        }

    return {
        "operator_state": "manual_review_needed",
        "primary_action_type": "inspect_artifacts",
        "primary_action": str(triage_action or acceptance_command or "Inspect acceptance, triage, readiness, and review-packet artifacts."),
        "primary_command": str(acceptance_command) if isinstance(acceptance_command, str) else None,
        "primary_video": live_composite,
        "failure_category": failure_category,
    }


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _list_value(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _summary_markdown(summary: dict[str, object]) -> str:
    inputs = summary.get("status_inputs", {})
    paths = summary.get("artifact_paths", {})
    blocking = summary.get("blocking_requirements", [])
    lines = [
        "# Realtime Demo Operator Outcome",
        "",
        f"- Operator state: `{summary.get('operator_state')}`",
        f"- Primary action type: `{summary.get('primary_action_type')}`",
        f"- Failure category: `{summary.get('failure_category')}`",
        f"- Primary action: {summary.get('primary_action')}",
        f"- Primary command: `{summary.get('primary_command')}`",
        f"- Primary video: `{summary.get('primary_video')}`",
        f"- Recommended exit code: `{summary.get('recommended_exit_code')}`",
        f"- Exit code meaning: `{summary.get('exit_code_meaning')}`",
        "",
        "## Status Inputs",
        "",
    ]
    if isinstance(inputs, dict):
        for key in sorted(inputs):
            lines.append(f"- `{key}`: `{inputs[key]}`")
    lines.extend(["", "## Blocking Requirements", ""])
    if isinstance(blocking, list) and blocking:
        lines.extend(f"- `{item}`" for item in blocking)
    else:
        lines.append("- None")
    lines.extend(["", "## Artifact Paths", ""])
    if isinstance(paths, dict):
        for key in sorted(paths):
            lines.append(f"- `{key}`: `{paths[key]}`")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "OPERATOR_EXIT_CODES",
    "RealtimeDemoOperatorReportConfig",
    "build_realtime_demo_operator_report",
]
