"""One-card summary for the current realtime demo run state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeDemoFinalRunCardConfig:
    """Input and output paths for the final realtime demo run card."""

    output_root: Path = Path("artifacts/realtime_demo_final_run_card_20260616")
    live_run_checklist: Path = Path("artifacts/realtime_demo_live_run_checklist_20260616/live_run_checklist.json")
    operator_outcome: Path = Path("artifacts/realtime_demo_operator_outcome_20260616/operator_outcome.json")
    submission_packet: Path = Path("artifacts/realtime_demo_submission_packet_20260616/submission_candidate_packet.json")
    launch_summary: Path = Path("artifacts/realtime_demo_launch_20260616/demo_launch_summary.json")


def build_realtime_demo_final_run_card(config: RealtimeDemoFinalRunCardConfig) -> dict[str, object]:
    """Write a single status card for recording, retake, research, or submission handoff."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    output_json = config.output_root / "final_run_card.json"
    output_md = config.output_root / "final_run_card.md"
    checklist = _read_json_if_exists(config.live_run_checklist) or {}
    operator = _read_json_if_exists(config.operator_outcome) or {}
    submission = _read_json_if_exists(config.submission_packet) or {}
    launch_summary = _read_json_if_exists(config.launch_summary) or {}

    summary = _card_summary(
        checklist=checklist,
        operator=operator,
        submission=submission,
        launch_summary=launch_summary,
        config=config,
        output_json=output_json,
        output_md=output_md,
    )
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(_card_markdown(summary), encoding="utf-8")
    return summary


def _card_summary(
    *,
    checklist: dict[str, Any],
    operator: dict[str, Any],
    submission: dict[str, Any],
    launch_summary: dict[str, Any],
    config: RealtimeDemoFinalRunCardConfig,
    output_json: Path,
    output_md: Path,
) -> dict[str, object]:
    status = _card_status(checklist=checklist, operator=operator, submission=submission)
    primary_video = _primary_video(submission)
    primary_command = _primary_command(
        status=status,
        checklist=checklist,
        operator=operator,
        submission=submission,
        launch_summary=launch_summary,
    )
    primary_command_absolute = _primary_command_absolute(status=status, launch_summary=launch_summary)
    next_steps = _next_steps(status=status, submission=submission)
    operator_status_inputs = _dict_value(operator, "status_inputs")
    return {
        "card_status": status,
        "ready_for_submission_linking": submission.get("ready_for_submission_linking") is True,
        "selected_run_id": submission.get("selected_run_id"),
        "primary_video": primary_video,
        "primary_command": primary_command,
        "primary_command_absolute": primary_command_absolute,
        "primary_action": operator.get("primary_action"),
        "response_prompt": checklist.get("response_prompt"),
        "recommended_first_expected_actual_gesture": checklist.get("recommended_first_expected_actual_gesture"),
        "operator_instruction_details": _instruction_details(checklist),
        "failure_category": operator.get("failure_category"),
        "live_rock_retake_gate_status": operator_status_inputs.get("live_rock_retake_gate_status"),
        "live_rock_retake_binary_decision_frame_count": operator_status_inputs.get(
            "live_rock_retake_binary_decision_frame_count"
        ),
        "live_rock_retake_confirmed_binary_decision_count": operator_status_inputs.get(
            "live_rock_retake_confirmed_binary_decision_count"
        ),
        "live_rock_retake_first_binary_decision": operator_status_inputs.get(
            "live_rock_retake_first_binary_decision"
        ),
        "live_rock_retake_first_binary_robot_action": operator_status_inputs.get(
            "live_rock_retake_first_binary_robot_action"
        ),
        "blocking_items": _list_value(checklist, "blocking_items"),
        "next_steps": next_steps,
        "status_inputs": {
            "checklist_status": checklist.get("checklist_status"),
            "ok_to_run_pipeline": checklist.get("ok_to_run_pipeline"),
            "operator_state": operator.get("operator_state"),
            "submission_packet_status": submission.get("packet_status"),
            "ready_for_submission_linking": submission.get("ready_for_submission_linking"),
            "recommended_first_expected_actual_gesture": checklist.get("recommended_first_expected_actual_gesture"),
            "live_rock_retake_gate_status": operator_status_inputs.get("live_rock_retake_gate_status"),
            "live_rock_retake_binary_decision_frame_count": operator_status_inputs.get(
                "live_rock_retake_binary_decision_frame_count"
            ),
            "live_rock_retake_confirmed_binary_decision_count": operator_status_inputs.get(
                "live_rock_retake_confirmed_binary_decision_count"
            ),
            "live_rock_retake_first_binary_decision": operator_status_inputs.get(
                "live_rock_retake_first_binary_decision"
            ),
            "live_rock_retake_first_binary_robot_action": operator_status_inputs.get(
                "live_rock_retake_first_binary_robot_action"
            ),
        },
        "inputs": {
            "live_run_checklist": config.live_run_checklist.as_posix(),
            "operator_outcome": config.operator_outcome.as_posix(),
            "submission_packet": config.submission_packet.as_posix(),
            "launch_summary": config.launch_summary.as_posix(),
        },
        "outputs": {
            "card_json": output_json.as_posix(),
            "card_md": output_md.as_posix(),
        },
        "claim_scope": "one-card summary over existing live-demo artifacts; does not run capture, inference, rendering, upload, or repository edits",
    }


def _card_status(
    *,
    checklist: dict[str, Any],
    operator: dict[str, Any],
    submission: dict[str, Any],
) -> str:
    operator_state = str(operator.get("operator_state") or "")
    if operator_state == "ready_to_record":
        return "ready_to_record_live_demo"
    if operator_state == "research_iteration_needed":
        return "simulation_first_research_needed"
    if submission.get("candidate_missing_reason") == "manual_review_rejected":
        return "ready_to_record_live_demo"
    if operator_state == "retake_capture":
        return "retake_live_capture"
    if operator_state == "postprocess_or_repair_capture":
        return "postprocess_or_repair_capture"
    if operator_state == "setup_fix_needed":
        return "setup_fix_needed"
    if submission.get("ready_for_submission_linking") is True:
        return "ready_for_final_submission_review"
    if checklist.get("ok_to_run_pipeline") is True:
        return "ready_to_record_live_demo"
    return "manual_review_needed"


def _primary_video(submission: dict[str, Any]) -> str | None:
    value = submission.get("final_video_path")
    return str(value) if value else None


def _primary_command(
    *,
    status: str,
    checklist: dict[str, Any],
    operator: dict[str, Any],
    submission: dict[str, Any],
    launch_summary: dict[str, Any],
) -> str | None:
    if status in {"ready_for_final_submission_review", "simulation_first_research_needed", "setup_fix_needed"}:
        return None
    if status == "postprocess_or_repair_capture":
        command = operator.get("primary_command")
        return str(command) if command else None
    if status == "ready_to_record_live_demo" and submission.get("candidate_missing_reason") == "manual_review_rejected":
        command = _strict_wrapper_relative_command(launch_summary)
        if command:
            return command
    command = checklist.get("live_pipeline_command") or operator.get("primary_command")
    return str(command) if command else None


def _primary_command_absolute(*, status: str, launch_summary: dict[str, Any]) -> str | None:
    if status in {"ready_for_final_submission_review", "simulation_first_research_needed", "setup_fix_needed"}:
        return None
    scripts_absolute = _dict_value(launch_summary, "scripts_absolute")
    strict_script = scripts_absolute.get("run_live_demo_operator_confirmed_strict")
    if strict_script:
        return f'powershell -ExecutionPolicy Bypass -File "{strict_script}"'
    return None


def _strict_wrapper_relative_command(launch_summary: dict[str, Any]) -> str | None:
    scripts = _dict_value(launch_summary, "scripts")
    strict_script = scripts.get("run_live_demo_operator_confirmed_strict")
    if strict_script:
        return f"powershell -ExecutionPolicy Bypass -File {strict_script}"
    return None


def _next_steps(*, status: str, submission: dict[str, Any]) -> list[str]:
    if status == "ready_for_final_submission_review":
        steps = _list_value(submission, "submission_next_steps")
        return steps or [
            "manual_video_review",
            "youtube_demo_video_upload",
            "readme_demo_video_link",
            "report_demo_video_link",
        ]
    if status == "ready_to_record_live_demo":
        return [
            "run_live_pipeline",
            "review_live_composite",
            "archive_and_select_final_candidate",
        ]
    if status == "simulation_first_research_needed":
        return [
            "inspect_live_frame_log",
            "simulation_first_dataset_iteration",
            "rerun_offline_and_live_validation",
        ]
    if status == "retake_live_capture":
        return [
            "adjust_camera_framing_or_lighting",
            "rerun_live_pipeline",
        ]
    if status == "postprocess_or_repair_capture":
        return [
            "rerun_live_verification",
            "rerun_schunk_composite",
            "refresh_acceptance_and_goal_audit",
        ]
    if status == "setup_fix_needed":
        return [
            "fix_local_setup",
            "rerun_preflight",
        ]
    return ["inspect_operator_outcome_and_artifacts"]


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


def _instruction_details(checklist: dict[str, Any]) -> list[dict[str, str]]:
    value = checklist.get("operator_instruction_details")
    if not isinstance(value, list):
        return []
    details: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        details.append(
            {
                "id": str(item.get("id") or ""),
                "text": str(item.get("text") or ""),
            }
        )
    return details


def _card_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Realtime Demo Final Run Card",
        "",
        f"- Card status: `{summary.get('card_status')}`",
        f"- Ready for submission linking: `{summary.get('ready_for_submission_linking')}`",
        f"- Selected run ID: `{summary.get('selected_run_id')}`",
        f"- Primary video: `{summary.get('primary_video')}`",
        f"- Primary command: `{summary.get('primary_command')}`",
        f"- Primary command absolute: `{summary.get('primary_command_absolute')}`",
        f"- Response prompt: `{summary.get('response_prompt')}`",
        f"- Recommended first expected actual gesture: `{summary.get('recommended_first_expected_actual_gesture')}`",
        f"- Failure category: `{summary.get('failure_category')}`",
        "",
        "## Blocking Items",
        "",
    ]
    blocking = summary.get("blocking_items", [])
    if isinstance(blocking, list) and blocking:
        lines.extend(f"- `{item}`" for item in blocking)
    else:
        lines.append("- None")
    lines.extend(["", "## Next Steps", ""])
    next_steps = summary.get("next_steps", [])
    if isinstance(next_steps, list):
        lines.extend(f"- `{step}`" for step in next_steps)
    instruction_details = summary.get("operator_instruction_details", [])
    lines.extend(["", "## Operator Instructions", ""])
    if isinstance(instruction_details, list) and instruction_details:
        for item in instruction_details:
            if not isinstance(item, dict):
                continue
            lines.append(f"- `{item.get('id')}`: {item.get('text')}")
    else:
        lines.append("- None")
    lines.extend(["", "## Status Inputs", ""])
    status_inputs = summary.get("status_inputs", {})
    if isinstance(status_inputs, dict):
        for key in sorted(status_inputs):
            lines.append(f"- `{key}`: `{status_inputs[key]}`")
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoFinalRunCardConfig", "build_realtime_demo_final_run_card"]
