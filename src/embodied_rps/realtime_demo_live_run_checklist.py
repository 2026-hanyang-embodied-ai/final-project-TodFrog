"""Operator checklist for starting the realtime RPS live demo run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LIVE_PIPELINE_COMMAND = (
    "powershell -ExecutionPolicy Bypass -File "
    "artifacts\\realtime_demo_launch_20260616\\24_run_live_demo_operator_confirmed_strict.ps1"
)
DEFAULT_OPERATOR_CONFIRMED_COMMAND = (
    "powershell -ExecutionPolicy Bypass -File artifacts\\realtime_demo_launch_20260616\\22_run_live_demo_operator_confirmed.ps1"
)


@dataclass(frozen=True)
class RealtimeDemoLiveRunChecklistConfig:
    """Input and output paths for the pre-run live demo checklist."""

    output_root: Path = Path("artifacts/realtime_demo_live_run_checklist_20260616")
    readiness_summary: Path = Path("artifacts/realtime_demo_readiness_20260616/readiness_summary.json")
    operator_outcome: Path = Path("artifacts/realtime_demo_operator_outcome_20260616/operator_outcome.json")
    preflight_summary: Path = Path("artifacts/realtime_demo_rehearsal_20260616/preflight/preflight_summary.json")
    launch_summary: Path = Path("artifacts/realtime_demo_launch_20260616/demo_launch_summary.json")


def build_realtime_demo_live_run_checklist(config: RealtimeDemoLiveRunChecklistConfig) -> dict[str, object]:
    """Write a concise operator checklist for the next live demo run."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    output_json = config.output_root / "live_run_checklist.json"
    output_md = config.output_root / "live_run_checklist.md"
    readiness = _read_json_if_exists(config.readiness_summary) or {}
    operator = _read_json_if_exists(config.operator_outcome) or {}
    preflight = _read_json_if_exists(config.preflight_summary) or {}
    launch = _read_json_if_exists(config.launch_summary) or {}

    summary = _checklist_summary(
        readiness=readiness,
        operator=operator,
        preflight=preflight,
        launch=launch,
        config=config,
        output_json=output_json,
        output_md=output_md,
    )
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(_checklist_markdown(summary), encoding="utf-8")
    return summary


def _checklist_summary(
    *,
    readiness: dict[str, Any],
    operator: dict[str, Any],
    preflight: dict[str, Any],
    launch: dict[str, Any],
    config: RealtimeDemoLiveRunChecklistConfig,
    output_json: Path,
    output_md: Path,
) -> dict[str, object]:
    blocking_items = _blocking_items(readiness=readiness, operator=operator, preflight=preflight, launch=launch)
    ok_to_run = not blocking_items
    response_prompt = _response_prompt(preflight)
    live_pipeline_command = _live_pipeline_command(launch=launch, operator=operator)
    camera = _dict_value(preflight, "camera")
    recommended_first_expected_actual_gesture = "rock"
    summary: dict[str, object] = {
        "checklist_status": "ready_to_start_live_recording" if ok_to_run else "blocked_before_live_recording",
        "ok_to_run_pipeline": ok_to_run,
        "live_pipeline_command": live_pipeline_command if ok_to_run else None,
        "response_prompt": response_prompt,
        "recommended_first_expected_actual_gesture": recommended_first_expected_actual_gesture if ok_to_run else None,
        "camera_index": preflight.get("camera_index"),
        "camera_frame_width": camera.get("frame_width"),
        "camera_frame_height": camera.get("frame_height"),
        "camera_max_frames": launch.get("camera_max_frames"),
        "hand_visibility_min_detection_rate": launch.get("hand_visibility_min_detection_rate"),
        "blocking_items": blocking_items,
        "operator_steps": _operator_steps(ok_to_run=ok_to_run, response_prompt=response_prompt),
        "operator_instruction_details": _operator_instruction_details(
            ok_to_run=ok_to_run,
            response_prompt=response_prompt,
            recommended_first_expected_actual_gesture=recommended_first_expected_actual_gesture,
        ),
        "status_inputs": {
            "readiness_status": readiness.get("status"),
            "operator_state": operator.get("operator_state"),
            "operator_recommended_exit_code": operator.get("recommended_exit_code"),
            "preflight_status": preflight.get("status"),
            "launch_script_count": launch.get("script_count"),
        },
        "inputs": {
            "readiness_summary": config.readiness_summary.as_posix(),
            "operator_outcome": config.operator_outcome.as_posix(),
            "preflight_summary": config.preflight_summary.as_posix(),
            "launch_summary": config.launch_summary.as_posix(),
        },
        "outputs": {
            "checklist_json": output_json.as_posix(),
            "checklist_md": output_md.as_posix(),
        },
        "claim_scope": "pre-run operator checklist over existing artifacts; does not run camera capture, model inference, or rendering",
    }
    return summary


def _blocking_items(
    *,
    readiness: dict[str, Any],
    operator: dict[str, Any],
    preflight: dict[str, Any],
    launch: dict[str, Any],
) -> list[str]:
    blocking: list[str] = []
    if not _readiness_allows_live_checklist(readiness):
        blocking.append("readiness_not_ready_for_live_capture")
    if str(operator.get("operator_state")) not in {"ready_to_record", "retake_capture"}:
        blocking.append("operator_not_ready_to_record")
    if not _preflight_allows_live_checklist(preflight):
        blocking.append("preflight_not_ready")
    if int(launch.get("script_count", 0) or 0) < 18:
        blocking.append("launch_scripts_incomplete")
    return blocking


def _readiness_allows_live_checklist(readiness: dict[str, Any]) -> bool:
    if readiness.get("status") == "ready_for_live_capture":
        return True
    remaining = readiness.get("remaining_actions")
    if not isinstance(remaining, list):
        return False
    return {str(item) for item in remaining} == {"operator_command_audit_not_passed"}


def _preflight_allows_live_checklist(preflight: dict[str, Any]) -> bool:
    if preflight.get("status") == "ready_for_live_demo" and preflight.get("ok") is True:
        return True
    failures = preflight.get("failures")
    if not isinstance(failures, list):
        return False
    failure_set = {str(item) for item in failures}
    if failure_set != {"hand_visibility_low"}:
        return False
    camera = _dict_value(preflight, "camera")
    return camera.get("opened") is True and camera.get("frame_read") is True


def _operator_steps(*, ok_to_run: bool, response_prompt: str | None) -> list[str]:
    if not ok_to_run:
        return [
            "fix_local_setup",
            "refresh_readiness",
            "rerun_preflight",
        ]
    prompt = response_prompt or "scissors"
    return [
        "show_hand_to_camera_before_starting",
        "keep_single_hand_centered_and_well_lit",
        "run_live_pipeline_command",
        f"perform_gesture_during_prompt_{prompt}",
        "hold_hand_until_robot_action_is_visible",
        "review_live_composite_and_archive_reports",
    ]


def _operator_instruction_details(
    *,
    ok_to_run: bool,
    response_prompt: str | None,
    recommended_first_expected_actual_gesture: str,
) -> list[dict[str, str]]:
    if not ok_to_run:
        return [
            {
                "id": "fix_blocking_items",
                "text": "Resolve the blocking items, then rebuild the live-run checklist before recording.",
            }
        ]
    prompt = (response_prompt or "scissors").upper()
    return [
        {
            "id": "enter_expected_actual_gesture",
            "text": (
                f"Enter `{recommended_first_expected_actual_gesture}` first when prompted for the expected actual "
                "gesture; this retests the previous rock-to-scissors false trigger."
            ),
        },
        {
            "id": "standby_prompts",
            "text": "Keep a clear rock/fist standby during PROMPT ROCK and PROMPT PAPER.",
        },
        {
            "id": "response_window",
            "text": (
                f"Treat PROMPT {prompt} as the response window. If the expected actual gesture is `rock`, keep "
                "rock; otherwise transition from rock to the selected actual gesture."
            ),
        },
        {
            "id": "hold_for_decision",
            "text": "Hold the final hand pose until the prediction, confidence, and robot action are visible.",
        },
    ]


def _live_pipeline_command(*, launch: dict[str, Any], operator: dict[str, Any]) -> str:
    scripts = _dict_value(launch, "scripts")
    strict_script = scripts.get("run_live_demo_operator_confirmed_strict")
    if strict_script:
        script_path = str(strict_script).replace("/", "\\")
        return f"powershell -ExecutionPolicy Bypass -File {script_path}"
    confirmed_script = scripts.get("run_live_demo_operator_confirmed")
    if confirmed_script:
        script_path = str(confirmed_script).replace("/", "\\")
        return f"powershell -ExecutionPolicy Bypass -File {script_path}"
    command = operator.get("primary_command")
    if command:
        return str(command)
    return DEFAULT_OPERATOR_CONFIRMED_COMMAND


def _response_prompt(preflight: dict[str, Any]) -> str | None:
    policy = _dict_value(preflight, "required_policy")
    value = policy.get("response_prompt")
    return str(value) if value is not None else None


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _checklist_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Realtime Demo Live Run Checklist",
        "",
        f"- Checklist status: `{summary.get('checklist_status')}`",
        f"- OK to run pipeline: `{summary.get('ok_to_run_pipeline')}`",
        f"- Live pipeline command: `{summary.get('live_pipeline_command')}`",
        f"- Response prompt: `{summary.get('response_prompt')}`",
        f"- Expected Actual Gesture First Retake: `{summary.get('recommended_first_expected_actual_gesture')}`",
        f"- Camera index: `{summary.get('camera_index')}`",
        f"- Camera frame: `{summary.get('camera_frame_width')}x{summary.get('camera_frame_height')}`",
        "",
        "## Blocking Items",
        "",
    ]
    blocking = summary.get("blocking_items", [])
    if isinstance(blocking, list) and blocking:
        lines.extend(f"- `{item}`" for item in blocking)
    else:
        lines.append("- None")
    lines.extend(["", "## Operator Steps", ""])
    steps = summary.get("operator_steps", [])
    if isinstance(steps, list):
        lines.extend(f"- `{step}`" for step in steps)
    lines.extend(["", "## Operator Instructions", ""])
    details = summary.get("operator_instruction_details", [])
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                lines.append(f"- `{item.get('id')}`: {item.get('text')}")
    lines.extend(["", "## Status Inputs", ""])
    status_inputs = summary.get("status_inputs", {})
    if isinstance(status_inputs, dict):
        for key in sorted(status_inputs):
            lines.append(f"- `{key}`: `{status_inputs[key]}`")
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoLiveRunChecklistConfig", "build_realtime_demo_live_run_checklist"]
