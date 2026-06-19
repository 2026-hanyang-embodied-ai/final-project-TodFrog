"""Operator-facing handoff card for the realtime RPS demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeDemoOperatorHandoffCardConfig:
    """Input and output paths for operator handoff card generation."""

    output_root: Path = Path("artifacts/realtime_demo_operator_handoff_card_20260616")
    live_status_snapshot: Path = Path("artifacts/realtime_demo_live_status_snapshot_20260616/live_status_snapshot.json")
    final_run_card: Path = Path("artifacts/realtime_demo_final_run_card_20260616/final_run_card.json")
    launch_summary: Path = Path("artifacts/realtime_demo_launch_20260616/demo_launch_summary.json")


def build_realtime_demo_operator_handoff_card(config: RealtimeDemoOperatorHandoffCardConfig) -> dict[str, object]:
    """Write a human-facing handoff card from current live-demo status artifacts."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    output_json = config.output_root / "operator_handoff_card.json"
    output_md = config.output_root / "operator_handoff_card.md"
    snapshot = _read_json_if_exists(config.live_status_snapshot) or {}
    final_card = _read_json_if_exists(config.final_run_card) or {}
    launch_summary = _read_json_if_exists(config.launch_summary) or {}
    card = _handoff_card(
        snapshot=snapshot,
        final_card=final_card,
        launch_summary=launch_summary,
        config=config,
        output_json=output_json,
        output_md=output_md,
    )
    output_json.write_text(json.dumps(card, indent=2), encoding="utf-8")
    output_md.write_text(_handoff_markdown(card), encoding="utf-8")
    return card


def _handoff_card(
    *,
    snapshot: dict[str, Any],
    final_card: dict[str, Any],
    launch_summary: dict[str, Any],
    config: RealtimeDemoOperatorHandoffCardConfig,
    output_json: Path,
    output_md: Path,
) -> dict[str, object]:
    completion_candidate = snapshot.get("completion_candidate") is True
    recommended_command = None if completion_candidate else _recommended_command(snapshot, final_card, launch_summary)
    manual_review_command = None if completion_candidate else _manual_review_command(launch_summary)
    raw_recommended_command_absolute = None if completion_candidate else _recommended_command_absolute_candidate(
        launch_summary
    )
    recommended_command_absolute = _ascii_safe_command(raw_recommended_command_absolute)
    recommended_first_expected_actual_gesture = None if completion_candidate else "rock"
    return {
        "handoff_status": _handoff_status(snapshot, final_card),
        "goal_status": snapshot.get("goal_status"),
        "goal_complete": snapshot.get("goal_complete") is True,
        "completion_candidate": completion_candidate,
        "recommended_track": snapshot.get("recommended_track"),
        "recommended_command": recommended_command,
        "manual_review_command": manual_review_command,
        "recommended_first_expected_actual_gesture": recommended_first_expected_actual_gesture,
        "recommended_command_absolute": recommended_command_absolute,
        "recommended_command_absolute_note": _absolute_command_note(
            raw_command=raw_recommended_command_absolute,
            displayed_command=recommended_command_absolute,
        ),
        "operator_steps": _operator_steps(
            completion_candidate=completion_candidate,
            manual_review_command=manual_review_command,
            recommended_first_expected_actual_gesture=recommended_first_expected_actual_gesture,
        ),
        "exit_code_guide": _exit_code_guide(),
        "review_artifacts": _review_artifacts(snapshot, final_card, config, output_json, output_md),
        "missing_requirements": _list_value(snapshot, "missing_requirements"),
        "status_inputs": {
            "live_status_snapshot": config.live_status_snapshot.as_posix(),
            "final_run_card": config.final_run_card.as_posix(),
            "launch_summary": config.launch_summary.as_posix(),
        },
        "launch_scripts": _dict_value(launch_summary, "scripts"),
        "outputs": {
            "operator_handoff_card_json": output_json.as_posix(),
            "operator_handoff_card_md": output_md.as_posix(),
        },
        "claim_scope": (
            "operator handoff card over existing artifacts; does not run camera capture, "
            "inference, rendering, training, upload, or repository edits"
        ),
    }


def _handoff_status(snapshot: dict[str, Any], final_card: dict[str, Any]) -> str:
    status = snapshot.get("snapshot_status") or final_card.get("card_status")
    return str(status) if status else "manual_review_needed"


def _recommended_command(
    snapshot: dict[str, Any],
    final_card: dict[str, Any],
    launch_summary: dict[str, Any],
) -> str | None:
    command = snapshot.get("recommended_command") or final_card.get("primary_command")
    if command:
        return str(command)
    scripts = _dict_value(launch_summary, "scripts")
    strict_script = scripts.get("run_live_demo_operator_confirmed_strict")
    if strict_script:
        return f"powershell -ExecutionPolicy Bypass -File {strict_script}"
    return None


def _recommended_command_absolute_candidate(launch_summary: dict[str, Any]) -> str | None:
    scripts_absolute = _dict_value(launch_summary, "scripts_absolute")
    strict_script = scripts_absolute.get("run_live_demo_operator_confirmed_strict")
    if strict_script:
        return f'powershell -ExecutionPolicy Bypass -File "{strict_script}"'
    project_root = launch_summary.get("project_root_absolute")
    scripts = _dict_value(launch_summary, "scripts")
    strict_relative = scripts.get("run_live_demo_operator_confirmed_strict")
    if not project_root or not strict_relative:
        return None
    strict_path = Path(str(strict_relative))
    if not strict_path.is_absolute():
        strict_path = Path(str(project_root)) / strict_path
    return f'powershell -ExecutionPolicy Bypass -File "{strict_path.as_posix()}"'


def _manual_review_command(launch_summary: dict[str, Any]) -> str | None:
    scripts = _dict_value(launch_summary, "scripts")
    review_script = scripts.get("record_manual_review_decision")
    if review_script:
        return f"powershell -ExecutionPolicy Bypass -File {review_script}"
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


def _operator_steps(
    *,
    completion_candidate: bool,
    manual_review_command: str | None = None,
    recommended_first_expected_actual_gesture: str | None = None,
) -> list[dict[str, str]]:
    if completion_candidate:
        return [
            {
                "id": "inspect_final_video_candidate",
                "text": "Open the selected live composite or final demo candidate and confirm the visible interaction.",
            },
            {
                "id": "manual_video_review",
                "text": "Check that the prompt, hand skeleton, probabilities, prediction, and robot response are readable.",
            },
            {
                "id": "link_submission_artifacts",
                "text": "Use the reviewed candidate for README, report, and YouTube demo-video linking.",
            },
        ]
    steps = [
        {
            "id": "run_recommended_command",
            "text": "Run the recommended PowerShell command from the project root; use the absolute command only when it is shown below.",
        },
        {
            "id": "enter_expected_actual_gesture",
            "text": (
                f"Enter `{recommended_first_expected_actual_gesture or 'rock'}` first as the expected actual gesture "
                "to retest the previous rock-to-scissors false trigger."
            ),
        },
        {
            "id": "press_enter_when_hand_centered",
            "text": "Place the hand in the camera view, then press Enter when the checklist prompt asks for confirmation.",
        },
        {
            "id": "perform_gesture_during_prompt_scissors",
            "text": (
                "Keep rock during PROMPT ROCK and PROMPT PAPER; during PROMPT SCISSORS, treat it as the "
                "response window and perform the selected actual gesture."
            ),
        },
        {
            "id": "hold_until_robot_action_visible",
            "text": "Hold the pose until the prediction and robot response action are visible in the overlay.",
        },
        {
            "id": "inspect_live_status_snapshot",
            "text": "After the command exits, inspect the live status snapshot and this handoff card.",
        },
    ]
    if manual_review_command:
        steps.append(
            {
                "id": "record_manual_review_decision",
                "text": (
                    "After inspecting the decision frame and composite video, record approved or rejected manual "
                    f"review with `{manual_review_command}`."
                ),
            }
        )
    steps.append(
        {
            "id": "review_exit_code",
            "text": "Use the exit-code guide to decide whether to retry capture, continue research, or package evidence.",
        }
    )
    return steps


def _exit_code_guide() -> dict[str, str]:
    return {
        "0": "final evidence ready",
        "10": "live capture still missing",
        "30": "postprocess repair needed",
        "40": "simulation-first research iteration needed",
        "60": "manual review needed",
    }


def _review_artifacts(
    snapshot: dict[str, Any],
    final_card: dict[str, Any],
    config: RealtimeDemoOperatorHandoffCardConfig,
    output_json: Path,
    output_md: Path,
) -> dict[str, object]:
    live_artifacts = _dict_value(snapshot, "live_artifacts")
    review_artifacts: dict[str, object] = {
        "live_status_snapshot": config.live_status_snapshot.as_posix(),
        "final_run_card": config.final_run_card.as_posix(),
        "launch_summary": config.launch_summary.as_posix(),
        "operator_handoff_card_json": output_json.as_posix(),
        "operator_handoff_card_md": output_md.as_posix(),
    }
    for key in (
        "live_overlay",
        "live_frame_log",
        "live_composite",
        "live_response_decision_frame",
        "live_response_prompt_diagnostic_frame",
    ):
        if key in live_artifacts:
            review_artifacts[key] = live_artifacts[key]
        exists_key = f"{key}_exists"
        if exists_key in live_artifacts:
            review_artifacts[exists_key] = live_artifacts[exists_key]
    diagnostic_reason = live_artifacts.get("live_response_prompt_diagnostic_frame_reason")
    if diagnostic_reason is not None:
        review_artifacts["live_response_prompt_diagnostic_frame_reason"] = diagnostic_reason
    primary_video = final_card.get("primary_video")
    if primary_video:
        review_artifacts["final_primary_video"] = str(primary_video)
    return review_artifacts


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


def _handoff_markdown(card: dict[str, object]) -> str:
    lines = [
        "# Realtime Demo Operator Handoff Card",
        "",
        f"- Handoff status: `{card.get('handoff_status')}`",
        f"- Goal status: `{card.get('goal_status')}`",
        f"- Goal complete: `{card.get('goal_complete')}`",
        f"- Completion candidate: `{card.get('completion_candidate')}`",
        f"- Recommended track: `{card.get('recommended_track')}`",
        f"- Recommended command: `{card.get('recommended_command')}`",
        f"- Manual Review Command: `{card.get('manual_review_command')}`",
        f"- Recommended first expected actual gesture: `{card.get('recommended_first_expected_actual_gesture')}`",
        f"- Recommended command absolute: `{card.get('recommended_command_absolute')}`",
        f"- Recommended command absolute note: `{card.get('recommended_command_absolute_note')}`",
        "",
        "## Operator Steps",
        "",
    ]
    steps = card.get("operator_steps", [])
    if isinstance(steps, list):
        for item in steps:
            if isinstance(item, dict):
                lines.append(f"- `{item.get('id')}`: {item.get('text')}")
    lines.extend(["", "## Exit Codes", ""])
    exit_code_guide = card.get("exit_code_guide", {})
    if isinstance(exit_code_guide, dict):
        for code in sorted(exit_code_guide, key=str):
            lines.append(f"- `{code}`: {exit_code_guide[code]}")
    lines.extend(["", "## Review Artifacts", ""])
    review_artifacts = card.get("review_artifacts", {})
    if isinstance(review_artifacts, dict):
        for key in sorted(review_artifacts):
            lines.append(f"- `{key}`: `{review_artifacts[key]}`")
    lines.extend(["", "## Missing Requirements", ""])
    missing = card.get("missing_requirements", [])
    if isinstance(missing, list) and missing:
        lines.extend(f"- `{item}`" for item in missing)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoOperatorHandoffCardConfig", "build_realtime_demo_operator_handoff_card"]
