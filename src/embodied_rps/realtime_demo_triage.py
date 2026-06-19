"""Failure triage for prompt-gated realtime RPS demo captures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STRICT_LIVE_DEMO_COMMAND = (
    "Run artifacts/realtime_demo_launch_20260616/24_run_live_demo_operator_confirmed_strict.ps1 "
    "and perform during PROMPT SCISSORS."
)


@dataclass(frozen=True)
class RealtimeDemoTriageConfig:
    """Configuration for summarizing the next action after demo verification."""

    output_root: Path
    readiness_summary: Path | None = Path("artifacts/realtime_demo_readiness_20260616/readiness_summary.json")
    preflight_summary: Path | None = Path("artifacts/realtime_demo_rehearsal_20260616/preflight/preflight_summary.json")
    postcapture_summary: Path | None = Path(
        "artifacts/realtime_demo_rehearsal_20260616/postcapture/postcapture_summary.json"
    )
    composite_manifest: Path | None = Path(
        "artifacts/realtime_schunk_live_demo_composite_20260616/realtime_schunk_demo_composite_manifest.json"
    )
    live_rock_retake_gate: Path | None = Path(
        "artifacts/realtime_demo_live_rock_retake_gate_20260616/live_rock_retake_gate.json"
    )


def triage_realtime_demo_capture(config: RealtimeDemoTriageConfig) -> dict[str, object]:
    """Classify the current demo state and recommend the next concrete action."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    summary_json = config.output_root / "triage_summary.json"
    summary_md = config.output_root / "triage_summary.md"

    readiness = _read_json_if_exists(config.readiness_summary)
    preflight = _read_json_if_exists(config.preflight_summary)
    postcapture = _read_json_if_exists(config.postcapture_summary)
    composite = _read_json_if_exists(config.composite_manifest)
    live_rock_gate = _read_json_if_exists(config.live_rock_retake_gate)
    evidence = _build_evidence(
        readiness=readiness,
        preflight=preflight,
        postcapture=postcapture,
        composite=composite,
        live_rock_gate=live_rock_gate,
        config=config,
    )
    status, category, action = _classify(
        readiness=readiness,
        preflight=preflight,
        postcapture=postcapture,
        composite=composite,
        live_rock_gate=live_rock_gate,
    )
    summary: dict[str, object] = {
        "status": status,
        "failure_category": category,
        "recommended_next_action": action,
        "evidence": evidence,
        "outputs": {
            "summary_json": summary_json.as_posix(),
            "summary_md": summary_md.as_posix(),
        },
        "claim_scope": "triage over existing readiness/postcapture/composite artifacts; does not run camera capture or model inference",
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _read_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _build_evidence(
    *,
    readiness: dict[str, Any] | None,
    preflight: dict[str, Any] | None,
    postcapture: dict[str, Any] | None,
    composite: dict[str, Any] | None,
    live_rock_gate: dict[str, Any] | None,
    config: RealtimeDemoTriageConfig,
) -> dict[str, object]:
    gate = _dict_value(postcapture, "demo_success_gate")
    frame_log = _dict_value(postcapture, "frame_log")
    first_binary = _dict_value(gate, "first_response_prompt_binary_decision")
    hand_visibility = _dict_value(preflight, "hand_visibility")
    response_prompt_diagnostic_frame = _dict_value(postcapture, "response_prompt_diagnostic_frame")
    outputs = _dict_value(postcapture, "outputs")
    live_rock_response = _dict_value(live_rock_gate, "response_window")
    live_rock_first_binary = _dict_value(live_rock_response, "first_binary_decision")
    diagnostic_frame_path = _first_present(
        response_prompt_diagnostic_frame.get("path"),
        outputs.get("response_prompt_diagnostic_frame_png") if response_prompt_diagnostic_frame else None,
    )
    return {
        "readiness_summary": config.readiness_summary.as_posix() if config.readiness_summary else None,
        "preflight_summary": config.preflight_summary.as_posix() if config.preflight_summary else None,
        "postcapture_summary": config.postcapture_summary.as_posix() if config.postcapture_summary else None,
        "composite_manifest": config.composite_manifest.as_posix() if config.composite_manifest else None,
        "live_rock_retake_gate": config.live_rock_retake_gate.as_posix() if config.live_rock_retake_gate else None,
        "readiness_status": readiness.get("status") if readiness else None,
        "remaining_actions": _list_value(readiness, "remaining_actions"),
        "preflight_status": preflight.get("status") if preflight else None,
        "preflight_failures": _list_value(preflight, "failures"),
        "hand_visibility_detection_rate": _optional_float(_value(hand_visibility, "detection_rate")),
        "hand_visibility_detected_frames": hand_visibility.get("detected_frames"),
        "hand_visibility_frame_count": hand_visibility.get("frame_count"),
        "hand_visibility_diagnostic_images": _list_value(hand_visibility, "diagnostic_image_paths"),
        "postcapture_status": postcapture.get("status") if postcapture else None,
        "postcapture_failures": _list_value(postcapture, "failures"),
        "gate_passed": gate.get("passed") if gate else None,
        "gate_failure_reasons": _list_value(gate, "failure_reasons"),
        "expected_actual_gesture": gate.get("expected_actual_gesture") if gate else None,
        "detection_rate": _first_present(
            _optional_float(_value(gate, "detection_rate")),
            _optional_float(_value(frame_log, "detection_rate")),
        ),
        "binary_decision_latency_s": _optional_float(_value(gate, "binary_decision_latency_s")),
        "first_binary_decision": first_binary.get("decision_state") if first_binary else None,
        "first_binary_robot_action": first_binary.get("robot_action") if first_binary else None,
        "live_rock_retake_gate_status": live_rock_gate.get("gate_status") if live_rock_gate else None,
        "live_rock_retake_gate_passed": live_rock_gate.get("passed") if live_rock_gate else None,
        "live_rock_retake_failure_reasons": _list_value(live_rock_gate, "failure_reasons"),
        "live_rock_retake_binary_decision_frame_count": live_rock_response.get("binary_decision_frame_count")
        if live_rock_response
        else None,
        "live_rock_retake_confirmed_binary_decision_count": live_rock_response.get(
            "confirmed_binary_decision_count"
        )
        if live_rock_response
        else None,
        "live_rock_retake_first_binary_decision": live_rock_first_binary.get("decision_state")
        if live_rock_first_binary
        else None,
        "live_rock_retake_first_binary_robot_action": live_rock_first_binary.get("robot_action")
        if live_rock_first_binary
        else None,
        "response_prompt_diagnostic_frame": diagnostic_frame_path,
        "response_prompt_diagnostic_frame_reason": response_prompt_diagnostic_frame.get("reason")
        if response_prompt_diagnostic_frame
        else None,
        "composite_status": composite.get("status") if composite else None,
        "composite_frame_count": composite.get("frame_count") if composite else None,
    }


def _classify(
    *,
    readiness: dict[str, Any] | None,
    preflight: dict[str, Any] | None,
    postcapture: dict[str, Any] | None,
    composite: dict[str, Any] | None,
    live_rock_gate: dict[str, Any] | None,
) -> tuple[str, str | None, str]:
    readiness_status = readiness.get("status") if readiness else None
    remaining = set(_list_value(readiness, "remaining_actions"))
    preflight_failures = set(_list_value(preflight, "failures"))
    post_failures = set(_list_value(postcapture, "failures"))
    gate = _dict_value(postcapture, "demo_success_gate")
    gate_reasons = set(_list_value(gate, "failure_reasons"))
    composite_status = composite.get("status") if composite else None
    live_rock_reasons = set(_list_value(live_rock_gate, "failure_reasons"))

    if (
        live_rock_gate
        and live_rock_gate.get("expected_actual_gesture") == "rock"
        and live_rock_gate.get("passed") is False
        and (
            "response_window_binary_decision_present" in live_rock_reasons
            or "response_window_confirmed_binary_decision_present" in live_rock_reasons
        )
    ):
        return (
            "needs_research_iteration",
            "rock_false_trigger",
            "Use the live rock-retake gate frame-log evidence as hard-negative data, add simulated 3D rock-wait examples, retrain or tune the guard, then retake.",
        )

    if gate.get("passed") is True and (
        composite_status == "passed" or composite_status is None or readiness_status == "live_demo_artifact_ready"
    ):
        return (
            "ready_for_final_demo",
            None,
            "Use the live composite as final demo evidence and record presentation/demo video narration.",
        )

    if postcapture is None and _readiness_indicates_live_capture_needed(readiness_status, remaining):
        return (
            "needs_live_capture",
            "live_capture_missing",
            STRICT_LIVE_DEMO_COMMAND,
        )

    # Once post-capture evidence exists, stale preflight hand-visibility artifacts
    # should not override the actual live capture outcome.
    preflight_can_block = postcapture is None

    if preflight_can_block and ("camera_unavailable" in preflight_failures or "camera_probe_failed" in preflight_failures):
        return (
            "needs_retake",
            "camera_unavailable",
            "Fix camera access, camera index, or another app using the camera, then rerun preflight.",
        )

    if preflight_can_block and ("hand_visibility_low" in preflight_failures or "hand_visibility_probe_failed" in preflight_failures):
        return (
            "needs_retake",
            "hand_visibility_low",
            "Fix hand framing, lighting, hand distance, or background using the preflight diagnostic images, then rerun.",
        )

    if preflight_can_block and preflight_failures.intersection(
        {"config_missing", "config_load_failed", "python_executable_missing", "profile_missing", "model_state_missing"}
    ):
        return (
            "needs_setup_fix",
            "preflight_config_or_model_failure",
            "Fix the realtime config, Python environment, profile paths, or model checkpoints before live capture.",
        )

    if "overlay_video_missing" in post_failures:
        return (
            "needs_live_capture",
            "live_capture_missing",
            "Re-run the strict live wrapper because no overlay video was produced.",
        )

    if "overlay_video_unreadable" in post_failures or "frame_count_too_short" in post_failures:
        return (
            "needs_retake",
            "stream_failure",
            "Retake the live demo after checking camera access, output path, and capture duration.",
        )

    if "frame_log_missing" in post_failures or "frame_log_empty" in post_failures:
        return (
            "needs_retake",
            "frame_log_missing",
            "Retake or rerun verification with --frame-log-jsonl so prediction timing can be audited.",
        )

    if "detection_rate_below_minimum" in gate_reasons or "detection_rate_missing" in gate_reasons:
        return (
            "needs_retake",
            "detection_loss",
            "Improve lighting, hand framing, and camera angle, then retake before changing model policy.",
        )

    expected_actual = str(gate.get("expected_actual_gesture") or "").lower()
    first_binary = _dict_value(gate, "first_response_prompt_binary_decision")
    first_binary_decision = str(first_binary.get("decision_state") or "")
    if (
        expected_actual == "rock"
        and (
            "unexpected_actual_gesture_decision" in gate_reasons
            or "unexpected_actual_gesture_robot_action" in gate_reasons
        )
        and first_binary_decision in {"paper", "scissors"}
    ):
        return (
            "needs_research_iteration",
            "rock_false_trigger",
            "Use the live rock-hold frame log as hard-negative evidence, add simulated 3D rock-wait examples, and retrain before retaking.",
        )

    if "unexpected_response_decision" in gate_reasons or "unexpected_actual_gesture_decision" in gate_reasons:
        return (
            "needs_research_iteration",
            "wrong_class",
            "Inspect the frame log and add simulated 3D skeleton augmentation around the confused class boundary.",
        )

    if "unexpected_robot_action" in gate_reasons or "unexpected_actual_gesture_robot_action" in gate_reasons:
        return (
            "needs_research_iteration",
            "robot_action_mismatch",
            "Check response-policy mapping before using this capture for the SCHUNK/demo layer.",
        )

    if "response_prompt_binary_decision_late" in gate_reasons:
        return (
            "needs_research_iteration",
            "late_binary_decision",
            "Add earlier-motion simulated 3D skeleton augmentation and rerun the strict latency gate.",
        )

    if "response_prompt_binary_decision_missing" in gate_reasons:
        return (
            "needs_research_iteration",
            "missing_binary_decision",
            "Add simulated 3D skeleton hard examples for the response prompt and rerun validation before retaking.",
        )

    if readiness_status == "live_capture_needs_postprocessing" or "live_composite_missing_or_not_ready" in remaining:
        return (
            "needs_postprocessing",
            "composite_missing",
            "Run 06_verify_live_capture.ps1, then 05_create_live_schunk_composite.ps1.",
        )

    if postcapture is None and readiness is None:
        return (
            "needs_live_capture",
            "live_capture_missing",
            "Generate readiness/postcapture artifacts by running the strict one-command live demo wrapper.",
        )

    return (
        "needs_manual_review",
        "unknown",
        "Inspect readiness, postcapture, and frame-log artifacts; the failure did not match a known category.",
    )


def _readiness_indicates_live_capture_needed(readiness_status: object, remaining: set[str]) -> bool:
    return (
        readiness_status == "ready_for_live_capture"
        or "live_capture_missing" in remaining
        or remaining == {"operator_command_audit_not_passed"}
    )


def _dict_value(payload: dict[str, Any] | None, key: str) -> dict[str, Any]:
    value = payload.get(key) if payload else None
    return value if isinstance(value, dict) else {}


def _list_value(payload: dict[str, Any] | None, key: str) -> list[str]:
    value = payload.get(key) if payload else None
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _value(payload: dict[str, Any], key: str) -> object:
    return payload.get(key)


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _first_present(*values: object) -> object:
    for value in values:
        if value is not None:
            return value
    return None


def _summary_markdown(summary: dict[str, object]) -> str:
    evidence = summary.get("evidence", {})
    lines = [
        "# Realtime Demo Failure Triage",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Failure category: `{summary.get('failure_category')}`",
        f"- Recommended next action: {summary.get('recommended_next_action')}",
        "",
        "## Evidence",
        "",
    ]
    if isinstance(evidence, dict):
        for key in sorted(evidence):
            lines.append(f"- `{key}`: `{evidence[key]}`")
    lines.append("")
    return "\n".join(lines)


__all__ = ["STRICT_LIVE_DEMO_COMMAND", "RealtimeDemoTriageConfig", "triage_realtime_demo_capture"]
