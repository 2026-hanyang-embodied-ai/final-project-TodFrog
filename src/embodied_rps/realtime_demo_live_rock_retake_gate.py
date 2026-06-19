"""Strict live-rock retake gate for response-window false-trigger suppression."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BINARY_DECISIONS = {"paper", "scissors"}
ROCK_WAIT_DECISIONS = {"rock", "wait_counter_paper"}


@dataclass(frozen=True)
class RealtimeDemoLiveRockRetakeGateConfig:
    """Input and output paths for the live rock-retake gate."""

    output_root: Path = Path("artifacts/realtime_demo_live_rock_retake_gate_20260616")
    frame_log: Path = Path("artifacts/realtime_demo_rehearsal_20260616/live_camera_frames.jsonl")
    postcapture_summary: Path | None = Path("artifacts/realtime_demo_rehearsal_20260616/postcapture/postcapture_summary.json")
    expected_actual_gesture: str | None = None
    response_prompt: str = "scissors"
    min_detection_rate: float = 0.80


def build_realtime_demo_live_rock_retake_gate(
    config: RealtimeDemoLiveRockRetakeGateConfig,
) -> dict[str, object]:
    """Write a strict gate for the expected-rock live retake."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    output_json = config.output_root / "live_rock_retake_gate.json"
    output_md = config.output_root / "live_rock_retake_gate.md"
    rows = _read_jsonl_if_exists(config.frame_log)
    postcapture = _read_json_if_exists(config.postcapture_summary) if config.postcapture_summary else {}
    expected = _expected_actual_gesture(config.expected_actual_gesture, rows, postcapture)
    response_rows = _response_rows(rows, response_prompt=config.response_prompt)
    detection_rate = _detection_rate(rows)
    response_summary = _response_window_summary(response_rows)
    failure_reasons = _failure_reasons(
        rows=rows,
        expected=expected,
        response_rows=response_rows,
        response_summary=response_summary,
        detection_rate=detection_rate,
        min_detection_rate=float(config.min_detection_rate),
        postcapture=postcapture,
    )
    if expected is not None and expected != "rock":
        status = "not_applicable"
        passed = True
        failure_reasons = []
    else:
        passed = not failure_reasons
        status = "passed" if passed else "failed"
    summary: dict[str, object] = {
        "gate_status": status,
        "passed": passed,
        "expected_actual_gesture": expected,
        "response_prompt": config.response_prompt,
        "frame_log": {
            "path": config.frame_log.as_posix(),
            "exists": config.frame_log.exists(),
            "record_count": len(rows),
            "detection_rate": detection_rate,
            "decision_state_counts": dict(Counter(str(row.get("decision_state")) for row in rows)),
        },
        "response_window": response_summary,
        "postcapture_gate": _postcapture_gate_summary(postcapture),
        "failure_reasons": failure_reasons,
        "outputs": {
            "summary_json": output_json.as_posix(),
            "summary_md": output_md.as_posix(),
        },
        "claim_scope": (
            "strict frame-log audit for expected-rock live retakes; detects any response-window "
            "paper/scissors decision_state leakage and does not run camera capture or inference"
        ),
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(_markdown(summary), encoding="utf-8")
    return summary


def _failure_reasons(
    *,
    rows: list[dict[str, Any]],
    expected: str | None,
    response_rows: list[dict[str, Any]],
    response_summary: dict[str, object],
    detection_rate: float | None,
    min_detection_rate: float,
    postcapture: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if not rows:
        failures.append("frame_log_missing_or_empty")
    if expected is None:
        failures.append("expected_actual_gesture_missing")
    elif expected != "rock":
        return failures
    if not response_rows:
        failures.append("response_window_missing")
    if detection_rate is None:
        failures.append("detection_rate_missing")
    elif detection_rate < min_detection_rate:
        failures.append("detection_rate_below_minimum")
    if int(response_summary.get("binary_decision_frame_count", 0)) > 0:
        failures.append("response_window_binary_decision_present")
    if int(response_summary.get("confirmed_binary_decision_count", 0)) > 0:
        failures.append("response_window_confirmed_binary_decision_present")
    if response_rows and int(response_summary.get("paper_robot_action_frame_count", 0)) == 0:
        failures.append("paper_robot_action_missing")
    gate = _dict_value(postcapture, "demo_success_gate")
    if gate and gate.get("expected_actual_gesture") == "rock" and gate.get("passed") is False:
        failures.append("postcapture_demo_success_gate_failed")
    return failures


def _response_window_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    binary_rows = [row for row in rows if row.get("decision_state") in BINARY_DECISIONS]
    confirmed_binary_rows = [
        row for row in binary_rows if row.get("confirmed_decision") is True
    ]
    return {
        "frame_count": len(rows),
        "decision_state_counts": dict(Counter(str(row.get("decision_state")) for row in rows)),
        "robot_action_counts": dict(Counter(str(row.get("robot_action")) for row in rows)),
        "binary_decision_frame_count": len(binary_rows),
        "confirmed_binary_decision_count": len(confirmed_binary_rows),
        "paper_robot_action_frame_count": sum(1 for row in rows if row.get("robot_action") == "paper"),
        "first_binary_decision": _compact_row(binary_rows[0]) if binary_rows else None,
        "first_confirmed_binary_decision": _compact_row(confirmed_binary_rows[0]) if confirmed_binary_rows else None,
        "mean_p_rock": _mean(row.get("p_rock") for row in rows),
        "mean_p_paper": _mean(row.get("p_paper") for row in rows),
        "mean_p_scissors": _mean(row.get("p_scissors") for row in rows),
    }


def _response_rows(rows: list[dict[str, Any]], *, response_prompt: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        response_window = row.get("response_window")
        if response_window is True or row.get("active_prompt") == response_prompt:
            selected.append(row)
    return selected


def _expected_actual_gesture(
    explicit: str | None,
    rows: list[dict[str, Any]],
    postcapture: dict[str, Any],
) -> str | None:
    if explicit:
        return _normalize_gesture(explicit)
    for row in rows:
        value = row.get("expected_actual_gesture")
        if isinstance(value, str) and value.strip():
            return _normalize_gesture(value)
    gate = _dict_value(postcapture, "demo_success_gate")
    value = gate.get("expected_actual_gesture")
    return _normalize_gesture(value) if isinstance(value, str) and value.strip() else None


def _normalize_gesture(value: str) -> str:
    parsed = value.strip().lower()
    if parsed not in {"rock", "paper", "scissors"}:
        raise ValueError(f"Unsupported gesture: {value}")
    return parsed


def _postcapture_gate_summary(postcapture: dict[str, Any]) -> dict[str, object]:
    gate = _dict_value(postcapture, "demo_success_gate")
    return {
        "configured": bool(gate),
        "passed": gate.get("passed"),
        "expected_actual_gesture": gate.get("expected_actual_gesture"),
        "ground_truth_match": gate.get("ground_truth_match"),
        "robot_action_match": gate.get("robot_action_match"),
        "failure_reasons": gate.get("failure_reasons"),
    }


def _compact_row(row: dict[str, Any]) -> dict[str, object]:
    return {
        "frame_index": row.get("frame_index"),
        "time_s": row.get("time_s"),
        "active_prompt": row.get("active_prompt"),
        "decision_state": row.get("decision_state"),
        "robot_action": row.get("robot_action"),
        "confirmed_decision": row.get("confirmed_decision"),
        "p_rock": row.get("p_rock"),
        "p_paper": row.get("p_paper"),
        "p_scissors": row.get("p_scissors"),
    }


def _detection_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row.get("detected") is True) / len(rows)


def _mean(values: object) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return sum(numeric) / len(numeric) if numeric else None


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            rows.append(loaded)
    return rows


def _read_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _markdown(summary: dict[str, object]) -> str:
    response_window = summary.get("response_window", {})
    lines = [
        "# Live Rock Retake Gate",
        "",
        f"- Status: `{summary.get('gate_status')}`",
        f"- Passed: `{summary.get('passed')}`",
        f"- Expected actual gesture: `{summary.get('expected_actual_gesture')}`",
        f"- Response prompt: `{summary.get('response_prompt')}`",
        f"- Binary decision frames: `{response_window.get('binary_decision_frame_count') if isinstance(response_window, dict) else None}`",
        f"- Confirmed binary decision frames: `{response_window.get('confirmed_binary_decision_count') if isinstance(response_window, dict) else None}`",
        f"- Paper robot-action frames: `{response_window.get('paper_robot_action_frame_count') if isinstance(response_window, dict) else None}`",
        "",
        "## Failure Reasons",
        "",
    ]
    failures = summary.get("failure_reasons", [])
    if isinstance(failures, list) and failures:
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoLiveRockRetakeGateConfig", "build_realtime_demo_live_rock_retake_gate"]
