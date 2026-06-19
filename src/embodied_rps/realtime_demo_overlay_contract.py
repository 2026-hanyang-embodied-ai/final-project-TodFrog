"""Contract audit for realtime RPS demo overlay evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-untyped]


@dataclass(frozen=True)
class RealtimeDemoOverlayContractConfig:
    """Configuration for auditing a demo overlay and its per-frame log."""

    overlay_video: Path
    frame_log_jsonl: Path
    output_root: Path
    response_prompt: str = "scissors"
    required_prompts: tuple[str, ...] = ("rock", "paper", "scissors")
    min_detection_rate: float = 0.80
    max_binary_latency_s: float = 0.50
    expected_actual_gesture: str | None = None


def audit_realtime_demo_overlay_contract(config: RealtimeDemoOverlayContractConfig) -> dict[str, object]:
    """Verify that overlay artifacts contain the demo evidence required by the goal."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    summary_json = config.output_root / "overlay_contract_summary.json"
    summary_md = config.output_root / "overlay_contract_summary.md"
    video = _probe_video(config.overlay_video)
    rows = _read_frame_log_rows(config.frame_log_jsonl) if config.frame_log_jsonl.exists() else []
    prompt_counts = _prompt_counts(rows)
    detection_rate = _detection_rate(rows)
    response_start_s = _response_prompt_start_time(rows, response_prompt=config.response_prompt)
    first_binary = _first_response_binary_decision(rows, response_prompt=config.response_prompt)
    first_expected = _first_response_expected_decision(
        rows,
        response_prompt=config.response_prompt,
        expected_actual_gesture=config.expected_actual_gesture,
    )
    binary_time_s = _optional_float(first_binary.get("time_s")) if first_binary else None
    expected_time_s = _optional_float(first_expected.get("time_s")) if first_expected else None
    binary_latency_s = (
        binary_time_s - response_start_s if binary_time_s is not None and response_start_s is not None else None
    )
    expected_latency_s = (
        expected_time_s - response_start_s if expected_time_s is not None and response_start_s is not None else None
    )
    checks = {
        "overlay_video_exists": config.overlay_video.exists(),
        "video_opened": video.get("opened") is True,
        "frame_log_exists": config.frame_log_jsonl.exists(),
        "frame_log_nonempty": bool(rows),
        "frame_count_matches_log": bool(video.get("frame_count") == len(rows) and rows),
        "prompt_cycle_present": all(prompt_counts.get(prompt, 0) > 0 for prompt in config.required_prompts),
        "probabilities_present": _all_numeric_fields(rows, ("p_rock", "p_paper", "p_scissors")),
        "confidence_margin_present": _all_numeric_fields(rows, ("confidence", "margin")),
        "transition_mass_present": _all_numeric_fields(rows, ("transition_mass",)),
        "robot_action_present": any(isinstance(row.get("robot_action"), str) and row.get("robot_action") for row in rows),
        "detection_rate_ok": detection_rate is not None and detection_rate >= config.min_detection_rate,
        "response_prompt_present": prompt_counts.get(config.response_prompt, 0) > 0,
        "response_prompt_binary_decision_present": first_binary is not None,
        "response_prompt_binary_decision_on_time": (
            binary_latency_s is not None and binary_latency_s <= config.max_binary_latency_s
        ),
        "response_prompt_expected_decision_present": first_expected is not None,
        "response_prompt_expected_decision_on_time": (
            expected_latency_s is not None and expected_latency_s <= config.max_binary_latency_s
        ),
    }
    failures = _failures_for_checks(
        checks,
        binary_latency_s=binary_latency_s,
        expected_latency_s=expected_latency_s,
        config=config,
    )
    summary: dict[str, object] = {
        "status": "passed" if not failures else "blocked",
        "contract_passed": not failures,
        "overlay_video": config.overlay_video.as_posix(),
        "frame_log_jsonl": config.frame_log_jsonl.as_posix(),
        "checks": checks,
        "failures": failures,
        "metrics": {
            "video_frame_count": video.get("frame_count"),
            "video_width": video.get("width"),
            "video_height": video.get("height"),
            "video_duration_s": video.get("duration_s"),
            "frame_log_records": len(rows),
            "detection_rate": detection_rate,
            "prompt_counts": prompt_counts,
            "response_prompt_start_time_s": response_start_s,
            "binary_decision_time_s": binary_time_s,
            "binary_decision_latency_s": binary_latency_s,
            "first_response_binary_decision": first_binary,
            "expected_actual_gesture": _normalize_expected_actual_gesture(config.expected_actual_gesture),
            "expected_decision_time_s": expected_time_s,
            "expected_decision_latency_s": expected_latency_s,
            "first_response_expected_decision": first_expected,
        },
        "outputs": {
            "summary_json": summary_json.as_posix(),
            "summary_md": summary_md.as_posix(),
        },
        "claim_scope": "overlay contract over existing video/frame-log artifacts; does not run model inference or OCR",
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _probe_video(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    try:
        opened = bool(capture.isOpened())
        if not opened:
            return {"opened": False}
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return {
            "opened": True,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_s": float(frame_count) / fps if fps > 0.0 else 0.0,
        }
    finally:
        capture.release()


def _read_frame_log_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        loaded = json.loads(stripped)
        if not isinstance(loaded, dict):
            raise ValueError("frame log JSONL rows must be objects")
        rows.append(loaded)
    return rows


def _prompt_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        prompt = row.get("active_prompt")
        if isinstance(prompt, str) and prompt:
            counts[prompt] = counts.get(prompt, 0) + 1
    return counts


def _detection_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row.get("detected") is True) / len(rows)


def _all_numeric_fields(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> bool:
    if not rows:
        return False
    for row in rows:
        for field in fields:
            value = row.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
    return True


def _response_prompt_start_time(rows: list[dict[str, Any]], *, response_prompt: str) -> float | None:
    for row in rows:
        if row.get("active_prompt") == response_prompt:
            return _optional_float(row.get("time_s"))
    return None


def _first_response_binary_decision(
    rows: list[dict[str, Any]],
    *,
    response_prompt: str,
) -> dict[str, object] | None:
    for row in rows:
        if (
            row.get("active_prompt") == response_prompt
            and row.get("confirmed_decision") is True
            and row.get("decision_state") in {"paper", "scissors"}
        ):
            return {
                "frame_index": row.get("frame_index"),
                "time_s": row.get("time_s"),
                "decision_state": row.get("decision_state"),
                "robot_action": row.get("robot_action"),
                "confidence": row.get("confidence"),
                "margin": row.get("margin"),
                "transition_mass": row.get("transition_mass"),
            }
    return None


def _first_response_expected_decision(
    rows: list[dict[str, Any]],
    *,
    response_prompt: str,
    expected_actual_gesture: str | None,
) -> dict[str, object] | None:
    expected_decisions = _expected_decisions_for_actual_gesture(_normalize_expected_actual_gesture(expected_actual_gesture))
    if expected_decisions is None:
        return _first_response_binary_decision(rows, response_prompt=response_prompt)
    for row in rows:
        if (
            row.get("active_prompt") == response_prompt
            and row.get("confirmed_decision") is True
            and row.get("decision_state") in expected_decisions
        ):
            return {
                "frame_index": row.get("frame_index"),
                "time_s": row.get("time_s"),
                "decision_state": row.get("decision_state"),
                "robot_action": row.get("robot_action"),
                "confidence": row.get("confidence"),
                "margin": row.get("margin"),
                "transition_mass": row.get("transition_mass"),
            }
    return None


def _normalize_expected_actual_gesture(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in {"rock", "paper", "scissors"} else None


def _expected_decisions_for_actual_gesture(expected_actual_gesture: str | None) -> set[str] | None:
    if expected_actual_gesture == "rock":
        return {"rock", "wait_counter_paper"}
    if expected_actual_gesture in {"paper", "scissors"}:
        return {expected_actual_gesture}
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _failures_for_checks(
    checks: dict[str, bool],
    *,
    binary_latency_s: float | None,
    expected_latency_s: float | None,
    config: RealtimeDemoOverlayContractConfig,
) -> list[str]:
    failures: list[str] = []
    expected_actual = _normalize_expected_actual_gesture(config.expected_actual_gesture)
    require_expected_decision = expected_actual is not None
    required_map = {
        "overlay_video_exists": "overlay_video_missing",
        "video_opened": "overlay_video_unreadable",
        "frame_log_exists": "frame_log_missing",
        "frame_log_nonempty": "frame_log_empty",
        "frame_count_matches_log": "frame_count_mismatch",
        "prompt_cycle_present": "prompt_cycle_missing",
        "probabilities_present": "probabilities_missing",
        "confidence_margin_present": "confidence_margin_missing",
        "transition_mass_present": "transition_mass_missing",
        "robot_action_present": "robot_action_missing",
        "detection_rate_ok": "detection_rate_below_minimum",
        "response_prompt_present": "response_prompt_missing",
    }
    if require_expected_decision:
        required_map["response_prompt_expected_decision_present"] = "response_prompt_expected_decision_missing"
    else:
        required_map["response_prompt_binary_decision_present"] = "response_prompt_binary_decision_missing"
    for check, failure in required_map.items():
        if not checks.get(check, False):
            failures.append(failure)
    timing_check = (
        "response_prompt_expected_decision_on_time"
        if require_expected_decision
        else "response_prompt_binary_decision_on_time"
    )
    timing_latency = expected_latency_s if require_expected_decision else binary_latency_s
    timing_missing_failure = (
        "response_prompt_expected_decision_timing_missing"
        if require_expected_decision
        else "response_prompt_binary_decision_timing_missing"
    )
    timing_late_failure = (
        "response_prompt_expected_decision_late"
        if require_expected_decision
        else "response_prompt_binary_decision_late"
    )
    if not checks.get(timing_check, False):
        if timing_latency is not None and timing_latency > config.max_binary_latency_s:
            failures.append(timing_late_failure)
        else:
            failures.append(timing_missing_failure)
    return failures


def _summary_markdown(summary: dict[str, object]) -> str:
    checks = summary.get("checks", {})
    metrics = summary.get("metrics", {})
    failures = summary.get("failures", [])
    lines = [
        "# Realtime Demo Overlay Contract Audit",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Contract passed: `{summary.get('contract_passed')}`",
        f"- Overlay video: `{summary.get('overlay_video')}`",
        f"- Frame log: `{summary.get('frame_log_jsonl')}`",
        "",
        "## Failures",
        "",
    ]
    if isinstance(failures, list) and failures:
        lines.extend(f"- `{failure}`" for failure in failures)
    else:
        lines.append("- None")
    lines.extend(["", "## Checks", ""])
    if isinstance(checks, dict):
        for key in sorted(checks):
            lines.append(f"- `{key}`: `{checks[key]}`")
    lines.extend(["", "## Metrics", ""])
    if isinstance(metrics, dict):
        for key in sorted(metrics):
            lines.append(f"- `{key}`: `{metrics[key]}`")
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoOverlayContractConfig", "audit_realtime_demo_overlay_contract"]
