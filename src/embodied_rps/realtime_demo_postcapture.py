"""Post-capture verification for prompt-gated realtime demo overlay videos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-untyped]


@dataclass(frozen=True)
class RealtimeDemoPostCaptureConfig:
    """Configuration for validating a captured realtime demo overlay."""

    overlay_video: Path
    output_root: Path
    response_preview_image: Path | None = None
    live_composite_output_root: Path | None = None
    frame_log_jsonl: Path | None = None
    response_prompt: str | None = "scissors"
    expected_actual_gesture: str | None = None
    expected_response_decision: str | None = None
    expected_robot_action: str | None = None
    enforce_demo_success_gate: bool = False
    max_response_binary_latency_s: float = 0.50
    min_detection_rate: float = 0.80
    prompt_sequence: tuple[str, ...] = ("rock", "paper", "scissors")
    prompt_cycle_s: float = 1.0
    min_frame_count: int = 30
    min_duration_s: float = 3.0


def verify_realtime_demo_capture(config: RealtimeDemoPostCaptureConfig) -> dict[str, object]:
    """Validate a realtime demo overlay and extract prompt-cycle review frames."""

    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "postcapture_summary.json"
    summary_md = output_root / "postcapture_summary.md"
    prompt_frame_dir = output_root / "prompt_frames"
    contact_sheet_path = output_root / "prompt_contact_sheet.png"
    response_decision_frame_path = output_root / "response_decision_frame.png"
    response_prompt_diagnostic_frame_path = output_root / "response_prompt_diagnostic_frame.png"

    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    outputs: dict[str, str] = {
        "summary_json": summary_json.as_posix(),
        "summary_md": summary_md.as_posix(),
        "contact_sheet_png": contact_sheet_path.as_posix(),
        "prompt_frame_dir": prompt_frame_dir.as_posix(),
        "response_decision_frame_png": response_decision_frame_path.as_posix(),
        "response_prompt_diagnostic_frame_png": response_prompt_diagnostic_frame_path.as_posix(),
    }

    overlay_exists = config.overlay_video.exists()
    checks["overlay_video_exists"] = overlay_exists
    if not overlay_exists:
        failures.append("overlay_video_missing")
        summary = _build_summary(
            config,
            status="blocked",
            ok=False,
            failures=failures,
            warnings=warnings,
            checks=checks,
            video={},
            prompt_cycle={"prompt_frame_count": 0, "prompt_frames": []},
            frame_log=_empty_frame_log_summary(config),
            response_decision_frame=None,
            response_prompt_diagnostic_frame=None,
            demo_success_gate=_build_demo_success_gate(config, frame_log=_empty_frame_log_summary(config)),
            outputs=outputs,
            next_command=None,
        )
        _write_summary(summary, summary_json, summary_md)
        return summary

    video = _probe_video(config.overlay_video)
    checks["video_opened"] = bool(video.get("opened"))
    checks["frame_count_ok"] = int(video.get("frame_count", 0)) >= int(config.min_frame_count)
    checks["duration_ok"] = float(video.get("duration_s", 0.0)) >= float(config.min_duration_s)
    checks["resolution_ok"] = int(video.get("width", 0)) > 0 and int(video.get("height", 0)) > 0
    if not checks["video_opened"]:
        failures.append("overlay_video_unreadable")
    if not checks["frame_count_ok"]:
        failures.append("frame_count_too_short")
    if not checks["duration_ok"]:
        failures.append("duration_too_short")
    if not checks["resolution_ok"]:
        failures.append("resolution_invalid")

    prompt_cycle: dict[str, object] = {"prompt_frame_count": 0, "prompt_frames": []}
    if checks["video_opened"]:
        prompt_cycle = _extract_prompt_frames(
            video_path=config.overlay_video,
            output_dir=prompt_frame_dir,
            contact_sheet_path=contact_sheet_path,
            prompt_sequence=config.prompt_sequence,
            prompt_cycle_s=float(config.prompt_cycle_s),
            fps=float(video["fps"]),
            frame_count=int(video["frame_count"]),
        )
        checks["prompt_frames_extracted"] = int(prompt_cycle["prompt_frame_count"]) == len(config.prompt_sequence)
        if not checks["prompt_frames_extracted"]:
            failures.append("prompt_frames_incomplete")
    else:
        checks["prompt_frames_extracted"] = False

    if config.response_preview_image is not None:
        response_preview_exists = config.response_preview_image.exists()
        checks["response_preview_image_exists"] = response_preview_exists
        if not response_preview_exists:
            failures.append("response_preview_image_missing")
    else:
        checks["response_preview_image_exists"] = False
        warnings.append("response_preview_image_not_configured")

    frame_log = _summarize_frame_log(config, video=video)
    for key, value in frame_log.get("checks", {}).items():
        checks[str(key)] = bool(value)
    failures.extend(str(failure) for failure in frame_log.get("failures", []))
    warnings.extend(str(warning) for warning in frame_log.get("warnings", []))
    response_decision_frame = _extract_response_decision_frame(
        video_path=config.overlay_video,
        frame_log=frame_log,
        output_path=response_decision_frame_path,
    )
    response_prompt_diagnostic_frame = _extract_response_prompt_diagnostic_frame(
        video_path=config.overlay_video,
        frame_log=frame_log,
        output_path=response_prompt_diagnostic_frame_path,
    )
    demo_success_gate = _build_demo_success_gate(config, frame_log=frame_log)
    checks["demo_success_gate_passed"] = bool(demo_success_gate.get("passed"))
    if config.enforce_demo_success_gate and not demo_success_gate.get("passed"):
        failures.append("demo_success_gate_failed")

    next_command = _build_next_command(config) if not failures and config.response_preview_image is not None else None
    status = "ready_for_composite" if not failures and config.response_preview_image is not None else ("passed" if not failures else "blocked")
    summary = _build_summary(
        config,
        status=status,
        ok=not failures,
        failures=failures,
        warnings=warnings,
        checks=checks,
        video=video,
        prompt_cycle=prompt_cycle,
        frame_log=frame_log,
        response_decision_frame=response_decision_frame,
        response_prompt_diagnostic_frame=response_prompt_diagnostic_frame,
        demo_success_gate=demo_success_gate,
        outputs=outputs,
        next_command=next_command,
    )
    _write_summary(summary, summary_json, summary_md)
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
        duration_s = float(frame_count) / fps if fps > 0.0 else 0.0
        return {
            "opened": True,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_s": duration_s,
        }
    finally:
        capture.release()


def _extract_prompt_frames(
    *,
    video_path: Path,
    output_dir: Path,
    contact_sheet_path: Path,
    prompt_sequence: tuple[str, ...],
    prompt_cycle_s: float,
    fps: float,
    frame_count: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    labeled_images: list[tuple[str, Image.Image]] = []
    for prompt_index, prompt in enumerate(prompt_sequence):
        timestamp_s = (prompt_index + 0.5) * prompt_cycle_s
        frame_index = min(max(0, int(round(timestamp_s * fps))), max(0, frame_count - 1))
        image = _read_frame(video_path, frame_index=frame_index)
        if image is None:
            continue
        frame_path = output_dir / f"prompt_{prompt_index:02d}_{prompt}.png"
        image.save(frame_path)
        records.append(
            {
                "prompt": prompt,
                "timestamp_s": timestamp_s,
                "frame_index": frame_index,
                "path": frame_path.as_posix(),
            }
        )
        labeled_images.append((f"{prompt} @ {timestamp_s:.2f}s", image))
    if labeled_images:
        _write_contact_sheet(contact_sheet_path, labeled_images)
    return {
        "prompt_sequence": list(prompt_sequence),
        "prompt_cycle_s": prompt_cycle_s,
        "prompt_frame_count": len(records),
        "prompt_frames": records,
    }


def _read_frame(path: Path, *, frame_index: int) -> Image.Image | None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            return None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    finally:
        capture.release()


def _write_contact_sheet(path: Path, labeled_images: list[tuple[str, Image.Image]]) -> None:
    font = ImageFont.load_default()
    thumbnail_w = 360
    thumbnail_h = 220
    label_h = 28
    margin = 16
    width = margin * 2 + len(labeled_images) * thumbnail_w
    height = margin * 2 + thumbnail_h + label_h
    canvas = Image.new("RGB", (width, height), (243, 245, 248))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(labeled_images):
        left = margin + index * thumbnail_w
        top = margin
        tile = Image.new("RGB", (thumbnail_w - 12, thumbnail_h), (226, 232, 240))
        copied = image.copy()
        copied.thumbnail((thumbnail_w - 24, thumbnail_h - label_h - 8), _resampling_lanczos())
        tile.paste(copied, ((tile.width - copied.width) // 2, label_h + (tile.height - label_h - copied.height) // 2))
        canvas.paste(tile, (left, top))
        draw.text((left + 8, top + 8), label, fill=(15, 23, 42), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _build_next_command(config: RealtimeDemoPostCaptureConfig) -> str:
    output_root = config.live_composite_output_root or (config.output_root / "composite")
    return (
        "python -m embodied_rps.tools.create_realtime_schunk_demo_composite "
        f"--overlay-video {config.overlay_video.as_posix()} "
        f"--response-preview-image {config.response_preview_image.as_posix() if config.response_preview_image else ''} "
        f"--output-root {output_root.as_posix()}"
    ).strip()


def _build_summary(
    config: RealtimeDemoPostCaptureConfig,
    *,
    status: str,
    ok: bool,
    failures: list[str],
    warnings: list[str],
    checks: dict[str, bool],
    video: dict[str, object],
    prompt_cycle: dict[str, object],
    frame_log: dict[str, object],
    response_decision_frame: dict[str, object] | None,
    response_prompt_diagnostic_frame: dict[str, object] | None,
    demo_success_gate: dict[str, object],
    outputs: dict[str, str],
    next_command: str | None,
) -> dict[str, object]:
    return {
        "status": status,
        "ok": ok,
        "overlay_video": config.overlay_video.as_posix(),
        "output_root": config.output_root.as_posix(),
        "response_preview_image": config.response_preview_image.as_posix() if config.response_preview_image else None,
        "live_composite_output_root": (
            config.live_composite_output_root.as_posix() if config.live_composite_output_root else None
        ),
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
        "video": video,
        "prompt_cycle": prompt_cycle,
        "frame_log": frame_log,
        "response_decision_frame": response_decision_frame,
        "response_prompt_diagnostic_frame": response_prompt_diagnostic_frame,
        "demo_success_gate": demo_success_gate,
        "outputs": outputs,
        "next_command": next_command,
        "claim_scope": "post-capture stream and prompt-frame verification; not model accuracy validation",
    }


def _empty_frame_log_summary(config: RealtimeDemoPostCaptureConfig) -> dict[str, object]:
    return {
        "configured": config.frame_log_jsonl is not None,
        "path": config.frame_log_jsonl.as_posix() if config.frame_log_jsonl is not None else None,
        "record_count": 0,
        "frame_count_matches_video": False,
        "detection_rate": None,
        "prompt_counts": {},
        "response_prompt_start_time_s": None,
        "first_response_prompt_confirmed_decision": None,
        "first_response_prompt_binary_decision": None,
        "first_response_prompt_ground_truth_decision": None,
        "last_response_prompt_frame": None,
        "checks": {
            "frame_log_exists": False,
            "frame_log_ready": False,
        },
        "failures": [],
        "warnings": ["frame_log_not_configured"] if config.frame_log_jsonl is None else [],
    }


def _extract_response_decision_frame(
    *,
    video_path: Path,
    frame_log: dict[str, object],
    output_path: Path,
) -> dict[str, object] | None:
    first_decision = frame_log.get("first_response_prompt_ground_truth_decision")
    if not isinstance(first_decision, dict):
        first_decision = frame_log.get("first_response_prompt_binary_decision")
    if not isinstance(first_decision, dict):
        return None
    source_frame_index = first_decision.get("frame_index")
    frame_index = _zero_based_frame_index(source_frame_index)
    if frame_index is None:
        return None
    image = _read_frame(video_path, frame_index=frame_index)
    if image is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "path": output_path.as_posix(),
        "source_frame_index": source_frame_index,
        "zero_based_frame_index": frame_index,
        "time_s": first_decision.get("time_s"),
        "decision_state": first_decision.get("decision_state"),
        "robot_action": first_decision.get("robot_action"),
        "confidence": first_decision.get("confidence"),
        "margin": first_decision.get("margin"),
    }


def _extract_response_prompt_diagnostic_frame(
    *,
    video_path: Path,
    frame_log: dict[str, object],
    output_path: Path,
) -> dict[str, object] | None:
    if isinstance(frame_log.get("first_response_prompt_binary_decision"), dict):
        return None
    last_response_frame = frame_log.get("last_response_prompt_frame")
    if not isinstance(last_response_frame, dict):
        return None
    source_frame_index = last_response_frame.get("frame_index")
    frame_index = _zero_based_frame_index(source_frame_index)
    if frame_index is None:
        return None
    image = _read_frame(video_path, frame_index=frame_index)
    if image is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "path": output_path.as_posix(),
        "reason": "response_prompt_binary_decision_missing",
        "source_frame_index": source_frame_index,
        "zero_based_frame_index": frame_index,
        "time_s": last_response_frame.get("time_s"),
        "decision_state": last_response_frame.get("decision_state"),
        "robot_action": last_response_frame.get("robot_action"),
        "confidence": last_response_frame.get("confidence"),
        "margin": last_response_frame.get("margin"),
    }


def _zero_based_frame_index(value: object) -> int | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return max(0, int(value) - 1)


def _summarize_frame_log(
    config: RealtimeDemoPostCaptureConfig,
    *,
    video: dict[str, object],
) -> dict[str, object]:
    if config.frame_log_jsonl is None:
        return _empty_frame_log_summary(config)
    checks = {
        "frame_log_exists": config.frame_log_jsonl.exists(),
        "frame_log_ready": False,
    }
    failures: list[str] = []
    warnings: list[str] = []
    if not checks["frame_log_exists"]:
        failures.append("frame_log_missing")
        return {
            "configured": True,
            "path": config.frame_log_jsonl.as_posix(),
            "record_count": 0,
            "frame_count_matches_video": False,
            "detection_rate": None,
            "prompt_counts": {},
            "response_prompt_start_time_s": None,
            "first_response_prompt_confirmed_decision": None,
            "first_response_prompt_binary_decision": None,
            "first_response_prompt_ground_truth_decision": None,
            "last_response_prompt_frame": None,
            "checks": checks,
            "failures": failures,
            "warnings": warnings,
        }

    rows = _read_frame_log_rows(config.frame_log_jsonl)
    record_count = len(rows)
    video_frame_count = int(video.get("frame_count", 0)) if video.get("opened") else 0
    frame_count_matches_video = record_count == video_frame_count
    detected_count = sum(1 for row in rows if row.get("detected") is True)
    prompt_counts: dict[str, int] = {}
    for row in rows:
        prompt = row.get("active_prompt")
        if isinstance(prompt, str) and prompt:
            prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
    first_response_decision = _first_response_prompt_confirmed_decision(
        rows,
        response_prompt=config.response_prompt,
        binary_only=False,
    )
    first_response_binary_decision = _first_response_prompt_confirmed_decision(
        rows,
        response_prompt=config.response_prompt,
        binary_only=True,
    )
    first_response_ground_truth_decision = _first_response_prompt_ground_truth_decision(
        first_confirmed=first_response_decision,
        first_binary=first_response_binary_decision,
        expected_actual_gesture=config.expected_actual_gesture,
    )
    response_prompt_start_time_s = _response_prompt_start_time_s(
        rows,
        response_prompt=config.response_prompt,
    )
    last_response_prompt_frame = _last_response_prompt_frame(
        rows,
        response_prompt=config.response_prompt,
    )

    if record_count == 0:
        failures.append("frame_log_empty")
    if not frame_count_matches_video:
        failures.append("frame_log_frame_count_mismatch")
    if first_response_decision is None and config.response_prompt is not None:
        warnings.append("response_prompt_confirmed_decision_missing")
    checks["frame_log_ready"] = bool(record_count > 0 and frame_count_matches_video)
    return {
        "configured": True,
        "path": config.frame_log_jsonl.as_posix(),
        "record_count": record_count,
        "frame_count_matches_video": frame_count_matches_video,
        "detection_rate": (detected_count / record_count) if record_count else None,
        "prompt_counts": prompt_counts,
        "response_prompt_start_time_s": response_prompt_start_time_s,
        "first_response_prompt_confirmed_decision": first_response_decision,
        "first_response_prompt_binary_decision": first_response_binary_decision,
        "first_response_prompt_ground_truth_decision": first_response_ground_truth_decision,
        "last_response_prompt_frame": last_response_prompt_frame,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
    }


def _read_frame_log_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        loaded = json.loads(stripped)
        if not isinstance(loaded, dict):
            raise ValueError("frame log JSONL rows must be objects")
        rows.append(dict(loaded))
    return rows


def _first_response_prompt_confirmed_decision(
    rows: list[dict[str, object]],
    *,
    response_prompt: str | None,
    binary_only: bool,
) -> dict[str, object] | None:
    if response_prompt is None:
        return None
    for row in rows:
        if row.get("active_prompt") == response_prompt and row.get("confirmed_decision") is True:
            if binary_only and row.get("decision_state") not in {"paper", "scissors"}:
                continue
            return {
                "frame_index": row.get("frame_index"),
                "time_s": row.get("time_s"),
                "decision_state": row.get("decision_state"),
                "robot_action": row.get("robot_action"),
                "confidence": row.get("confidence"),
                "margin": row.get("margin"),
                "p_rock": row.get("p_rock"),
                "p_paper": row.get("p_paper"),
                "p_scissors": row.get("p_scissors"),
            }
    return None


def _first_response_prompt_ground_truth_decision(
    *,
    first_confirmed: dict[str, object] | None,
    first_binary: dict[str, object] | None,
    expected_actual_gesture: str | None,
) -> dict[str, object] | None:
    expected = _normalize_expected_actual_gesture(expected_actual_gesture)
    if expected is None:
        return first_binary
    if expected == "rock":
        return first_binary if first_binary is not None else first_confirmed
    return first_binary


def _response_prompt_start_time_s(
    rows: list[dict[str, object]],
    *,
    response_prompt: str | None,
) -> float | None:
    if response_prompt is None:
        return None
    for row in rows:
        if row.get("active_prompt") == response_prompt:
            value = row.get("time_s")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def _last_response_prompt_frame(
    rows: list[dict[str, object]],
    *,
    response_prompt: str | None,
) -> dict[str, object] | None:
    if response_prompt is None:
        return None
    for row in reversed(rows):
        if row.get("active_prompt") != response_prompt:
            continue
        return {
            "frame_index": row.get("frame_index"),
            "time_s": row.get("time_s"),
            "detected": row.get("detected"),
            "decision_state": row.get("decision_state"),
            "robot_action": row.get("robot_action"),
            "confidence": row.get("confidence"),
            "margin": row.get("margin"),
            "p_rock": row.get("p_rock"),
            "p_paper": row.get("p_paper"),
            "p_scissors": row.get("p_scissors"),
            "confirmed_decision": row.get("confirmed_decision"),
        }
    return None


def _build_demo_success_gate(
    config: RealtimeDemoPostCaptureConfig,
    *,
    frame_log: dict[str, object],
) -> dict[str, object]:
    failure_reasons: list[str] = []
    expected_actual = _normalize_expected_actual_gesture(config.expected_actual_gesture)
    detection_rate = _optional_float_value(frame_log.get("detection_rate"))
    first_binary = frame_log.get("first_response_prompt_binary_decision")
    first_binary_record = first_binary if isinstance(first_binary, dict) else None
    first_ground_truth = frame_log.get("first_response_prompt_ground_truth_decision")
    first_ground_truth_record = first_ground_truth if isinstance(first_ground_truth, dict) else None
    timing_record = first_ground_truth_record if expected_actual is not None else first_binary_record
    response_prompt_start_time_s = _optional_float_value(frame_log.get("response_prompt_start_time_s"))
    binary_time_s = (
        _optional_float_value(timing_record.get("time_s"))
        if timing_record is not None
        else None
    )
    latency_s = (
        binary_time_s - response_prompt_start_time_s
        if binary_time_s is not None and response_prompt_start_time_s is not None
        else None
    )

    if detection_rate is None:
        failure_reasons.append("detection_rate_missing")
    elif detection_rate < config.min_detection_rate:
        failure_reasons.append("detection_rate_below_minimum")
    if expected_actual is None and first_binary_record is None:
        failure_reasons.append("response_prompt_binary_decision_missing")
    if expected_actual is not None and first_ground_truth_record is None:
        failure_reasons.append("response_prompt_ground_truth_decision_missing")
    if latency_s is None:
        failure_reasons.append(
            "response_prompt_ground_truth_latency_missing"
            if expected_actual is not None
            else "response_prompt_binary_latency_missing"
        )
    elif latency_s > config.max_response_binary_latency_s:
        failure_reasons.append(
            "response_prompt_ground_truth_decision_late"
            if expected_actual is not None
            else "response_prompt_binary_decision_late"
        )

    expected_decisions = _expected_decisions_for_actual_gesture(expected_actual)
    expected_robot_action = _expected_robot_action_for_actual_gesture(expected_actual) or config.expected_robot_action
    actual_decision = first_ground_truth_record.get("decision_state") if first_ground_truth_record is not None else None
    actual_robot_action = first_ground_truth_record.get("robot_action") if first_ground_truth_record is not None else None
    ground_truth_match = (
        actual_decision in expected_decisions
        if expected_decisions is not None and first_ground_truth_record is not None
        else None
    )
    robot_action_match = (
        actual_robot_action == expected_robot_action
        if expected_robot_action is not None and first_ground_truth_record is not None
        else None
    )
    if ground_truth_match is False:
        failure_reasons.append("unexpected_actual_gesture_decision")
    if robot_action_match is False:
        failure_reasons.append("unexpected_actual_gesture_robot_action")

    if (
        config.expected_response_decision is not None
        and first_binary_record is not None
        and first_binary_record.get("decision_state") != config.expected_response_decision
    ):
        failure_reasons.append("unexpected_response_decision")
    if (
        config.expected_robot_action is not None
        and first_binary_record is not None
        and first_binary_record.get("robot_action") != config.expected_robot_action
    ):
        failure_reasons.append("unexpected_robot_action")

    return {
        "configured": True,
        "enforced": bool(config.enforce_demo_success_gate),
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "response_prompt": config.response_prompt,
        "expected_actual_gesture": expected_actual,
        "expected_response_decision": (
            "wait_counter_paper"
            if expected_actual == "rock"
            else (next(iter(expected_decisions)) if expected_decisions and len(expected_decisions) == 1 else config.expected_response_decision)
        ),
        "expected_response_decisions": sorted(expected_decisions) if expected_decisions is not None else None,
        "expected_robot_action": expected_robot_action,
        "ground_truth_match": ground_truth_match,
        "robot_action_match": robot_action_match,
        "min_detection_rate": float(config.min_detection_rate),
        "max_response_binary_latency_s": float(config.max_response_binary_latency_s),
        "detection_rate": detection_rate,
        "response_prompt_start_time_s": response_prompt_start_time_s,
        "binary_decision_time_s": binary_time_s,
        "binary_decision_latency_s": latency_s,
        "first_response_prompt_binary_decision": first_binary_record,
        "first_response_prompt_ground_truth_decision": first_ground_truth_record,
    }


def _normalize_expected_actual_gesture(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip().lower()
    if parsed == "":
        return None
    if parsed not in {"rock", "paper", "scissors"}:
        raise ValueError(f"Unsupported expected actual gesture: {value}")
    return parsed


def _expected_decisions_for_actual_gesture(value: str | None) -> set[str] | None:
    if value is None:
        return None
    if value == "rock":
        return {"rock", "wait_counter_paper"}
    return {value}


def _expected_robot_action_for_actual_gesture(value: str | None) -> str | None:
    if value == "rock":
        return "paper"
    if value == "paper":
        return "scissors"
    if value == "scissors":
        return "rock"
    return None


def _optional_float_value(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _write_summary(summary: dict[str, object], json_path: Path, markdown_path: Path) -> None:
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path.write_text(_summary_markdown(summary), encoding="utf-8")


def _summary_markdown(summary: dict[str, object]) -> str:
    video = summary.get("video", {})
    prompt_cycle = summary.get("prompt_cycle", {})
    frame_log = summary.get("frame_log", {})
    response_decision_frame = summary.get("response_decision_frame", {})
    response_prompt_diagnostic_frame = summary.get("response_prompt_diagnostic_frame", {})
    lines = [
        "# Realtime Demo Post-Capture Verification",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Overlay video: `{summary.get('overlay_video')}`",
        f"- Frame count: `{video.get('frame_count') if isinstance(video, dict) else None}`",
        f"- Resolution: `{video.get('width') if isinstance(video, dict) else None}x{video.get('height') if isinstance(video, dict) else None}`",
        f"- Duration seconds: `{video.get('duration_s') if isinstance(video, dict) else None}`",
        f"- Prompt frames: `{prompt_cycle.get('prompt_frame_count') if isinstance(prompt_cycle, dict) else None}`",
        f"- Frame-log records: `{frame_log.get('record_count') if isinstance(frame_log, dict) else None}`",
        f"- Frame-log detection rate: `{frame_log.get('detection_rate') if isinstance(frame_log, dict) else None}`",
        f"- Response decision frame: `{response_decision_frame.get('path') if isinstance(response_decision_frame, dict) else None}`",
        f"- Response prompt diagnostic frame: `{response_prompt_diagnostic_frame.get('path') if isinstance(response_prompt_diagnostic_frame, dict) else None}`",
        "",
        "## Failures",
        "",
    ]
    failures = summary.get("failures", [])
    if isinstance(failures, list) and failures:
        lines.extend(f"- `{failure}`" for failure in failures)
    else:
        lines.append("- None")
    next_command = summary.get("next_command")
    if next_command:
        lines.extend(["", "## Next Command", "", f"```text\n{next_command}\n```"])
    lines.append("")
    return "\n".join(lines)


def _resampling_lanczos() -> int:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return int(resampling.LANCZOS)
    return int(Image.LANCZOS)


__all__ = ["RealtimeDemoPostCaptureConfig", "verify_realtime_demo_capture"]
