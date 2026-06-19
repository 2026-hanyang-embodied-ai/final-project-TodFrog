"""Opt-in webcam recorder for non-held-out v4 staging MP4s."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER

DEFAULT_V4_RECORDING_CAPTURE_ROOT = Path("artifacts/real_skeleton_v4_recording_capture_20260612")

RecorderCallable = Callable[[Path, "V4StagingRecorderConfig"], Mapping[str, object]]


@dataclass(frozen=True)
class V4StagingRecorderConfig:
    """Configuration for recording one non-held-out MP4 into v4 staging."""

    staging_root: Path
    label: str
    output_root: Path = DEFAULT_V4_RECORDING_CAPTURE_ROOT
    camera_index: int = 0
    pre_roll_s: float = 0.0
    duration_s: float = 3.0
    fps: float = 30.0
    width: int | None = None
    height: int | None = None
    filename: str | None = None
    prefix: str = "v4"
    codec: str = "mp4v"
    dry_run: bool = False


def validate_staging_recording_label(label: str) -> str:
    """Return a normalized label or raise for unsupported labels."""

    normalized = label.strip().lower()
    if normalized not in REVIEW_LABEL_ORDER:
        raise ValueError(f"Unsupported label {label!r}; expected one of {', '.join(REVIEW_LABEL_ORDER)}")
    return normalized


def next_staging_recording_filename(label_dir: Path, *, label: str, prefix: str = "v4") -> str:
    """Return the next deterministic staging filename for a label folder."""

    normalized_label = validate_staging_recording_label(label)
    if not prefix:
        raise ValueError("prefix must be non-empty")
    pattern = re.compile(rf"^{re.escape(prefix)}_{re.escape(normalized_label)}_(\d{{6}})\.mp4$", re.IGNORECASE)
    max_index = 0
    if label_dir.exists():
        for path in label_dir.glob("*.mp4"):
            match = pattern.match(path.name)
            if match is not None:
                max_index = max(max_index, int(match.group(1)))
    return f"{prefix}_{normalized_label}_{max_index + 1:06d}.mp4"


def planned_staging_recording_path(config: V4StagingRecorderConfig) -> Path:
    """Return the exact MP4 path that would be recorded for this config."""

    label = validate_staging_recording_label(config.label)
    label_dir = config.staging_root / label
    filename = config.filename or next_staging_recording_filename(label_dir, label=label, prefix=config.prefix)
    if Path(filename).name != filename:
        raise ValueError(f"filename must be a basename, got {filename!r}")
    if not filename.lower().endswith(".mp4"):
        filename = f"{filename}.mp4"
    output_path = label_dir / filename
    _assert_within(output_path, label_dir)
    return output_path


def record_v4_staging_clip(
    config: V4StagingRecorderConfig,
    *,
    recorder: RecorderCallable | None = None,
) -> dict[str, object]:
    """Record or plan one MP4 in the v4 staging folder."""

    _validate_config(config)
    label = validate_staging_recording_label(config.label)
    output_path = planned_staging_recording_path(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if config.dry_run:
        summary = _summary_payload(
            config=config,
            status="planned",
            label=label,
            output_path=output_path,
            recorder_metadata={"dry_run": True},
        )
        _write_summary(config.output_root, summary)
        return summary

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing staging MP4: {output_path}")
    active_recorder = recorder or _record_with_opencv
    recorder_metadata = dict(active_recorder(output_path, config))
    if not output_path.exists():
        raise RuntimeError(f"Recorder did not create output MP4: {output_path}")
    if output_path.stat().st_size <= 0:
        raise RuntimeError(f"Recorder created an empty MP4: {output_path}")

    summary = _summary_payload(
        config=config,
        status="recorded",
        label=label,
        output_path=output_path,
        recorder_metadata=recorder_metadata,
    )
    metadata_path = config.output_root / "recordings" / label / f"{output_path.stem}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["metadata_path"] = metadata_path.as_posix()
    _write_summary(config.output_root, summary)
    return summary


def _record_with_opencv(output_path: Path, config: V4StagingRecorderConfig) -> dict[str, object]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("opencv-python is required for live v4 staging recording") from exc

    capture = cv2.VideoCapture(int(config.camera_index))
    if config.width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(config.width))
    if config.height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.height))
    if config.fps > 0:
        capture.set(cv2.CAP_PROP_FPS, float(config.fps))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera index {config.camera_index}")
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise RuntimeError(f"Camera index {config.camera_index} did not return a frame")
        pre_roll_frames = 0
        if config.pre_roll_s > 0:
            pre_roll_until = time.monotonic() + float(config.pre_roll_s)
            while time.monotonic() < pre_roll_until:
                ok, next_frame = capture.read()
                if not ok or next_frame is None:
                    break
                first_frame = next_frame
                pre_roll_frames += 1
        frame_height, frame_width = first_frame.shape[:2]
        writer_fps = _safe_positive_float(config.fps, default=30.0)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*config.codec),
            writer_fps,
            (int(frame_width), int(frame_height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open MP4 writer for {output_path}")
        try:
            target_frames = max(1, int(round(config.duration_s * writer_fps)))
            started_at = time.monotonic()
            written_frames = 0
            frame = first_frame
            while written_frames < target_frames:
                writer.write(frame)
                written_frames += 1
                if written_frames >= target_frames:
                    break
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
            elapsed_s = time.monotonic() - started_at
        finally:
            writer.release()
    finally:
        capture.release()

    if written_frames <= 0:
        raise RuntimeError("No frames were recorded")
    return {
        "camera_index": int(config.camera_index),
        "frame_count": int(written_frames),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "fps": writer_fps,
        "duration_s": round(float(elapsed_s), 6),
        "requested_duration_s": float(config.duration_s),
        "requested_pre_roll_s": float(config.pre_roll_s),
        "pre_roll_frames_discarded": int(pre_roll_frames),
        "codec": config.codec,
    }


def _summary_payload(
    *,
    config: V4StagingRecorderConfig,
    status: str,
    label: str,
    output_path: Path,
    recorder_metadata: Mapping[str, object],
) -> dict[str, object]:
    return {
        "status": status,
        "label": label,
        "staging_root": config.staging_root.as_posix(),
        "output_path": output_path.as_posix(),
        "output_root": config.output_root.as_posix(),
        "camera_index": int(config.camera_index),
        "requested_pre_roll_s": float(config.pre_roll_s),
        "requested_duration_s": float(config.duration_s),
        "requested_fps": float(config.fps),
        "requested_width": config.width,
        "requested_height": config.height,
        "filename": output_path.name,
        "recorder_metadata": dict(recorder_metadata),
        "next_action": "run_monitor_v4_recording_ingest",
        "next_command": (
            "python -m embodied_rps.tools.monitor_v4_recording_ingest "
            "--source-root <staging_root> "
            "--calibration-root <calibration_root> "
            "--heldout-root <heldout_test_root> "
            "--iterations 1 --poll-interval-s 0"
        ),
    }


def _write_summary(output_root: Path, summary: Mapping[str, object]) -> None:
    (output_root / "recording_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _validate_config(config: V4StagingRecorderConfig) -> None:
    validate_staging_recording_label(config.label)
    if config.pre_roll_s < 0 or not math.isfinite(config.pre_roll_s):
        raise ValueError("pre_roll_s must be a finite non-negative value")
    if config.duration_s <= 0 or not math.isfinite(config.duration_s):
        raise ValueError("duration_s must be a finite positive value")
    if config.fps <= 0 or not math.isfinite(config.fps):
        raise ValueError("fps must be a finite positive value")
    if config.width is not None and config.width <= 0:
        raise ValueError("width must be positive when provided")
    if config.height is not None and config.height <= 0:
        raise ValueError("height must be positive when provided")
    if len(config.codec) != 4:
        raise ValueError("codec must be a four-character code")


def _safe_positive_float(value: float, *, default: float) -> float:
    return float(value) if math.isfinite(value) and value > 0 else default


def _assert_within(path: Path, parent: Path) -> None:
    resolved_path = path.expanduser().resolve(strict=False)
    resolved_parent = parent.expanduser().resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_parent)
    except ValueError as exc:
        raise ValueError(f"planned output escapes label folder: {path}") from exc


__all__ = [
    "DEFAULT_V4_RECORDING_CAPTURE_ROOT",
    "V4StagingRecorderConfig",
    "next_staging_recording_filename",
    "planned_staging_recording_path",
    "record_v4_staging_clip",
    "validate_staging_recording_label",
]
