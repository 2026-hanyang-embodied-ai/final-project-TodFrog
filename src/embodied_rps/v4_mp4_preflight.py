"""MP4 preflight audit for v4 calibration recordings."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.v4_calibration_intake import discover_calibration_videos, validate_calibration_discovery

VideoProbe = Callable[[Path], dict[str, object]]


@dataclass(frozen=True)
class V4Mp4PreflightConfig:
    """Configuration for auditing v4 calibration MP4 files before skeleton review."""

    input_root: Path
    heldout_roots: tuple[Path, ...]
    output_root: Path
    expected_min_per_label: int = 20
    min_frame_count: int = 5
    min_fps: float = 1.0
    min_width: int = 1
    min_height: int = 1


def audit_v4_calibration_mp4s(config: V4Mp4PreflightConfig, *, video_probe: VideoProbe | None = None) -> dict[str, object]:
    """Audit v4 calibration MP4 count, labels, and basic stream metadata."""

    if video_probe is None:
        video_probe = probe_video_with_opencv
    config.output_root.mkdir(parents=True, exist_ok=True)
    videos = discover_calibration_videos(config.input_root, heldout_roots=config.heldout_roots)
    discovery = validate_calibration_discovery(videos, expected_min_per_label=config.expected_min_per_label)
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for video in videos:
        probe = video_probe(video.source_path)
        record = {
            "video_id": video.output_stem,
            "label": video.label,
            "source_path": video.source_path.as_posix(),
            **probe,
        }
        video_failures = _video_failures(record, config)
        record["status"] = "passed" if not video_failures else "failed"
        record["failure_codes"] = [str(item["code"]) for item in video_failures]
        records.append(record)
        failures.extend({"video_id": video.output_stem, **item} for item in video_failures)
    if discovery["status"] != "ready_for_skeleton_review":
        for label, issue in dict(discovery["missing_or_low_labels"]).items():
            failures.append({"code": "missing_or_low_label_count", "label": label, **dict(issue)})
    failed_video_count = sum(1 for record in records if record["status"] != "passed")
    status = "passed" if discovery["status"] == "ready_for_skeleton_review" and failed_video_count == 0 else str(discovery["status"])
    if status == "ready_for_skeleton_review" and failed_video_count > 0:
        status = "failed_video_metadata"
    summary = {
        "status": status,
        "input_root": config.input_root.as_posix(),
        "output_root": config.output_root.as_posix(),
        "video_count": len(videos),
        "expected_min_per_label": config.expected_min_per_label,
        "label_counts": dict(discovery["label_counts"]),
        "missing_or_low_labels": dict(discovery["missing_or_low_labels"]),
        "duplicate_count": int(discovery["duplicate_count"]),
        "failed_video_count": failed_video_count,
        "failures": failures,
        "records": records,
        "preflight_table": (config.output_root / "mp4_preflight_table.csv").as_posix(),
    }
    _write_table(config.output_root / "mp4_preflight_table.csv", records)
    _write_json(config.output_root / "mp4_preflight_summary.json", summary)
    _write_markdown(config.output_root / "mp4_preflight_summary.md", summary)
    return summary


def probe_video_with_opencv(path: Path) -> dict[str, object]:
    """Probe MP4 stream metadata with OpenCV."""

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("MP4 preflight requires opencv-python or a supplied video_probe.") from exc
    from embodied_rps.real_skeleton_review import _open_capture_with_optional_ascii_copy, safe_fps

    capture, temp_dir = _open_capture_with_optional_ascii_copy(cv2, path, path.stem)
    try:
        opened = bool(capture.isOpened())
        if not opened:
            return {
                "path": path.as_posix(),
                "opened": False,
                "width": 0,
                "height": 0,
                "frame_count": 0,
                "fps": 0.0,
                "duration_s": None,
                "failure_reason": "opencv_could_not_open",
            }
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = safe_fps(cv2, capture)
        duration_s = frame_count / fps if fps > 0 else None
        return {
            "path": path.as_posix(),
            "opened": True,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "fps": fps,
            "duration_s": duration_s,
            "failure_reason": None,
        }
    finally:
        capture.release()
        if temp_dir is not None:
            temp_dir.cleanup()


def _video_failures(record: dict[str, object], config: V4Mp4PreflightConfig) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    if not bool(record.get("opened")):
        failures.append({"code": "video_not_opened", "reason": record.get("failure_reason")})
    width = _int(record.get("width"))
    height = _int(record.get("height"))
    frame_count = _int(record.get("frame_count"))
    fps = _float(record.get("fps"))
    if width < config.min_width:
        failures.append({"code": "width_too_small", "actual": width, "minimum": config.min_width})
    if height < config.min_height:
        failures.append({"code": "height_too_small", "actual": height, "minimum": config.min_height})
    if frame_count < config.min_frame_count:
        failures.append({"code": "frame_count_too_small", "actual": frame_count, "minimum": config.min_frame_count})
    if not math.isfinite(fps) or fps < config.min_fps:
        failures.append({"code": "fps_too_small", "actual": fps, "minimum": config.min_fps})
    return failures


def _write_table(path: Path, records: list[dict[str, object]]) -> None:
    fieldnames = [
        "video_id",
        "label",
        "status",
        "source_path",
        "opened",
        "width",
        "height",
        "frame_count",
        "fps",
        "duration_s",
        "failure_codes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fieldnames}
            if isinstance(row["failure_codes"], list):
                row["failure_codes"] = ";".join(str(item) for item in row["failure_codes"])
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# V4 MP4 Preflight Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Input root: `{summary['input_root']}`",
        f"- Video count: `{summary['video_count']}`",
        f"- Label counts: `{summary['label_counts']}`",
        f"- Failed video count: `{summary['failed_video_count']}`",
        "",
        "## Blocking Issues",
        "",
    ]
    failures = summary.get("failures")
    if isinstance(failures, list) and failures:
        for failure in failures:
            lines.append(f"- `{json.dumps(failure, ensure_ascii=False)}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Next Step", "", "Run MediaPipe skeleton review only after this preflight passes.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["V4Mp4PreflightConfig", "audit_v4_calibration_mp4s", "probe_video_with_opencv"]
