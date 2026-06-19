"""Utilities for long scissors-pose collection sessions."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def summarize_scissors_pose_collection(
    *,
    frame_log_jsonl: Path,
    skeleton_npz: Path,
    output_root: Path,
    overlay_video: Path | None = None,
    collection_label: str = "scissors",
) -> dict[str, object]:
    """Summarize a long prompt-pose collection run and write crop-review helpers."""

    output_root.mkdir(parents=True, exist_ok=True)
    label = _normalized_collection_label(collection_label)
    records = _read_jsonl(frame_log_jsonl)
    skeleton_summary = _summarize_skeleton_npz(skeleton_npz)
    frame_count = len(records)
    detected_frame_count = sum(1 for record in records if record.get("detected") is True)
    decision_counts = Counter(str(record.get("decision_state")) for record in records if record.get("decision_state"))
    prompt_counts = Counter(str(record.get("active_prompt")) for record in records if record.get("active_prompt"))
    mismatch_reason: str | None = None
    if skeleton_summary["frame_count"] != frame_count:
        mismatch_reason = "frame_log_and_skeleton_npz_frame_count_mismatch"
    contact_sheet = _write_contact_sheet_if_possible(
        overlay_video=overlay_video,
        output_path=output_root / "review_contact_sheet.png",
    )

    quality_summary_json = output_root / "quality_summary.json"
    quality_summary_csv = output_root / "quality_summary.csv"
    segment_template = output_root / "segment_ranges_template.csv"
    summary: dict[str, object] = {
        "status": "passed" if frame_count > 0 and mismatch_reason is None else "failed",
        "failure_reason": mismatch_reason,
        "frame_log_jsonl": frame_log_jsonl.as_posix(),
        "skeleton_npz": skeleton_npz.as_posix(),
        "overlay_video": overlay_video.as_posix() if overlay_video is not None else None,
        "collection_label": label,
        "frame_count": frame_count,
        "detected_frame_count": detected_frame_count,
        "detection_rate": float(detected_frame_count / frame_count) if frame_count else 0.0,
        "decision_counts": dict(sorted(decision_counts.items())),
        "prompt_counts": dict(sorted(prompt_counts.items())),
        "skeleton_summary": skeleton_summary,
        "outputs": {
            "quality_summary_json": quality_summary_json.as_posix(),
            "quality_summary_csv": quality_summary_csv.as_posix(),
            "segment_ranges_template_csv": segment_template.as_posix(),
            "review_contact_sheet_png": contact_sheet.as_posix() if contact_sheet is not None else None,
        },
        "claim_scope": (
            "post-capture summary over existing long prompt pose collection artifacts; "
            "does not train a model or select final demo media"
        ),
    }
    quality_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_quality_csv(quality_summary_csv, summary)
    _write_segment_template(segment_template, collection_label=label)
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing frame log: {path}")
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            records.append(loaded)
    return records


def _summarize_skeleton_npz(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing skeleton NPZ: {path}")
    with np.load(path, allow_pickle=False) as data:
        landmarks = np.asarray(data["canonical_landmarks"], dtype=np.float32)
        detected = np.asarray(data["detected"], dtype=np.bool_)
        metadata_json = str(data["metadata_json"].item()) if "metadata_json" in data else "{}"
    if landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):
        raise ValueError(f"{path} canonical_landmarks must have shape (T,21,3)")
    frame_count = int(landmarks.shape[0])
    detected_count = int(np.count_nonzero(detected))
    finite = bool(np.isfinite(landmarks).all())
    return {
        "frame_count": frame_count,
        "detected_frame_count": detected_count,
        "detection_rate": float(detected_count / frame_count) if frame_count else 0.0,
        "finite": finite,
        "metadata": json.loads(metadata_json),
    }


def _write_quality_csv(path: Path, summary: dict[str, object]) -> None:
    fieldnames = [
        "status",
        "failure_reason",
        "frame_count",
        "detected_frame_count",
        "detection_rate",
        "decision_counts",
        "prompt_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "status": summary.get("status"),
                "failure_reason": summary.get("failure_reason"),
                "frame_count": summary.get("frame_count"),
                "detected_frame_count": summary.get("detected_frame_count"),
                "detection_rate": summary.get("detection_rate"),
                "decision_counts": json.dumps(summary.get("decision_counts", {}), sort_keys=True),
                "prompt_counts": json.dumps(summary.get("prompt_counts", {}), sort_keys=True),
            }
        )


def _write_segment_template(path: Path, *, collection_label: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["segment_id", "start_s", "end_s", "pose_tag", "notes"])
        writer.writerow([f"{collection_label}_pose_001", "", "", f"{collection_label}_prompt_window", "fill after visual review"])


def _normalized_collection_label(value: str) -> str:
    label = value.strip().lower().replace("-", "_")
    return label if label else "scissors"


def _write_contact_sheet_if_possible(*, overlay_video: Path | None, output_path: Path) -> Path | None:
    if overlay_video is None or not overlay_video.exists():
        return None
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return None
    capture = cv2.VideoCapture(str(overlay_video))
    if not capture.isOpened():
        return None
    frames = []
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        capture.release()
        return None
    sample_indices = np.linspace(0, max(0, total_frames - 1), num=min(12, total_frames), dtype=np.int64)
    for index in sample_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    capture.release()
    if not frames:
        return None
    thumb_h = 160
    thumbs = []
    for frame in frames:
        height, width = frame.shape[:2]
        thumb_w = max(1, int(round(width * thumb_h / max(height, 1))))
        thumbs.append(cv2.resize(frame, (thumb_w, thumb_h)))
    rows = []
    for start in range(0, len(thumbs), 4):
        row = thumbs[start : start + 4]
        max_w = max(thumb.shape[1] for thumb in row)
        padded = [
            cv2.copyMakeBorder(thumb, 0, 0, 0, max_w - thumb.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for thumb in row
        ]
        rows.append(np.hstack(padded))
    max_row_w = max(row.shape[1] for row in rows)
    padded_rows = [
        cv2.copyMakeBorder(row, 0, 0, 0, max_row_w - row.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for row in rows
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack(padded_rows))
    return output_path


__all__ = ["summarize_scissors_pose_collection"]
