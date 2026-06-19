"""MediaPipe hand-skeleton review utilities for real MP4 clips."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REVIEW_LABEL_FOLDERS: Mapping[str, str] = {
    "rock": "rock",
    "paper": "paper",
    "scissors": "scissors",
}

REVIEW_LABEL_ORDER: tuple[str, ...] = ("rock", "paper", "scissors")

FINGER_COLORS: Mapping[str, tuple[int, int, int]] = {
    "thumb": (255, 170, 60),
    "index": (80, 220, 120),
    "middle": (90, 190, 255),
    "ring": (220, 120, 255),
    "pinky": (255, 130, 130),
    "palm": (220, 220, 220),
}

HAND_EDGES: tuple[tuple[str, int, int], ...] = (
    ("palm", 0, 1),
    ("thumb", 1, 2),
    ("thumb", 2, 3),
    ("thumb", 3, 4),
    ("palm", 0, 5),
    ("index", 5, 6),
    ("index", 6, 7),
    ("index", 7, 8),
    ("palm", 0, 9),
    ("middle", 9, 10),
    ("middle", 10, 11),
    ("middle", 11, 12),
    ("palm", 0, 13),
    ("ring", 13, 14),
    ("ring", 14, 15),
    ("ring", 15, 16),
    ("palm", 0, 17),
    ("pinky", 17, 18),
    ("pinky", 18, 19),
    ("pinky", 19, 20),
    ("palm", 5, 9),
    ("palm", 9, 13),
    ("palm", 13, 17),
)


@dataclass(frozen=True)
class SkeletonReviewVideo:
    """One MP4 selected for visual MediaPipe skeleton review."""

    source_path: Path
    label: str
    source_folder: str
    output_stem: str
    source_stem: str


def natural_key(value: str | Path) -> list[object]:
    """Return a stable natural-sort key for filenames with embedded numbers."""

    text = value.as_posix() if isinstance(value, Path) else value
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def discover_skeleton_review_videos(input_root: Path, *, output_prefix: str = "test") -> list[SkeletonReviewVideo]:
    """Discover `rock`, `paper`, and `scissors` MP4s for skeleton review."""

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    videos: list[SkeletonReviewVideo] = []
    for label in REVIEW_LABEL_ORDER:
        folder = input_root / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing review label folder: {folder}")
        paths = sorted(folder.glob("*.mp4"), key=natural_key)
        for index, path in enumerate(paths, start=1):
            videos.append(
                SkeletonReviewVideo(
                    source_path=path,
                    label=REVIEW_LABEL_FOLDERS[label],
                    source_folder=label,
                    output_stem=f"{output_prefix}_{label}_{index:06d}",
                    source_stem=path.stem,
                )
            )
    return videos


def validate_skeleton_review_discovery(
    videos: Sequence[SkeletonReviewVideo],
    *,
    expected_count: int = 15,
    expected_per_label: int = 5,
) -> dict[str, object]:
    """Validate count, duplicate paths, and final-label balance."""

    resolved = [video.source_path.resolve() for video in videos]
    duplicate_count = len(resolved) - len(set(resolved))
    label_counts = Counter(video.label for video in videos)
    passed = (
        len(videos) == expected_count
        and duplicate_count == 0
        and all(label_counts.get(label, 0) == expected_per_label for label in REVIEW_LABEL_ORDER)
    )
    return {
        "passed": passed,
        "video_count": len(videos),
        "expected_count": expected_count,
        "expected_per_label": expected_per_label,
        "duplicate_count": duplicate_count,
        "label_counts": dict(sorted(label_counts.items())),
    }


def process_review_video(
    *,
    cv2: Any,
    mp: Any,
    item: SkeletonReviewVideo,
    output_root: Path,
) -> dict[str, object]:
    """Extract MediaPipe landmarks and write review videos/tables for one clip."""

    clip_dir = output_root / "clips" / item.label / item.output_stem
    clip_dir.mkdir(parents=True, exist_ok=True)
    capture, temp_dir = _open_capture_with_optional_ascii_copy(cv2, item.source_path, item.output_stem)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {item.source_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = safe_fps(cv2, capture)
        duration_s = expected_frames / fps if fps > 0 else None

        skeleton_path = output_root / "videos" / "skeleton_only" / item.label / f"{item.output_stem}.mp4"
        side_by_side_path = output_root / "videos" / "side_by_side" / item.label / f"{item.output_stem}.mp4"
        json_path = output_root / "landmarks_json" / item.label / f"{item.output_stem}.json"
        csv_path = output_root / "landmarks_csv" / item.label / f"{item.output_stem}.csv"

        skeleton_writer = mp4_writer(cv2, skeleton_path, fps, (width, height))
        side_by_side_writer = mp4_writer(cv2, side_by_side_path, fps, (width * 2, height))

        frame_records: list[dict[str, object]] = []
        csv_rows: list[dict[str, object]] = []
        detected_frames = 0
        total_score = 0.0
        scored_frames = 0
        multi_hand_frames = 0
        contact_frame: NDArray[np.uint8] | None = None

        hands_module = mp.solutions.hands
        hands = hands_module.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            frame_index = 0
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                timestamp_s = frame_index / fps if fps > 0 else 0.0
                result = hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                primary_index, detections = select_primary_hand(result)
                detection_count = len(detections)
                if detection_count > 1:
                    multi_hand_frames += 1

                landmarks = _primary_landmarks(result, primary_index)
                handedness: str | None = None
                score: float | None = None
                if primary_index is not None and primary_index < len(detections):
                    primary = detections[primary_index]
                    handedness = _optional_string(primary.get("label"))
                    score = _optional_float(primary.get("score"))
                if landmarks is not None:
                    detected_frames += 1
                    if score is not None:
                        total_score += score
                        scored_frames += 1

                skeleton_canvas = np.zeros_like(frame_bgr)
                draw_hand_skeleton(cv2, skeleton_canvas, landmarks)
                status = (
                    f"OK {handedness or 'unknown'} score={score:.3f}"
                    if landmarks is not None and score is not None
                    else "NO HAND DETECTED"
                )
                draw_status(cv2, skeleton_canvas, status, landmarks is not None)
                side_by_side = np.concatenate([frame_bgr, skeleton_canvas], axis=1)
                draw_status(cv2, side_by_side, status, landmarks is not None)

                skeleton_writer.write(skeleton_canvas)
                side_by_side_writer.write(side_by_side)
                if expected_frames <= 0 or frame_index == expected_frames // 2:
                    contact_frame = side_by_side.copy()

                frame_record = {
                    "video_id": item.output_stem,
                    "source_path": item.source_path.as_posix(),
                    "label": item.label,
                    "source_folder": item.source_folder,
                    "source_stem": item.source_stem,
                    "frame_index": frame_index,
                    "timestamp_s": round(timestamp_s, 6),
                    "detected": landmarks is not None,
                    "detection_count": detection_count,
                    "primary_detection_index": primary_index,
                    "primary_handedness": handedness,
                    "primary_score": score,
                    "all_detections": detections,
                    "landmarks": landmarks,
                }
                frame_records.append(frame_record)
                csv_rows.extend(
                    landmark_rows(
                        item=item,
                        frame_index=frame_index,
                        timestamp_s=round(timestamp_s, 6),
                        width=width,
                        height=height,
                        detection_count=detection_count,
                        primary_index=primary_index,
                        handedness=handedness,
                        score=score,
                        landmarks=landmarks,
                    )
                )
                frame_index += 1
        finally:
            hands.close()
            skeleton_writer.release()
            side_by_side_writer.release()

        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "video_id": item.output_stem,
                    "source_path": item.source_path.as_posix(),
                    "label": item.label,
                    "source_folder": item.source_folder,
                    "source_stem": item.source_stem,
                    "frame_width": width,
                    "frame_height": height,
                    "fps": fps,
                    "expected_frames": expected_frames,
                    "processed_frames": len(frame_records),
                    "frames": frame_records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        write_landmarks_csv(csv_path, csv_rows)

        missing_frames = len(frame_records) - detected_frames
        detection_coverage = detected_frames / len(frame_records) if frame_records else 0.0
        average_score = total_score / scored_frames if scored_frames else None
        return {
            "video_id": item.output_stem,
            "source_path": item.source_path.as_posix(),
            "label": item.label,
            "source_folder": item.source_folder,
            "source_stem": item.source_stem,
            "frame_width": width,
            "frame_height": height,
            "fps": fps,
            "duration_s": duration_s,
            "expected_frames": expected_frames,
            "processed_frames": len(frame_records),
            "detected_frames": detected_frames,
            "missing_frames": missing_frames,
            "detection_coverage": detection_coverage,
            "average_primary_score": average_score,
            "multi_hand_frames": multi_hand_frames,
            "needs_review": detection_coverage < 0.95 or missing_frames > 0,
            "skeleton_video": skeleton_path.as_posix(),
            "side_by_side_video": side_by_side_path.as_posix(),
            "landmarks_json": json_path.as_posix(),
            "landmarks_csv": csv_path.as_posix(),
            "contact_frame": contact_frame,
        }
    finally:
        capture.release()
        if temp_dir is not None:
            temp_dir.cleanup()


def select_primary_hand(result: Any) -> tuple[int | None, list[dict[str, object]]]:
    """Select the highest-confidence detected hand and summarize all detections."""

    landmarks = list(result.multi_hand_landmarks or [])
    handedness_list = list(result.multi_handedness or [])
    if len(landmarks) == 0:
        return None, []
    detections: list[dict[str, object]] = []
    best_index = 0
    best_score = -1.0
    for index in range(len(landmarks)):
        label: str | None = None
        score = 0.0
        if index < len(handedness_list) and handedness_list[index].classification:
            category = handedness_list[index].classification[0]
            label = str(category.label)
            score = float(category.score)
        detections.append({"index": index, "label": label, "score": score})
        if score > best_score:
            best_index = index
            best_score = score
    return best_index, detections


def draw_hand_skeleton(cv2: Any, frame: NDArray[np.uint8], landmarks: Sequence[Mapping[str, object]] | None) -> None:
    """Draw the standard 21-landmark MediaPipe hand skeleton."""

    if landmarks is None:
        return
    height, width = frame.shape[:2]
    points: dict[int, tuple[int, int]] = {}
    for landmark in landmarks:
        index = int(_required_float(landmark["index"]))
        x_px = int(round(_required_float(landmark["x"]) * width))
        y_px = int(round(_required_float(landmark["y"]) * height))
        points[index] = (x_px, y_px)

    for group, start, end in HAND_EDGES:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], FINGER_COLORS[group], 4, cv2.LINE_AA)
    for index, point in points.items():
        radius = 6 if index in {0, 4, 8, 12, 16, 20} else 5
        cv2.circle(frame, point, radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, point, radius, (20, 20, 20), 1, cv2.LINE_AA)


def draw_status(cv2: Any, frame: NDArray[np.uint8], text: str, ok: bool) -> None:
    """Draw compact detection-quality status text."""

    color = (80, 220, 120) if ok else (80, 80, 255)
    width = min(frame.shape[1] - 12, 12 + max(300, 10 * len(text)))
    cv2.rectangle(frame, (12, 12), (width, 50), (0, 0, 0), -1)
    cv2.putText(frame, text, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def landmark_rows(
    *,
    item: SkeletonReviewVideo,
    frame_index: int,
    timestamp_s: float,
    width: int,
    height: int,
    detection_count: int,
    primary_index: int | None,
    handedness: str | None,
    score: float | None,
    landmarks: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    """Return per-landmark CSV rows for one frame."""

    common: dict[str, object] = {
        "video_id": item.output_stem,
        "label": item.label,
        "frame_index": frame_index,
        "timestamp_s": timestamp_s,
        "detection_count": detection_count,
        "primary_detection_index": primary_index,
        "handedness": handedness,
        "score": score,
    }
    if landmarks is None:
        return [
            {
                **common,
                "detected": False,
                "landmark_index": None,
                "x_norm": None,
                "y_norm": None,
                "z_norm": None,
                "x_px": None,
                "y_px": None,
            }
        ]
    rows: list[dict[str, object]] = []
    for landmark in landmarks:
        x_norm = _required_float(landmark["x"])
        y_norm = _required_float(landmark["y"])
        rows.append(
            {
                **common,
                "detected": True,
                "landmark_index": int(_required_float(landmark["index"])),
                "x_norm": x_norm,
                "y_norm": y_norm,
                "z_norm": _required_float(landmark["z"]),
                "x_px": round(x_norm * width, 3),
                "y_px": round(y_norm * height, 3),
            }
        )
    return rows


def write_landmarks_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write per-frame/per-landmark CSV rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def write_quality_table(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    """Write clip-level detection quality table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "video_id",
        "label",
        "processed_frames",
        "detected_frames",
        "missing_frames",
        "detection_coverage",
        "average_primary_score",
        "multi_hand_frames",
        "needs_review",
        "side_by_side_video",
        "skeleton_video",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column) for column in columns})


def validate_review_outputs(records: Sequence[Mapping[str, object]], output_root: Path) -> dict[str, object]:
    """Validate output review MP4 dimensions and frame counts with ffprobe."""

    validations: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for record in records:
        for kind in ("skeleton_video", "side_by_side_video"):
            output_path = Path(str(record[kind]))
            probe = ffprobe_video(output_path)
            width = int(probe["width"])
            height = int(probe["height"])
            frames = int(probe.get("nb_frames") or 0)
            expected_width = int(record["frame_width"]) if kind == "skeleton_video" else int(record["frame_width"]) * 2
            expected_height = int(record["frame_height"])
            expected_frames = int(record["processed_frames"])
            passed = width == expected_width and height == expected_height and frames == expected_frames
            validation = {
                "video_id": record["video_id"],
                "kind": kind,
                "path": output_path.as_posix(),
                "width": width,
                "height": height,
                "frames": frames,
                "expected_width": expected_width,
                "expected_height": expected_height,
                "expected_frames": expected_frames,
                "status": "passed" if passed else "failed",
            }
            validations.append(validation)
            if not passed:
                failed.append(validation)
    summary = {
        "status": "passed" if not failed else "failed",
        "validation_count": len(validations),
        "failed_count": len(failed),
        "validations": validations,
        "failed": failed,
    }
    (output_root / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_review_manifest(
    *,
    input_root: Path,
    output_root: Path,
    records: Sequence[Mapping[str, object]],
    discovery: Mapping[str, object],
    validation: Mapping[str, object],
    review_stage: str = "held_out_test_skeleton_review",
    training_policy: str = "These 15 clips remain validation-only and are not training seeds.",
) -> dict[str, object]:
    """Build the global review manifest."""

    clean_records = [{key: value for key, value in record.items() if key != "contact_frame"} for record in records]
    return {
        "status": "passed" if bool(discovery.get("passed")) and validation.get("status") == "passed" else "failed",
        "review_stage": review_stage,
        "input_root": input_root.as_posix(),
        "output_root": output_root.as_posix(),
        "video_count": len(records),
        "discovery": dict(discovery),
        "artifact_contract": {
            "skeleton_video": "skeleton-only MP4, same resolution and frame count as source",
            "side_by_side_video": "source frame next to skeleton-only frame, double source width",
            "landmarks_json": "per-frame detection metadata and 21 normalized landmarks when detected",
            "landmarks_csv": "per-landmark table with normalized and image-space coordinates",
            "quality_table": "clip-level detection coverage and review flags",
            "contact_sheet": "middle-frame side-by-side tiles for quick review",
        },
        "training_policy": training_policy,
        "records": clean_records,
        "quality_table_path": (output_root / "quality_table.csv").as_posix(),
        "validation_summary_path": (output_root / "validation_summary.json").as_posix(),
        "contact_sheet_path": (output_root / "review_contact_sheet.png").as_posix(),
        "validation": dict(validation),
    }


def write_contact_sheet(cv2: Any, path: Path, labeled_frames: Sequence[tuple[str, NDArray[np.uint8]]]) -> None:
    """Write a compact contact sheet from representative review frames."""

    if len(labeled_frames) == 0:
        return
    tile_width = 320
    tile_height = 180
    columns = 5
    rows = int(math.ceil(len(labeled_frames) / columns))
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, (label, frame) in enumerate(labeled_frames):
        resized = cv2.resize(frame, (tile_width, tile_height))
        row = index // columns
        column = index % columns
        y0 = row * tile_height
        x0 = column * tile_width
        sheet[y0 : y0 + tile_height, x0 : x0 + tile_width] = resized
        cv2.putText(sheet, label[:32], (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 240, 20), 1, cv2.LINE_AA)
    imwrite_unicode(cv2, path, sheet)


def ffprobe_video(path: Path) -> dict[str, object]:
    """Return basic video stream metadata from ffprobe."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(completed.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found: {path}")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise RuntimeError(f"Invalid ffprobe stream payload for {path}")
    return dict(stream)


def mp4_writer(cv2: Any, path: Path, fps: float, size: tuple[int, int]) -> Any:
    """Open an MP4 writer with a broadly supported codec."""

    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Could not open MP4 writer for {path}")


def safe_fps(cv2: Any, capture: Any) -> float:
    """Return capture FPS with a safe fallback."""

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0.0:
        return 30.0
    return fps


def imwrite_unicode(cv2: Any, path: Path, image: NDArray[np.uint8]) -> None:
    """Write an image using cv2 encoding plus Python file I/O."""

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def _open_capture_with_optional_ascii_copy(
    cv2: Any,
    path: Path,
    output_stem: str,
) -> tuple[Any, tempfile.TemporaryDirectory[str] | None]:
    capture = cv2.VideoCapture(str(path))
    if capture.isOpened():
        return capture, None
    capture.release()
    temp_dir = tempfile.TemporaryDirectory(prefix="rps_skeleton_review_")
    temp_path = Path(temp_dir.name) / f"{output_stem}.mp4"
    shutil.copy2(path, temp_path)
    capture = cv2.VideoCapture(str(temp_path))
    return capture, temp_dir


def _primary_landmarks(result: Any, primary_index: int | None) -> list[dict[str, object]] | None:
    if primary_index is None:
        return None
    hands = list(result.multi_hand_landmarks or [])
    if primary_index >= len(hands):
        return None
    return [
        {"index": index, "x": float(landmark.x), "y": float(landmark.y), "z": float(landmark.z)}
        for index, landmark in enumerate(hands[primary_index].landmark)
    ]


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _required_float(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Expected numeric value, got {value!r}")
    return float(value)
