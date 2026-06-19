"""Feature-level diagnostics for real skeleton prediction failures."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from embodied_rps.tools.run_realtime_skeleton_predictor import canonicalize_mediapipe_landmarks

FINGER_CHAINS: dict[str, tuple[int, int, int, int]] = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_NAMES: tuple[str, ...] = tuple(FINGER_CHAINS)
REFERENCE_FEATURE_COLUMNS: tuple[str, ...] = (
    "motion_energy_mean",
    "jitter_energy_mean",
    "spread_mean",
    "early_index_extension_mean",
    "early_middle_extension_mean",
    "early_ring_extension_mean",
    "early_pinky_extension_mean",
    "final_index_extension_mean",
    "final_middle_extension_mean",
    "final_ring_extension_mean",
    "final_pinky_extension_mean",
)


@dataclass(frozen=True)
class ReviewClip:
    """Canonical landmarks loaded from one skeleton-review JSON file."""

    video_id: str
    label: str
    source_path: str
    canonical_landmarks: NDArray[np.float32]


@dataclass(frozen=True)
class ClipFeatureSummary:
    """Compact feature summary for a canonical skeleton clip."""

    clip_id: str
    label: str
    frame_count: int
    early_frame_count: int
    early_mean_extension: dict[str, float]
    final_mean_extension: dict[str, float]
    early_mean_velocity: dict[str, float]
    motion_energy_mean: float
    jitter_energy_mean: float
    spread_mean: float

    def to_flat_row(self) -> dict[str, object]:
        """Return a CSV/JSON-friendly flat representation."""

        row: dict[str, object] = {
            "clip_id": self.clip_id,
            "label": self.label,
            "frame_count": self.frame_count,
            "early_frame_count": self.early_frame_count,
            "motion_energy_mean": self.motion_energy_mean,
            "jitter_energy_mean": self.jitter_energy_mean,
            "spread_mean": self.spread_mean,
        }
        for finger in FINGER_NAMES:
            row[f"early_{finger}_extension_mean"] = self.early_mean_extension[finger]
            row[f"final_{finger}_extension_mean"] = self.final_mean_extension[finger]
            row[f"early_{finger}_velocity_mean"] = self.early_mean_velocity[finger]
        return row


def load_review_clip(path: Path) -> ReviewClip:
    """Load one review JSON and canonicalize detected MediaPipe landmarks."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{path} does not contain a frames list")

    canonical_frames: list[NDArray[np.float32]] = []
    for frame in frames:
        if not isinstance(frame, dict) or not bool(frame.get("detected")):
            continue
        raw_landmarks = frame.get("landmarks")
        if not isinstance(raw_landmarks, list):
            continue
        points = _landmark_array(raw_landmarks, path)
        canonical_frames.append(canonicalize_mediapipe_landmarks(points))

    if not canonical_frames:
        raise ValueError(f"{path} contains no detected landmark frames")

    return ReviewClip(
        video_id=str(payload.get("video_id", path.stem)),
        label=str(payload.get("label", path.parent.name)),
        source_path=str(payload.get("source_path", "")),
        canonical_landmarks=np.stack(canonical_frames).astype(np.float32),
    )


def summarize_canonical_sequence(
    canonical_landmarks: NDArray[np.float32],
    *,
    label: str,
    clip_id: str = "",
    progress_cutoff: float = 0.5,
) -> ClipFeatureSummary:
    """Summarize early/final finger extension and temporal stability."""

    frames = np.asarray(canonical_landmarks, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (21, 3):
        raise ValueError("canonical_landmarks must have shape (T,21,3)")
    frame_count = int(frames.shape[0])
    if frame_count == 0:
        raise ValueError("canonical_landmarks must contain at least one frame")

    clip_progress = (np.arange(frame_count, dtype=np.float32) + 1.0) / np.float32(frame_count)
    early_mask = clip_progress <= np.float32(progress_cutoff)
    if not bool(np.any(early_mask)):
        early_mask[0] = True

    extensions_by_finger = _finger_extensions(frames)
    velocities_by_finger = _finger_tip_velocities(frames)
    early_mean_extension = {finger: float(np.mean(values[early_mask])) for finger, values in extensions_by_finger.items()}
    final_window_start = max(0, frame_count - max(2, int(round(frame_count * 0.2))))
    final_mean_extension = {finger: float(np.mean(values[final_window_start:])) for finger, values in extensions_by_finger.items()}
    early_velocity_mask = early_mask[1:] if frame_count > 1 else np.asarray([], dtype=np.bool_)
    early_mean_velocity = {
        finger: float(np.mean(values[early_velocity_mask])) if values.size and bool(np.any(early_velocity_mask)) else 0.0
        for finger, values in velocities_by_finger.items()
    }

    frame_deltas = np.linalg.norm(np.diff(frames, axis=0), axis=2).mean(axis=1) if frame_count > 1 else np.asarray([0.0], dtype=np.float32)
    second_deltas = (
        np.linalg.norm(np.diff(frames, n=2, axis=0), axis=2).mean(axis=1)
        if frame_count > 2
        else np.asarray([0.0], dtype=np.float32)
    )
    spread = np.linalg.norm(frames[:, 8, :] - frames[:, 20, :], axis=1)

    return ClipFeatureSummary(
        clip_id=clip_id,
        label=label,
        frame_count=frame_count,
        early_frame_count=int(np.count_nonzero(early_mask)),
        early_mean_extension=early_mean_extension,
        final_mean_extension=final_mean_extension,
        early_mean_velocity=early_mean_velocity,
        motion_energy_mean=float(np.mean(frame_deltas)),
        jitter_energy_mean=float(np.mean(second_deltas)),
        spread_mean=float(np.mean(spread[early_mask])),
    )


def build_augmentation_recommendation(
    summary: ClipFeatureSummary,
    *,
    failure_reason: str | None,
    decision_state: str | None,
) -> dict[str, object]:
    """Map a failure and its feature signature to a simulation target family."""

    reason = failure_reason or "unknown"
    decision = decision_state or ""
    label = summary.label
    if label == "rock" and reason == "false_trigger":
        target = "rock_wait_false_transition_hard_negatives"
        rationale = (
            f"Generate rock-wait sequences with index/middle extension-like noise and wrist/view jitter; "
            f"failed decision was {decision or 'unknown'}."
        )
    elif label == "paper" and reason in {"no_stable_decision", "late_decision"}:
        target = "paper_no_stable_decision_hard_positives"
        rationale = "Generate paper openings with stronger early ring/pinky extension evidence and varied opening speed."
    elif label == "paper" and reason == "wrong_prediction":
        target = "paper_scissors_boundary_hard_examples"
        rationale = f"Generate paper trajectories whose early index/middle evidence looks like {decision or 'the wrong class'} but ring/pinky open before the deadline."
    elif label == "scissors" and reason in {"no_stable_decision", "late_decision"}:
        target = "shaky_scissors_early_transition_positives"
        rationale = "Generate scissors trajectories with shaky roll/yaw and early index/middle extension that remains stable by progress 0.50."
    elif label == "scissors" and reason == "wrong_prediction":
        target = "scissors_paper_boundary_hard_examples"
        rationale = f"Generate scissors trajectories with paper-like early spread but ring/pinky stay curled; failed decision was {decision or 'unknown'}."
    else:
        target = f"{label}_{reason}_review"
        rationale = "Inspect this failure before adding a new simulation family."
    return {
        "clip_id": summary.clip_id,
        "label": label,
        "failure_reason": reason,
        "decision_state": decision,
        "target_family": target,
        "rationale": rationale,
    }


def summarize_prediction_artifacts(
    *,
    review_root: Path,
    prediction_root: Path,
    progress_cutoff: float = 0.5,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Join skeleton review JSONs with prediction metrics and produce diagnostic rows."""

    review_by_source = _load_review_index(review_root)
    rows: list[dict[str, object]] = []
    recommendations: list[dict[str, object]] = []
    missing_reviews: list[str] = []

    metric_paths = sorted((prediction_root / "clips").rglob("metrics.json"))
    if not metric_paths:
        raise ValueError(f"no metrics.json files found under {prediction_root / 'clips'}")

    for metrics_path in metric_paths:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        source_path = str(metrics.get("source_path", ""))
        review = review_by_source.get(_normalize_source_key(source_path))
        if review is None:
            missing_reviews.append(source_path)
            continue
        clip_id = str(metrics.get("clip_id", review.video_id))
        label = str(metrics.get("true_gesture", review.label))
        feature_summary = summarize_canonical_sequence(
            review.canonical_landmarks,
            label=label,
            clip_id=clip_id,
            progress_cutoff=progress_cutoff,
        )
        flat = feature_summary.to_flat_row()
        flat.update(
            {
                "source_path": source_path,
                "passed": bool(metrics.get("passed")),
                "failure_reason": metrics.get("failure_reason"),
                "decision_state": metrics.get("decision_state"),
                "decision_progress": metrics.get("decision_progress"),
                "decision_confidence": metrics.get("decision_confidence"),
            }
        )
        rows.append(flat)
        if not bool(metrics.get("passed")):
            recommendations.append(
                build_augmentation_recommendation(
                    feature_summary,
                    failure_reason=str(metrics.get("failure_reason") or ""),
                    decision_state=str(metrics.get("decision_state") or ""),
                )
            )

    label_reference_means = _label_reference_means(rows)
    rows_by_clip_id = {str(row["clip_id"]): row for row in rows}
    for recommendation in recommendations:
        row = rows_by_clip_id.get(str(recommendation.get("clip_id", "")))
        reference = label_reference_means.get(str(recommendation.get("label", "")))
        if row is None or reference is None:
            continue
        for column, reference_value in reference.items():
            value = row.get(column)
            if isinstance(value, int | float):
                recommendation[f"{column}_delta_from_passed_label"] = float(value) - reference_value

    recommendation_counts = Counter(item["target_family"] for item in recommendations)
    summary = {
        "review_root": review_root.as_posix(),
        "prediction_root": prediction_root.as_posix(),
        "clip_count": len(rows),
        "failed_clip_count": sum(1 for row in rows if not bool(row["passed"])),
        "missing_review_count": len(missing_reviews),
        "missing_reviews": missing_reviews,
        "label_reference_means": label_reference_means,
        "recommendation_counts": dict(sorted(recommendation_counts.items())),
        "recommendations": recommendations,
    }
    return rows, summary


def _load_review_index(review_root: Path) -> dict[str, ReviewClip]:
    review_paths = sorted((review_root / "landmarks_json").rglob("*.json"))
    if not review_paths:
        raise ValueError(f"no review JSON files found under {review_root / 'landmarks_json'}")
    by_source: dict[str, ReviewClip] = {}
    for path in review_paths:
        review = load_review_clip(path)
        by_source[_normalize_source_key(review.source_path)] = review
    return by_source


def _finger_extensions(frames: NDArray[np.float32]) -> dict[str, NDArray[np.float64]]:
    denominator = np.maximum(
        np.linalg.norm(frames[:, 9, :] - frames[:, 0, :], axis=1).astype(np.float64),
        0.75,
    )
    values: dict[str, NDArray[np.float64]] = {}
    for finger, chain in FINGER_CHAINS.items():
        mcp = chain[0]
        tip = chain[-1]
        values[finger] = np.linalg.norm(frames[:, tip, :] - frames[:, mcp, :], axis=1).astype(np.float64) / denominator
    return values


def _finger_tip_velocities(frames: NDArray[np.float32]) -> dict[str, NDArray[np.float64]]:
    values: dict[str, NDArray[np.float64]] = {}
    if frames.shape[0] <= 1:
        return {finger: np.asarray([], dtype=np.float64) for finger in FINGER_NAMES}
    for finger, chain in FINGER_CHAINS.items():
        tip = chain[-1]
        values[finger] = np.linalg.norm(np.diff(frames[:, tip, :], axis=0), axis=1).astype(np.float64)
    return values


def _label_reference_means(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_label: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if not bool(row.get("passed")):
            continue
        by_label.setdefault(str(row.get("label", "")), []).append(row)

    references: dict[str, dict[str, float]] = {}
    for label, label_rows in by_label.items():
        references[label] = {}
        for column in REFERENCE_FEATURE_COLUMNS:
            values = [float(row[column]) for row in label_rows if isinstance(row.get(column), int | float)]
            if values:
                references[label][column] = float(np.mean(values))
    return references


def _landmark_array(landmarks: list[Any], path: Path) -> NDArray[np.float32]:
    if len(landmarks) != 21:
        raise ValueError(f"{path} expected 21 landmarks, got {len(landmarks)}")
    points = np.zeros((21, 3), dtype=np.float32)
    for fallback_index, item in enumerate(landmarks):
        if not isinstance(item, dict):
            raise ValueError(f"{path} landmark {fallback_index} is not an object")
        index = int(item.get("index", fallback_index))
        points[index] = [
            float(item.get("x_norm", item.get("x"))),
            float(item.get("y_norm", item.get("y"))),
            float(item.get("z_norm", item.get("z"))),
        ]
    return points


def _normalize_source_key(path: str) -> str:
    return path.replace("\\", "/").lower()


__all__ = [
    "ClipFeatureSummary",
    "ReviewClip",
    "build_augmentation_recommendation",
    "load_review_clip",
    "summarize_canonical_sequence",
    "summarize_prediction_artifacts",
]
