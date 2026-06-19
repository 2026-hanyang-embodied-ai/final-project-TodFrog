"""Few-shot classifier over temporal hand-skeleton curl trajectories."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from embodied_rps.real_skeleton_failure_features import FINGER_CHAINS, FINGER_NAMES, load_review_clip

TemporalLabel = Literal["rock", "paper", "scissors"]


@dataclass(frozen=True)
class TemporalTrajectoryConfig:
    """Configuration for a temporal curl trajectory classifier."""

    observation_progress: float = 0.5
    sample_count: int = 12
    confidence_temperature: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.observation_progress <= 1.0:
            raise ValueError("observation_progress must be in (0, 1]")
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least 2")
        if self.confidence_temperature <= 0.0:
            raise ValueError("confidence_temperature must be positive")


@dataclass(frozen=True)
class TemporalReviewClip:
    """One loaded skeleton-review clip with canonical landmarks."""

    clip_id: str
    label: TemporalLabel
    source_path: str
    canonical_landmarks: NDArray[np.float32]


@dataclass(frozen=True)
class TemporalTrajectoryClassifier:
    """Nearest-centroid classifier over sampled finger curl trajectories."""

    labels: tuple[TemporalLabel, ...]
    feature_names: tuple[str, ...]
    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    centroids: Mapping[TemporalLabel, NDArray[np.float64]]
    config: TemporalTrajectoryConfig
    train_clip_count: int

    def predict(self, canonical_landmarks: NDArray[np.float32]) -> tuple[TemporalLabel, dict[str, object]]:
        """Predict a gesture from one canonical hand-skeleton sequence."""

        feature_vector, feature_names = extract_temporal_curl_feature_vector(
            canonical_landmarks,
            observation_progress=self.config.observation_progress,
            sample_count=self.config.sample_count,
        )
        if tuple(feature_names) != self.feature_names:
            raise ValueError("feature contract mismatch")
        normalized = (feature_vector.astype(np.float64) - self.feature_mean) / self.feature_scale
        distances = {
            label: float(np.linalg.norm(normalized - centroid))
            for label, centroid in self.centroids.items()
        }
        scores = np.asarray(
            [-distances[label] / self.config.confidence_temperature for label in self.labels],
            dtype=np.float64,
        )
        scores = scores - float(np.max(scores))
        probabilities_array = np.exp(scores)
        probabilities_array = probabilities_array / float(probabilities_array.sum())
        probabilities = {
            label: float(probability)
            for label, probability in zip(self.labels, probabilities_array.tolist(), strict=True)
        }
        prediction = max(probabilities.items(), key=lambda item: item[1])[0]
        ordered = sorted(probabilities.values(), reverse=True)
        diagnostics: dict[str, object] = {
            "predicted_label": prediction,
            "confidence": float(ordered[0]),
            "confidence_margin": float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0]),
            "distances": distances,
            "probabilities": probabilities,
        }
        return prediction, diagnostics

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable classifier representation."""

        return {
            "labels": list(self.labels),
            "feature_names": list(self.feature_names),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "centroids": {label: centroid.tolist() for label, centroid in self.centroids.items()},
            "config": {
                "observation_progress": self.config.observation_progress,
                "sample_count": self.config.sample_count,
                "confidence_temperature": self.config.confidence_temperature,
            },
            "train_clip_count": self.train_clip_count,
        }


def extract_temporal_curl_feature_vector(
    canonical_landmarks: NDArray[np.float32],
    *,
    observation_progress: float,
    sample_count: int,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Sample finger extension, differential curl, and velocity over the observed prefix."""

    frames = np.asarray(canonical_landmarks, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (21, 3):
        raise ValueError("canonical_landmarks must have shape (T,21,3)")
    if frames.shape[0] == 0:
        raise ValueError("canonical_landmarks must not be empty")
    if not 0.0 < observation_progress <= 1.0:
        raise ValueError("observation_progress must be in (0, 1]")
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")

    observed = _observed_prefix(frames, observation_progress)
    extensions = _finger_extensions(observed)
    spread = np.linalg.norm(observed[:, 8, :] - observed[:, 20, :], axis=1).astype(np.float64)
    motion = _finger_tip_motion(observed)

    feature_values: list[float] = []
    feature_names: list[str] = []
    for finger in FINGER_NAMES:
        sampled = _resample_series(extensions[finger], sample_count)
        for sample_index, value in enumerate(sampled):
            feature_names.append(f"{finger}_extension_t{sample_index:02d}")
            feature_values.append(float(value))
    for finger in FINGER_NAMES:
        sampled_velocity = _resample_series(np.gradient(extensions[finger]), sample_count)
        for sample_index, value in enumerate(sampled_velocity):
            feature_names.append(f"{finger}_extension_velocity_t{sample_index:02d}")
            feature_values.append(float(value))

    index_middle = (extensions["index"] + extensions["middle"]) * 0.5
    ring_pinky = (extensions["ring"] + extensions["pinky"]) * 0.5
    grouped_series = {
        "index_middle_minus_ring_pinky": index_middle - ring_pinky,
        "spread": spread,
        "tip_motion": motion,
    }
    for name, values in grouped_series.items():
        sampled = _resample_series(values, sample_count)
        for sample_index, value in enumerate(sampled):
            feature_names.append(f"{name}_t{sample_index:02d}")
            feature_values.append(float(value))

    return np.asarray(feature_values, dtype=np.float64), tuple(feature_names)


def load_temporal_review_clips_from_root(root: Path) -> list[TemporalReviewClip]:
    """Load all skeleton-review JSON clips under a review artifact root."""

    paths = sorted((root / "landmarks_json").rglob("*.json"))
    if not paths:
        raise ValueError(f"no landmark JSON files found under {root / 'landmarks_json'}")
    clips: list[TemporalReviewClip] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        review = load_review_clip(path)
        clips.append(
            TemporalReviewClip(
                clip_id=str(payload.get("video_id") or path.stem),
                label=_infer_temporal_label(payload, path),
                source_path=review.source_path,
                canonical_landmarks=review.canonical_landmarks,
            )
        )
    return clips


def fit_temporal_trajectory_classifier(
    clips: Sequence[tuple[str, str, NDArray[np.float32]] | TemporalReviewClip],
    *,
    config: TemporalTrajectoryConfig,
) -> TemporalTrajectoryClassifier:
    """Fit a nearest-centroid classifier from labeled temporal skeleton clips."""

    normalized_clips = [_normalize_clip(item) for item in clips]
    if not normalized_clips:
        raise ValueError("at least one training clip is required")
    labels = tuple(label for label in ("rock", "paper", "scissors") if any(clip.label == label for clip in normalized_clips))
    if len(labels) < 2:
        raise ValueError("at least two labels are required")

    feature_rows: list[NDArray[np.float64]] = []
    feature_names: tuple[str, ...] | None = None
    for clip in normalized_clips:
        vector, names = extract_temporal_curl_feature_vector(
            clip.canonical_landmarks,
            observation_progress=config.observation_progress,
            sample_count=config.sample_count,
        )
        feature_rows.append(vector)
        feature_names = names if feature_names is None else feature_names
        if names != feature_names:
            raise ValueError("feature contract mismatch")
    assert feature_names is not None

    matrix = np.asarray(feature_rows, dtype=np.float64)
    feature_mean = matrix.mean(axis=0)
    feature_scale = matrix.std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    normalized_matrix = (matrix - feature_mean) / feature_scale
    centroids: dict[TemporalLabel, NDArray[np.float64]] = {}
    for label in labels:
        label_indices = [index for index, clip in enumerate(normalized_clips) if clip.label == label]
        centroids[label] = normalized_matrix[label_indices].mean(axis=0)

    return TemporalTrajectoryClassifier(
        labels=labels,
        feature_names=feature_names,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        centroids=centroids,
        config=config,
        train_clip_count=len(normalized_clips),
    )


def evaluate_temporal_trajectory_classifier(
    classifier: TemporalTrajectoryClassifier,
    clips: Sequence[tuple[str, str, NDArray[np.float32]] | TemporalReviewClip],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate a temporal trajectory classifier on labeled clips."""

    normalized_clips = [_normalize_clip(item) for item in clips]
    rows: list[dict[str, object]] = []
    for clip in normalized_clips:
        prediction, diagnostics = classifier.predict(clip.canonical_landmarks)
        passed = prediction == clip.label
        probabilities = diagnostics["probabilities"]
        row = {
            "clip_id": clip.clip_id,
            "source_path": clip.source_path,
            "true_label": clip.label,
            "predicted_label": prediction,
            "passed": passed,
            "confidence": diagnostics["confidence"],
            "confidence_margin": diagnostics["confidence_margin"],
            "rock_probability": _mapping_number(probabilities, "rock"),
            "paper_probability": _mapping_number(probabilities, "paper"),
            "scissors_probability": _mapping_number(probabilities, "scissors"),
            "selected_robot_action": _robot_action_for_prediction(prediction),
            "failure_reason": None if passed else "wrong_prediction",
        }
        rows.append(row)
    summary = _summarize_rows(rows)
    return rows, summary


def summarize_temporal_trajectory_experiment(
    *,
    train_roots: Sequence[Path],
    eval_root: Path,
    config: TemporalTrajectoryConfig,
) -> tuple[list[dict[str, object]], dict[str, object], TemporalTrajectoryClassifier]:
    """Train from review roots and evaluate on one review root."""

    train_clips: list[TemporalReviewClip] = []
    train_root_summaries: list[dict[str, object]] = []
    for root in train_roots:
        clips = load_temporal_review_clips_from_root(root)
        train_clips.extend(clips)
        train_root_summaries.append(_root_summary(root, clips))
    eval_clips = load_temporal_review_clips_from_root(eval_root)
    classifier = fit_temporal_trajectory_classifier(train_clips, config=config)
    rows, summary = evaluate_temporal_trajectory_classifier(classifier, eval_clips)
    summary.update(
        {
            "train_clip_count": len(train_clips),
            "train_roots": [root.as_posix() for root in train_roots],
            "eval_root": eval_root.as_posix(),
            "train_root_summaries": train_root_summaries,
            "eval_root_summary": _root_summary(eval_root, eval_clips),
            "classifier": classifier.to_json_dict(),
        }
    )
    return rows, summary, classifier


def _normalize_clip(item: tuple[str, str, NDArray[np.float32]] | TemporalReviewClip) -> TemporalReviewClip:
    if isinstance(item, TemporalReviewClip):
        return item
    clip_id, label, landmarks = item
    return TemporalReviewClip(
        clip_id=clip_id,
        label=_validate_label(label),
        source_path="",
        canonical_landmarks=np.asarray(landmarks, dtype=np.float32),
    )


def _observed_prefix(frames: NDArray[np.float32], observation_progress: float) -> NDArray[np.float32]:
    frame_count = int(frames.shape[0])
    progress = (np.arange(frame_count, dtype=np.float32) + 1.0) / np.float32(frame_count)
    observed_mask = progress <= np.float32(observation_progress)
    if not bool(np.any(observed_mask)):
        observed_mask[0] = True
    return frames[observed_mask]


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


def _finger_tip_motion(frames: NDArray[np.float32]) -> NDArray[np.float64]:
    if frames.shape[0] <= 1:
        return np.zeros((frames.shape[0],), dtype=np.float64)
    tip_indices = [chain[-1] for chain in FINGER_CHAINS.values()]
    velocities = np.linalg.norm(np.diff(frames[:, tip_indices, :], axis=0), axis=2).mean(axis=1)
    return np.concatenate([velocities[:1], velocities]).astype(np.float64)


def _resample_series(values: NDArray[np.float64], sample_count: int) -> NDArray[np.float64]:
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError("series must be 1-D")
    if series.size == 0:
        return np.zeros((sample_count,), dtype=np.float64)
    if series.size == 1:
        return np.full((sample_count,), float(series[0]), dtype=np.float64)
    source_x = np.linspace(0.0, 1.0, num=series.size)
    target_x = np.linspace(0.0, 1.0, num=sample_count)
    return np.interp(target_x, source_x, series).astype(np.float64)


def _infer_temporal_label(payload: Mapping[str, Any], path: Path) -> TemporalLabel:
    raw_label = str(payload.get("label") or "")
    if raw_label in {"rock", "paper", "scissors"}:
        return _validate_label(raw_label)
    transition_label = str(payload.get("transition_label") or "")
    if transition_label == "rock_to_paper":
        return "paper"
    if transition_label == "rock_to_scissors":
        return "scissors"
    path_text = path.as_posix().lower()
    path_text_windows = str(path).lower()
    if "rock_to_paper" in path_text:
        return "paper"
    if "rock_to_scissors" in path_text:
        return "scissors"
    if "/rock/" in path_text or "\\rock\\" in path_text_windows:
        return "rock"
    if "/paper/" in path_text or "\\paper\\" in path_text_windows:
        return "paper"
    if "/scissors/" in path_text or "\\scissors\\" in path_text_windows:
        return "scissors"
    raise ValueError(f"could not infer temporal label for {path}")


def _validate_label(label: str) -> TemporalLabel:
    if label not in {"rock", "paper", "scissors"}:
        raise ValueError(f"unsupported temporal label: {label}")
    return label  # type: ignore[return-value]


def _summarize_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    label_counts = Counter(str(row["true_label"]) for row in rows)
    passed_counts = Counter(str(row["true_label"]) for row in rows if bool(row["passed"]))
    failed = [row for row in rows if not bool(row["passed"])]
    return {
        "eval_clip_count": len(rows),
        "passed_clip_count": sum(1 for row in rows if bool(row["passed"])),
        "failed_clip_count": len(failed),
        "accuracy": _safe_rate(sum(1 for row in rows if bool(row["passed"])), len(rows)),
        "per_class": {
            label: {
                "clip_count": int(label_counts.get(label, 0)),
                "passed_count": int(passed_counts.get(label, 0)),
                "accuracy": _safe_rate(int(passed_counts.get(label, 0)), int(label_counts.get(label, 0))),
            }
            for label in ("rock", "paper", "scissors")
        },
        "failed_clip_ids": [str(row["clip_id"]) for row in failed],
        "failure_reason_counts": dict(sorted(Counter(str(row["failure_reason"]) for row in failed).items())),
    }


def _root_summary(root: Path, clips: Sequence[TemporalReviewClip]) -> dict[str, object]:
    return {
        "root": root.as_posix(),
        "clip_count": len(clips),
        "label_counts": dict(sorted(Counter(clip.label for clip in clips).items())),
    }


def _mapping_number(mapping: object, key: str) -> float:
    if isinstance(mapping, Mapping):
        value = mapping.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 0.0


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _robot_action_for_prediction(prediction: str) -> str:
    if prediction == "rock":
        return "paper"
    if prediction == "paper":
        return "scissors"
    if prediction == "scissors":
        return "rock"
    return ""


__all__ = [
    "TemporalReviewClip",
    "TemporalTrajectoryClassifier",
    "TemporalTrajectoryConfig",
    "evaluate_temporal_trajectory_classifier",
    "extract_temporal_curl_feature_vector",
    "fit_temporal_trajectory_classifier",
    "load_temporal_review_clips_from_root",
    "summarize_temporal_trajectory_experiment",
]
