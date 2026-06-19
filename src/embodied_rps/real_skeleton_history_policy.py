"""Few-shot probability-history policies for saved real-skeleton validation clips."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from embodied_rps.real_skeleton_video_eval import (
    COUNTER_MOVES,
    WAIT_COUNTER_PAPER_STATE,
    EvaluationGesture,
    robot_action_for_decision_state,
)

HistoryLabel = Literal["rock", "paper", "scissors"]


@dataclass(frozen=True)
class HistoryPolicyConfig:
    """Configuration for a clip-history centroid policy."""

    observation_progress: float = 0.50
    max_decision_progress: float = 0.50
    confidence_temperature: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < self.observation_progress <= 1.0:
            raise ValueError("observation_progress must be in (0, 1]")
        if not 0.0 < self.max_decision_progress <= 1.0:
            raise ValueError("max_decision_progress must be in (0, 1]")
        if self.confidence_temperature <= 0.0:
            raise ValueError("confidence_temperature must be positive")


@dataclass(frozen=True)
class HistoryTrainingClip:
    """A labeled saved-output clip used by the history policy."""

    clip_id: str
    true_gesture: EvaluationGesture
    rows: Sequence[Mapping[str, object]]


@dataclass(frozen=True)
class HistoryCentroidPolicy:
    """Nearest-centroid classifier over per-clip probability-history features."""

    labels: tuple[HistoryLabel, ...]
    feature_names: tuple[str, ...]
    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    centroids: Mapping[HistoryLabel, NDArray[np.float64]]
    config: HistoryPolicyConfig


def feature_vector_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    observation_progress: float,
) -> dict[str, float]:
    """Build a fixed probability-history feature mapping from rows up to a progress cutoff."""

    observed = [
        row
        for row in rows
        if bool(row.get("detected", False)) and _number(row.get("clip_progress")) <= observation_progress
    ]
    if len(observed) == 0:
        observed = [row for row in rows if bool(row.get("detected", False))]
    if len(observed) == 0:
        return {name: 0.0 for name in _FEATURE_NAMES}
    probability_rows = np.asarray(
        [
            [
                _number(row.get("rock_probability")),
                _number(row.get("paper_probability")),
                _number(row.get("scissors_probability")),
            ]
            for row in observed
        ],
        dtype=np.float64,
    )
    latest = probability_rows[-1]
    first = probability_rows[0]
    mean = probability_rows.mean(axis=0)
    maximum = probability_rows.max(axis=0)
    std = probability_rows.std(axis=0)
    prediction_counts = {
        label: sum(1 for row in observed if row.get("prediction") == label) / float(len(observed))
        for label in ("rock", "paper", "scissors")
    }
    features = {
        "observed_frame_count": float(len(observed)),
        "latest_rock_probability": float(latest[0]),
        "latest_paper_probability": float(latest[1]),
        "latest_scissors_probability": float(latest[2]),
        "mean_rock_probability": float(mean[0]),
        "mean_paper_probability": float(mean[1]),
        "mean_scissors_probability": float(mean[2]),
        "max_rock_probability": float(maximum[0]),
        "max_paper_probability": float(maximum[1]),
        "max_scissors_probability": float(maximum[2]),
        "std_rock_probability": float(std[0]),
        "std_paper_probability": float(std[1]),
        "std_scissors_probability": float(std[2]),
        "delta_rock_probability": float(latest[0] - first[0]),
        "delta_paper_probability": float(latest[1] - first[1]),
        "delta_scissors_probability": float(latest[2] - first[2]),
        "rock_prediction_fraction": float(prediction_counts["rock"]),
        "paper_prediction_fraction": float(prediction_counts["paper"]),
        "scissors_prediction_fraction": float(prediction_counts["scissors"]),
    }
    return {name: features[name] for name in _FEATURE_NAMES}


def fit_history_centroid_policy(
    clips: Sequence[HistoryTrainingClip],
    *,
    config: HistoryPolicyConfig,
) -> HistoryCentroidPolicy:
    """Fit a nearest-centroid history policy from labeled calibration clips."""

    if len(clips) == 0:
        raise ValueError("clips must not be empty")
    feature_names = _FEATURE_NAMES
    matrix = np.asarray(
        [
            _feature_array(feature_vector_from_rows(clip.rows, observation_progress=config.observation_progress), feature_names)
            for clip in clips
        ],
        dtype=np.float64,
    )
    feature_mean = matrix.mean(axis=0)
    feature_scale = matrix.std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    normalized = (matrix - feature_mean) / feature_scale
    labels = tuple(label for label in ("rock", "paper", "scissors") if any(clip.true_gesture == label for clip in clips))
    centroids: dict[HistoryLabel, NDArray[np.float64]] = {}
    for label in labels:
        label_rows = normalized[[index for index, clip in enumerate(clips) if clip.true_gesture == label]]
        centroids[cast(HistoryLabel, label)] = label_rows.mean(axis=0)
    return HistoryCentroidPolicy(
        labels=cast(tuple[HistoryLabel, ...], labels),
        feature_names=feature_names,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        centroids=centroids,
        config=config,
    )


def summarize_history_clip(
    clip: HistoryTrainingClip,
    *,
    policy: HistoryCentroidPolicy,
) -> dict[str, object]:
    """Summarize one saved clip under a fitted history policy."""

    rows = [row for row in clip.rows if bool(row.get("detected", False))]
    decision_progress = _decision_progress(rows, observation_progress=policy.config.observation_progress)
    if len(rows) == 0:
        return {
            "clip_id": clip.clip_id,
            "true_gesture": clip.true_gesture,
            "passed": False,
            "failure_reason": "no_detection",
            "predicted_gesture": None,
            "decision_state": None,
            "selected_robot_action": None,
            "decision_progress": None,
        }
    prediction, probabilities = predict_history_label(clip.rows, policy=policy)
    decision_state = WAIT_COUNTER_PAPER_STATE if prediction == "rock" else prediction
    robot_action = robot_action_for_decision_state(decision_state)
    confidence = float(probabilities[prediction])
    sorted_probabilities = sorted(probabilities.values(), reverse=True)
    margin = sorted_probabilities[0] - sorted_probabilities[1] if len(sorted_probabilities) > 1 else sorted_probabilities[0]
    early_enough = decision_progress <= policy.config.max_decision_progress
    if clip.true_gesture == "rock":
        correct = prediction == "rock"
        failure_reason = None if correct and early_enough else "false_trigger" if not correct and early_enough else "late_wait_decision"
    else:
        correct = prediction == clip.true_gesture
        failure_reason = None if correct and early_enough else "wrong_prediction" if not correct else "late_decision"
    return {
        "clip_id": clip.clip_id,
        "true_gesture": clip.true_gesture,
        "passed": bool(correct and early_enough),
        "failure_reason": failure_reason,
        "predicted_gesture": prediction,
        "decision_state": decision_state,
        "selected_robot_action": robot_action,
        "decision_progress": decision_progress,
        "decision_confidence": confidence,
        "decision_confidence_margin": float(margin),
        "history_probabilities": probabilities,
    }


def predict_history_label(
    rows: Sequence[Mapping[str, object]],
    *,
    policy: HistoryCentroidPolicy,
) -> tuple[HistoryLabel, dict[HistoryLabel, float]]:
    """Predict a label from probability-history features."""

    feature_map = feature_vector_from_rows(rows, observation_progress=policy.config.observation_progress)
    feature_array = _feature_array(feature_map, policy.feature_names)
    normalized = (feature_array - policy.feature_mean) / policy.feature_scale
    distances = {
        label: float(np.linalg.norm(normalized - centroid))
        for label, centroid in policy.centroids.items()
    }
    scores = np.asarray([-distances[label] / policy.config.confidence_temperature for label in policy.labels], dtype=np.float64)
    scores = scores - float(np.max(scores))
    exp_scores = np.exp(scores)
    exp_scores = exp_scores / float(exp_scores.sum())
    probabilities = {
        label: float(probability)
        for label, probability in zip(policy.labels, exp_scores.tolist(), strict=True)
    }
    prediction = max(probabilities.items(), key=lambda item: item[1])[0]
    return prediction, probabilities


def _feature_array(feature_map: Mapping[str, float], feature_names: Sequence[str]) -> NDArray[np.float64]:
    return np.asarray([float(feature_map[name]) for name in feature_names], dtype=np.float64)


def _decision_progress(rows: Sequence[Mapping[str, object]], *, observation_progress: float) -> float:
    candidates = [row for row in rows if _number(row.get("clip_progress")) <= observation_progress]
    if not candidates:
        candidates = list(rows)
    return _number(candidates[-1].get("clip_progress")) if candidates else observation_progress


def _number(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


_FEATURE_NAMES: tuple[str, ...] = (
    "observed_frame_count",
    "latest_rock_probability",
    "latest_paper_probability",
    "latest_scissors_probability",
    "mean_rock_probability",
    "mean_paper_probability",
    "mean_scissors_probability",
    "max_rock_probability",
    "max_paper_probability",
    "max_scissors_probability",
    "std_rock_probability",
    "std_paper_probability",
    "std_scissors_probability",
    "delta_rock_probability",
    "delta_paper_probability",
    "delta_scissors_probability",
    "rock_prediction_fraction",
    "paper_prediction_fraction",
    "scissors_prediction_fraction",
)


__all__ = [
    "HistoryCentroidPolicy",
    "HistoryPolicyConfig",
    "HistoryTrainingClip",
    "feature_vector_from_rows",
    "fit_history_centroid_policy",
    "predict_history_label",
    "summarize_history_clip",
]
