"""Prediction confidence gating and RPS counter-move policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

OpponentGesture: TypeAlias = Literal["rock", "paper", "scissors"]


@dataclass(frozen=True)
class PredictionResult:
    """A classifier decision made at one partial-observation point."""

    predicted_gesture: OpponentGesture
    confidence: float
    observation_ratio: float
    decision_frame: int
    confidence_margin: float = 0.0


class CounterMovePolicy:
    """Map an opponent gesture to the RPS gesture that beats it."""

    def counter(self, opponent_gesture: OpponentGesture) -> OpponentGesture:
        """Return the winning response for the predicted opponent gesture."""

        if opponent_gesture == "rock":
            return "paper"
        if opponent_gesture == "paper":
            return "scissors"
        if opponent_gesture == "scissors":
            return "rock"
        raise ValueError(f"Unsupported opponent gesture: {opponent_gesture}")


class ConfidenceGate:
    """Act only when the top class probability reaches the configured threshold."""

    def __init__(self, threshold: float, min_margin: float = 0.0) -> None:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if min_margin < 0.0 or min_margin > 1.0:
            raise ValueError("min_margin must be in [0, 1]")
        self.threshold = threshold
        self.min_margin = min_margin

    def decide(
        self,
        probabilities: Sequence[float],
        label_names: Sequence[str],
        *,
        observation_ratio: float,
        decision_frame: int,
    ) -> PredictionResult | None:
        """Return a prediction only if the maximum probability clears the threshold."""

        if len(probabilities) == 0:
            raise ValueError("probabilities must not be empty")
        if len(probabilities) != len(label_names):
            raise ValueError("probabilities and label_names must have the same length")
        sorted_indices = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)
        best_index = sorted_indices[0]
        confidence = float(probabilities[best_index])
        runner_up = float(probabilities[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = confidence - runner_up
        if confidence < self.threshold:
            return None
        if margin < self.min_margin:
            return None
        predicted_label = label_names[best_index]
        if predicted_label not in ("rock", "paper", "scissors"):
            return None
        return PredictionResult(
            predicted_gesture=_opponent_gesture(predicted_label),
            confidence=confidence,
            observation_ratio=float(observation_ratio),
            decision_frame=int(decision_frame),
            confidence_margin=margin,
        )


def _opponent_gesture(value: str) -> OpponentGesture:
    if value in ("rock", "paper", "scissors"):
        return cast(OpponentGesture, value)
    raise ValueError(f"Expected a non-neutral RPS gesture, got {value}")
