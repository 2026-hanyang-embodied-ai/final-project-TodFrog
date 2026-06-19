"""Classification metrics and model selection helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ClassificationMetrics:
    """Core metrics for one RPS classification evaluation."""

    accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serializable metrics dictionary."""

        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "confusion_matrix": self.confusion_matrix,
        }


def classification_metrics(
    labels: NDArray[np.int64],
    predictions: NDArray[np.int64],
    *,
    num_classes: int,
) -> ClassificationMetrics:
    """Compute accuracy, macro F1, and confusion matrix without sklearn."""

    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, prediction in zip(labels.tolist(), predictions.tolist(), strict=True):
        confusion[int(label), int(prediction)] += 1

    accuracy = float(np.trace(confusion) / max(1, int(confusion.sum())))
    f1_scores: list[float] = []
    for class_index in range(num_classes):
        true_positive = float(confusion[class_index, class_index])
        false_positive = float(confusion[:, class_index].sum() - confusion[class_index, class_index])
        false_negative = float(confusion[class_index, :].sum() - confusion[class_index, class_index])
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0.0 else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0.0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
        f1_scores.append(f1)

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=float(np.mean(np.asarray(f1_scores, dtype=np.float64))),
        confusion_matrix=confusion.astype(int).tolist(),
    )


def select_best_run(runs: Sequence[dict[str, object]], *, ratio: str) -> dict[str, object]:
    """Select the best run by macro F1, then latency, then parameter count."""

    if len(runs) == 0:
        raise ValueError("runs must not be empty")

    def key(run: dict[str, object]) -> tuple[float, float, int]:
        metrics = _mapping_value(run, "metrics")
        ratio_metrics = _mapping_value(metrics, ratio)
        macro_f1 = _float_value(ratio_metrics, "macro_f1")
        latency = _float_value(run, "latency_ms")
        parameter_count = _int_value(run, "parameter_count")
        return (macro_f1, -latency, -parameter_count)

    return max(runs, key=key)


def _mapping_value(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    parsed: dict[str, object] = {}
    for nested_key, nested_value in value.items():
        if not isinstance(nested_key, str):
            raise TypeError(f"{key} must use string keys")
        parsed[nested_key] = nested_value
    return parsed


def _float_value(mapping: Mapping[str, object], key: str) -> float:
    value = mapping[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def _int_value(mapping: Mapping[str, object], key: str) -> int:
    value = mapping[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return value
