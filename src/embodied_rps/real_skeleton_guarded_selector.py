"""Guarded model selector over saved real-skeleton validation outputs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from embodied_rps.real_skeleton_model_disagreement import summarize_model_disagreement

SelectorSource = Literal["baseline", "candidate"]
FeatureSet = Literal["basic", "temporal"]


@dataclass(frozen=True)
class GuardedSelectorConfig:
    """Conservative nearest-centroid guard configuration."""

    observation_progress: float = 0.5
    positive_distance_multiplier: float = 1.0
    min_positive_radius: float = 0.25
    min_distance_margin: float = 0.0
    feature_set: FeatureSet = "basic"

    def __post_init__(self) -> None:
        if not 0.0 < self.observation_progress <= 1.0:
            raise ValueError("observation_progress must be in (0, 1]")
        if self.positive_distance_multiplier <= 0.0:
            raise ValueError("positive_distance_multiplier must be positive")
        if self.min_positive_radius < 0.0:
            raise ValueError("min_positive_radius must be non-negative")
        if self.min_distance_margin < 0.0:
            raise ValueError("min_distance_margin must be non-negative")
        if self.feature_set not in {"basic", "temporal"}:
            raise ValueError("feature_set must be basic or temporal")


@dataclass(frozen=True)
class GuardedSelector:
    """A fitted guard that selects baseline or candidate clip output."""

    feature_names: tuple[str, ...]
    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    positive_centroid: NDArray[np.float64]
    negative_centroid: NDArray[np.float64]
    positive_radius: float
    config: GuardedSelectorConfig
    train_clip_count: int
    positive_count: int
    negative_count: int

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable selector summary."""

        return {
            "feature_names": list(self.feature_names),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "positive_centroid": self.positive_centroid.tolist(),
            "negative_centroid": self.negative_centroid.tolist(),
            "positive_radius": self.positive_radius,
            "config": {
                "observation_progress": self.config.observation_progress,
                "positive_distance_multiplier": self.config.positive_distance_multiplier,
                "min_positive_radius": self.config.min_positive_radius,
                "min_distance_margin": self.config.min_distance_margin,
                "feature_set": self.config.feature_set,
            },
            "train_clip_count": self.train_clip_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }


def fit_guarded_selector(
    rows: Sequence[Mapping[str, object]],
    *,
    config: GuardedSelectorConfig,
) -> GuardedSelector:
    """Fit a nearest-centroid guard from non-held-out disagreement rows."""

    if len(rows) == 0:
        raise ValueError("rows must not be empty")
    positive_indices = [index for index, row in enumerate(rows) if row.get("disagreement_category") == "candidate_fix"]
    negative_indices = [index for index, row in enumerate(rows) if index not in positive_indices]
    if not positive_indices:
        raise ValueError("at least one candidate_fix row is required to fit the guard")
    if not negative_indices:
        raise ValueError("at least one non-candidate_fix row is required to fit the guard")

    feature_names = _feature_names_for_config(config)
    matrix = np.asarray([_feature_array(row, feature_names) for row in rows], dtype=np.float64)
    feature_mean = matrix.mean(axis=0)
    feature_scale = matrix.std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    normalized = (matrix - feature_mean) / feature_scale
    positive = normalized[positive_indices]
    negative = normalized[negative_indices]
    positive_centroid = positive.mean(axis=0)
    negative_centroid = negative.mean(axis=0)
    positive_distances = np.linalg.norm(positive - positive_centroid, axis=1)
    centroid_distance = float(np.linalg.norm(positive_centroid - negative_centroid))
    positive_radius = max(
        float(np.max(positive_distances)),
        float(config.min_positive_radius),
        centroid_distance * 0.5,
    )
    return GuardedSelector(
        feature_names=feature_names,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        positive_centroid=positive_centroid,
        negative_centroid=negative_centroid,
        positive_radius=positive_radius,
        config=config,
        train_clip_count=len(rows),
        positive_count=len(positive_indices),
        negative_count=len(negative_indices),
    )


def predict_selector_source(selector: GuardedSelector, row: Mapping[str, object]) -> tuple[SelectorSource, dict[str, float]]:
    """Predict whether the candidate output is close enough to the trusted-fix region."""

    features = _feature_array(row, selector.feature_names)
    normalized = (features - selector.feature_mean) / selector.feature_scale
    positive_distance = float(np.linalg.norm(normalized - selector.positive_centroid))
    negative_distance = float(np.linalg.norm(normalized - selector.negative_centroid))
    radius_limit = selector.positive_radius * selector.config.positive_distance_multiplier
    choose_candidate = (
        positive_distance <= radius_limit
        and positive_distance + selector.config.min_distance_margin <= negative_distance
    )
    diagnostics = {
        "guard_positive_distance": positive_distance,
        "guard_negative_distance": negative_distance,
        "guard_radius_limit": radius_limit,
    }
    return ("candidate" if choose_candidate else "baseline"), diagnostics


def evaluate_guarded_selector(
    selector: GuardedSelector,
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Apply a fitted selector to disagreement rows and summarize pass counts."""

    selected_rows: list[dict[str, object]] = []
    for row in rows:
        selected_source, diagnostics = predict_selector_source(selector, row)
        baseline_passed = bool(row.get("baseline_passed"))
        candidate_passed = bool(row.get("candidate_passed"))
        selected_passed = candidate_passed if selected_source == "candidate" else baseline_passed
        selected = dict(row)
        selected.update(diagnostics)
        selected["selected_source"] = selected_source
        selected["selected_passed"] = selected_passed
        selected["selected_failure_reason"] = (
            row.get("candidate_failure_reason") if selected_source == "candidate" else row.get("baseline_failure_reason")
        )
        selected_rows.append(selected)

    source_counts = Counter(str(row["selected_source"]) for row in selected_rows)
    category_counts = Counter(str(row.get("disagreement_category")) for row in selected_rows)
    failed_rows = [row for row in selected_rows if not bool(row.get("selected_passed"))]
    summary: dict[str, object] = {
        "eval_clip_count": len(selected_rows),
        "baseline_passed_clip_count": sum(1 for row in selected_rows if bool(row.get("baseline_passed"))),
        "candidate_passed_clip_count": sum(1 for row in selected_rows if bool(row.get("candidate_passed"))),
        "selected_passed_clip_count": sum(1 for row in selected_rows if bool(row.get("selected_passed"))),
        "candidate_selection_count": int(source_counts.get("candidate", 0)),
        "baseline_selection_count": int(source_counts.get("baseline", 0)),
        "selected_accuracy": _safe_rate(sum(1 for row in selected_rows if bool(row.get("selected_passed"))), len(selected_rows)),
        "selection_source_counts": dict(sorted(source_counts.items())),
        "disagreement_category_counts": dict(sorted(category_counts.items())),
        "selected_failed_clip_ids": [str(row.get("clip_id")) for row in failed_rows],
        "selected_failed_count": len(failed_rows),
    }
    return selected_rows, summary


def summarize_guarded_selector_experiment(
    *,
    train_pairs: Sequence[tuple[Path, Path]],
    eval_baseline_root: Path,
    eval_candidate_root: Path,
    config: GuardedSelectorConfig,
) -> tuple[list[dict[str, object]], dict[str, object], GuardedSelector]:
    """Fit a guard from train root pairs and evaluate it on an eval root pair."""

    train_rows: list[dict[str, object]] = []
    train_pair_summaries: list[dict[str, object]] = []
    for baseline_root, candidate_root in train_pairs:
        pair_rows, pair_summary = summarize_model_disagreement(
            baseline_root=baseline_root,
            candidate_root=candidate_root,
            observation_progress=config.observation_progress,
        )
        train_rows.extend(pair_rows)
        train_pair_summaries.append(pair_summary)

    selector = fit_guarded_selector(train_rows, config=config)
    eval_rows, eval_pair_summary = summarize_model_disagreement(
        baseline_root=eval_baseline_root,
        candidate_root=eval_candidate_root,
        observation_progress=config.observation_progress,
    )
    selected_rows, summary = evaluate_guarded_selector(selector, eval_rows)
    summary.update(
        {
            "train_clip_count": len(train_rows),
            "train_pair_count": len(train_pairs),
            "train_positive_count": selector.positive_count,
            "train_negative_count": selector.negative_count,
            "train_pair_summaries": train_pair_summaries,
            "eval_pair_summary": eval_pair_summary,
            "selector": selector.to_json_dict(),
        }
    )
    return selected_rows, summary, selector


def _feature_array(row: Mapping[str, object], feature_names: Sequence[str]) -> NDArray[np.float64]:
    return np.asarray([_number(row.get(name)) for name in feature_names], dtype=np.float64)


def _number(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


GUARD_FEATURE_NAMES: tuple[str, ...] = (
    "baseline_latest_rock_probability",
    "baseline_latest_paper_probability",
    "baseline_latest_scissors_probability",
    "candidate_latest_rock_probability",
    "candidate_latest_paper_probability",
    "candidate_latest_scissors_probability",
    "candidate_minus_baseline_latest_rock_probability",
    "candidate_minus_baseline_latest_paper_probability",
    "candidate_minus_baseline_latest_scissors_probability",
    "baseline_mean_rock_probability",
    "baseline_mean_paper_probability",
    "baseline_mean_scissors_probability",
    "candidate_mean_rock_probability",
    "candidate_mean_paper_probability",
    "candidate_mean_scissors_probability",
    "candidate_minus_baseline_mean_rock_probability",
    "candidate_minus_baseline_mean_paper_probability",
    "candidate_minus_baseline_mean_scissors_probability",
    "baseline_paper_prediction_fraction",
    "baseline_scissors_prediction_fraction",
    "candidate_paper_prediction_fraction",
    "candidate_scissors_prediction_fraction",
)

GUARD_TEMPORAL_FEATURE_NAMES: tuple[str, ...] = GUARD_FEATURE_NAMES + (
    "baseline_max_rock_probability",
    "baseline_max_paper_probability",
    "baseline_max_scissors_probability",
    "candidate_max_rock_probability",
    "candidate_max_paper_probability",
    "candidate_max_scissors_probability",
    "candidate_minus_baseline_max_rock_probability",
    "candidate_minus_baseline_max_paper_probability",
    "candidate_minus_baseline_max_scissors_probability",
    "baseline_delta_rock_probability",
    "baseline_delta_paper_probability",
    "baseline_delta_scissors_probability",
    "candidate_delta_rock_probability",
    "candidate_delta_paper_probability",
    "candidate_delta_scissors_probability",
    "candidate_minus_baseline_delta_rock_probability",
    "candidate_minus_baseline_delta_paper_probability",
    "candidate_minus_baseline_delta_scissors_probability",
)


def _feature_names_for_config(config: GuardedSelectorConfig) -> tuple[str, ...]:
    if config.feature_set == "temporal":
        return GUARD_TEMPORAL_FEATURE_NAMES
    return GUARD_FEATURE_NAMES


__all__ = [
    "FeatureSet",
    "GUARD_FEATURE_NAMES",
    "GUARD_TEMPORAL_FEATURE_NAMES",
    "GuardedSelector",
    "GuardedSelectorConfig",
    "evaluate_guarded_selector",
    "fit_guarded_selector",
    "predict_selector_source",
    "summarize_guarded_selector_experiment",
]
