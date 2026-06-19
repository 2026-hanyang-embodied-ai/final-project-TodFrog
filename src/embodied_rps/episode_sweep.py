"""Bundled threshold sweeps for the GRU episode policy runner."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import yaml

from embodied_rps.episode import EpisodePolicyConfig, EpisodePredictor, load_episode_policy_config, run_episode_policy


def load_threshold_sweep_config(
    path: Path,
) -> tuple[EpisodePolicyConfig, tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[int, ...], Path, Path]:
    """Load a GRU episode threshold sweep config."""

    with path.open("r", encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("threshold sweep config must be a mapping")
    policy_config_path = _string_value(loaded, "policy_config")
    thresholds = tuple(_float_sequence(_required(loaded, "thresholds"), "thresholds"))
    velocity_scales = tuple(_float_sequence(loaded.get("actuator_velocity_scales", [1.0]), "actuator_velocity_scales"))
    margins = tuple(_float_sequence(loaded.get("confidence_margins", [0.0]), "confidence_margins"))
    confirmation_counts = tuple(_int_sequence(loaded.get("confirmation_counts", [1]), "confirmation_counts"))
    output_path = Path(_string_value(loaded, "output_path"))
    log_dir = Path(_string_value(loaded, "log_dir"))
    return load_episode_policy_config(Path(policy_config_path)), thresholds, velocity_scales, margins, confirmation_counts, output_path, log_dir


def run_episode_threshold_sweep(
    *,
    base_config: EpisodePolicyConfig,
    thresholds: Sequence[float],
    output_path: Path,
    log_dir: Path,
    velocity_scales: Sequence[float] | None = None,
    confidence_margins: Sequence[float] | None = None,
    confirmation_counts: Sequence[int] | None = None,
    predictor: EpisodePredictor | None = None,
) -> dict[str, object]:
    """Run the episode policy across confidence thresholds and write an aggregate summary."""

    threshold_summaries: list[dict[str, object]] = []
    log_dir.mkdir(parents=True, exist_ok=True)
    scales = tuple(float(scale) for scale in (velocity_scales if velocity_scales is not None else (base_config.actuator_velocity_scale,)))
    margins = tuple(float(margin) for margin in (confidence_margins if confidence_margins is not None else (base_config.confidence_margin,)))
    confirmations = tuple(int(count) for count in (confirmation_counts if confirmation_counts is not None else (base_config.confirmation_count,)))
    include_scale_in_name = len(scales) > 1 or any(scale != base_config.actuator_velocity_scale for scale in scales)
    include_margin_in_name = len(margins) > 1 or any(margin != base_config.confidence_margin for margin in margins)
    include_confirmation_in_name = len(confirmations) > 1 or any(count != base_config.confirmation_count for count in confirmations)
    for velocity_scale in scales:
        for margin in margins:
            for confirmation_count in confirmations:
                for threshold in thresholds:
                    threshold_label = f"{threshold:.2f}"
                    scale_label = f"{velocity_scale:.2f}"
                    margin_label = f"{margin:.2f}"
                    name_parts = [f"episode_policy_threshold_{threshold_label}"]
                    if include_scale_in_name:
                        name_parts.append(f"velocity_{scale_label}")
                    if include_margin_in_name:
                        name_parts.append(f"margin_{margin_label}")
                    if include_confirmation_in_name:
                        name_parts.append(f"confirm_{confirmation_count}")
                    log_stem = "_".join(name_parts)
                    config = replace(
                        base_config,
                        confidence_threshold=float(threshold),
                        confidence_margin=float(margin),
                        confirmation_count=int(confirmation_count),
                        actuator_velocity_scale=float(velocity_scale),
                        log_path=log_dir / f"{log_stem}.jsonl",
                        summary_path=log_dir / f"{log_stem}_summary.json",
                    )
                    summary = run_episode_policy(config, predictor=predictor)
                    summary["actuator_velocity_scale"] = float(velocity_scale)
                    summary["confidence_margin"] = float(margin)
                    summary["confirmation_count"] = int(confirmation_count)
                    threshold_summaries.append(summary)
    aggregate: dict[str, object] = {
        "profile": base_config.model_profile,
        "profile_path": base_config.profile_path.as_posix(),
        "dataset_path": base_config.dataset_path.as_posix(),
        "thresholds": [float(threshold) for threshold in thresholds],
        "actuator_velocity_scales": [float(scale) for scale in scales],
        "confidence_margins": [float(margin) for margin in margins],
        "confirmation_counts": [int(count) for count in confirmations],
        "threshold_summaries": threshold_summaries,
        "best_by_actuator_feasible_win_rate": _best_summary(threshold_summaries, "actuator_feasible_win_rate"),
        "best_by_clear_actuator_feasible_win_rate": _best_summary(
            threshold_summaries,
            "clear_actuator_feasible_win_rate",
        ),
        "best_loss_free_by_clear_actuator_feasible_win_rate": _best_loss_free_summary(
            threshold_summaries,
            "clear_actuator_feasible_win_rate",
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def _best_summary(summaries: Sequence[dict[str, object]], key: str) -> dict[str, object] | None:
    if len(summaries) == 0:
        return None
    return max(summaries, key=lambda summary: _float_value(summary, key))


def _best_loss_free_summary(summaries: Sequence[dict[str, object]], key: str) -> dict[str, object] | None:
    loss_free = [summary for summary in summaries if _float_value(summary, "loss_rate") == 0.0]
    if len(loss_free) == 0:
        return None
    return _best_summary(loss_free, key)


def _required(mapping: dict[object, object], key: str) -> object:
    if key not in mapping:
        raise ValueError(f"Missing required key: {key}")
    return mapping[key]


def _string_value(mapping: dict[object, object], key: str) -> str:
    value = _required(mapping, key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _float_value(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _float_sequence(value: object, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    parsed: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"{label} must contain numbers")
        parsed.append(float(item))
    return parsed


def _int_sequence(value: object, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    parsed: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{label} must contain integers")
        if item <= 0:
            raise ValueError(f"{label} must contain positive integers")
        parsed.append(int(item))
    return parsed
