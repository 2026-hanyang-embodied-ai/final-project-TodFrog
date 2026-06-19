"""Two-stage skeleton prediction helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def combine_two_stage_probabilities(
    *,
    stage1_labels: Sequence[str],
    stage1_probabilities: Sequence[float],
    stage2_labels: Sequence[str],
    stage2_probabilities: Sequence[float],
) -> dict[str, float]:
    """Combine stage1 rock/transition and stage2 paper/scissors probabilities."""

    stage1 = _probability_mapping(stage1_labels, stage1_probabilities)
    stage2 = _probability_mapping(stage2_labels, stage2_probabilities)
    if set(stage1) != {"rock", "transition"}:
        raise ValueError("stage1 labels must be exactly rock and transition")
    if set(stage2) != {"paper", "scissors"}:
        raise ValueError("stage2 labels must be exactly paper and scissors")
    transition_mass = float(stage1["transition"])
    return {
        "rock_probability": float(stage1["rock"]),
        "paper_probability": transition_mass * float(stage2["paper"]),
        "scissors_probability": transition_mass * float(stage2["scissors"]),
        "transition_mass": transition_mass,
    }


def two_stage_frame_row(
    *,
    frame_index: int,
    time_s: float,
    clip_progress: float,
    model_progress: float,
    detected: bool,
    stage1_labels: Sequence[str],
    stage1_probabilities: Sequence[float],
    stage2_labels: Sequence[str],
    stage2_probabilities: Sequence[float],
) -> dict[str, object]:
    """Build one strict-evaluator-compatible row from two-stage outputs."""

    if not detected:
        return {
            "frame_index": int(frame_index),
            "time_s": float(time_s),
            "detected": False,
            "prediction": None,
            "rock_probability": 0.0,
            "paper_probability": 0.0,
            "scissors_probability": 0.0,
            "transition_mass": 0.0,
            "confidence": 0.0,
            "confidence_margin": 0.0,
            "clip_progress": float(clip_progress),
            "model_progress": float(model_progress),
        }
    combined = combine_two_stage_probabilities(
        stage1_labels=stage1_labels,
        stage1_probabilities=stage1_probabilities,
        stage2_labels=stage2_labels,
        stage2_probabilities=stage2_probabilities,
    )
    probabilities = {
        "rock": combined["rock_probability"],
        "paper": combined["paper_probability"],
        "scissors": combined["scissors_probability"],
    }
    prediction = max(probabilities.items(), key=lambda item: item[1])[0]
    ordered = sorted(probabilities.values(), reverse=True)
    confidence = float(ordered[0])
    margin = confidence - float(ordered[1]) if len(ordered) > 1 else confidence
    return {
        "frame_index": int(frame_index),
        "time_s": float(time_s),
        "detected": True,
        "prediction": prediction,
        **combined,
        "confidence": confidence,
        "confidence_margin": margin,
        "clip_progress": float(clip_progress),
        "model_progress": float(model_progress),
    }


def validate_two_stage_label_names(stage1_labels: Sequence[str], stage2_labels: Sequence[str]) -> None:
    """Validate exported profile labels for the two-stage predictor contract."""

    if set(str(label) for label in stage1_labels) != {"rock", "transition"}:
        raise ValueError("stage1 profile labels must be exactly rock and transition")
    if set(str(label) for label in stage2_labels) != {"paper", "scissors"}:
        raise ValueError("stage2 profile labels must be exactly paper and scissors")


def _probability_mapping(labels: Sequence[str], probabilities: Sequence[float]) -> Mapping[str, float]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if len(labels) == 0:
        raise ValueError("labels must not be empty")
    mapping = {str(label): float(probability) for label, probability in zip(labels, probabilities, strict=True)}
    total = sum(mapping.values())
    if total <= 0.0:
        raise ValueError("probabilities must have positive total mass")
    return {label: probability / total for label, probability in mapping.items()}


__all__ = ["combine_two_stage_probabilities", "two_stage_frame_row", "validate_two_stage_label_names"]
