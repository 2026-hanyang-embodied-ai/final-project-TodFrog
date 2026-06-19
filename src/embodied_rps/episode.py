"""Actuator-feasible episode policy runner for synthetic skeleton RPS data."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from embodied_rps.checkpoints import LoadedModelProfile, load_model_profile
from embodied_rps.config import KinematicConfig, load_kinematic_config
from embodied_rps.dataset import SyntheticDataset, build_observed_batch, load_synthetic_dataset, observation_length
from embodied_rps.domain import ActuatorLimits
from embodied_rps.feasibility import check_actuator_feasibility
from embodied_rps.policy import ConfidenceGate, CounterMovePolicy, OpponentGesture, PredictionResult

EpisodeRecord: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class EpisodePolicyConfig:
    """Configuration for classifier-to-actuator episode evaluation."""

    model_profile: str
    profile_path: Path
    dataset_path: Path
    hand_config_path: Path
    split: str
    confidence_threshold: float
    confidence_margin: float
    confirmation_count: int
    observation_ratios: tuple[float, ...]
    deadline_s: float
    actuator_velocity_scale: float
    response_delay_s: float | None
    log_path: Path
    summary_path: Path
    max_episodes: int | None = None
    device: str = "auto"


class EpisodePredictor(Protocol):
    """Predict one gesture from one episode sequence at incremental observation ratios."""

    def predict(self, sequence: NDArray[np.float32], observation_ratios: tuple[float, ...]) -> PredictionResult | None:
        """Return the first confident prediction, or abstain."""


class ModelEpisodePredictor:
    """Run a saved PyTorch profile with a confidence gate."""

    def __init__(
        self,
        loaded_profile: LoadedModelProfile,
        gate: ConfidenceGate,
        device: torch.device,
        confirmation_count: int = 1,
    ) -> None:
        if confirmation_count <= 0:
            raise ValueError("confirmation_count must be positive")
        self._loaded_profile = loaded_profile
        self._gate = gate
        self._device = device
        self._confirmation_count = confirmation_count

    def predict(self, sequence: NDArray[np.float32], observation_ratios: tuple[float, ...]) -> PredictionResult | None:
        """Evaluate observation ratios in order and stop at the first confident prediction."""

        confirmed_label: str | None = None
        confirmed_count = 0
        with torch.no_grad():
            for ratio in observation_ratios:
                observed = build_observed_batch(sequence[np.newaxis, :, :], ratio)
                logits = self._loaded_profile.model(torch.from_numpy(observed).to(self._device))
                probability_tensor = torch.softmax(logits, dim=1)[0].detach().cpu()
                probabilities = [float(probability_tensor[index].item()) for index in range(int(probability_tensor.shape[0]))]
                decision_frame = observation_length(int(sequence.shape[0]), ratio)
                result = self._gate.decide(
                    probabilities,
                    self._loaded_profile.metadata.label_names,
                    observation_ratio=ratio,
                    decision_frame=decision_frame,
                )
                if result is None:
                    confirmed_label = None
                    confirmed_count = 0
                    continue
                if result.predicted_gesture == confirmed_label:
                    confirmed_count += 1
                else:
                    confirmed_label = result.predicted_gesture
                    confirmed_count = 1
                if confirmed_count >= self._confirmation_count:
                    return result
        return None


def load_episode_policy_config(path: Path) -> EpisodePolicyConfig:
    """Load and validate an episode policy YAML config."""

    with path.open("r", encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle)
    root = _mapping(loaded, "episode policy config")
    return EpisodePolicyConfig(
        model_profile=_string_value(root, "model_profile"),
        profile_path=Path(_string_value(root, "profile_path")),
        dataset_path=Path(_string_value(root, "dataset_path")),
        hand_config_path=Path(_string_value(root, "hand_config_path")),
        split=_string_value(root, "split"),
        confidence_threshold=_probability(_float_value(root, "confidence_threshold"), "confidence_threshold"),
        confidence_margin=_probability(_optional_float_value(root, "confidence_margin", default=0.0), "confidence_margin"),
        confirmation_count=_positive_int(_optional_int(root, "confirmation_count", default=1), "confirmation_count"),
        observation_ratios=tuple(_ratio_list(_required(root, "observation_ratios"), "observation_ratios")),
        deadline_s=_positive_float(_float_value(root, "deadline_s"), "deadline_s"),
        actuator_velocity_scale=_positive_float(
            _float_value(root, "actuator_velocity_scale"),
            "actuator_velocity_scale",
        ),
        response_delay_s=_optional_non_negative_float(root, "response_delay_s"),
        log_path=Path(_string_value(root, "log_path")),
        summary_path=Path(_string_value(root, "summary_path")),
        max_episodes=_optional_positive_int(root, "max_episodes"),
        device=_optional_string(root, "device", default="auto"),
    )


def run_episode_policy(config: EpisodePolicyConfig, predictor: EpisodePredictor | None = None) -> dict[str, object]:
    """Run the configured synthetic episodes and write JSONL logs plus summary metrics."""

    dataset = load_synthetic_dataset(config.dataset_path)
    hand_config = load_kinematic_config(config.hand_config_path)
    effective_predictor = predictor
    latency_ms: float | None = None
    if effective_predictor is None:
        device = _select_device(config.device)
        loaded_profile = load_model_profile(config.profile_path, device=device)
        effective_predictor = ModelEpisodePredictor(
            loaded_profile,
            ConfidenceGate(config.confidence_threshold, min_margin=config.confidence_margin),
            device,
            confirmation_count=config.confirmation_count,
        )
        latency_ms = _optional_float(loaded_profile.metadata.metrics, "latency_ms")

    records = _run_episode_records(
        config=config,
        dataset=dataset,
        hand_config=hand_config,
        predictor=effective_predictor,
        latency_ms=latency_ms,
    )
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    with config.log_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary = summarize_episode_records(
        records,
        model_profile=config.model_profile,
        confidence_threshold=config.confidence_threshold,
        observation_ratios=config.observation_ratios,
        latency_ms=latency_ms,
    )
    summary["actuator_velocity_scale"] = config.actuator_velocity_scale
    summary["deadline_s"] = config.deadline_s
    summary["response_delay_s"] = config.response_delay_s
    summary["confidence_margin"] = config.confidence_margin
    summary["confirmation_count"] = config.confirmation_count
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)
    config.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize_episode_records(
    records: Sequence[Mapping[str, object]],
    *,
    model_profile: str,
    confidence_threshold: float,
    observation_ratios: tuple[float, ...],
    latency_ms: float | None,
) -> dict[str, object]:
    """Summarize JSONL-compatible episode records."""

    total = len(records)
    result_counts = Counter(_record_result(record) for record in records)
    clear_records = [record for record in records if _is_clear_rps_label(record.get("true_gesture"))]
    ambiguous_records = [record for record in records if record.get("true_gesture") == "ambiguous"]
    non_abstain = [record for record in records if _record_result(record) != "abstain"]
    prediction_correct = sum(1 for record in non_abstain if record.get("raw_prediction_correct") is True)
    clear_wins = sum(1 for record in clear_records if _record_result(record) == "win")
    ambiguous_abstains = sum(1 for record in ambiguous_records if _record_result(record) == "abstain")
    decision_ratios = [_number(record.get("observation_ratio")) for record in non_abstain]
    response_times = [_number(record.get("actuator_response_time_s")) for record in non_abstain]
    failure_counts = Counter(
        reason
        for reason in (_string_or_none(record.get("failure_reason")) for record in records)
        if reason is not None
    )
    return {
        "model_profile": model_profile,
        "total_episodes": total,
        "clear_episode_count": len(clear_records),
        "ambiguous_episode_count": len(ambiguous_records),
        "confidence_threshold": confidence_threshold,
        "observation_ratios": list(observation_ratios),
        "model_latency_ms": latency_ms,
        "raw_prediction_accuracy": _safe_rate(prediction_correct, len(non_abstain)),
        "average_confidence_crossing_observation_ratio": _mean(decision_ratios),
        "average_actuator_response_time_s": _mean(response_times),
        "result_counts": dict(sorted(result_counts.items())),
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "actuator_feasible_win_rate": _safe_rate(result_counts["win"], total),
        "abstention_rate": _safe_rate(result_counts["abstain"], total),
        "infeasible_rate": _safe_rate(result_counts["infeasible"], total),
        "loss_rate": _safe_rate(result_counts["loss"], total),
        "threshold_latency_settings": {
            "confidence_threshold": confidence_threshold,
            "model_latency_ms": latency_ms,
        },
        "clear_actuator_feasible_win_rate": _safe_rate(clear_wins, len(clear_records)),
        "ambiguous_abstention_rate": _safe_rate(ambiguous_abstains, len(ambiguous_records)) if len(ambiguous_records) > 0 else None,
    }


def summarize_episode_log(
    *,
    log_path: Path,
    out_path: Path,
    model_profile: str | None = None,
    confidence_threshold: float | None = None,
    latency_ms: float | None = None,
) -> dict[str, object]:
    """Load a JSONL episode log and write a summary report."""

    records: list[EpisodeRecord] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue
        loaded: object = json.loads(line)
        records.append(dict(_mapping(loaded, "episode log record")))
    inferred_profile = model_profile if model_profile is not None else _infer_string(records, "model_profile", "unknown")
    inferred_threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else _optional_float(records[0], "confidence_threshold")
        if len(records) > 0
        else 0.0
    )
    inferred_latency = latency_ms if latency_ms is not None else _optional_float(records[0], "model_latency_ms") if len(records) > 0 else None
    ratio_values: list[float] = []
    for record in records:
        ratio = _number(record.get("observation_ratio"))
        if ratio is not None:
            ratio_values.append(ratio)
    configured_ratios = _optional_number_tuple(records[0], "configured_observation_ratios") if len(records) > 0 else None
    ratios = configured_ratios if configured_ratios is not None else tuple(sorted(set(ratio_values)))
    summary = summarize_episode_records(
        records,
        model_profile=inferred_profile,
        confidence_threshold=0.0 if inferred_threshold is None else inferred_threshold,
        observation_ratios=ratios,
        latency_ms=inferred_latency,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _run_episode_records(
    *,
    config: EpisodePolicyConfig,
    dataset: SyntheticDataset,
    hand_config: KinematicConfig,
    predictor: EpisodePredictor,
    latency_ms: float | None,
) -> list[EpisodeRecord]:
    split_index = _split_index(dataset, config.split)
    indices = [int(index) for index in np.where(dataset.splits == split_index)[0]]
    if config.max_episodes is not None:
        indices = indices[: config.max_episodes]
    limits = _scaled_actuator_limits(hand_config, config)
    counter_policy = CounterMovePolicy()
    records: list[EpisodeRecord] = []
    for episode_number, dataset_index in enumerate(indices):
        sequence = cast(NDArray[np.float32], dataset.sequences[dataset_index])
        true_label = str(dataset.label_names[int(dataset.labels[dataset_index])])
        if not _is_clear_rps_label(true_label):
            records.append(_non_rps_ground_truth_record(config, episode_number, dataset_index, true_label, latency_ms))
            continue
        true_gesture = _opponent_gesture(true_label)
        prediction = predictor.predict(sequence, config.observation_ratios)
        if prediction is None:
            records.append(_abstain_record(config, episode_number, dataset_index, true_gesture, latency_ms))
            continue
        selected_counter = counter_policy.counter(prediction.predicted_gesture)
        remaining_time_s = max(0.0, config.deadline_s * (1.0 - prediction.observation_ratio))
        feasibility = check_actuator_feasibility(
            current_pose=hand_config.gestures["neutral"],
            target_pose=hand_config.gestures[selected_counter],
            limits=limits,
            remaining_time_s=remaining_time_s,
        )
        raw_correct = prediction.predicted_gesture == true_gesture
        if not feasibility.feasible:
            result = "infeasible"
            failure_reason = feasibility.failure_reason
        elif raw_correct:
            result = "win"
            failure_reason = None
        else:
            result = "loss"
            failure_reason = "prediction_wrong"
        records.append(
            {
                "episode_id": episode_number,
                "dataset_index": dataset_index,
                "model_profile": config.model_profile,
                "model_latency_ms": latency_ms,
                "configured_observation_ratios": list(config.observation_ratios),
                "true_gesture": true_gesture,
                "predicted_gesture": prediction.predicted_gesture,
                "raw_prediction_correct": raw_correct,
                "confidence": prediction.confidence,
                "confidence_margin": prediction.confidence_margin,
                "confidence_threshold": config.confidence_threshold,
                "confidence_margin_threshold": config.confidence_margin,
                "confirmation_count": config.confirmation_count,
                "observation_ratio": prediction.observation_ratio,
                "decision_frame": prediction.decision_frame,
                "selected_counter_move": selected_counter,
                "remaining_time_s": remaining_time_s,
                "actuator_response_time_s": feasibility.required_time_s,
                "limiting_joint": feasibility.limiting_joint,
                "feasible": feasibility.feasible,
                "result": result,
                "failure_reason": failure_reason,
            }
        )
    return records


def _abstain_record(
    config: EpisodePolicyConfig,
    episode_number: int,
    dataset_index: int,
    true_gesture: str,
    latency_ms: float | None,
) -> EpisodeRecord:
    return {
        "episode_id": episode_number,
        "dataset_index": dataset_index,
        "model_profile": config.model_profile,
        "model_latency_ms": latency_ms,
        "configured_observation_ratios": list(config.observation_ratios),
        "true_gesture": true_gesture,
        "predicted_gesture": None,
        "raw_prediction_correct": False,
        "confidence": None,
        "confidence_margin": None,
        "confidence_threshold": config.confidence_threshold,
        "confidence_margin_threshold": config.confidence_margin,
        "confirmation_count": config.confirmation_count,
        "observation_ratio": None,
        "decision_frame": None,
        "selected_counter_move": None,
        "remaining_time_s": None,
        "actuator_response_time_s": None,
        "limiting_joint": None,
        "feasible": None,
        "result": "abstain",
        "failure_reason": "low_confidence",
    }


def _non_rps_ground_truth_record(
    config: EpisodePolicyConfig,
    episode_number: int,
    dataset_index: int,
    true_label: str,
    latency_ms: float | None,
) -> EpisodeRecord:
    """Record an abstention for non-clear-RPS labels such as ambiguous."""

    return {
        "episode_id": episode_number,
        "dataset_index": dataset_index,
        "model_profile": config.model_profile,
        "model_latency_ms": latency_ms,
        "configured_observation_ratios": list(config.observation_ratios),
        "true_gesture": true_label,
        "predicted_gesture": None,
        "raw_prediction_correct": False,
        "confidence": None,
        "confidence_margin": None,
        "confidence_threshold": config.confidence_threshold,
        "confidence_margin_threshold": config.confidence_margin,
        "confirmation_count": config.confirmation_count,
        "observation_ratio": None,
        "decision_frame": None,
        "selected_counter_move": None,
        "remaining_time_s": None,
        "actuator_response_time_s": None,
        "limiting_joint": None,
        "feasible": None,
        "result": "abstain",
        "failure_reason": "ambiguous_ground_truth",
    }


def _scaled_actuator_limits(hand_config: KinematicConfig, config: EpisodePolicyConfig) -> ActuatorLimits:
    response_delay_s = hand_config.response_delay_s if config.response_delay_s is None else config.response_delay_s
    return ActuatorLimits(
        velocity_limits_rad_s={
            joint_name: limit * config.actuator_velocity_scale
            for joint_name, limit in hand_config.velocity_limits_rad_s.items()
        },
        response_delay_s=response_delay_s,
    )


def _split_index(dataset: SyntheticDataset, split_name: str) -> int:
    for index, name in enumerate(dataset.split_names):
        if name == split_name:
            return index
    raise ValueError(f"Unknown dataset split: {split_name}")


def _opponent_gesture(value: str) -> OpponentGesture:
    if value == "rock":
        return "rock"
    if value == "paper":
        return "paper"
    if value == "scissors":
        return "scissors"
    raise ValueError(f"Expected rock, paper, or scissors, got {value}")


def _is_clear_rps_label(value: object) -> bool:
    return value in ("rock", "paper", "scissors")


def _select_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured)


def _record_result(record: Mapping[str, object]) -> str:
    value = record.get("result")
    if isinstance(value, str):
        return value
    raise ValueError("episode record missing string result")


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mean(values: Sequence[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if len(filtered) == 0:
        return None
    return float(sum(filtered) / len(filtered))


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"Expected string or None, got {value}")


def _infer_string(records: Sequence[Mapping[str, object]], key: str, default: str) -> str:
    if len(records) == 0:
        return default
    value = records[0].get(key)
    return value if isinstance(value, str) and value != "" else default


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    parsed: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} must use string keys")
        parsed[key] = item
    return parsed


def _required(mapping: Mapping[str, object], key: str) -> object:
    if key not in mapping:
        raise ValueError(f"Missing required key: {key}")
    return mapping[key]


def _string_value(mapping: Mapping[str, object], key: str) -> str:
    value = _required(mapping, key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(mapping: Mapping[str, object], key: str, *, default: str) -> str:
    if key not in mapping:
        return default
    return _string_value(mapping, key)


def _optional_int(mapping: Mapping[str, object], key: str, *, default: int) -> int:
    if key not in mapping:
        return default
    value = mapping[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _float_value(mapping: Mapping[str, object], key: str) -> float:
    value = _required(mapping, key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _optional_float(mapping: Mapping[str, object], key: str, *, default: float | None = None) -> float | None:
    value = mapping.get(key)
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value)


def _optional_float_value(mapping: Mapping[str, object], key: str, *, default: float) -> float:
    value = mapping.get(key)
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _optional_number_tuple(mapping: Mapping[str, object], key: str) -> tuple[float, ...] | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    parsed: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            return None
        parsed.append(float(item))
    return tuple(parsed)


def _positive_float(value: float, label: str) -> float:
    if value <= 0.0:
        raise ValueError(f"{label} must be positive")
    return value


def _positive_int(value: int, label: str) -> int:
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _probability(value: float, label: str) -> float:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return value


def _optional_non_negative_float(mapping: Mapping[str, object], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number or null")
    parsed = float(value)
    if parsed < 0.0:
        raise ValueError(f"{key} must be non-negative")
    return parsed


def _optional_positive_int(mapping: Mapping[str, object], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer or null")
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _ratio_list(value: object, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    ratios: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"{label} must contain numbers")
        ratios.append(_probability(float(item), label))
    if len(ratios) == 0:
        raise ValueError(f"{label} must not be empty")
    return ratios
