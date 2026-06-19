"""Synthetic skeleton/joint trajectory generation for supervised RPS learning."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import yaml
from numpy.typing import NDArray

from embodied_rps.config import ConfigError, KinematicConfig
from embodied_rps.domain import Gesture

LABEL_NAMES: tuple[Gesture, ...] = ("rock", "paper", "scissors")
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """Configuration for the synthetic skeleton trajectory dataset."""

    seed: int
    episodes_per_class: int
    sequence_length: int
    duration_s_range: tuple[float, float]
    train_fraction: float
    val_fraction: float
    test_fraction: float
    noise_std_range: tuple[float, float]
    finger_delay_s_range: tuple[float, float]
    hesitation_probability: float
    output_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class SyntheticDataset:
    """Generated synthetic RPS trajectory tensors and metadata."""

    sequences: NDArray[np.float32]
    positions: NDArray[np.float32]
    velocities: NDArray[np.float32]
    labels: NDArray[np.int64]
    splits: NDArray[np.int64]
    label_names: tuple[Gesture, ...]
    split_names: tuple[str, ...]
    joint_names: tuple[str, ...]


def load_synthetic_dataset_config(path: Path) -> SyntheticDatasetConfig:
    """Load and validate synthetic dataset generation config."""

    with path.open("r", encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle)
    root = _as_mapping(loaded, "dataset config")

    train_fraction = _positive_fraction(_required_float(root, "train_fraction"), "train_fraction")
    val_fraction = _positive_fraction(_required_float(root, "val_fraction"), "val_fraction")
    test_fraction = _positive_fraction(_required_float(root, "test_fraction"), "test_fraction")
    if not math.isclose(train_fraction + val_fraction + test_fraction, 1.0, abs_tol=1e-6):
        raise ConfigError("train_fraction + val_fraction + test_fraction must equal 1.0")

    return SyntheticDatasetConfig(
        seed=_required_int(root, "seed"),
        episodes_per_class=_positive_int(_required_int(root, "episodes_per_class"), "episodes_per_class"),
        sequence_length=_positive_int(_required_int(root, "sequence_length"), "sequence_length"),
        duration_s_range=_range_pair(_required(root, "duration_s_range"), "duration_s_range"),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        noise_std_range=_range_pair(_required(root, "noise_std_range"), "noise_std_range"),
        finger_delay_s_range=_range_pair(_required(root, "finger_delay_s_range"), "finger_delay_s_range"),
        hesitation_probability=_probability(
            _required_float(root, "hesitation_probability"),
            "hesitation_probability",
        ),
        output_path=Path(_required_string(root, "output_path")),
        metadata_path=Path(_required_string(root, "metadata_path")),
    )


def generate_synthetic_dataset(
    config: SyntheticDatasetConfig,
    hand_config: KinematicConfig,
) -> SyntheticDataset:
    """Generate balanced synthetic neutral-to-RPS joint trajectories."""

    rng = np.random.default_rng(config.seed)
    joint_names = hand_config.joint_names
    neutral = _pose_vector(hand_config, "neutral")
    positions: list[NDArray[np.float32]] = []
    velocities: list[NDArray[np.float32]] = []
    labels: list[int] = []
    splits: list[int] = []

    for label_index, gesture in enumerate(LABEL_NAMES):
        split_ids = _balanced_split_ids(config.episodes_per_class, config)
        for episode_index in range(config.episodes_per_class):
            target = _pose_vector(hand_config, gesture)
            duration_s = rng.uniform(config.duration_s_range[0], config.duration_s_range[1])
            trajectory = _generate_single_trajectory(
                rng=rng,
                neutral=neutral,
                target=target,
                sequence_length=config.sequence_length,
                duration_s=duration_s,
                noise_std_range=config.noise_std_range,
                finger_delay_s_range=config.finger_delay_s_range,
                hesitation_probability=config.hesitation_probability,
            )
            positions.append(trajectory)
            velocities.append(_compute_velocities(trajectory, duration_s))
            labels.append(label_index)
            splits.append(split_ids[episode_index])

    position_array = cast(NDArray[np.float32], np.stack(positions).astype(np.float32))
    velocity_array = cast(NDArray[np.float32], np.stack(velocities).astype(np.float32))
    sequence_array = cast(NDArray[np.float32], np.concatenate([position_array, velocity_array], axis=2))
    label_array = cast(NDArray[np.int64], np.asarray(labels, dtype=np.int64))
    split_array = cast(NDArray[np.int64], np.asarray(splits, dtype=np.int64))

    return SyntheticDataset(
        sequences=sequence_array,
        positions=position_array,
        velocities=velocity_array,
        labels=label_array,
        splits=split_array,
        label_names=LABEL_NAMES,
        split_names=SPLIT_NAMES,
        joint_names=joint_names,
    )


def save_synthetic_dataset(dataset: SyntheticDataset, config: SyntheticDatasetConfig) -> None:
    """Save generated dataset tensors and metadata."""

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output_path,
        sequences=dataset.sequences,
        positions=dataset.positions,
        velocities=dataset.velocities,
        labels=dataset.labels,
        splits=dataset.splits,
        label_names=np.asarray(dataset.label_names),
        split_names=np.asarray(dataset.split_names),
        joint_names=np.asarray(dataset.joint_names),
    )
    metadata = {
        "config": _json_ready_config(config),
        "num_samples": int(dataset.labels.shape[0]),
        "sequence_length": int(dataset.sequences.shape[1]),
        "feature_dim": int(dataset.sequences.shape[2]),
        "label_names": list(dataset.label_names),
        "split_names": list(dataset.split_names),
        "joint_names": list(dataset.joint_names),
    }
    config.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_synthetic_dataset(path: Path) -> SyntheticDataset:
    """Load a saved synthetic RPS dataset."""

    loaded = np.load(path, allow_pickle=False)
    return SyntheticDataset(
        sequences=cast(NDArray[np.float32], loaded["sequences"].astype(np.float32)),
        positions=cast(NDArray[np.float32], loaded["positions"].astype(np.float32)),
        velocities=cast(NDArray[np.float32], loaded["velocities"].astype(np.float32)),
        labels=cast(NDArray[np.int64], loaded["labels"].astype(np.int64)),
        splits=cast(NDArray[np.int64], loaded["splits"].astype(np.int64)),
        label_names=tuple(cast(Sequence[Gesture], loaded["label_names"].tolist())),
        split_names=tuple(cast(Sequence[str], loaded["split_names"].tolist())),
        joint_names=tuple(cast(Sequence[str], loaded["joint_names"].tolist())),
    )


def observation_length(sequence_length: int, ratio: float) -> int:
    """Return the number of observed frames for a partial observation ratio."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError("ratio must be in (0, 1]")
    return max(1, min(sequence_length, int(math.ceil(sequence_length * ratio))))


def build_observed_batch(sequences: NDArray[np.float32], ratio: float) -> NDArray[np.float32]:
    """Keep only the observed prefix and pad the rest with the last observed frame."""

    length = observation_length(int(sequences.shape[1]), ratio)
    observed = sequences.copy()
    if length < sequences.shape[1]:
        observed[:, length:, :] = sequences[:, length - 1 : length, :]
    return observed


def _generate_single_trajectory(
    *,
    rng: np.random.Generator,
    neutral: NDArray[np.float32],
    target: NDArray[np.float32],
    sequence_length: int,
    duration_s: float,
    noise_std_range: tuple[float, float],
    finger_delay_s_range: tuple[float, float],
    hesitation_probability: float,
) -> NDArray[np.float32]:
    times = np.linspace(0.0, duration_s, sequence_length, dtype=np.float32)
    trajectory: NDArray[np.float32] = np.zeros((sequence_length, neutral.shape[0]), dtype=np.float32)
    for joint_index in range(neutral.shape[0]):
        delay_s = float(rng.uniform(finger_delay_s_range[0], finger_delay_s_range[1]))
        denominator = max(duration_s - delay_s, 1e-6)
        progress = np.clip((times - delay_s) / denominator, 0.0, 1.0)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        if rng.random() < hesitation_probability:
            center = float(rng.uniform(0.35, 0.65))
            width = float(rng.uniform(0.05, 0.12))
            amplitude = float(rng.uniform(0.04, 0.11))
            hesitation = amplitude * np.exp(-((progress - center) ** 2) / (2.0 * width * width))
            smooth = np.clip(smooth - hesitation, 0.0, 1.0)
        trajectory[:, joint_index] = neutral[joint_index] + (target[joint_index] - neutral[joint_index]) * smooth

    noise_std = float(rng.uniform(noise_std_range[0], noise_std_range[1]))
    if noise_std > 0.0:
        noise = cast(NDArray[np.float32], rng.normal(0.0, noise_std, size=trajectory.shape).astype(np.float32))
        trajectory = cast(NDArray[np.float32], trajectory + noise)
    trajectory = cast(NDArray[np.float32], np.clip(trajectory, 0.0, 1.05))
    trajectory[0, :] = neutral
    trajectory[-1, :] = target
    return trajectory.astype(np.float32)


def _compute_velocities(positions: NDArray[np.float32], duration_s: float) -> NDArray[np.float32]:
    frame_dt = duration_s / max(1, positions.shape[0] - 1)
    velocities = np.zeros_like(positions, dtype=np.float32)
    velocities[1:, :] = (positions[1:, :] - positions[:-1, :]) / frame_dt
    velocities[0, :] = velocities[1, :]
    return velocities


def _balanced_split_ids(num_samples: int, config: SyntheticDatasetConfig) -> NDArray[np.int64]:
    train_count = int(num_samples * config.train_fraction)
    val_count = int(num_samples * config.val_fraction)
    test_count = num_samples - train_count - val_count
    split_ids = np.asarray([0] * train_count + [1] * val_count + [2] * test_count, dtype=np.int64)
    return split_ids


def _pose_vector(hand_config: KinematicConfig, gesture: Gesture) -> NDArray[np.float32]:
    values = [hand_config.gestures[gesture].joints[joint_name] for joint_name in hand_config.joint_names]
    return cast(NDArray[np.float32], np.asarray(values, dtype=np.float32))


def _json_ready_config(config: SyntheticDatasetConfig) -> dict[str, object]:
    raw = asdict(config)
    raw["output_path"] = str(config.output_path)
    raw["metadata_path"] = str(config.metadata_path)
    return raw


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    parsed: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError(f"{label} must use string keys")
        parsed[key] = item
    return parsed


def _required(mapping: Mapping[str, object], key: str) -> object:
    if key not in mapping:
        raise ConfigError(f"Missing required dataset config key: {key}")
    return mapping[key]


def _required_int(mapping: Mapping[str, object], key: str) -> int:
    value = _required(mapping, key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer")
    return value


def _required_float(mapping: Mapping[str, object], key: str) -> float:
    value = _required(mapping, key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{key} must be a number")
    return float(value)


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = _required(mapping, key)
    if not isinstance(value, str) or value == "":
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _range_pair(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ConfigError(f"{label} must be a two-item range")
    low, high = value
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise ConfigError(f"{label} must contain numbers")
    parsed = (float(low), float(high))
    if parsed[0] < 0.0 or parsed[1] < parsed[0]:
        raise ConfigError(f"{label} must be non-negative and ordered")
    return parsed


def _positive_int(value: int, label: str) -> int:
    if value <= 0:
        raise ConfigError(f"{label} must be positive")
    return value


def _positive_fraction(value: float, label: str) -> float:
    if value <= 0.0 or value >= 1.0:
        raise ConfigError(f"{label} must be in (0, 1)")
    return value


def _probability(value: float, label: str) -> float:
    if value < 0.0 or value > 1.0:
        raise ConfigError(f"{label} must be in [0, 1]")
    return value
