"""Dependency-free MiniRocket-style validation baseline for RPS trajectories."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from embodied_rps.dataset import SyntheticDataset, build_observed_batch
from embodied_rps.metrics import classification_metrics


@dataclass(frozen=True)
class MiniRocketConfig:
    """One MiniRocket-style deterministic convolutional feature run."""

    seed: int
    num_kernels: int
    alpha: float
    kernel_sizes: tuple[int, ...]
    dilations: tuple[int, ...]

    def run_id(self) -> str:
        """Return a stable run identifier."""

        return f"minirocket_seed{self.seed}_k{self.num_kernels}_alpha{self.alpha:g}"


@dataclass(frozen=True)
class RocketKernel:
    """One deterministic 1D multivariate convolution kernel."""

    channels: NDArray[np.int64]
    weights: NDArray[np.float32]
    dilation: int


@dataclass(frozen=True)
class FittedMiniRocket:
    """A fitted MiniRocket-style feature classifier."""

    config: MiniRocketConfig
    kernels: tuple[RocketKernel, ...]
    feature_mean: NDArray[np.float32]
    feature_std: NDArray[np.float32]
    weights: NDArray[np.float32]
    label_count: int


def evaluate_minirocket_config(
    dataset: SyntheticDataset,
    *,
    config: MiniRocketConfig,
    observation_ratios: Sequence[float],
) -> dict[str, object]:
    """Fit on the training split and evaluate each requested observation ratio."""

    train_mask = dataset.splits == 0
    test_mask = dataset.splits == 2
    train_sequences = cast(NDArray[np.float32], dataset.sequences[train_mask])
    train_labels = cast(NDArray[np.int64], dataset.labels[train_mask])
    test_sequences = cast(NDArray[np.float32], dataset.sequences[test_mask])
    test_labels = cast(NDArray[np.int64], dataset.labels[test_mask])
    fitted = fit_minirocket_classifier(
        train_sequences,
        train_labels,
        config=config,
        label_count=len(dataset.label_names),
    )

    metrics_by_ratio: dict[str, object] = {}
    for ratio in observation_ratios:
        observed = build_observed_batch(test_sequences, ratio)
        predictions = predict_minirocket(fitted, observed)
        metrics = classification_metrics(test_labels, predictions, num_classes=len(dataset.label_names))
        metrics_by_ratio[f"{ratio:.2f}"] = metrics.to_json()

    latency_ms = measure_minirocket_latency_ms(fitted, test_sequences[:1])
    return {
        "run_id": config.run_id(),
        "model": "minirocket",
        "config": {
            "seed": config.seed,
            "num_kernels": config.num_kernels,
            "alpha": config.alpha,
            "kernel_sizes": list(config.kernel_sizes),
            "dilations": list(config.dilations),
        },
        "parameter_count": int(fitted.weights.size),
        "latency_ms": latency_ms,
        "metrics": metrics_by_ratio,
    }


def fit_minirocket_classifier(
    sequences: NDArray[np.float32],
    labels: NDArray[np.int64],
    *,
    config: MiniRocketConfig,
    label_count: int,
) -> FittedMiniRocket:
    """Fit deterministic convolutional features with a ridge one-vs-rest head."""

    kernels = _make_kernels(config, feature_count=int(sequences.shape[2]))
    features = _extract_features(sequences, kernels)
    mean = cast(NDArray[np.float32], features.mean(axis=0).astype(np.float32))
    std = cast(NDArray[np.float32], features.std(axis=0).astype(np.float32))
    std = cast(NDArray[np.float32], np.where(std < 1e-6, 1.0, std).astype(np.float32))
    normalized = cast(NDArray[np.float32], ((features - mean) / std).astype(np.float32))
    design = _with_bias(normalized)
    targets = np.full((labels.shape[0], label_count), -1.0, dtype=np.float32)
    for row, label in enumerate(labels.tolist()):
        targets[row, int(label)] = 1.0
    regularizer = np.eye(design.shape[1], dtype=np.float32) * float(config.alpha)
    regularizer[-1, -1] = 0.0
    lhs = design.T @ design + regularizer
    rhs = design.T @ targets
    weights = cast(NDArray[np.float32], np.linalg.solve(lhs, rhs).astype(np.float32))
    return FittedMiniRocket(
        config=config,
        kernels=kernels,
        feature_mean=mean,
        feature_std=std,
        weights=weights,
        label_count=label_count,
    )


def predict_minirocket(fitted: FittedMiniRocket, sequences: NDArray[np.float32]) -> NDArray[np.int64]:
    """Predict integer labels for sequences."""

    features = _extract_features(sequences, fitted.kernels)
    normalized = cast(NDArray[np.float32], ((features - fitted.feature_mean) / fitted.feature_std).astype(np.float32))
    scores = _with_bias(normalized) @ fitted.weights
    return cast(NDArray[np.int64], np.argmax(scores, axis=1).astype(np.int64))


def measure_minirocket_latency_ms(fitted: FittedMiniRocket, single_sequence: NDArray[np.float32], repeats: int = 50) -> float:
    """Measure single-sample inference latency in milliseconds."""

    if int(single_sequence.shape[0]) != 1:
        raise ValueError("single_sequence must contain exactly one sample")
    for _ in range(3):
        _ = predict_minirocket(fitted, single_sequence)
    start = time.perf_counter()
    for _ in range(repeats):
        _ = predict_minirocket(fitted, single_sequence)
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / repeats


def load_minirocket_configs(root: Mapping[str, object], seeds: Sequence[int]) -> list[MiniRocketConfig]:
    """Expand MiniRocket-style validation sweep config."""

    configs: list[MiniRocketConfig] = []
    num_kernels = _int_list(_required(root, "num_kernels"), "minirocket.num_kernels")
    alphas = _float_list(_required(root, "alphas"), "minirocket.alphas")
    kernel_sizes = tuple(_int_list(_required(root, "kernel_sizes"), "minirocket.kernel_sizes"))
    dilations = tuple(_int_list(_required(root, "dilations"), "minirocket.dilations"))
    for seed in seeds:
        for kernel_count in num_kernels:
            for alpha in alphas:
                configs.append(
                    MiniRocketConfig(
                        seed=seed,
                        num_kernels=kernel_count,
                        alpha=alpha,
                        kernel_sizes=kernel_sizes,
                        dilations=dilations,
                    )
                )
    return configs


def _make_kernels(config: MiniRocketConfig, *, feature_count: int) -> tuple[RocketKernel, ...]:
    rng = np.random.default_rng(config.seed)
    kernels: list[RocketKernel] = []
    for _ in range(config.num_kernels):
        kernel_size = int(rng.choice(np.asarray(config.kernel_sizes, dtype=np.int64)))
        dilation = int(rng.choice(np.asarray(config.dilations, dtype=np.int64)))
        channel_count = int(rng.integers(1, min(4, feature_count) + 1))
        channels = cast(NDArray[np.int64], rng.choice(feature_count, size=channel_count, replace=False).astype(np.int64))
        weights = cast(NDArray[np.float32], rng.normal(0.0, 1.0, size=(channel_count, kernel_size)).astype(np.float32))
        weights = cast(NDArray[np.float32], (weights - weights.mean(axis=1, keepdims=True)).astype(np.float32))
        kernels.append(RocketKernel(channels=channels, weights=weights, dilation=dilation))
    return tuple(kernels)


def _extract_features(sequences: NDArray[np.float32], kernels: Sequence[RocketKernel]) -> NDArray[np.float32]:
    features = np.zeros((int(sequences.shape[0]), len(kernels) * 2), dtype=np.float32)
    for row in range(int(sequences.shape[0])):
        sequence = sequences[row]
        for kernel_index, kernel in enumerate(kernels):
            response = _convolve_sequence(sequence, kernel)
            features[row, kernel_index * 2] = float(np.mean(response > 0.0))
            features[row, kernel_index * 2 + 1] = float(np.max(response))
    return features


def _convolve_sequence(sequence: NDArray[np.float32], kernel: RocketKernel) -> NDArray[np.float32]:
    time_length = int(sequence.shape[0])
    kernel_size = int(kernel.weights.shape[1])
    receptive = (kernel_size - 1) * kernel.dilation + 1
    if receptive > time_length:
        return np.asarray([0.0], dtype=np.float32)
    response_length = time_length - receptive + 1
    responses = np.zeros(response_length, dtype=np.float32)
    selected = sequence[:, kernel.channels]
    for offset in range(response_length):
        total = 0.0
        for kernel_step in range(kernel_size):
            frame = offset + kernel_step * kernel.dilation
            total += float(np.dot(selected[frame, :], kernel.weights[:, kernel_step]))
        responses[offset] = total
    return responses


def _with_bias(features: NDArray[np.float32]) -> NDArray[np.float32]:
    ones = np.ones((features.shape[0], 1), dtype=np.float32)
    return cast(NDArray[np.float32], np.concatenate([features, ones], axis=1).astype(np.float32))


def _required(mapping: Mapping[str, object], key: str) -> object:
    if key not in mapping:
        raise ValueError(f"Missing required key: {key}")
    return mapping[key]


def _int_list(value: object, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    parsed: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{label} must contain integers")
        parsed.append(item)
    return parsed


def _float_list(value: object, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    parsed: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"{label} must contain numbers")
        parsed.append(float(item))
    return parsed
