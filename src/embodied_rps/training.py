"""Training and evaluation utilities for supervised skeleton RPS classifiers."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from embodied_rps.dataset import SyntheticDataset, build_observed_batch, load_synthetic_dataset
from embodied_rps.metrics import classification_metrics, select_best_run
from embodied_rps.models import RpsClassifier, build_classifier, parameter_count
from embodied_rps.training_types import ModelRunConfig


@dataclass(frozen=True)
class TrainedRun:
    """A trained model and the metrics written for that run."""

    model: RpsClassifier
    result: dict[str, object]


def load_sweep_config(path: Path) -> dict[str, object]:
    """Load a model sweep YAML file as a string-keyed dictionary."""

    with path.open("r", encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ValueError("model sweep config must be a mapping")
    parsed: dict[str, object] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise ValueError("model sweep config must use string keys")
        parsed[key] = value
    return parsed


def iter_model_run_configs(sweep_config: Mapping[str, object], requested_model: str) -> Iterable[ModelRunConfig]:
    """Expand model sweep YAML into run configs."""

    seeds = _int_list(_required(sweep_config, "seeds"), "seeds")
    models = _mapping(_required(sweep_config, "models"), "models")
    requested = tuple(models.keys()) if requested_model == "all" else (requested_model,)
    for model_name in requested:
        if model_name not in models:
            raise ValueError(f"Unknown model in sweep: {model_name}")
        model_config = _mapping(models[model_name], f"models.{model_name}")
        if model_name == "mlp":
            for seed in seeds:
                for hidden_size in _int_list(_required(model_config, "hidden_sizes"), "hidden_sizes"):
                    for dropout in _float_list(_required(model_config, "dropout"), "dropout"):
                        yield ModelRunConfig(model="mlp", seed=seed, hidden_size=hidden_size, dropout=dropout)
        elif model_name == "gru":
            for seed in seeds:
                for hidden_size in _int_list(_required(model_config, "hidden_sizes"), "hidden_sizes"):
                    for layers in _int_list(_required(model_config, "layers"), "layers"):
                        for dropout in _float_list(_required(model_config, "dropout"), "dropout"):
                            yield ModelRunConfig(
                                model="gru",
                                seed=seed,
                                hidden_size=hidden_size,
                                layers=layers,
                                dropout=dropout,
                            )
        elif model_name == "tcn":
            for seed in seeds:
                for channels in _int_list(_required(model_config, "channels"), "channels"):
                    for kernel_size in _int_list(_required(model_config, "kernel_sizes"), "kernel_sizes"):
                        for dropout in _float_list(_required(model_config, "dropout"), "dropout"):
                            yield ModelRunConfig(
                                model="tcn",
                                seed=seed,
                                hidden_size=channels,
                                kernel_size=kernel_size,
                                dropout=dropout,
                            )
        elif model_name == "transformer":
            for seed in seeds:
                for dim in _int_list(_required(model_config, "dims"), "dims"):
                    for heads in _int_list(_required(model_config, "heads"), "heads"):
                        for layers in _int_list(_required(model_config, "layers"), "layers"):
                            for dropout in _float_list(_required(model_config, "dropout"), "dropout"):
                                yield ModelRunConfig(
                                    model="transformer",
                                    seed=seed,
                                    hidden_size=dim,
                                    heads=heads,
                                    layers=layers,
                                    dropout=dropout,
                                )
        elif model_name == "stgcn":
            for seed in seeds:
                for channels in _int_list(_required(model_config, "channels"), "channels"):
                    for layers in _int_list(_required(model_config, "layers"), "layers"):
                        for kernel_size in _int_list(_required(model_config, "kernel_sizes"), "kernel_sizes"):
                            for dropout in _float_list(_required(model_config, "dropout"), "dropout"):
                                yield ModelRunConfig(
                                    model="stgcn",
                                    seed=seed,
                                    hidden_size=channels,
                                    layers=layers,
                                    kernel_size=kernel_size,
                                    dropout=dropout,
                                )
        else:
            raise ValueError(f"Unsupported model: {model_name}")


def train_model_runs(
    *,
    sweep_config: Mapping[str, object],
    requested_model: str,
    smoke: bool,
    max_runs: int | None,
) -> list[dict[str, object]]:
    """Train one or more model runs and write run metrics."""

    dataset_path = Path(_string_value(sweep_config, "dataset_path"))
    dataset = load_synthetic_dataset(dataset_path)
    runs_dir = Path(_string_value(sweep_config, "runs_dir"))
    runs_dir.mkdir(parents=True, exist_ok=True)
    device = _select_device(_string_value(sweep_config, "device"))
    epochs = _int_value(sweep_config, "smoke_epochs" if smoke else "epochs")
    batch_size = _int_value(sweep_config, "batch_size")
    learning_rate = _float_value(sweep_config, "learning_rate")
    ratios = _float_list(_required(sweep_config, "observation_ratios"), "observation_ratios")
    training_ratios = _float_list(
        sweep_config.get("training_observation_ratios", ratios),
        "training_observation_ratios",
    )

    completed: list[dict[str, object]] = []
    for index, run_config in enumerate(iter_model_run_configs(sweep_config, requested_model)):
        if max_runs is not None and index >= max_runs:
            break
        completed.append(
            train_single_run(
                dataset=dataset,
                dataset_path=dataset_path,
                run_config=run_config,
                runs_dir=runs_dir,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                observation_ratios=ratios,
                training_observation_ratios=training_ratios,
            )
        )
    return completed


def train_single_run(
    *,
    dataset: SyntheticDataset,
    dataset_path: Path | None = None,
    run_config: ModelRunConfig,
    runs_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    observation_ratios: Sequence[float],
    training_observation_ratios: Sequence[float] | None = None,
) -> dict[str, object]:
    """Train and evaluate one model run."""

    return train_single_run_with_model(
        dataset=dataset,
        dataset_path=dataset_path,
        run_config=run_config,
        runs_dir=runs_dir,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        observation_ratios=observation_ratios,
        training_observation_ratios=training_observation_ratios,
    ).result


def train_single_run_with_model(
    *,
    dataset: SyntheticDataset,
    dataset_path: Path | None = None,
    run_config: ModelRunConfig,
    runs_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    observation_ratios: Sequence[float],
    training_observation_ratios: Sequence[float] | None = None,
) -> TrainedRun:
    """Train and evaluate one model run while retaining the model object."""

    torch.manual_seed(run_config.seed)
    np.random.seed(run_config.seed)
    effective_training_ratios = tuple(training_observation_ratios) if training_observation_ratios is not None else tuple(observation_ratios)
    train_x, train_y = _observed_split_arrays(dataset, split_index=0, ratios=effective_training_ratios)
    model = build_classifier(
        run_config,
        input_dim=int(dataset.sequences.shape[2]),
        sequence_length=int(dataset.sequences.shape[1]),
        num_classes=len(dataset.label_names),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    metrics_by_ratio = evaluate_model(model, dataset, device=device, ratios=observation_ratios)
    latency_ms = measure_latency_ms(model, dataset, device=device, ratio=0.50)
    resolved_dataset_path = dataset_path if dataset_path is not None else Path("<in-memory>")
    run_result: dict[str, object] = {
        "run_id": run_config.run_id(),
        "model": run_config.model,
        "config": run_config.__dict__,
        "device": str(device),
        "dataset_path": resolved_dataset_path.as_posix(),
        "dataset_fingerprint": dataset_fingerprint(resolved_dataset_path),
        "label_names": list(dataset.label_names),
        "sequence_length": int(dataset.sequences.shape[1]),
        "feature_dim": int(dataset.sequences.shape[2]),
        "training_observation_ratios": [float(ratio) for ratio in effective_training_ratios],
        "evaluation_observation_ratios": [float(ratio) for ratio in observation_ratios],
        "epochs": epochs,
        "parameter_count": parameter_count(model),
        "latency_ms": latency_ms,
        "metrics": metrics_by_ratio,
    }
    run_dir = runs_dir / run_config.run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(run_result, indent=2), encoding="utf-8")
    return TrainedRun(model=model, result=run_result)


def evaluate_model(
    model: RpsClassifier,
    dataset: SyntheticDataset,
    *,
    device: torch.device,
    ratios: Sequence[float],
) -> dict[str, object]:
    """Evaluate a model on every requested observation ratio."""

    model.eval()
    results: dict[str, object] = {}
    test_mask = dataset.splits == 2
    labels = cast(NDArray[np.int64], dataset.labels[test_mask])
    with torch.no_grad():
        for ratio in ratios:
            observed = build_observed_batch(dataset.sequences[test_mask], ratio)
            logits = model(torch.from_numpy(observed).to(device))
            predictions = cast(NDArray[np.int64], logits.argmax(dim=1).cpu().numpy().astype(np.int64))
            metrics = classification_metrics(labels, predictions, num_classes=len(dataset.label_names))
            results[f"{ratio:.2f}"] = metrics.to_json()
    return results


def measure_latency_ms(
    model: RpsClassifier,
    dataset: SyntheticDataset,
    *,
    device: torch.device,
    ratio: float,
    repeats: int = 100,
) -> float:
    """Measure single-sample inference latency on the active device."""

    model.eval()
    test_mask = dataset.splits == 2
    observed = build_observed_batch(dataset.sequences[test_mask][:1], ratio)
    sample = torch.from_numpy(observed).to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / repeats


def write_model_comparison(runs_dir: Path, output_path: Path) -> dict[str, object]:
    """Aggregate saved model run metrics into a comparison report."""

    runs: list[dict[str, object]] = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid metrics file: {metrics_path}")
        parsed: dict[str, object] = {}
        for key, value in loaded.items():
            if not isinstance(key, str):
                raise ValueError(f"Invalid key in metrics file: {metrics_path}")
            parsed[key] = value
        runs.append(parsed)
    if len(runs) == 0:
        raise ValueError(f"No metrics files found under {runs_dir}")

    ratios = _metric_ratios(runs)
    best_30 = select_best_run(runs, ratio="0.30")
    best_50 = select_best_run(runs, ratio="0.50")
    comparison: dict[str, object] = {
        "num_runs": len(runs),
        "dataset_fingerprints": _unique_dataset_fingerprints(runs),
        "best_for_early_prediction_30": _summary(best_30, "0.30"),
        "best_for_clear_distinction_50": _summary(best_50, "0.50"),
        "best_for_clear_rps_50": _summary(_select_best_clear_rps_run(runs, ratio="0.50"), "0.50"),
        "best_by_ratio": {ratio: _summary(select_best_run(runs, ratio=ratio), ratio) for ratio in ratios},
        "best_clear_rps_by_ratio": {ratio: _summary(_select_best_clear_rps_run(runs, ratio=ratio), ratio) for ratio in ratios},
        "best_by_model_by_ratio": {
            ratio: {
                model_name: _summary(select_best_run(_runs_for_model(runs, model_name), ratio=ratio), ratio)
                for model_name in _model_names(runs)
            }
            for ratio in ratios
        },
        "runs_at_50": [_summary(run, "0.50") for run in runs],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def dataset_fingerprint(dataset_path: Path) -> dict[str, object]:
    """Return a stable file fingerprint for stale-result detection."""

    if str(dataset_path) == "<in-memory>":
        return {"path": "<in-memory>", "exists": False}
    if not dataset_path.exists():
        return {"path": dataset_path.as_posix(), "exists": False}
    digest = hashlib.sha256()
    with dataset_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = dataset_path.stat()
    return {
        "path": dataset_path.as_posix(),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _unique_dataset_fingerprints(runs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for run in runs:
        value = run.get("dataset_fingerprint")
        if not isinstance(value, Mapping):
            continue
        parsed = {str(key): item for key, item in value.items() if isinstance(key, str)}
        marker = json.dumps(parsed, sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(parsed)
    return unique


def _metric_ratios(runs: Sequence[Mapping[str, object]]) -> list[str]:
    metrics = _mapping(_required(runs[0], "metrics"), "metrics")
    return sorted(metrics.keys(), key=float)


def _model_names(runs: Sequence[Mapping[str, object]]) -> list[str]:
    return sorted({_string_value(run, "model") for run in runs})


def _runs_for_model(runs: Sequence[dict[str, object]], model_name: str) -> list[dict[str, object]]:
    return [run for run in runs if _string_value(run, "model") == model_name]


def _summary(run: Mapping[str, object], ratio: str) -> dict[str, object]:
    metrics = _mapping(_required(run, "metrics"), "metrics")
    ratio_metrics = _mapping(_required(metrics, ratio), ratio)
    clear_rps = _clear_rps_metrics(ratio_metrics)
    return {
        "run_id": _string_value(run, "run_id"),
        "model": _string_value(run, "model"),
        "ratio": ratio,
        "macro_f1": _float_value(ratio_metrics, "macro_f1"),
        "accuracy": _float_value(ratio_metrics, "accuracy"),
        "clear_rps_accuracy": clear_rps["accuracy"],
        "clear_rps_prediction_rate": clear_rps["prediction_rate"],
        "clear_rps_to_ambiguous_rate": clear_rps["to_ambiguous_rate"],
        "latency_ms": _float_value(run, "latency_ms"),
        "parameter_count": _int_value(run, "parameter_count"),
    }


def _select_best_clear_rps_run(runs: Sequence[dict[str, object]], *, ratio: str) -> dict[str, object]:
    return max(
        runs,
        key=lambda run: (
            _clear_rps_metrics(_mapping(_required(_mapping(_required(run, "metrics"), "metrics"), ratio), ratio))["accuracy"],
            -_float_value(run, "latency_ms"),
            -_int_value(run, "parameter_count"),
        ),
    )


def _clear_rps_metrics(ratio_metrics: Mapping[str, object]) -> dict[str, float]:
    matrix_value = _required(ratio_metrics, "confusion_matrix")
    if not isinstance(matrix_value, Sequence) or isinstance(matrix_value, (str, bytes)):
        raise ValueError("confusion_matrix must be a sequence")
    matrix: list[list[int]] = []
    for row in matrix_value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError("confusion_matrix rows must be sequences")
        parsed_row: list[int] = []
        for item in row:
            if not isinstance(item, int) or isinstance(item, bool):
                raise ValueError("confusion_matrix values must be integers")
            parsed_row.append(item)
        matrix.append(parsed_row)
    if len(matrix) < 3 or any(len(row) < 3 for row in matrix[:3]):
        return {"accuracy": 0.0, "prediction_rate": 0.0, "to_ambiguous_rate": 0.0}
    total_clear = sum(sum(matrix[row_index]) for row_index in range(3))
    if total_clear == 0:
        return {"accuracy": 0.0, "prediction_rate": 0.0, "to_ambiguous_rate": 0.0}
    correct_clear = sum(matrix[index][index] for index in range(3))
    clear_predictions = sum(matrix[row_index][column_index] for row_index in range(3) for column_index in range(3))
    clear_to_ambiguous = sum(matrix[row_index][3] for row_index in range(3)) if all(len(row) > 3 for row in matrix[:3]) else 0
    return {
        "accuracy": float(correct_clear) / float(total_clear),
        "prediction_rate": float(clear_predictions) / float(total_clear),
        "to_ambiguous_rate": float(clear_to_ambiguous) / float(total_clear),
    }


def _observed_split_arrays(
    dataset: SyntheticDataset,
    *,
    split_index: int,
    ratios: Sequence[float],
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    mask = dataset.splits == split_index
    sequences = cast(NDArray[np.float32], dataset.sequences[mask])
    labels = cast(NDArray[np.int64], dataset.labels[mask])
    observed_batches = [build_observed_batch(sequences, ratio) for ratio in ratios]
    train_x = cast(NDArray[np.float32], np.concatenate(observed_batches, axis=0).astype(np.float32))
    train_y = cast(NDArray[np.int64], np.tile(labels, len(ratios)).astype(np.int64))
    return train_x, train_y


def _select_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured)


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


def _int_value(mapping: Mapping[str, object], key: str) -> int:
    value = _required(mapping, key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _float_value(mapping: Mapping[str, object], key: str) -> float:
    value = _required(mapping, key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _int_list(value: object, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a list of integers")
    parsed: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{label} must contain integers")
        parsed.append(item)
    return parsed


def _float_list(value: object, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a list of numbers")
    parsed: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"{label} must contain numbers")
        parsed.append(float(item))
    return parsed
