"""Validation sweeps for MiniRocket-style and ST-GCN skeleton baselines."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from embodied_rps.dataset import load_synthetic_dataset
from embodied_rps.metrics import select_best_run
from embodied_rps.minirocket import evaluate_minirocket_config, load_minirocket_configs
from embodied_rps.training import (
    _float_list,
    _float_value,
    _int_value,
    _mapping,
    _required,
    _select_device,
    iter_model_run_configs,
    train_single_run,
)


def load_validation_sweep_config(path: Path) -> dict[str, object]:
    """Load a validation sweep YAML file."""

    with path.open("r", encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ValueError("validation sweep config must be a mapping")
    parsed: dict[str, object] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise ValueError("validation sweep config must use string keys")
        parsed[key] = value
    return parsed


def run_validation_sweeps(config: Mapping[str, object]) -> dict[str, object]:
    """Run MiniRocket-style and ST-GCN validation sweeps from config."""

    dataset = load_synthetic_dataset(Path(_string_value(config, "dataset_path")))
    output_dir = Path(_string_value(config, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = _float_list(_required(config, "observation_ratios"), "observation_ratios")
    seeds = _int_list(_required(config, "seeds"), "seeds")
    runs: list[dict[str, object]] = []

    if _bool_value(config, "run_minirocket", default=True):
        minirocket_root = _mapping(_required(config, "minirocket"), "minirocket")
        for run_config in load_minirocket_configs(minirocket_root, seeds):
            run = evaluate_minirocket_config(dataset, config=run_config, observation_ratios=ratios)
            _write_run(output_dir / "minirocket" / str(run["run_id"]) / "metrics.json", run)
            runs.append(run)

    if _bool_value(config, "run_stgcn", default=True):
        stgcn_root = _mapping(_required(config, "stgcn"), "stgcn")
        sweep_config: dict[str, object] = {
            "seeds": seeds,
            "models": {"stgcn": stgcn_root},
        }
        for stgcn_run_config in iter_model_run_configs(sweep_config, requested_model="stgcn"):
            run = train_single_run(
                dataset=dataset,
                run_config=stgcn_run_config,
                runs_dir=output_dir / "stgcn",
                device=_select_device(_string_value(config, "device")),
                epochs=_int_value(config, "epochs"),
                batch_size=_int_value(config, "batch_size"),
                learning_rate=_float_value(config, "learning_rate"),
                observation_ratios=ratios,
            )
            runs.append(run)

    ratio_labels = tuple(f"{ratio:.2f}" for ratio in ratios)
    summary = summarize_validation_runs(runs, ratios=ratio_labels)
    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize_validation_runs(runs: Sequence[dict[str, object]], *, ratios: Sequence[str]) -> dict[str, object]:
    """Summarize validation runs by model and observation ratio."""

    model_names = sorted({_string_from_mapping(run, "model") for run in runs})
    return {
        "num_runs": len(runs),
        "models": model_names,
        "ratios": list(ratios),
        "best_by_ratio": {
            ratio: _summary(select_best_run(list(runs), ratio=ratio), ratio) for ratio in ratios if len(runs) > 0
        },
        "best_by_model_by_ratio": {
            ratio: {
                model_name: _summary(select_best_run(_runs_for_model(runs, model_name), ratio=ratio), ratio)
                for model_name in model_names
            }
            for ratio in ratios
        },
        "runs": [_compact_run(run) for run in runs],
    }


def _write_run(path: Path, run: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2), encoding="utf-8")


def _runs_for_model(runs: Sequence[dict[str, object]], model_name: str) -> list[dict[str, object]]:
    return [run for run in runs if _string_from_mapping(run, "model") == model_name]


def _summary(run: Mapping[str, object], ratio: str) -> dict[str, object]:
    metrics = _mapping(_required(run, "metrics"), "metrics")
    ratio_metrics = _mapping(_required(metrics, ratio), ratio)
    return {
        "run_id": _string_from_mapping(run, "run_id"),
        "model": _string_from_mapping(run, "model"),
        "ratio": ratio,
        "macro_f1": _float_value(ratio_metrics, "macro_f1"),
        "accuracy": _float_value(ratio_metrics, "accuracy"),
        "latency_ms": _float_value(run, "latency_ms"),
        "parameter_count": _int_value(run, "parameter_count"),
    }


def _compact_run(run: Mapping[str, object]) -> dict[str, object]:
    return {
        "run_id": _string_from_mapping(run, "run_id"),
        "model": _string_from_mapping(run, "model"),
        "latency_ms": _float_value(run, "latency_ms"),
        "parameter_count": _int_value(run, "parameter_count"),
    }


def _string_value(mapping: Mapping[str, object], key: str) -> str:
    value = _required(mapping, key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _string_from_mapping(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _int_list(value: object, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    parsed: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{label} must contain integers")
        parsed.append(item)
    return parsed


def _bool_value(mapping: Mapping[str, object], key: str, *, default: bool) -> bool:
    value = mapping.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value
