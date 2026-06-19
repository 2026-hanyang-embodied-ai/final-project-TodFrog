"""Checkpoint export and loading for supervised RPS classifiers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import torch

from embodied_rps.dataset import load_synthetic_dataset
from embodied_rps.models import RpsClassifier, build_classifier
from embodied_rps.training import (
    _float_list,
    _float_value,
    _int_value,
    _required,
    _select_device,
    _string_value,
    iter_model_run_configs,
    load_sweep_config,
    train_single_run_with_model,
)
from embodied_rps.training_types import ModelName, ModelRunConfig

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class ModelProfileMetadata:
    """Metadata needed to reconstruct a saved classifier."""

    profile: str
    run_id: str
    checkpoint_path: Path
    dataset_path: Path
    label_names: tuple[str, ...]
    input_dim: int
    sequence_length: int
    num_classes: int
    run_config: ModelRunConfig
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class LoadedModelProfile:
    """A reconstructed classifier plus its metadata."""

    metadata: ModelProfileMetadata
    model: RpsClassifier


def save_model_profile(
    *,
    model: RpsClassifier,
    profile_dir: Path,
    profile: str,
    run_id: str,
    run_config: ModelRunConfig,
    dataset_path: Path,
    label_names: Sequence[str],
    input_dim: int,
    sequence_length: int,
    num_classes: int,
    metrics: Mapping[str, object],
) -> Path:
    """Save model weights and JSON metadata for a named inference profile."""

    profile_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = profile_dir / f"{profile}.pt"
    metadata_path = profile_dir / f"{profile}.json"
    torch.save(model.state_dict(), checkpoint_path)
    metadata: dict[str, JsonValue] = {
        "profile": profile,
        "run_id": run_id,
        "checkpoint_path": checkpoint_path.as_posix(),
        "dataset_path": dataset_path.as_posix(),
        "label_names": [str(label) for label in label_names],
        "input_dim": int(input_dim),
        "sequence_length": int(sequence_length),
        "num_classes": int(num_classes),
        "run_config": _run_config_json(run_config),
        "metrics": _json_ready_mapping(metrics),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def load_model_profile(metadata_path: Path, *, device: torch.device) -> LoadedModelProfile:
    """Load a saved inference profile and reconstruct the classifier."""

    metadata = _load_metadata(metadata_path)
    model = build_classifier(
        metadata.run_config,
        input_dim=metadata.input_dim,
        sequence_length=metadata.sequence_length,
        num_classes=metadata.num_classes,
    )
    loaded_state: object = torch.load(metadata.checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(loaded_state, Mapping):
        raise ValueError(f"Checkpoint must contain a state dict: {metadata.checkpoint_path}")
    state_dict = cast(Mapping[str, torch.Tensor], _string_keyed_mapping(loaded_state, "checkpoint state dict"))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return LoadedModelProfile(metadata=metadata, model=model)


def find_model_run_config(sweep_config: Mapping[str, object], run_id: str) -> ModelRunConfig:
    """Find one expanded model run config by stable run id."""

    for run_config in iter_model_run_configs(sweep_config, "all"):
        if run_config.run_id() == run_id:
            return run_config
    raise ValueError(f"No configured run_id found: {run_id}")


def export_model_profile(*, config_path: Path, run_id: str, profile: str) -> Path:
    """Retrain one configured run and save a loadable model profile."""

    sweep_config = load_sweep_config(config_path)
    dataset_path = Path(_string_value(sweep_config, "dataset_path"))
    dataset = load_synthetic_dataset(dataset_path)
    runs_dir = Path(_string_value(sweep_config, "runs_dir"))
    run_config = find_model_run_config(sweep_config, run_id)
    trained = train_single_run_with_model(
        dataset=dataset,
        run_config=run_config,
        runs_dir=runs_dir,
        device=_select_device(_string_value(sweep_config, "device")),
        epochs=_int_value(sweep_config, "epochs"),
        batch_size=_int_value(sweep_config, "batch_size"),
        learning_rate=_float_value(sweep_config, "learning_rate"),
        observation_ratios=_float_list(_required(sweep_config, "observation_ratios"), "observation_ratios"),
        training_observation_ratios=_float_list(
            sweep_config.get("training_observation_ratios", _required(sweep_config, "observation_ratios")),
            "training_observation_ratios",
        ),
    )
    return save_model_profile(
        model=trained.model,
        profile_dir=Path("results/model_profiles"),
        profile=profile,
        run_id=run_id,
        run_config=run_config,
        dataset_path=dataset_path,
        label_names=dataset.label_names,
        input_dim=int(dataset.sequences.shape[2]),
        sequence_length=int(dataset.sequences.shape[1]),
        num_classes=len(dataset.label_names),
        metrics=trained.result,
    )


def _load_metadata(metadata_path: Path) -> ModelProfileMetadata:
    loaded: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    root = _mapping(loaded, "model profile")
    run_config_root = _mapping(_required(root, "run_config"), "run_config")
    checkpoint_path = Path(_string_value(root, "checkpoint_path"))
    if not checkpoint_path.is_absolute():
        checkpoint_path = metadata_path.parent / checkpoint_path.name
    return ModelProfileMetadata(
        profile=_string_value(root, "profile"),
        run_id=_string_value(root, "run_id"),
        checkpoint_path=checkpoint_path,
        dataset_path=Path(_string_value(root, "dataset_path")),
        label_names=_string_tuple(_required(root, "label_names"), "label_names"),
        input_dim=_int_value(root, "input_dim"),
        sequence_length=_int_value(root, "sequence_length"),
        num_classes=_int_value(root, "num_classes"),
        run_config=ModelRunConfig(
            model=_model_name(_string_value(run_config_root, "model")),
            seed=_int_value(run_config_root, "seed"),
            hidden_size=_int_value(run_config_root, "hidden_size"),
            dropout=_float_value(run_config_root, "dropout"),
            layers=_int_value(run_config_root, "layers"),
            heads=_int_value(run_config_root, "heads"),
            kernel_size=_int_value(run_config_root, "kernel_size"),
        ),
        metrics=_mapping(_required(root, "metrics"), "metrics"),
    )


def _run_config_json(config: ModelRunConfig) -> dict[str, JsonValue]:
    return {
        "model": config.model,
        "seed": config.seed,
        "hidden_size": config.hidden_size,
        "dropout": config.dropout,
        "layers": config.layers,
        "heads": config.heads,
        "kernel_size": config.kernel_size,
    }


def _json_ready_mapping(mapping: Mapping[str, object]) -> dict[str, JsonValue]:
    return {key: _json_ready(value) for key, value in mapping.items()}


def _json_ready(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        parsed: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON metadata mappings must use string keys")
            parsed[key] = _json_ready(item)
        return parsed
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_ready(item) for item in value]
    return str(value)


def _model_name(value: str) -> ModelName:
    if value in ("mlp", "gru", "tcn", "transformer", "stgcn"):
        return cast(ModelName, value)
    raise ValueError(f"Unsupported model name: {value}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    parsed: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} must use string keys")
        parsed[key] = item
    return parsed


def _string_keyed_mapping(value: Mapping[object, object], label: str) -> Mapping[str, object]:
    parsed: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} must use string keys")
        parsed[key] = item
    return parsed


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "":
            raise ValueError(f"{label} must contain non-empty strings")
        parsed.append(item)
    return tuple(parsed)
