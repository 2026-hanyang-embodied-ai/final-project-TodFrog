"""Training utilities for real MediaPipe skeleton final-gesture prediction."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from embodied_rps.metrics import classification_metrics, select_best_run
from embodied_rps.models import RpsClassifier, build_classifier, parameter_count
from embodied_rps.training import _select_device, iter_model_run_configs, load_sweep_config
from embodied_rps.training_types import ModelRunConfig

BINARY_FINAL_GESTURE_LABELS: Mapping[str, int] = {"paper": 0, "scissors": 1}
THREE_CLASS_GESTURE_LABELS: Mapping[str, int] = {"rock": 0, "paper": 1, "scissors": 2}
ROCK_TRANSITION_LABELS: Mapping[str, int] = {"rock": 0, "transition": 1}
FINAL_GESTURE_LABELS: Mapping[str, int] = BINARY_FINAL_GESTURE_LABELS
SPLIT_LABELS: Mapping[str, int] = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class RealSkeletonDataset:
    """Real skeleton shard dataset after feature extraction."""

    sequences: NDArray[np.float32]
    labels: NDArray[np.int64]
    splits: NDArray[np.int64]
    lengths: NDArray[np.int64]
    mask: NDArray[np.bool_]
    sample_ids: tuple[str, ...]
    source_shards: tuple[str, ...]
    label_names: tuple[str, ...] = ("paper", "scissors")
    split_names: tuple[str, ...] = ("train", "val", "test")
    feature_name: str = "landmark_velocity_126"


@dataclass(frozen=True)
class CalibrationAnchorDataset:
    """Small non-held-out real calibration anchor set for domain alignment."""

    sequences: NDArray[np.float32]
    labels: NDArray[np.int64]
    lengths: NDArray[np.int64]
    mask: NDArray[np.bool_]
    sample_ids: tuple[str, ...]
    source_paths: tuple[str, ...]
    label_names: tuple[str, ...]
    feature_name: str = "landmark_velocity_126"


@dataclass(frozen=True)
class RealTrainedRun:
    """A trained real-skeleton model and its serialized metrics."""

    model: RpsClassifier
    result: dict[str, object]


def load_real_skeleton_dataset(
    dataset_root: Path,
    *,
    feature_name: str = "landmark_velocity_126",
) -> RealSkeletonDataset:
    """Load sharded real skeleton NPZ files into the model feature contract."""

    if feature_name != "landmark_velocity_126":
        raise ValueError(f"Unsupported real skeleton feature: {feature_name}")
    shards_root = dataset_root / "shards"
    if not shards_root.exists():
        raise FileNotFoundError(f"Missing real skeleton shard root: {shards_root}")

    all_sequences: list[NDArray[np.float32]] = []
    all_targets: list[list[str]] = []
    all_splits: list[NDArray[np.int64]] = []
    all_lengths: list[NDArray[np.int64]] = []
    all_masks: list[NDArray[np.bool_]] = []
    sample_ids: list[str] = []
    source_shards: list[str] = []

    for split_name, split_index in SPLIT_LABELS.items():
        split_dir = shards_root / split_name
        shard_paths = sorted(split_dir.glob("*.npz"))
        if len(shard_paths) == 0:
            raise FileNotFoundError(f"No NPZ shards found under {split_dir}")
        for shard_path in shard_paths:
            with np.load(shard_path, allow_pickle=False) as shard:
                landmarks = cast(NDArray[np.float32], np.asarray(shard["canonical_landmarks"], dtype=np.float32))
                if landmarks.ndim != 4 or landmarks.shape[2:] != (21, 3):
                    raise ValueError(f"{shard_path} canonical_landmarks must have shape (N,T,21,3)")
                sample_count = int(landmarks.shape[0])
                sequence_length = int(landmarks.shape[1])

                if "lengths" in shard:
                    lengths = cast(NDArray[np.int64], np.asarray(shard["lengths"], dtype=np.int64))
                else:
                    lengths = np.full((sample_count,), sequence_length, dtype=np.int64)
                if lengths.shape != (sample_count,):
                    raise ValueError(f"{shard_path} lengths must have shape (N,)")

                if "mask" in shard:
                    mask = cast(NDArray[np.bool_], np.asarray(shard["mask"], dtype=np.bool_))
                else:
                    mask = _mask_from_lengths(lengths, sequence_length)
                if mask.shape != (sample_count, sequence_length):
                    raise ValueError(f"{shard_path} mask must have shape (N,T)")

                targets = _decode_string_array(np.asarray(shard["target_names"]))
                if len(targets) != sample_count:
                    raise ValueError(f"{shard_path} target_names must contain one value per sample")

                features = landmark_velocity_features(landmarks, mask=mask, lengths=lengths)
                _assert_finite_valid(features, mask, shard_path)

                all_sequences.append(features)
                all_targets.append(targets)
                all_splits.append(np.full((sample_count,), split_index, dtype=np.int64))
                all_lengths.append(lengths.astype(np.int64, copy=False))
                all_masks.append(mask.astype(np.bool_, copy=False))
                source_shards.extend([shard_path.as_posix()] * sample_count)

                if "sample_ids" in shard:
                    sample_ids.extend(_decode_string_array(np.asarray(shard["sample_ids"])))
                else:
                    base = shard_path.stem
                    sample_ids.extend([f"{split_name}/{base}/{index:05d}" for index in range(sample_count)])

    target_values = [target for targets in all_targets for target in targets]
    label_names = _label_names_for_targets(target_values)
    label_to_index = {label_name: index for index, label_name in enumerate(label_names)}
    sequences = cast(NDArray[np.float32], np.concatenate(all_sequences, axis=0).astype(np.float32))
    labels = cast(NDArray[np.int64], np.asarray([label_to_index[target] for target in target_values], dtype=np.int64))
    splits = cast(NDArray[np.int64], np.concatenate(all_splits, axis=0).astype(np.int64))
    lengths = cast(NDArray[np.int64], np.concatenate(all_lengths, axis=0).astype(np.int64))
    mask = cast(NDArray[np.bool_], np.concatenate(all_masks, axis=0).astype(np.bool_))
    return RealSkeletonDataset(
        sequences=sequences,
        labels=labels,
        splits=splits,
        lengths=lengths,
        mask=mask,
        sample_ids=tuple(sample_ids),
        source_shards=tuple(source_shards),
        label_names=label_names,
        feature_name=feature_name,
    )


def load_calibration_anchor_dataset(
    seed_package_root: Path,
    *,
    label_names: Sequence[str],
    feature_name: str = "landmark_velocity_126",
) -> CalibrationAnchorDataset:
    """Load approved non-held-out v4 calibration seeds as real-domain anchors."""

    if feature_name != "landmark_velocity_126":
        raise ValueError(f"Unsupported real skeleton feature: {feature_name}")
    seed_npz = seed_package_root / "v4_calibration_seed_dataset.npz"
    summary_path = seed_package_root / "seed_package_summary.json"
    if not seed_npz.exists():
        raise FileNotFoundError(f"Missing calibration seed dataset: {seed_npz}")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, Mapping) or summary.get("status") != "passed":
            raise ValueError(f"Calibration seed package is not passed: {summary_path}")

    label_tuple = tuple(str(label_name) for label_name in label_names)
    label_to_index = {label_name: index for index, label_name in enumerate(label_tuple)}
    with np.load(seed_npz, allow_pickle=False) as seed_data:
        landmarks = cast(NDArray[np.float32], np.asarray(seed_data["canonical_landmarks"], dtype=np.float32))
        if landmarks.ndim != 4 or landmarks.shape[2:] != (21, 3):
            raise ValueError(f"{seed_npz} canonical_landmarks must have shape (N,T,21,3)")
        sample_count = int(landmarks.shape[0])
        sequence_length = int(landmarks.shape[1])
        lengths = cast(NDArray[np.int64], np.asarray(seed_data["lengths"], dtype=np.int64))
        mask = cast(NDArray[np.bool_], np.asarray(seed_data["mask"], dtype=np.bool_))
        target_names = _decode_string_array(np.asarray(seed_data["target_names"]))
        sample_ids = _decode_string_array(np.asarray(seed_data["sample_ids"]))
        source_paths = (
            _decode_string_array(np.asarray(seed_data["source_paths"]))
            if "source_paths" in seed_data
            else [""] * sample_count
        )

    if lengths.shape != (sample_count,):
        raise ValueError(f"{seed_npz} lengths must have shape (N,)")
    if mask.shape != (sample_count, sequence_length):
        raise ValueError(f"{seed_npz} mask must have shape (N,T)")
    if len(target_names) != sample_count or len(sample_ids) != sample_count:
        raise ValueError(f"{seed_npz} target_names/sample_ids length mismatch")
    if len(source_paths) != sample_count:
        raise ValueError(f"{seed_npz} source_paths length mismatch")
    _reject_heldout_seed_paths(source_paths, seed_npz)
    unknown_targets = sorted({target_name for target_name in target_names if target_name not in label_to_index})
    if unknown_targets:
        raise ValueError(f"{seed_npz} contains labels not present in training dataset: {unknown_targets}")

    features = landmark_velocity_features(landmarks, mask=mask, lengths=lengths)
    _assert_finite_valid(features, mask, seed_npz)
    labels = cast(NDArray[np.int64], np.asarray([label_to_index[target] for target in target_names], dtype=np.int64))
    return CalibrationAnchorDataset(
        sequences=features,
        labels=labels,
        lengths=lengths.astype(np.int64, copy=False),
        mask=mask.astype(np.bool_, copy=False),
        sample_ids=tuple(sample_ids),
        source_paths=tuple(source_paths),
        label_names=label_tuple,
        feature_name=feature_name,
    )


def landmark_velocity_features(
    canonical_landmarks: NDArray[np.float32],
    *,
    mask: NDArray[np.bool_],
    lengths: NDArray[np.int64],
) -> NDArray[np.float32]:
    """Convert canonical landmarks to flattened position plus velocity features."""

    if canonical_landmarks.ndim != 4 or canonical_landmarks.shape[2:] != (21, 3):
        raise ValueError("canonical_landmarks must have shape (N,T,21,3)")
    sample_count = int(canonical_landmarks.shape[0])
    sequence_length = int(canonical_landmarks.shape[1])
    flattened = canonical_landmarks.reshape(sample_count, sequence_length, 63).astype(np.float32, copy=True)
    filled = _fill_invalid_frames(flattened, mask=mask, lengths=lengths)
    velocity = np.zeros_like(filled, dtype=np.float32)
    velocity[:, 1:, :] = filled[:, 1:, :] - filled[:, :-1, :]
    features = np.concatenate((filled, velocity), axis=2).astype(np.float32)
    return cast(NDArray[np.float32], features)


def build_observed_batch_by_lengths(
    sequences: NDArray[np.float32],
    lengths: NDArray[np.int64],
    ratio: float,
) -> NDArray[np.float32]:
    """Build partial observations without leaking future valid frames."""

    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if sequences.ndim != 3:
        raise ValueError("sequences must have shape (N,T,F)")
    if lengths.shape != (sequences.shape[0],):
        raise ValueError("lengths must have shape (N,)")

    observed = sequences.astype(np.float32, copy=True)
    sequence_length = int(sequences.shape[1])
    for sample_index, raw_length in enumerate(lengths.tolist()):
        valid_length = max(1, min(int(raw_length), sequence_length))
        observed_length = max(1, min(valid_length, int(math.ceil(float(valid_length) * ratio))))
        fill_frame = observed[sample_index, observed_length - 1, :].copy()
        observed[sample_index, observed_length:, :] = fill_frame
    return cast(NDArray[np.float32], observed)


def train_real_model_runs(
    *,
    sweep_config: Mapping[str, object],
    requested_model: str,
    smoke: bool,
    max_runs: int | None,
) -> list[dict[str, object]]:
    """Train one or more real-skeleton final-gesture model runs."""

    dataset_root = Path(_string_value(sweep_config, "dataset_root"))
    dataset = load_real_skeleton_dataset(dataset_root)
    runs_dir = Path(_string_value(sweep_config, "runs_dir"))
    if smoke:
        runs_dir = runs_dir / "smoke"
    runs_dir.mkdir(parents=True, exist_ok=True)
    device = _select_device(_string_value(sweep_config, "device"))
    epochs = _int_value(sweep_config, "smoke_epochs" if smoke else "epochs")
    batch_size = _int_value(sweep_config, "batch_size")
    learning_rate = _float_value(sweep_config, "learning_rate")
    contrastive_loss_weight = _optional_float_value(sweep_config, "contrastive_loss_weight", 0.0)
    contrastive_temperature = _optional_float_value(sweep_config, "contrastive_temperature", 0.10)
    domain_alignment_loss_weight = _optional_float_value(sweep_config, "domain_alignment_loss_weight", 0.0)
    domain_alignment_seed_package_root = _optional_path_value(sweep_config, "domain_alignment_seed_package_root")
    domain_alignment_scope = _optional_string_value(sweep_config, "domain_alignment_scope", "class")
    if contrastive_loss_weight < 0.0:
        raise ValueError("contrastive_loss_weight must be non-negative")
    if contrastive_temperature <= 0.0:
        raise ValueError("contrastive_temperature must be positive")
    if domain_alignment_loss_weight < 0.0:
        raise ValueError("domain_alignment_loss_weight must be non-negative")
    _validate_domain_alignment_scope(domain_alignment_scope)
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
            train_real_single_run(
                dataset=dataset,
                dataset_root=dataset_root,
                run_config=run_config,
                runs_dir=runs_dir,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                observation_ratios=ratios,
                training_observation_ratios=training_ratios,
                contrastive_loss_weight=contrastive_loss_weight,
                contrastive_temperature=contrastive_temperature,
                domain_alignment_seed_package_root=domain_alignment_seed_package_root,
                domain_alignment_loss_weight=domain_alignment_loss_weight,
                domain_alignment_scope=domain_alignment_scope,
            ).result
        )
    return completed


def train_real_single_run(
    *,
    dataset: RealSkeletonDataset,
    dataset_root: Path,
    run_config: ModelRunConfig,
    runs_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    observation_ratios: Sequence[float],
    training_observation_ratios: Sequence[float],
    contrastive_loss_weight: float = 0.0,
    contrastive_temperature: float = 0.10,
    domain_alignment_seed_package_root: Path | None = None,
    domain_alignment_loss_weight: float = 0.0,
    domain_alignment_scope: str = "class",
) -> RealTrainedRun:
    """Train and evaluate one real-skeleton model run."""

    if contrastive_loss_weight < 0.0:
        raise ValueError("contrastive_loss_weight must be non-negative")
    if contrastive_temperature <= 0.0:
        raise ValueError("contrastive_temperature must be positive")
    if domain_alignment_loss_weight < 0.0:
        raise ValueError("domain_alignment_loss_weight must be non-negative")
    if domain_alignment_loss_weight > 0.0 and domain_alignment_seed_package_root is None:
        raise ValueError("domain_alignment_seed_package_root is required when domain_alignment_loss_weight > 0")
    _validate_domain_alignment_scope(domain_alignment_scope)
    torch.manual_seed(run_config.seed)
    np.random.seed(run_config.seed)
    train_x, train_y, train_prefix_ids = _observed_real_split_arrays_with_prefix_ids(
        dataset,
        split_index=0,
        ratios=training_observation_ratios,
    )
    anchor_dataset = (
        load_calibration_anchor_dataset(
            domain_alignment_seed_package_root,
            label_names=dataset.label_names,
            feature_name=dataset.feature_name,
        )
        if domain_alignment_seed_package_root is not None and domain_alignment_loss_weight > 0.0
        else None
    )
    anchor_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
    if anchor_dataset is not None:
        anchor_x, anchor_y, anchor_prefix_ids = _observed_anchor_arrays_with_prefix_ids(
            anchor_dataset,
            ratios=training_observation_ratios,
        )
        anchor_loader = DataLoader(
            TensorDataset(torch.from_numpy(anchor_x), torch.from_numpy(anchor_y), torch.from_numpy(anchor_prefix_ids)),
            batch_size=min(batch_size, max(1, int(anchor_x.shape[0]))),
            shuffle=True,
        )
    model = build_classifier(
        run_config,
        input_dim=int(dataset.sequences.shape[2]),
        sequence_length=int(dataset.sequences.shape[1]),
        num_classes=len(dataset.label_names),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_prefix_ids)),
        batch_size=batch_size,
        shuffle=True,
    )

    epoch_losses: list[float] = []
    epoch_contrastive_losses: list[float] = []
    epoch_domain_alignment_losses: list[float] = []
    model.train()
    for _ in range(epochs):
        running_loss = 0.0
        running_contrastive_loss = 0.0
        running_domain_alignment_loss = 0.0
        batch_count = 0
        anchor_iter = iter(anchor_loader) if anchor_loader is not None else None
        for batch_x, batch_y, batch_prefix_ids in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_prefix_ids = batch_prefix_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            contrastive_loss = logits.new_zeros(())
            domain_alignment_loss = logits.new_zeros(())
            if contrastive_loss_weight > 0.0:
                embeddings = model.encode(batch_x)
                contrastive_loss = supervised_contrastive_loss(
                    embeddings,
                    batch_y,
                    temperature=contrastive_temperature,
                )
                loss = loss + contrastive_loss_weight * contrastive_loss
            if domain_alignment_loss_weight > 0.0 and anchor_iter is not None and anchor_loader is not None:
                try:
                    anchor_x, anchor_y, anchor_prefix_ids = next(anchor_iter)
                except StopIteration:
                    anchor_iter = iter(anchor_loader)
                    anchor_x, anchor_y, anchor_prefix_ids = next(anchor_iter)
                anchor_x = anchor_x.to(device)
                anchor_y = anchor_y.to(device)
                anchor_prefix_ids = anchor_prefix_ids.to(device)
                batch_embeddings = model.encode(batch_x)
                anchor_embeddings = model.encode(anchor_x)
                if domain_alignment_scope == "class_prefix":
                    domain_alignment_loss = class_prefix_alignment_loss(
                        batch_embeddings,
                        batch_y,
                        batch_prefix_ids,
                        anchor_embeddings,
                        anchor_y,
                        anchor_prefix_ids,
                    )
                else:
                    domain_alignment_loss = class_conditional_alignment_loss(
                        batch_embeddings,
                        batch_y,
                        anchor_embeddings,
                        anchor_y,
                    )
                loss = loss + domain_alignment_loss_weight * domain_alignment_loss
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu().item())
            running_contrastive_loss += float(contrastive_loss.detach().cpu().item())
            running_domain_alignment_loss += float(domain_alignment_loss.detach().cpu().item())
            batch_count += 1
        epoch_losses.append(running_loss / max(1, batch_count))
        epoch_contrastive_losses.append(running_contrastive_loss / max(1, batch_count))
        epoch_domain_alignment_losses.append(running_domain_alignment_loss / max(1, batch_count))

    metrics_by_ratio = evaluate_real_model(model, dataset, device=device, ratios=observation_ratios)
    latency_ms = measure_real_latency_ms(model, dataset, device=device, ratio=0.50)
    dataset_summary = real_dataset_summary(dataset, dataset_root)
    run_result: dict[str, object] = {
        "run_id": run_config.run_id(),
        "model": run_config.model,
        "config": run_config.__dict__,
        "device": str(device),
        "dataset_root": dataset_root.as_posix(),
        "dataset_fingerprint": dataset_fingerprint_for_shards(dataset_root),
        "dataset_summary": dataset_summary,
        "label_names": list(dataset.label_names),
        "sequence_length": int(dataset.sequences.shape[1]),
        "feature_dim": int(dataset.sequences.shape[2]),
        "feature_name": dataset.feature_name,
        "training_observation_ratios": [float(ratio) for ratio in training_observation_ratios],
        "evaluation_observation_ratios": [float(ratio) for ratio in observation_ratios],
        "contrastive_loss_weight": float(contrastive_loss_weight),
        "contrastive_temperature": float(contrastive_temperature),
        "domain_alignment_loss_weight": float(domain_alignment_loss_weight),
        "domain_alignment_scope": domain_alignment_scope,
        "domain_alignment_seed_package_root": domain_alignment_seed_package_root.as_posix()
        if domain_alignment_seed_package_root is not None
        else None,
        "domain_alignment_anchor_summary": _calibration_anchor_summary(anchor_dataset),
        "epochs": epochs,
        "epoch_losses": epoch_losses,
        "epoch_contrastive_losses": epoch_contrastive_losses,
        "epoch_domain_alignment_losses": epoch_domain_alignment_losses,
        "parameter_count": parameter_count(model),
        "latency_ms": latency_ms,
        "metrics": metrics_by_ratio,
    }
    run_dir = runs_dir / run_config.run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(run_result, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "run_result": run_result}, run_dir / "model_state.pt")
    return RealTrainedRun(model=model, result=run_result)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Return supervised contrastive loss over same-label positives in a batch."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N,D)")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must have shape (N,)")
    sample_count = int(embeddings.shape[0])
    if sample_count <= 1:
        return embeddings.sum() * 0.0

    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    logits = normalized @ normalized.T / temperature
    self_mask = torch.eye(sample_count, dtype=torch.bool, device=embeddings.device)
    label_matches = labels[:, None] == labels[None, :]
    positive_mask = label_matches & ~self_mask
    anchor_mask = positive_mask.any(dim=1)
    if not bool(anchor_mask.any()):
        return embeddings.sum() * 0.0

    logits = logits.masked_fill(self_mask, float("-inf"))
    stable_logits = logits - torch.max(logits, dim=1, keepdim=True).values
    exp_logits = torch.exp(stable_logits).masked_fill(self_mask, 0.0)
    denominator = exp_logits.sum(dim=1).clamp_min(torch.finfo(exp_logits.dtype).eps)
    numerator = (exp_logits * positive_mask.to(exp_logits.dtype)).sum(dim=1).clamp_min(
        torch.finfo(exp_logits.dtype).eps
    )
    losses = -torch.log(numerator / denominator)
    selected_losses = losses[anchor_mask]
    output = selected_losses.mean()
    if not isinstance(output, torch.Tensor):
        raise TypeError("supervised contrastive loss must be a tensor")
    return output


def class_conditional_alignment_loss(
    source_embeddings: torch.Tensor,
    source_labels: torch.Tensor,
    anchor_embeddings: torch.Tensor,
    anchor_labels: torch.Tensor,
) -> torch.Tensor:
    """Align embedding means per class between synthetic batches and real anchors."""

    if source_embeddings.ndim != 2 or anchor_embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N,D)")
    if source_embeddings.shape[1] != anchor_embeddings.shape[1]:
        raise ValueError("source and anchor embeddings must share feature dimension")
    if source_labels.ndim != 1 or source_labels.shape[0] != source_embeddings.shape[0]:
        raise ValueError("source_labels must have shape (N,)")
    if anchor_labels.ndim != 1 or anchor_labels.shape[0] != anchor_embeddings.shape[0]:
        raise ValueError("anchor_labels must have shape (N,)")

    shared_labels = torch.unique(source_labels)
    losses: list[torch.Tensor] = []
    for label in shared_labels.tolist():
        source_mask = source_labels == int(label)
        anchor_mask = anchor_labels == int(label)
        if not bool(anchor_mask.any()) or not bool(source_mask.any()):
            continue
        source_mean = source_embeddings[source_mask].mean(dim=0)
        anchor_mean = anchor_embeddings[anchor_mask].mean(dim=0)
        losses.append(torch.mean((source_mean - anchor_mean) ** 2))
    if not losses:
        return source_embeddings.sum() * 0.0
    output = torch.stack(losses).mean()
    if not isinstance(output, torch.Tensor):
        raise TypeError("class-conditional alignment loss must be a tensor")
    return output


def class_prefix_alignment_loss(
    source_embeddings: torch.Tensor,
    source_labels: torch.Tensor,
    source_prefix_ids: torch.Tensor,
    anchor_embeddings: torch.Tensor,
    anchor_labels: torch.Tensor,
    anchor_prefix_ids: torch.Tensor,
) -> torch.Tensor:
    """Align embedding means per class and observed-prefix id."""

    if source_prefix_ids.ndim != 1 or source_prefix_ids.shape[0] != source_embeddings.shape[0]:
        raise ValueError("source_prefix_ids must have shape (N,)")
    if anchor_prefix_ids.ndim != 1 or anchor_prefix_ids.shape[0] != anchor_embeddings.shape[0]:
        raise ValueError("anchor_prefix_ids must have shape (N,)")
    _validate_alignment_inputs(source_embeddings, source_labels, anchor_embeddings, anchor_labels)

    losses: list[torch.Tensor] = []
    for label in torch.unique(source_labels).tolist():
        for prefix_id in torch.unique(source_prefix_ids).tolist():
            source_mask = (source_labels == int(label)) & (source_prefix_ids == int(prefix_id))
            anchor_mask = (anchor_labels == int(label)) & (anchor_prefix_ids == int(prefix_id))
            if not bool(anchor_mask.any()) or not bool(source_mask.any()):
                continue
            source_mean = source_embeddings[source_mask].mean(dim=0)
            anchor_mean = anchor_embeddings[anchor_mask].mean(dim=0)
            losses.append(torch.mean((source_mean - anchor_mean) ** 2))
    if not losses:
        return source_embeddings.sum() * 0.0
    output = torch.stack(losses).mean()
    if not isinstance(output, torch.Tensor):
        raise TypeError("class-prefix alignment loss must be a tensor")
    return output


def evaluate_real_model(
    model: RpsClassifier,
    dataset: RealSkeletonDataset,
    *,
    device: torch.device,
    ratios: Sequence[float],
    split_index: int = 2,
) -> dict[str, object]:
    """Evaluate a model on each requested observation ratio."""

    model.eval()
    results: dict[str, object] = {}
    test_mask = dataset.splits == split_index
    labels = cast(NDArray[np.int64], dataset.labels[test_mask])
    lengths = cast(NDArray[np.int64], dataset.lengths[test_mask])
    with torch.no_grad():
        for ratio in ratios:
            observed = build_observed_batch_by_lengths(dataset.sequences[test_mask], lengths, ratio)
            logits = model(torch.from_numpy(observed).to(device))
            probabilities = torch.softmax(logits, dim=1)
            predictions = cast(NDArray[np.int64], logits.argmax(dim=1).cpu().numpy().astype(np.int64))
            confidences = probabilities.max(dim=1).values.detach().cpu().numpy().astype(np.float64)
            metrics = classification_metrics(labels, predictions, num_classes=len(dataset.label_names)).to_json()
            metrics["mean_confidence"] = float(np.mean(confidences)) if confidences.size else 0.0
            metrics["median_confidence"] = float(np.median(confidences)) if confidences.size else 0.0
            results[f"{ratio:.2f}"] = metrics
    return results


def measure_real_latency_ms(
    model: RpsClassifier,
    dataset: RealSkeletonDataset,
    *,
    device: torch.device,
    ratio: float,
    repeats: int = 100,
) -> float:
    """Measure single-sample inference latency."""

    model.eval()
    test_mask = dataset.splits == 2
    sequences = dataset.sequences[test_mask][:1]
    lengths = dataset.lengths[test_mask][:1]
    observed = build_observed_batch_by_lengths(sequences, lengths, ratio)
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


def write_real_model_comparison(runs_dir: Path, output_path: Path, *, ready_ratio: str = "0.50") -> dict[str, object]:
    """Aggregate real-skeleton model metrics into a comparison report."""

    runs = _load_run_metrics(runs_dir)
    ratios = _metric_ratios(runs)
    best_30 = select_best_run(runs, ratio="0.30") if "0.30" in ratios else select_best_run(runs, ratio=ratios[0])
    best_50 = select_best_run(runs, ratio=ready_ratio)
    ready_metrics = _mapping(_required(_mapping(_required(best_50, "metrics"), "metrics"), ready_ratio), ready_ratio)
    ready_accuracy = _float_value(ready_metrics, "accuracy")
    comparison: dict[str, object] = {
        "num_runs": len(runs),
        "runs_dir": runs_dir.as_posix(),
        "dataset_fingerprints": _unique_dataset_fingerprints(runs),
        "label_names": _string_list(runs[0].get("label_names", []), "label_names"),
        "quality_target": {"ratio": ready_ratio, "accuracy": 0.90},
        "model_ready": ready_accuracy >= 0.90,
        "best_for_early_prediction_30": _summary(best_30, "0.30" if "0.30" in ratios else ratios[0]),
        "best_for_clear_distinction_50": _summary(best_50, ready_ratio),
        "best_by_ratio": {ratio: _summary(select_best_run(runs, ratio=ratio), ratio) for ratio in ratios},
        "best_by_model_by_ratio": {
            ratio: {
                model_name: _summary(select_best_run(_runs_for_model(runs, model_name), ratio=ratio), ratio)
                for model_name in _model_names(runs)
            }
            for ratio in ratios
        },
        "runs_at_50": [_summary(run, ready_ratio) for run in runs if ready_ratio in _mapping(run, "metrics")],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def export_best_real_profile(
    *,
    runs_dir: Path,
    output_dir: Path,
    profile_name: str,
    preferred_model: str = "gru",
    ratio: str = "0.50",
) -> dict[str, object]:
    """Export the best trained model profile for realtime inference."""

    runs = _load_run_metrics(runs_dir)
    candidates = _runs_for_model(runs, preferred_model)
    if len(candidates) == 0:
        candidates = runs
    best_run = select_best_run(candidates, ratio=ratio)
    run_id = _string_value(best_run, "run_id")
    run_dir = runs_dir / run_id
    state_path = run_dir / "model_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint for best run: {state_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_pt = output_dir / f"{profile_name}.pt"
    profile_json = output_dir / f"{profile_name}.json"
    checkpoint = torch.load(state_path, map_location="cpu")
    torch.save(checkpoint, profile_pt)
    metrics = _mapping(_required(_mapping(_required(best_run, "metrics"), "metrics"), ratio), ratio)
    profile: dict[str, object] = {
        "profile_name": profile_name,
        "model_state_path": profile_pt.as_posix(),
        "run_id": run_id,
        "selected_ratio": ratio,
        "selected_accuracy": _float_value(metrics, "accuracy"),
        "selected_macro_f1": _float_value(metrics, "macro_f1"),
        "model": _string_value(best_run, "model"),
        "config": dict(_mapping(_required(best_run, "config"), "config")),
        "label_names": _string_list(best_run.get("label_names", []), "label_names"),
        "sequence_length": _int_value(best_run, "sequence_length"),
        "feature_dim": _int_value(best_run, "feature_dim"),
        "feature_name": _string_value(best_run, "feature_name"),
        "dataset_root": _string_value(best_run, "dataset_root"),
        "dataset_fingerprint": dict(_mapping(_required(best_run, "dataset_fingerprint"), "dataset_fingerprint")),
        "metrics": dict(_mapping(_required(best_run, "metrics"), "metrics")),
    }
    profile_json.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def real_dataset_summary(dataset: RealSkeletonDataset, dataset_root: Path) -> dict[str, object]:
    """Return a compact dataset validation summary."""

    split_counts = {
        split_name: int(np.sum(dataset.splits == split_index))
        for split_name, split_index in SPLIT_LABELS.items()
    }
    label_counts = {
        label_name: int(np.sum(dataset.labels == label_index))
        for label_index, label_name in enumerate(dataset.label_names)
    }
    return {
        "dataset_root": dataset_root.as_posix(),
        "sample_count": int(dataset.sequences.shape[0]),
        "sequence_length": int(dataset.sequences.shape[1]),
        "feature_dim": int(dataset.sequences.shape[2]),
        "split_counts": split_counts,
        "label_counts": label_counts,
        "length_min": int(np.min(dataset.lengths)),
        "length_max": int(np.max(dataset.lengths)),
        "length_mean": float(np.mean(dataset.lengths)),
        "source_shard_count": len(set(dataset.source_shards)),
    }


def dataset_fingerprint_for_shards(dataset_root: Path) -> dict[str, object]:
    """Return a stable fingerprint over the real-skeleton shard files."""

    shard_paths = sorted((dataset_root / "shards").glob("*/*.npz"))
    digest = hashlib.sha256()
    total_size = 0
    max_mtime_ns = 0
    for shard_path in shard_paths:
        stat = shard_path.stat()
        total_size += int(stat.st_size)
        max_mtime_ns = max(max_mtime_ns, int(stat.st_mtime_ns))
        digest.update(shard_path.relative_to(dataset_root).as_posix().encode("utf-8"))
        with shard_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "path": dataset_root.as_posix(),
        "exists": dataset_root.exists(),
        "shard_count": len(shard_paths),
        "size_bytes": total_size,
        "mtime_ns": max_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _fill_invalid_frames(
    flattened: NDArray[np.float32],
    *,
    mask: NDArray[np.bool_],
    lengths: NDArray[np.int64],
) -> NDArray[np.float32]:
    filled = flattened.astype(np.float32, copy=True)
    sequence_length = int(filled.shape[1])
    for sample_index, raw_length in enumerate(lengths.tolist()):
        valid_length = max(1, min(int(raw_length), sequence_length))
        valid_indices = np.flatnonzero(mask[sample_index, :valid_length])
        if valid_indices.size == 0:
            raise ValueError(f"Sample {sample_index} has no valid skeleton frames")
        first_valid = int(valid_indices[0])
        if first_valid > 0:
            filled[sample_index, :first_valid, :] = filled[sample_index, first_valid, :]
        for frame_index in range(first_valid + 1, sequence_length):
            if frame_index >= valid_length or not bool(mask[sample_index, frame_index]):
                filled[sample_index, frame_index, :] = filled[sample_index, frame_index - 1, :]
    return cast(NDArray[np.float32], filled)


def _observed_real_split_arrays(
    dataset: RealSkeletonDataset,
    *,
    split_index: int,
    ratios: Sequence[float],
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    train_x, train_y, _prefix_ids = _observed_real_split_arrays_with_prefix_ids(
        dataset,
        split_index=split_index,
        ratios=ratios,
    )
    return train_x, train_y


def _observed_real_split_arrays_with_prefix_ids(
    dataset: RealSkeletonDataset,
    *,
    split_index: int,
    ratios: Sequence[float],
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]]:
    split_mask = dataset.splits == split_index
    sequences = cast(NDArray[np.float32], dataset.sequences[split_mask])
    labels = cast(NDArray[np.int64], dataset.labels[split_mask])
    lengths = cast(NDArray[np.int64], dataset.lengths[split_mask])
    observed_batches = [build_observed_batch_by_lengths(sequences, lengths, ratio) for ratio in ratios]
    train_x = cast(NDArray[np.float32], np.concatenate(observed_batches, axis=0).astype(np.float32))
    train_y = cast(NDArray[np.int64], np.tile(labels, len(ratios)).astype(np.int64))
    prefix_ids = _prefix_ids_for_ratios(sample_count=int(labels.shape[0]), ratios=ratios)
    return train_x, train_y, prefix_ids


def _observed_anchor_arrays(
    dataset: CalibrationAnchorDataset,
    *,
    ratios: Sequence[float],
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    anchor_x, anchor_y, _prefix_ids = _observed_anchor_arrays_with_prefix_ids(dataset, ratios=ratios)
    return anchor_x, anchor_y


def _observed_anchor_arrays_with_prefix_ids(
    dataset: CalibrationAnchorDataset,
    *,
    ratios: Sequence[float],
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]]:
    observed_batches = [build_observed_batch_by_lengths(dataset.sequences, dataset.lengths, ratio) for ratio in ratios]
    anchor_x = cast(NDArray[np.float32], np.concatenate(observed_batches, axis=0).astype(np.float32))
    anchor_y = cast(NDArray[np.int64], np.tile(dataset.labels, len(ratios)).astype(np.int64))
    prefix_ids = _prefix_ids_for_ratios(sample_count=int(dataset.labels.shape[0]), ratios=ratios)
    return anchor_x, anchor_y, prefix_ids


def _prefix_ids_for_ratios(*, sample_count: int, ratios: Sequence[float]) -> NDArray[np.int64]:
    prefix_ids = np.concatenate(
        [np.full((sample_count,), ratio_index, dtype=np.int64) for ratio_index, _ratio in enumerate(ratios)],
        axis=0,
    )
    return cast(NDArray[np.int64], prefix_ids.astype(np.int64, copy=False))


def _calibration_anchor_summary(dataset: CalibrationAnchorDataset | None) -> dict[str, object] | None:
    if dataset is None:
        return None
    return {
        "sample_count": int(dataset.sequences.shape[0]),
        "label_counts": {
            label_name: int(np.sum(dataset.labels == label_index))
            for label_index, label_name in enumerate(dataset.label_names)
        },
    }


def _mask_from_lengths(lengths: NDArray[np.int64], sequence_length: int) -> NDArray[np.bool_]:
    frame_indices = np.arange(sequence_length, dtype=np.int64)[None, :]
    return cast(NDArray[np.bool_], frame_indices < lengths[:, None])


def _decode_string_array(values: NDArray[object]) -> list[str]:
    decoded: list[str] = []
    for value in values.reshape(-1).tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _label_names_for_targets(targets: Sequence[str]) -> tuple[str, ...]:
    unique_targets = set(targets)
    if unique_targets == set(BINARY_FINAL_GESTURE_LABELS):
        return tuple(BINARY_FINAL_GESTURE_LABELS.keys())
    if unique_targets == set(THREE_CLASS_GESTURE_LABELS):
        return tuple(THREE_CLASS_GESTURE_LABELS.keys())
    if unique_targets == set(ROCK_TRANSITION_LABELS):
        return tuple(ROCK_TRANSITION_LABELS.keys())
    raise ValueError(f"Unsupported target labels: {sorted(unique_targets)}")


def _assert_finite_valid(features: NDArray[np.float32], mask: NDArray[np.bool_], shard_path: Path) -> None:
    valid_values = features[np.repeat(mask[:, :, None], features.shape[2], axis=2)]
    if not np.all(np.isfinite(valid_values)):
        raise ValueError(f"{shard_path} contains NaN or Inf in valid feature frames")


def _reject_heldout_seed_paths(source_paths: Sequence[str], seed_npz: Path) -> None:
    for source_path in source_paths:
        normalized = source_path.replace("\\", "/").lower()
        if "/test/" in normalized or normalized.endswith("/test"):
            raise ValueError(f"{seed_npz} contains held-out test source path: {source_path}")


def _validate_alignment_inputs(
    source_embeddings: torch.Tensor,
    source_labels: torch.Tensor,
    anchor_embeddings: torch.Tensor,
    anchor_labels: torch.Tensor,
) -> None:
    if source_embeddings.ndim != 2 or anchor_embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N,D)")
    if source_embeddings.shape[1] != anchor_embeddings.shape[1]:
        raise ValueError("source and anchor embeddings must share feature dimension")
    if source_labels.ndim != 1 or source_labels.shape[0] != source_embeddings.shape[0]:
        raise ValueError("source_labels must have shape (N,)")
    if anchor_labels.ndim != 1 or anchor_labels.shape[0] != anchor_embeddings.shape[0]:
        raise ValueError("anchor_labels must have shape (N,)")


def _validate_domain_alignment_scope(value: str) -> None:
    if value not in {"class", "class_prefix"}:
        raise ValueError("domain_alignment_scope must be 'class' or 'class_prefix'")


def _load_run_metrics(runs_dir: Path) -> list[dict[str, object]]:
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
    return runs


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
    return {
        "run_id": _string_value(run, "run_id"),
        "model": _string_value(run, "model"),
        "ratio": ratio,
        "macro_f1": _float_value(ratio_metrics, "macro_f1"),
        "accuracy": _float_value(ratio_metrics, "accuracy"),
        "mean_confidence": _float_value(ratio_metrics, "mean_confidence"),
        "latency_ms": _float_value(run, "latency_ms"),
        "parameter_count": _int_value(run, "parameter_count"),
    }


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
    return int(value)


def _float_value(mapping: Mapping[str, object], key: str) -> float:
    value = _required(mapping, key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _optional_float_value(mapping: Mapping[str, object], key: str, default: float) -> float:
    if key not in mapping:
        return float(default)
    return _float_value(mapping, key)


def _optional_path_value(mapping: Mapping[str, object], key: str) -> Path | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{key} must be a non-empty string")
    return Path(value)


def _optional_string_value(mapping: Mapping[str, object], key: str, default: str) -> str:
    if key not in mapping:
        return default
    value = mapping[key]
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return [str(item) for item in value]


def _float_list(value: object, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    parsed: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"{label} must contain numbers")
        parsed.append(float(item))
    return parsed


__all__ = [
    "FINAL_GESTURE_LABELS",
    "CalibrationAnchorDataset",
    "RealSkeletonDataset",
    "build_observed_batch_by_lengths",
    "class_conditional_alignment_loss",
    "class_prefix_alignment_loss",
    "dataset_fingerprint_for_shards",
    "evaluate_real_model",
    "export_best_real_profile",
    "landmark_velocity_features",
    "load_calibration_anchor_dataset",
    "load_real_skeleton_dataset",
    "load_sweep_config",
    "measure_real_latency_ms",
    "real_dataset_summary",
    "supervised_contrastive_loss",
    "train_real_model_runs",
    "train_real_single_run",
    "write_real_model_comparison",
]
