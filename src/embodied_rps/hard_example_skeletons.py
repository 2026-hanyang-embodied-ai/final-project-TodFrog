"""View-robust hard-example skeleton dataset generation."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from embodied_rps.pose_family import (
    FINGER_NAMES,
    KEYPOINT_NAMES,
    FingerName,
    Handedness,
    PersonIdentity,
    build_keypoints_from_semantics,
)
from embodied_rps.real_skeleton_training import landmark_velocity_features

TargetName: TypeAlias = Literal["paper", "scissors"]
SplitName: TypeAlias = Literal["train", "val", "test"]

TARGET_NAMES: tuple[TargetName, ...] = ("paper", "scissors")
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "test")
TARGET_TO_TRANSITION: dict[TargetName, str] = {
    "paper": "rock_to_paper",
    "scissors": "rock_to_scissors",
}
TARGET_TO_LABEL: dict[TargetName, int] = {"paper": 0, "scissors": 1}
CORE_FIELDS: tuple[str, ...] = (
    "sample_ids",
    "labels",
    "label_names",
    "target_names",
    "split_names",
    "lengths",
    "mask",
    "progress",
    "canonical_landmarks",
    "features",
    "source_names",
    "hard_example_flags",
)


@dataclass(frozen=True)
class HardExpansionConfig:
    """Configuration for hard-example skeleton expansion."""

    output_root: Path
    base_dataset_root: Path | None = None
    generated_per_target: int = 2500
    sequence_length: int = 72
    min_length: int = 48
    shard_size: int = 512
    seed: int = 20260611
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    overwrite: bool = False


@dataclass(frozen=True)
class SkeletonSample:
    """One padded skeleton sample ready for sharding."""

    sample_id: str
    split: SplitName
    target_name: TargetName
    canonical_landmarks: NDArray[np.float32]
    mask: NDArray[np.bool_]
    progress: NDArray[np.float32]
    source_name: str
    hard_example: bool
    metadata: dict[str, object]

    @property
    def length(self) -> int:
        return int(np.sum(self.mask))

    @property
    def transition_label(self) -> str:
        return TARGET_TO_TRANSITION[self.target_name]

    @property
    def label_id(self) -> int:
        return TARGET_TO_LABEL[self.target_name]


def generate_hard_expanded_dataset(config: HardExpansionConfig) -> dict[str, object]:
    """Generate, shard, validate, and document a hard-expanded skeleton dataset."""

    _validate_config(config)
    _prepare_output(config.output_root, overwrite=config.overwrite)
    rng = np.random.default_rng(config.seed)
    samples: list[SkeletonSample] = []
    if config.base_dataset_root is not None:
        samples.extend(load_base_dataset_samples(config.base_dataset_root, sequence_length=config.sequence_length))
    samples.extend(generate_hard_samples(config, rng=rng))
    samples = sorted(samples, key=lambda sample: (sample.split, sample.target_name, sample.sample_id))
    shard_rows = write_shards(config.output_root, samples, shard_size=config.shard_size, sequence_length=config.sequence_length)
    metadata_path = write_sample_metadata(config.output_root, samples)
    validation = validate_hard_expanded_dataset(config.output_root, shard_rows=shard_rows, expected_sample_count=len(samples))
    write_csv(config.output_root / "shard_index.csv", shard_rows, fieldnames=list(shard_rows[0].keys()) if shard_rows else [])
    write_json(config.output_root / "validation_summary.json", validation)
    write_json(config.output_root / "generation_config.json", _json_ready(asdict(config)))
    dataset_card_path = write_dataset_card(config.output_root, validation)
    run_summary = {
        "status": validation["status"],
        "output_root": config.output_root.as_posix(),
        "sample_count": len(samples),
        "generated_per_target": config.generated_per_target,
        "base_dataset_root": config.base_dataset_root.as_posix() if config.base_dataset_root is not None else None,
        "dataset_card": dataset_card_path.as_posix(),
        "sample_metadata": metadata_path.as_posix(),
        "validation_summary": (config.output_root / "validation_summary.json").as_posix(),
        "validation": validation,
    }
    write_json(config.output_root / "run_summary.json", run_summary)
    return run_summary


def load_base_dataset_samples(base_root: Path, *, sequence_length: int) -> list[SkeletonSample]:
    """Load existing real-guided shards into the hard-expanded sample contract."""

    shards_root = base_root / "shards"
    if not shards_root.exists():
        raise FileNotFoundError(f"Missing base shard root: {shards_root}")
    samples: list[SkeletonSample] = []
    for split in SPLIT_NAMES:
        for shard_path in sorted((shards_root / split).glob("*.npz")):
            with np.load(shard_path, allow_pickle=False) as shard:
                landmarks = cast(NDArray[np.float32], np.asarray(shard["canonical_landmarks"], dtype=np.float32))
                lengths = cast(NDArray[np.int64], np.asarray(shard["lengths"], dtype=np.int64))
                mask = cast(NDArray[np.bool_], np.asarray(shard["mask"], dtype=np.bool_))
                target_names = _decode_string_array(np.asarray(shard["target_names"]))
                sample_ids = (
                    _decode_string_array(np.asarray(shard["sample_ids"]))
                    if "sample_ids" in shard.files
                    else [f"{split}_{shard_path.stem}_{index:05d}" for index in range(landmarks.shape[0])]
                )
                progress = (
                    cast(NDArray[np.float32], np.asarray(shard["progress"], dtype=np.float32))
                    if "progress" in shard.files
                    else _progress_from_lengths(lengths, sequence_length)
                )
                if landmarks.shape[1:] != (sequence_length, 21, 3):
                    raise ValueError(f"{shard_path} canonical_landmarks must have shape (N,{sequence_length},21,3)")
                for index in range(landmarks.shape[0]):
                    target_name = _target_name(target_names[index])
                    samples.append(
                        SkeletonSample(
                            sample_id=f"base_{sample_ids[index]}",
                            split=split,
                            target_name=target_name,
                            canonical_landmarks=landmarks[index].astype(np.float32, copy=True),
                            mask=mask[index].astype(np.bool_, copy=True),
                            progress=progress[index].astype(np.float32, copy=True),
                            source_name="base_real_guided",
                            hard_example=False,
                            metadata={
                                "source_shard": shard_path.as_posix(),
                                "source_sample_id": sample_ids[index],
                                "source_length": int(lengths[index]),
                            },
                        )
                    )
    return samples


def generate_hard_samples(config: HardExpansionConfig, *, rng: np.random.Generator) -> list[SkeletonSample]:
    """Generate balanced hard paper and scissors control skeleton samples."""

    samples: list[SkeletonSample] = []
    for target_name in TARGET_NAMES:
        for index in range(config.generated_per_target):
            split = split_for_index(index, config.generated_per_target, train_fraction=config.train_fraction, val_fraction=config.val_fraction)
            samples.append(
                generate_one_hard_sample(
                    target_name=target_name,
                    sample_index=index,
                    split=split,
                    rng=rng,
                    sequence_length=config.sequence_length,
                    min_length=config.min_length,
                )
            )
    return samples


def generate_one_hard_sample(
    *,
    target_name: TargetName,
    sample_index: int,
    split: SplitName,
    rng: np.random.Generator,
    sequence_length: int,
    min_length: int,
) -> SkeletonSample:
    """Generate one hard paper or scissors control trajectory."""

    length = int(rng.integers(min_length, sequence_length + 1))
    mask = np.zeros((sequence_length,), dtype=np.bool_)
    mask[:length] = True
    progress = np.zeros((sequence_length,), dtype=np.float32)
    progress[:length] = np.linspace(0.0, 1.0, length, dtype=np.float32)
    if length < sequence_length:
        progress[length:] = 1.0

    identity = _sample_identity(rng, person_id=sample_index)
    handedness = identity.handedness
    yaw = float(rng.uniform(0.0, 360.0))
    pitch = float(rng.uniform(-42.0, 42.0))
    roll = float(rng.uniform(-35.0, 35.0))
    global_scale = float(rng.uniform(0.82, 1.22))
    depth_scale = float(rng.uniform(0.86, 1.18))
    translation = cast(NDArray[np.float32], rng.normal(0.0, 0.035, size=(3,)).astype(np.float32))
    noise_std = float(rng.uniform(0.0004, 0.0020))
    start_curls = {finger: float(rng.uniform(0.88, 0.99)) for finger in FINGER_NAMES}
    start_spreads = {finger: float(rng.normal(0.0, 0.015)) for finger in FINGER_NAMES}
    final_curls, final_spreads = _target_pose(target_name, rng)
    onset = _finger_onsets(target_name, rng)
    hesitation_center = float(rng.uniform(0.42, 0.68))
    hesitation_width = float(rng.uniform(0.06, 0.14))
    hesitation_strength = float(rng.uniform(0.0, 0.18))

    canonical = np.zeros((sequence_length, 21, 3), dtype=np.float32)
    last_valid = np.zeros((21, 3), dtype=np.float32)
    for frame_index in range(sequence_length):
        frame_progress = float(progress[min(frame_index, length - 1)])
        curls: dict[FingerName, float] = {}
        spreads: dict[FingerName, float] = {}
        for finger in FINGER_NAMES:
            motion = _motion_progress(
                frame_progress,
                onset=onset[finger],
                hesitation_center=hesitation_center,
                hesitation_width=hesitation_width,
                hesitation_strength=hesitation_strength,
            )
            curls[finger] = start_curls[finger] + (final_curls[finger] - start_curls[finger]) * motion
            spreads[finger] = start_spreads[finger] + (final_spreads[finger] - start_spreads[finger]) * motion
        raw = build_keypoints_from_semantics(curls, spreads, identity)
        if handedness == "left":
            raw = raw.copy()
            raw[:, 0] *= -1.0
        transformed = apply_3d_view_transform(
            raw * np.float32(global_scale),
            yaw_degrees=yaw,
            pitch_degrees=pitch,
            roll_degrees=roll,
            depth_scale=depth_scale,
        )
        transformed = transformed + translation
        if frame_index < length:
            transformed = transformed + cast(NDArray[np.float32], rng.normal(0.0, noise_std, size=transformed.shape).astype(np.float32))
            last_valid = canonicalize_mediapipe_style(transformed)
            canonical[frame_index] = last_valid
        else:
            canonical[frame_index] = last_valid

    sample_id = f"hard_{TARGET_TO_TRANSITION[target_name]}_{sample_index + 1:05d}"
    return SkeletonSample(
        sample_id=sample_id,
        split=split,
        target_name=target_name,
        canonical_landmarks=canonical,
        mask=mask,
        progress=progress,
        source_name="hard_procedural_3d",
        hard_example=True,
        metadata={
            "target_name": target_name,
            "transition_label": TARGET_TO_TRANSITION[target_name],
            "generated_length": length,
            "handedness": handedness,
            "yaw_degrees": yaw,
            "pitch_degrees": pitch,
            "roll_degrees": roll,
            "global_scale": global_scale,
            "depth_scale": depth_scale,
            "noise_std": noise_std,
            "onsets": onset,
            "hard_case": "slow_fist_like_paper" if target_name == "paper" else "matched_scissors_control",
        },
    )


def canonicalize_mediapipe_style(keypoints: NDArray[np.float32]) -> NDArray[np.float32]:
    """Canonicalize a 21-point 3D hand frame with the realtime MediaPipe contract."""

    points = cast(NDArray[np.float32], np.asarray(keypoints, dtype=np.float32))
    if points.shape != (21, 3):
        raise ValueError(f"keypoints must have shape (21,3), got {points.shape}")
    wrist = points[0]
    middle_mcp = points[9]
    index_mcp = points[5]
    pinky_mcp = points[17]
    x_axis = _normalize(pinky_mcp - index_mcp, fallback=np.asarray((1.0, 0.0, 0.0), dtype=np.float32))
    y_raw = _normalize(middle_mcp - wrist, fallback=np.asarray((0.0, 1.0, 0.0), dtype=np.float32))
    y_axis = y_raw - np.float32(np.dot(y_raw, x_axis)) * x_axis
    y_axis = _normalize(y_axis, fallback=np.asarray((0.0, 1.0, 0.0), dtype=np.float32))
    z_axis = _normalize(np.cross(x_axis, y_axis).astype(np.float32), fallback=np.asarray((0.0, 0.0, 1.0), dtype=np.float32))
    y_axis = _normalize(np.cross(z_axis, x_axis).astype(np.float32), fallback=y_axis)
    scale = float(np.linalg.norm(middle_mcp - wrist))
    if scale < 1.0e-6:
        scale = float(np.linalg.norm(pinky_mcp - index_mcp))
    if scale < 1.0e-6:
        scale = 1.0
    rel = (points - wrist) / np.float32(scale)
    return cast(NDArray[np.float32], np.stack((rel @ x_axis, rel @ y_axis, rel @ z_axis), axis=1).astype(np.float32))


def apply_3d_view_transform(
    keypoints: NDArray[np.float32],
    *,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
    depth_scale: float,
) -> NDArray[np.float32]:
    """Apply 3D camera/view variation before canonicalization."""

    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    roll = math.radians(roll_degrees)
    rz = np.asarray(((math.cos(yaw), -math.sin(yaw), 0.0), (math.sin(yaw), math.cos(yaw), 0.0), (0.0, 0.0, 1.0)), dtype=np.float32)
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, math.cos(pitch), -math.sin(pitch)), (0.0, math.sin(pitch), math.cos(pitch))), dtype=np.float32)
    ry = np.asarray(((math.cos(roll), 0.0, math.sin(roll)), (0.0, 1.0, 0.0), (-math.sin(roll), 0.0, math.cos(roll))), dtype=np.float32)
    transform = rz @ rx @ ry
    transformed = cast(NDArray[np.float32], (keypoints @ transform.T).astype(np.float32))
    transformed[:, 2] *= np.float32(depth_scale)
    return transformed


def split_for_index(index: int, count: int, *, train_fraction: float, val_fraction: float) -> SplitName:
    """Return deterministic split assignment for generated samples."""

    ratio = index / float(max(1, count))
    if ratio < train_fraction:
        return "train"
    if ratio < train_fraction + val_fraction:
        return "val"
    return "test"


def write_shards(output_root: Path, samples: list[SkeletonSample], *, shard_size: int, sequence_length: int) -> list[dict[str, object]]:
    """Write samples to split shards and return shard index rows."""

    shard_rows: list[dict[str, object]] = []
    for split in SPLIT_NAMES:
        split_samples = [sample for sample in samples if sample.split == split]
        for shard_index, start in enumerate(range(0, len(split_samples), shard_size)):
            shard_samples = split_samples[start : start + shard_size]
            shard_dir = output_root / "shards" / split
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / f"{split}_shard_{shard_index:03d}.npz"
            arrays = _arrays_for_samples(shard_samples, sequence_length=sequence_length)
            np.savez_compressed(shard_path, **arrays)
            valid = arrays["mask"]
            labels = [str(value) for value in arrays["target_names"].tolist()]
            row: dict[str, object] = {
                "split": split,
                "shard_index": shard_index,
                "path": shard_path.as_posix(),
                "sample_count": len(shard_samples),
                "valid_frame_count": int(np.sum(valid)),
                "target_counts": dict(sorted(Counter(labels).items())),
                "hard_example_count": int(np.sum(arrays["hard_example_flags"])),
            }
            shard_rows.append(row)
    return shard_rows


def validate_hard_expanded_dataset(
    output_root: Path,
    *,
    shard_rows: list[dict[str, object]],
    expected_sample_count: int,
) -> dict[str, object]:
    """Validate shape, balance, finite values, and canonical invariants."""

    failures: list[str] = []
    target_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    lengths: list[int] = []
    hard_count = 0
    wrist_errors: list[float] = []
    wrist_middle_scales: list[float] = []
    yaw_values: list[float] = []
    pitch_values: list[float] = []
    roll_values: list[float] = []
    sample_ids: list[str] = []

    for row in shard_rows:
        shard_path = Path(str(row["path"]))
        if not shard_path.exists():
            failures.append(f"missing shard {shard_path}")
            continue
        with np.load(shard_path, allow_pickle=False) as shard:
            missing = set(CORE_FIELDS) - set(shard.files)
            if missing:
                failures.append(f"{shard_path} missing fields {sorted(missing)}")
                continue
            landmarks = cast(NDArray[np.float32], np.asarray(shard["canonical_landmarks"], dtype=np.float32))
            mask = cast(NDArray[np.bool_], np.asarray(shard["mask"], dtype=np.bool_))
            features = cast(NDArray[np.float32], np.asarray(shard["features"], dtype=np.float32))
            if landmarks.ndim != 4 or landmarks.shape[1:] != (72, 21, 3):
                failures.append(f"{shard_path} has unexpected canonical_landmarks shape {landmarks.shape}")
            if features.ndim != 3 or features.shape[1:] != (72, 126):
                failures.append(f"{shard_path} has unexpected features shape {features.shape}")
            if not np.all(np.isfinite(landmarks[mask])):
                failures.append(f"{shard_path} contains non-finite valid landmarks")
            if not np.all(np.isfinite(features[mask])):
                failures.append(f"{shard_path} contains non-finite valid features")
            sample_ids.extend(_decode_string_array(np.asarray(shard["sample_ids"])))
            targets = _decode_string_array(np.asarray(shard["target_names"]))
            splits = _decode_string_array(np.asarray(shard["split_names"]))
            sources = _decode_string_array(np.asarray(shard["source_names"]))
            target_counts.update(targets)
            split_counts.update(splits)
            source_counts.update(sources)
            lengths.extend(int(value) for value in np.asarray(shard["lengths"], dtype=np.int64).tolist())
            hard_flags = np.asarray(shard["hard_example_flags"], dtype=np.bool_)
            hard_count += int(np.sum(hard_flags))
            valid_landmarks = landmarks[mask]
            if valid_landmarks.size:
                wrist_errors.extend(np.linalg.norm(valid_landmarks[:, 0, :], axis=1).astype(float).tolist())
                wrist_middle_scales.extend(np.linalg.norm(valid_landmarks[:, 9, :] - valid_landmarks[:, 0, :], axis=1).astype(float).tolist())

    metadata_path = output_root / "sample_metadata.jsonl"
    if metadata_path.exists():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("source_name") == "hard_procedural_3d":
                yaw_values.append(float(record.get("yaw_degrees", 0.0)))
                pitch_values.append(float(record.get("pitch_degrees", 0.0)))
                roll_values.append(float(record.get("roll_degrees", 0.0)))
    else:
        failures.append("missing sample_metadata.jsonl")

    if len(sample_ids) != expected_sample_count:
        failures.append(f"expected {expected_sample_count} samples, got {len(sample_ids)}")
    if len(set(sample_ids)) != len(sample_ids):
        failures.append("sample IDs are not unique")
    if set(target_counts) != set(TARGET_NAMES):
        failures.append(f"unexpected target labels {dict(target_counts)}")
    if target_counts.get("paper", 0) != target_counts.get("scissors", 0):
        failures.append(f"target counts are not balanced: {dict(target_counts)}")
    if wrist_errors and max(wrist_errors) > 1.0e-4:
        failures.append(f"wrist origin invariant failed: max {max(wrist_errors)}")

    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "sample_count": len(sample_ids),
        "target_counts": dict(sorted(target_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "hard_example_count": hard_count,
        "hard_example_ratio": float(hard_count / max(1, len(sample_ids))),
        "length_min": int(min(lengths)) if lengths else None,
        "length_max": int(max(lengths)) if lengths else None,
        "length_mean": float(np.mean(lengths)) if lengths else None,
        "wrist_origin_max_error": float(max(wrist_errors)) if wrist_errors else None,
        "wrist_to_middle_mcp_scale_mean": float(np.mean(wrist_middle_scales)) if wrist_middle_scales else None,
        "viewpoint_distribution": {
            "yaw_min": float(min(yaw_values)) if yaw_values else None,
            "yaw_max": float(max(yaw_values)) if yaw_values else None,
            "pitch_min": float(min(pitch_values)) if pitch_values else None,
            "pitch_max": float(max(pitch_values)) if pitch_values else None,
            "roll_min": float(min(roll_values)) if roll_values else None,
            "roll_max": float(max(roll_values)) if roll_values else None,
        },
    }


def write_sample_metadata(output_root: Path, samples: list[SkeletonSample]) -> Path:
    """Write one metadata JSONL record per sample."""

    path = output_root / "sample_metadata.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            record = {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "target_name": sample.target_name,
                "transition_label": sample.transition_label,
                "length": sample.length,
                "source_name": sample.source_name,
                "hard_example": sample.hard_example,
                **sample.metadata,
            }
            handle.write(json.dumps(_json_ready(record), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_dataset_card(output_root: Path, validation: dict[str, object]) -> Path:
    """Write a short dataset card for the hard-expanded skeleton artifact."""

    path = output_root / "dataset_card.md"
    text = f"""# View-Robust Hard-Expanded Skeleton Dataset, 2026-06-11

This dataset preserves the existing real-guided skeleton shard contract and adds balanced 3D procedural hard examples for early final-gesture prediction.

## Status

- validation: `{validation["status"]}`
- samples: `{validation["sample_count"]}`
- target counts: `{validation["target_counts"]}`
- split counts: `{validation["split_counts"]}`
- source counts: `{validation["source_counts"]}`
- hard example ratio: `{validation["hard_example_ratio"]}`

## Contract

Each shard contains padded `canonical_landmarks` with shape `(N, 72, 21, 3)`, `lengths`, `mask`, `progress`, `target_names`, and `sample_ids`.

The generated hard examples use wrist-origin, palm-local 3D canonical coordinates and are compatible with the existing `landmark_velocity_126` feature loader.
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[dict[str, object]], *, fieldnames: list[str]) -> None:
    """Write CSV rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def write_json(path: Path, data: object) -> None:
    """Write indented JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _arrays_for_samples(samples: list[SkeletonSample], *, sequence_length: int) -> dict[str, NDArray[np.generic]]:
    count = len(samples)
    landmarks = np.zeros((count, sequence_length, 21, 3), dtype=np.float32)
    mask = np.zeros((count, sequence_length), dtype=np.bool_)
    lengths = np.zeros((count,), dtype=np.int64)
    progress = np.zeros((count, sequence_length), dtype=np.float32)
    target_names = np.empty((count,), dtype="<U16")
    label_names = np.empty((count,), dtype="<U24")
    split_names = np.empty((count,), dtype="<U5")
    sample_ids = np.empty((count,), dtype="<U64")
    labels = np.zeros((count,), dtype=np.int64)
    source_names = np.empty((count,), dtype="<U24")
    hard_flags = np.zeros((count,), dtype=np.bool_)
    for index, sample in enumerate(samples):
        landmarks[index] = sample.canonical_landmarks
        mask[index] = sample.mask
        lengths[index] = sample.length
        progress[index] = sample.progress
        target_names[index] = sample.target_name
        label_names[index] = sample.transition_label
        split_names[index] = sample.split
        sample_ids[index] = sample.sample_id
        labels[index] = sample.label_id
        source_names[index] = sample.source_name
        hard_flags[index] = sample.hard_example
    features = landmark_velocity_features(landmarks, mask=mask, lengths=lengths)
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "label_names": label_names,
        "target_names": target_names,
        "split_names": split_names,
        "lengths": lengths,
        "mask": mask,
        "progress": progress,
        "canonical_landmarks": landmarks,
        "features": features,
        "source_names": source_names,
        "hard_example_flags": hard_flags,
        "feature_names": np.asarray(["canonical_xyz", "velocity_xyz"]),
    }


def _target_pose(target_name: TargetName, rng: np.random.Generator) -> tuple[dict[FingerName, float], dict[FingerName, float]]:
    if target_name == "paper":
        curls = {
            "thumb": float(rng.uniform(0.04, 0.30)),
            "index": float(rng.uniform(0.02, 0.18)),
            "middle": float(rng.uniform(0.02, 0.18)),
            "ring": float(rng.uniform(0.02, 0.20)),
            "pinky": float(rng.uniform(0.02, 0.22)),
        }
        spread = float(rng.uniform(0.08, 0.30))
    else:
        curls = {
            "thumb": float(rng.uniform(0.18, 0.78)),
            "index": float(rng.uniform(0.02, 0.18)),
            "middle": float(rng.uniform(0.02, 0.18)),
            "ring": float(rng.uniform(0.82, 0.99)),
            "pinky": float(rng.uniform(0.82, 0.99)),
        }
        spread = float(rng.uniform(0.04, 0.24))
    spreads: dict[FingerName, float] = {
        "thumb": spread * 1.2,
        "index": -spread,
        "middle": spread * 0.10,
        "ring": spread * 0.40,
        "pinky": spread,
    }
    return curls, spreads


def _finger_onsets(target_name: TargetName, rng: np.random.Generator) -> dict[FingerName, float]:
    base = float(rng.uniform(0.26, 0.42))
    if target_name == "paper":
        return {
            "thumb": min(0.48, base + float(rng.uniform(0.00, 0.10))),
            "index": min(0.46, base + float(rng.uniform(0.00, 0.06))),
            "middle": min(0.46, base + float(rng.uniform(0.00, 0.06))),
            "ring": min(0.49, base + float(rng.uniform(0.06, 0.16))),
            "pinky": min(0.50, base + float(rng.uniform(0.08, 0.18))),
        }
    return {
        "thumb": min(0.52, base + float(rng.uniform(0.06, 0.20))),
        "index": min(0.46, base + float(rng.uniform(0.00, 0.06))),
        "middle": min(0.46, base + float(rng.uniform(0.00, 0.06))),
        "ring": 0.98,
        "pinky": 0.98,
    }


def _motion_progress(
    progress: float,
    *,
    onset: float,
    hesitation_center: float,
    hesitation_width: float,
    hesitation_strength: float,
) -> float:
    if progress <= onset:
        return 0.0
    normalized = max(0.0, min(1.0, (progress - onset) / max(1.0 - onset, 1.0e-6)))
    smooth = normalized * normalized * (3.0 - 2.0 * normalized)
    hesitation = hesitation_strength * math.exp(-((progress - hesitation_center) ** 2) / (2.0 * hesitation_width * hesitation_width))
    return max(0.0, min(1.0, smooth - hesitation))


def _sample_identity(rng: np.random.Generator, *, person_id: int) -> PersonIdentity:
    handedness: Handedness = "left" if bool(rng.integers(0, 2)) else "right"
    finger_scales = {finger: float(rng.uniform(0.82, 1.22)) for finger in FINGER_NAMES}
    return PersonIdentity(
        person_id=person_id,
        handedness=handedness,
        palm_width_m=float(rng.uniform(0.075, 0.110)),
        finger_length_scales=finger_scales,
        thumb_angle_offset=float(rng.uniform(-0.18, 0.18)),
    )


def _progress_from_lengths(lengths: NDArray[np.int64], sequence_length: int) -> NDArray[np.float32]:
    progress = np.zeros((lengths.shape[0], sequence_length), dtype=np.float32)
    for index, raw_length in enumerate(lengths.tolist()):
        length = max(1, min(int(raw_length), sequence_length))
        progress[index, :length] = np.linspace(0.0, 1.0, length, dtype=np.float32)
        if length < sequence_length:
            progress[index, length:] = 1.0
    return progress


def _prepare_output(output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to replace generated files.")
    if output_root.exists():
        shards_root = output_root / "shards"
        if shards_root.exists():
            shutil.rmtree(shards_root)
        for name in (
            "sample_metadata.jsonl",
            "shard_index.csv",
            "validation_summary.json",
            "generation_config.json",
            "dataset_card.md",
            "run_summary.json",
        ):
            path = output_root / name
            if path.exists():
                path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)


def _validate_config(config: HardExpansionConfig) -> None:
    if config.generated_per_target <= 0:
        raise ValueError("generated_per_target must be positive")
    if config.sequence_length != 72:
        raise ValueError("sequence_length must be 72 for the current predictor contract")
    if not 1 <= config.min_length <= config.sequence_length:
        raise ValueError("min_length must be in [1, sequence_length]")
    if config.shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= config.val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if config.train_fraction + config.val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")


def _target_name(value: str) -> TargetName:
    if value not in TARGET_NAMES:
        raise ValueError(f"Unsupported target name: {value}")
    return cast(TargetName, value)


def _decode_string_array(values: NDArray[np.generic]) -> list[str]:
    return [str(value) for value in values.tolist()]


def _normalize(vector: NDArray[np.float32], *, fallback: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-8:
        return fallback.astype(np.float32, copy=True)
    return cast(NDArray[np.float32], (vector / np.float32(norm)).astype(np.float32))


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True)
    return _json_ready(value)
