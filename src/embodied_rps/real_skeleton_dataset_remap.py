"""Remap real-skeleton shard targets for two-stage RPS experiments."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np

RemapMode = Literal["rock_vs_transition", "paper_vs_scissors"]


def remap_real_skeleton_dataset(source_root: Path, output_root: Path, *, mode: RemapMode) -> dict[str, object]:
    """Write a remapped shard dataset for a two-stage skeleton classifier."""

    shards_root = source_root / "shards"
    if not shards_root.exists():
        raise FileNotFoundError(f"Missing source shard root: {shards_root}")
    output_shards = output_root / "shards"
    output_root.mkdir(parents=True, exist_ok=True)
    written_shards: list[str] = []
    target_counts: Counter[str] = Counter()
    sample_count = 0
    for split in ("train", "val", "test"):
        split_dir = shards_root / split
        output_split_dir = output_shards / split
        output_split_dir.mkdir(parents=True, exist_ok=True)
        for shard_path in sorted(split_dir.glob("*.npz")):
            with np.load(shard_path, allow_pickle=False) as shard:
                targets = _decode_strings(np.asarray(shard["target_names"]))
                keep_indices, remapped_targets = _remap_targets(targets, mode=mode)
                if not keep_indices:
                    continue
                output_path = output_split_dir / shard_path.name
                payload = {
                    key: np.asarray(shard[key])[keep_indices] if _is_sample_axis_array(np.asarray(shard[key]), len(targets)) else np.asarray(shard[key])
                    for key in shard.files
                    if key != "target_names"
                }
                payload["target_names"] = np.asarray(remapped_targets)
                np.savez_compressed(output_path, **payload)
                written_shards.append(output_path.as_posix())
                target_counts.update(remapped_targets)
                sample_count += len(remapped_targets)
    if sample_count == 0:
        raise ValueError(f"Remap produced no samples for mode: {mode}")
    summary = {
        "source_root": source_root.as_posix(),
        "output_root": output_root.as_posix(),
        "mode": mode,
        "sample_count": sample_count,
        "target_counts": dict(sorted(target_counts.items())),
        "written_shards": written_shards,
    }
    (output_root / "remap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _remap_targets(targets: list[str], *, mode: RemapMode) -> tuple[list[int], list[str]]:
    keep_indices: list[int] = []
    remapped: list[str] = []
    for index, target in enumerate(targets):
        if mode == "rock_vs_transition":
            if target == "rock":
                label = "rock"
            elif target in {"paper", "scissors"}:
                label = "transition"
            else:
                raise ValueError(f"Unsupported source target: {target}")
            keep_indices.append(index)
            remapped.append(label)
        elif mode == "paper_vs_scissors":
            if target in {"paper", "scissors"}:
                keep_indices.append(index)
                remapped.append(target)
            elif target != "rock":
                raise ValueError(f"Unsupported source target: {target}")
        else:
            raise ValueError(f"Unsupported remap mode: {mode}")
    return keep_indices, remapped


def _is_sample_axis_array(value: np.ndarray, sample_count: int) -> bool:
    return value.ndim > 0 and int(value.shape[0]) == sample_count


def _decode_strings(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for value in values.reshape(-1).tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


__all__ = ["RemapMode", "remap_real_skeleton_dataset"]
