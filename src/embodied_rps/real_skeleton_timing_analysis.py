"""Timing-gap analysis between real skeleton reviews and generated metadata."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from embodied_rps.real_skeleton_failure_features import FINGER_CHAINS, FINGER_NAMES, load_review_clip


@dataclass(frozen=True)
class FingerTiming:
    """Opening timing estimate for one finger trajectory."""

    opened: bool
    open_progress: float | None
    start_extension: float
    final_extension: float
    extension_delta: float


def normalize_final_label(label: str) -> str:
    """Normalize review/generator labels to final opponent gesture labels."""

    value = label.strip().lower().replace("-", "_").replace(" ", "_")
    if "scissors" in value or "가위" in label:
        return "scissors"
    if "paper" in value or "보" in label:
        return "paper"
    if "rock" in value or "바위" in label:
        return "rock"
    return value


def estimate_finger_open_progress(
    canonical_landmarks: NDArray[np.float32],
    *,
    threshold_fraction: float = 0.5,
    minimum_delta: float = 0.08,
) -> dict[str, FingerTiming]:
    """Estimate first progress where each finger crosses its own opening midpoint."""

    frames = np.asarray(canonical_landmarks, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (21, 3):
        raise ValueError("canonical_landmarks must have shape (T,21,3)")
    frame_count = int(frames.shape[0])
    if frame_count == 0:
        raise ValueError("canonical_landmarks must contain at least one frame")

    extensions = _finger_extensions(frames)
    progress = np.linspace(0.0, 1.0, frame_count, dtype=np.float64) if frame_count > 1 else np.asarray([0.0])
    edge_count = max(1, int(round(frame_count * 0.1)))
    final_count = max(1, int(round(frame_count * 0.2)))

    timing: dict[str, FingerTiming] = {}
    for finger, values in extensions.items():
        start_extension = float(np.median(values[:edge_count]))
        final_extension = float(np.median(values[-final_count:]))
        extension_delta = final_extension - start_extension
        if extension_delta < minimum_delta:
            timing[finger] = FingerTiming(
                opened=False,
                open_progress=None,
                start_extension=start_extension,
                final_extension=final_extension,
                extension_delta=extension_delta,
            )
            continue

        threshold = start_extension + extension_delta * threshold_fraction
        crossing_indices = np.flatnonzero(values >= threshold)
        open_progress = float(progress[int(crossing_indices[0])]) if crossing_indices.size else None
        timing[finger] = FingerTiming(
            opened=open_progress is not None,
            open_progress=open_progress,
            start_extension=start_extension,
            final_extension=final_extension,
            extension_delta=extension_delta,
        )
    return timing


def analyze_timing_gap(
    *,
    review_roots: list[Path],
    synthetic_metadata_paths: list[Path],
    threshold_fraction: float = 0.5,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Analyze real review opening timing and synthetic onset metadata."""

    real_rows = summarize_real_review_timing(
        review_roots=review_roots,
        threshold_fraction=threshold_fraction,
    )
    synthetic_rows, synthetic_summary = summarize_synthetic_metadata(synthetic_metadata_paths)
    summary = {
        "review_roots": [path.as_posix() for path in review_roots],
        "synthetic_metadata_paths": [path.as_posix() for path in synthetic_metadata_paths],
        "real": _summarize_real_rows(real_rows),
        "synthetic": synthetic_summary,
        "gap": _summarize_gap(real_rows, synthetic_rows),
    }
    return real_rows, synthetic_rows, summary


def summarize_real_review_timing(
    *,
    review_roots: list[Path],
    threshold_fraction: float = 0.5,
) -> list[dict[str, object]]:
    """Load review JSON files and summarize per-clip finger opening progress."""

    rows: list[dict[str, object]] = []
    for root in review_roots:
        review_paths = sorted((root / "landmarks_json").rglob("*.json"))
        if not review_paths:
            raise ValueError(f"no review JSON files found under {root / 'landmarks_json'}")
        for path in review_paths:
            clip = load_review_clip(path)
            raw_label = _infer_review_label(path=path, fallback=clip.label)
            label = normalize_final_label(raw_label)
            timing = estimate_finger_open_progress(
                clip.canonical_landmarks,
                threshold_fraction=threshold_fraction,
            )
            row: dict[str, object] = {
                "review_root": root.as_posix(),
                "clip_id": clip.video_id,
                "label": label,
                "raw_label": raw_label,
                "source_path": clip.source_path,
                "frame_count": int(clip.canonical_landmarks.shape[0]),
            }
            for finger in FINGER_NAMES:
                item = timing[finger]
                row[f"{finger}_opened"] = item.opened
                row[f"{finger}_open_progress"] = item.open_progress
                row[f"{finger}_start_extension"] = item.start_extension
                row[f"{finger}_final_extension"] = item.final_extension
                row[f"{finger}_extension_delta"] = item.extension_delta
            row["ring_pinky_open_progress_mean"] = _mean_optional(
                [timing["ring"].open_progress, timing["pinky"].open_progress]
            )
            row["index_middle_open_progress_mean"] = _mean_optional(
                [timing["index"].open_progress, timing["middle"].open_progress]
            )
            rows.append(row)
    return rows


def summarize_synthetic_metadata(metadata_paths: list[Path]) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Summarize generator onset metadata by profile and target label."""

    groups: dict[tuple[str, str], dict[str, list[float]]] = {}
    group_counts: Counter[tuple[str, str]] = Counter()
    records_total = 0
    records_with_onsets = 0
    records_without_onsets = 0
    metadata_files: list[str] = []

    for path in metadata_paths:
        if not path.exists():
            raise ValueError(f"synthetic metadata path does not exist: {path}")
        metadata_files.append(path.as_posix())
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                records_total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} contains invalid JSON") from exc
                onsets = record.get("onsets")
                if not isinstance(onsets, dict):
                    records_without_onsets += 1
                    continue
                target_name = normalize_final_label(str(record.get("target_name", record.get("transition_label", ""))))
                source_name = str(record.get("source_name", record.get("augmentation_profile", "unknown")))
                key = (source_name, target_name)
                group_counts[key] += 1
                group = groups.setdefault(key, {finger: [] for finger in FINGER_NAMES})
                for finger in FINGER_NAMES:
                    value = onsets.get(finger)
                    if isinstance(value, int | float):
                        group[finger].append(float(value))
                records_with_onsets += 1

    rows: list[dict[str, object]] = []
    for (source_name, target_name), values_by_finger in sorted(groups.items()):
        row: dict[str, object] = {
            "source_name": source_name,
            "target_name": target_name,
            "count": int(group_counts[(source_name, target_name)]),
        }
        for finger in FINGER_NAMES:
            values = values_by_finger[finger]
            row[f"{finger}_onset_mean"] = _mean(values)
            row[f"{finger}_onset_median"] = _median(values)
            row[f"{finger}_onset_p25"] = _percentile(values, 25.0)
            row[f"{finger}_onset_p75"] = _percentile(values, 75.0)
        row["ring_pinky_onset_median_mean"] = _mean_optional(
            [row["ring_onset_median"], row["pinky_onset_median"]]
        )
        row["index_middle_onset_median_mean"] = _mean_optional(
            [row["index_onset_median"], row["middle_onset_median"]]
        )
        rows.append(row)

    summary = {
        "metadata_files": metadata_files,
        "records_total": records_total,
        "records_with_onsets": records_with_onsets,
        "records_without_onsets": records_without_onsets,
        "group_count": len(rows),
        "target_counts_with_onsets": dict(
            sorted(Counter(target for (_, target), count in group_counts.items() for _ in range(count)).items())
        ),
    }
    return rows, summary


def expand_synthetic_metadata_paths(inputs: list[Path]) -> list[Path]:
    """Expand metadata files or dataset roots into concrete sample_metadata.jsonl paths."""

    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            direct = item / "sample_metadata.jsonl"
            if direct.exists():
                paths.append(direct)
            else:
                paths.extend(sorted(item.rglob("sample_metadata.jsonl")))
        else:
            paths.append(item)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = path.resolve().as_posix().lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _infer_review_label(*, path: Path, fallback: str) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if isinstance(payload, dict):
        for key in ("label", "transition_label", "target_name", "source_folder", "video_id", "clip_id"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                normalized = normalize_final_label(value)
                if normalized in {"rock", "paper", "scissors"}:
                    return value
    for candidate in (path.stem, path.parent.name, fallback):
        normalized = normalize_final_label(candidate)
        if normalized in {"rock", "paper", "scissors"}:
            return candidate
    return fallback


def _summarize_real_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    label_counts = Counter(str(row["label"]) for row in rows)
    by_label: dict[str, dict[str, float | None]] = {}
    for label in sorted(label_counts):
        label_rows = [row for row in rows if row["label"] == label]
        by_label[label] = {}
        for finger in FINGER_NAMES:
            values = [
                float(row[f"{finger}_open_progress"])
                for row in label_rows
                if isinstance(row.get(f"{finger}_open_progress"), int | float)
            ]
            by_label[label][f"{finger}_open_progress_median"] = _median(values)
            by_label[label][f"{finger}_open_progress_mean"] = _mean(values)
        by_label[label]["ring_pinky_open_progress_median_mean"] = _mean_optional(
            [
                by_label[label].get("ring_open_progress_median"),
                by_label[label].get("pinky_open_progress_median"),
            ]
        )
        by_label[label]["index_middle_open_progress_median_mean"] = _mean_optional(
            [
                by_label[label].get("index_open_progress_median"),
                by_label[label].get("middle_open_progress_median"),
            ]
        )
    return {
        "clip_count": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "by_label": by_label,
    }


def _summarize_gap(real_rows: list[dict[str, object]], synthetic_rows: list[dict[str, object]]) -> dict[str, object]:
    gap: dict[str, object] = {}
    real_paper_ring = _median(_column(real_rows, "ring_open_progress", label="paper"))
    real_paper_pinky = _median(_column(real_rows, "pinky_open_progress", label="paper"))
    synthetic_paper_ring = _median(_column(synthetic_rows, "ring_onset_median", target_name="paper"))
    synthetic_paper_pinky = _median(_column(synthetic_rows, "pinky_onset_median", target_name="paper"))
    if real_paper_ring is not None and synthetic_paper_ring is not None:
        gap["paper_ring_open_minus_synthetic_onset_median"] = real_paper_ring - synthetic_paper_ring
    if real_paper_pinky is not None and synthetic_paper_pinky is not None:
        gap["paper_pinky_open_minus_synthetic_onset_median"] = real_paper_pinky - synthetic_paper_pinky
    if real_paper_ring is not None:
        gap["real_paper_ring_open_progress_median"] = real_paper_ring
    if real_paper_pinky is not None:
        gap["real_paper_pinky_open_progress_median"] = real_paper_pinky
    if synthetic_paper_ring is not None:
        gap["synthetic_paper_ring_onset_median"] = synthetic_paper_ring
    if synthetic_paper_pinky is not None:
        gap["synthetic_paper_pinky_onset_median"] = synthetic_paper_pinky
    return gap


def _finger_extensions(frames: NDArray[np.float32]) -> dict[str, NDArray[np.float64]]:
    denominator = np.maximum(
        np.linalg.norm(frames[:, 9, :] - frames[:, 0, :], axis=1).astype(np.float64),
        0.75,
    )
    values: dict[str, NDArray[np.float64]] = {}
    for finger, chain in FINGER_CHAINS.items():
        values[finger] = (
            np.linalg.norm(frames[:, chain[-1], :] - frames[:, chain[0], :], axis=1).astype(np.float64) / denominator
        )
    return values


def _column(rows: list[dict[str, object]], column: str, **filters: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if any(row.get(key) != value for key, value in filters.items()):
            continue
        item = row.get(column)
        if isinstance(item, int | float):
            values.append(float(item))
    return values


def _mean_optional(values: list[object]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    return _mean(numeric)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


__all__ = [
    "FingerTiming",
    "analyze_timing_gap",
    "estimate_finger_open_progress",
    "expand_synthetic_metadata_paths",
    "normalize_final_label",
    "summarize_real_review_timing",
    "summarize_synthetic_metadata",
]
