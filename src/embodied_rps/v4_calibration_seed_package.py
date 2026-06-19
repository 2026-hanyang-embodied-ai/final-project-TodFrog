"""Build canonical v4 real-seed packages from approved skeleton-review outputs."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from embodied_rps.hard_example_skeletons import canonicalize_mediapipe_style
from embodied_rps.real_skeleton_training import landmark_velocity_features
from embodied_rps.three_class_wait_skeletons import TARGET_TO_LABEL
from embodied_rps.v4_calibration_intake import validate_v4_review_manifest_for_dataset

TARGET_NAMES: tuple[str, ...] = ("rock", "paper", "scissors")


@dataclass(frozen=True)
class V4SeedPackageConfig:
    """Configuration for converting approved v4 review landmarks into seed NPZ."""

    review_manifest_path: Path
    skeleton_review_plan_path: Path
    output_root: Path
    sequence_length: int = 72
    min_detection_coverage: float = 0.98
    allow_missing_review: bool = False


def build_v4_calibration_seed_package(config: V4SeedPackageConfig) -> dict[str, object]:
    """Build canonical seed NPZ/metadata from an approved v4 skeleton review manifest."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    if not config.review_manifest_path.exists():
        if not config.allow_missing_review:
            raise FileNotFoundError(f"Review manifest does not exist: {config.review_manifest_path}")
        summary = _awaiting_review_summary(config)
        _write_json(config.output_root / "seed_package_summary.json", summary)
        _write_markdown(config.output_root / "seed_package_summary.md", summary)
        return summary

    skeleton_plan = _load_json(config.skeleton_review_plan_path)
    review_manifest = _load_json(config.review_manifest_path)
    failures = validate_v4_review_manifest_for_dataset(
        review_manifest,
        skeleton_plan=skeleton_plan,
        min_detection_coverage=config.min_detection_coverage,
    )
    if failures:
        summary = {
            "status": "review_not_ready_for_seed_package",
            "review_manifest": config.review_manifest_path.as_posix(),
            "output_root": config.output_root.as_posix(),
            "failures": failures,
        }
        _write_json(config.output_root / "seed_package_summary.json", summary)
        _write_markdown(config.output_root / "seed_package_summary.md", summary)
        return summary

    records = _mapping_sequence(review_manifest.get("records"))
    seed_records = [_seed_from_review_record(record, sequence_length=config.sequence_length) for record in records]
    validation = validate_seed_records(seed_records, expected_count=int(review_manifest.get("video_count", len(seed_records))))
    summary_status = "passed" if validation["status"] == "passed" else "failed"
    npz_path = config.output_root / "v4_calibration_seed_dataset.npz"
    metadata_path = config.output_root / "seed_metadata.jsonl"
    quality_path = config.output_root / "seed_quality_summary.csv"
    _write_seed_npz(npz_path, seed_records, sequence_length=config.sequence_length)
    _write_seed_metadata(metadata_path, seed_records)
    _write_seed_quality_csv(quality_path, seed_records)
    summary = {
        "status": summary_status,
        "review_manifest": config.review_manifest_path.as_posix(),
        "skeleton_review_plan": config.skeleton_review_plan_path.as_posix(),
        "output_root": config.output_root.as_posix(),
        "seed_npz": npz_path.as_posix(),
        "seed_metadata": metadata_path.as_posix(),
        "seed_quality_summary": quality_path.as_posix(),
        "sequence_length": config.sequence_length,
        "sample_count": len(seed_records),
        "target_counts": dict(sorted(Counter(record["target_name"] for record in seed_records).items())),
        "validation": validation,
    }
    _write_json(config.output_root / "seed_package_summary.json", summary)
    _write_markdown(config.output_root / "seed_package_summary.md", summary)
    return summary


def validate_seed_records(seed_records: Sequence[Mapping[str, object]], *, expected_count: int) -> dict[str, object]:
    """Validate canonical v4 seed records before writing downstream datasets."""

    failures: list[dict[str, object]] = []
    target_counts = Counter(str(record.get("target_name")) for record in seed_records)
    sample_ids = [str(record.get("sample_id")) for record in seed_records]
    if len(seed_records) != expected_count:
        failures.append({"code": "seed_count_mismatch", "actual": len(seed_records), "expected": expected_count})
    if len(set(sample_ids)) != len(sample_ids):
        failures.append({"code": "duplicate_seed_sample_ids"})
    if set(target_counts) != set(TARGET_NAMES):
        failures.append({"code": "missing_seed_target", "target_counts": dict(sorted(target_counts.items()))})
    for record in seed_records:
        canonical = cast(NDArray[np.float32], record["canonical_landmarks"])
        mask = cast(NDArray[np.bool_], record["mask"])
        features = cast(NDArray[np.float32], record["features"])
        sample_id = str(record.get("sample_id"))
        if canonical.ndim != 3 or canonical.shape[1:] != (21, 3):
            failures.append({"code": "bad_canonical_shape", "sample_id": sample_id, "shape": list(canonical.shape)})
        if mask.ndim != 1 or mask.shape[0] != canonical.shape[0]:
            failures.append({"code": "bad_mask_shape", "sample_id": sample_id})
        if not np.all(np.isfinite(canonical[mask])):
            failures.append({"code": "non_finite_canonical", "sample_id": sample_id})
        if not np.all(np.isfinite(features[mask])):
            failures.append({"code": "non_finite_features", "sample_id": sample_id})
        valid = canonical[mask]
        if valid.size:
            wrist_error = float(np.max(np.linalg.norm(valid[:, 0, :], axis=1)))
            middle_scale = float(np.mean(np.linalg.norm(valid[:, 9, :] - valid[:, 0, :], axis=1)))
            if wrist_error > 1.0e-4:
                failures.append({"code": "wrist_origin_invariant_failed", "sample_id": sample_id, "wrist_error": wrist_error})
            if not 0.95 <= middle_scale <= 1.05:
                failures.append({"code": "middle_mcp_scale_invariant_failed", "sample_id": sample_id, "scale": middle_scale})
    return {
        "status": "passed" if not failures else "failed",
        "sample_count": len(seed_records),
        "expected_count": expected_count,
        "target_counts": dict(sorted(target_counts.items())),
        "failures": failures,
    }


def _seed_from_review_record(record: Mapping[str, object], *, sequence_length: int) -> dict[str, object]:
    landmarks_json = Path(str(record["landmarks_json"]))
    payload = _load_json(landmarks_json)
    frames = _mapping_sequence(payload.get("frames"))
    detected_frames = [frame for frame in frames if bool(frame.get("detected"))]
    if len(detected_frames) != len(frames):
        raise ValueError(f"Review JSON contains missing detections: {landmarks_json}")
    selected_indices = _selected_frame_indices(len(detected_frames), sequence_length)
    valid_length = len(selected_indices)
    canonical = np.zeros((sequence_length, 21, 3), dtype=np.float32)
    mask = np.zeros((sequence_length,), dtype=np.bool_)
    progress = np.zeros((sequence_length,), dtype=np.float32)
    last_valid = np.zeros((21, 3), dtype=np.float32)
    for out_index, source_index in enumerate(selected_indices):
        frame = detected_frames[source_index]
        raw = _landmarks_array(_mapping_sequence(frame.get("landmarks")))
        last_valid = canonicalize_mediapipe_style(raw)
        canonical[out_index] = last_valid
        mask[out_index] = True
    if valid_length > 0:
        progress[:valid_length] = np.linspace(0.0, 1.0, valid_length, dtype=np.float32)
        if valid_length < sequence_length:
            canonical[valid_length:] = last_valid
            progress[valid_length:] = 1.0
    lengths = np.asarray([valid_length], dtype=np.int64)
    features = landmark_velocity_features(canonical[None, ...], mask=mask[None, ...], lengths=lengths)[0]
    target_name = str(record["label"])
    return {
        "sample_id": f"v4_seed_{target_name}_{str(record['video_id']).split('_')[-1]}",
        "video_id": str(record["video_id"]),
        "source_path": str(record["source_path"]),
        "landmarks_json": landmarks_json.as_posix(),
        "target_name": target_name,
        "label": int(TARGET_TO_LABEL[target_name]),
        "canonical_landmarks": canonical,
        "features": features,
        "mask": mask,
        "progress": progress,
        "length": valid_length,
        "source_frame_count": len(frames),
        "selected_frame_indices": selected_indices,
        "detection_coverage": float(record.get("detection_coverage", 0.0)),
        "average_primary_score": record.get("average_primary_score"),
    }


def _selected_frame_indices(frame_count: int, sequence_length: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= sequence_length:
        return list(range(frame_count))
    return [int(round(value)) for value in np.linspace(0, frame_count - 1, sequence_length)]


def _landmarks_array(landmarks: Sequence[Mapping[str, object]]) -> NDArray[np.float32]:
    if len(landmarks) != 21:
        raise ValueError(f"expected 21 landmarks, got {len(landmarks)}")
    points = np.zeros((21, 3), dtype=np.float32)
    for item in landmarks:
        index = int(float(item["index"]))
        points[index] = np.asarray((float(item["x"]), float(item["y"]), float(item["z"])), dtype=np.float32)
    return points


def _write_seed_npz(path: Path, seed_records: Sequence[Mapping[str, object]], *, sequence_length: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(seed_records)
    canonical = np.stack([cast(NDArray[np.float32], record["canonical_landmarks"]) for record in seed_records]).astype(np.float32)
    features = np.stack([cast(NDArray[np.float32], record["features"]) for record in seed_records]).astype(np.float32)
    mask = np.stack([cast(NDArray[np.bool_], record["mask"]) for record in seed_records]).astype(np.bool_)
    progress = np.stack([cast(NDArray[np.float32], record["progress"]) for record in seed_records]).astype(np.float32)
    if canonical.shape != (count, sequence_length, 21, 3):
        raise ValueError(f"unexpected canonical seed shape {canonical.shape}")
    np.savez_compressed(
        path,
        sample_ids=np.asarray([str(record["sample_id"]) for record in seed_records], dtype="<U96"),
        labels=np.asarray([int(record["label"]) for record in seed_records], dtype=np.int64),
        target_names=np.asarray([str(record["target_name"]) for record in seed_records], dtype="<U16"),
        lengths=np.asarray([int(record["length"]) for record in seed_records], dtype=np.int64),
        mask=mask,
        progress=progress,
        canonical_landmarks=canonical,
        features=features,
        source_paths=np.asarray([str(record["source_path"]) for record in seed_records], dtype="<U512"),
        landmarks_json=np.asarray([str(record["landmarks_json"]) for record in seed_records], dtype="<U512"),
    )


def _write_seed_metadata(path: Path, seed_records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in seed_records:
            payload = {key: value for key, value in record.items() if key not in {"canonical_landmarks", "features", "mask", "progress"}}
            handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _write_seed_quality_csv(path: Path, seed_records: Sequence[Mapping[str, object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "video_id",
        "target_name",
        "length",
        "source_frame_count",
        "detection_coverage",
        "average_primary_score",
        "source_path",
        "landmarks_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in seed_records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def _awaiting_review_summary(config: V4SeedPackageConfig) -> dict[str, object]:
    return {
        "status": "awaiting_skeleton_review",
        "review_manifest": config.review_manifest_path.as_posix(),
        "skeleton_review_plan": config.skeleton_review_plan_path.as_posix(),
        "output_root": config.output_root.as_posix(),
        "seed_npz": (config.output_root / "v4_calibration_seed_dataset.npz").as_posix(),
        "message": "Seed package cannot be built until the v4 skeleton review manifest exists and passes readiness checks.",
    }


def _write_markdown(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "# V4 Calibration Seed Package Summary",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Review manifest: `{summary.get('review_manifest')}`",
        f"- Output root: `{summary.get('output_root')}`",
        f"- Seed NPZ: `{summary.get('seed_npz')}`",
        "",
    ]
    if summary.get("message"):
        lines.extend(["## Message", "", str(summary["message"]), ""])
    validation = summary.get("validation")
    if isinstance(validation, Mapping):
        lines.extend(["## Validation", "", f"- Status: `{validation.get('status')}`", f"- Sample count: `{validation.get('sample_count')}`", ""])
    failures = summary.get("failures")
    if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes)) and failures:
        lines.extend(["## Blocking Issues", ""])
        for failure in failures:
            lines.append(f"- `{json.dumps(failure, ensure_ascii=False)}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping_sequence(value: object) -> Sequence[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


__all__ = [
    "V4SeedPackageConfig",
    "build_v4_calibration_seed_package",
    "validate_seed_records",
]
