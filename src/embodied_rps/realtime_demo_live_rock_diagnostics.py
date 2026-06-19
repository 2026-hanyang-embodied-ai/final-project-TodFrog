"""Diagnostics and seed packaging for live rock-hold false triggers."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from embodied_rps.real_skeleton_training import landmark_velocity_features
from embodied_rps.tools.run_realtime_skeleton_predictor import canonicalize_mediapipe_landmarks


@dataclass(frozen=True)
class LiveRockDiagnosticsConfig:
    """Input/output paths for a saved live rock-hold false-trigger run."""

    output_root: Path
    frame_log: Path
    postcapture_summary: Path
    archive_run_id: str = "run_20260616_163603"


def build_live_rock_false_trigger_diagnostics(config: LiveRockDiagnosticsConfig) -> dict[str, object]:
    """Write compact diagnostics for an expected-rock live run."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(config.frame_log)
    postcapture = _read_json(config.postcapture_summary)
    gate = _dict_value(postcapture, "demo_success_gate")
    first_false = _first_false_trigger(gate, rows)
    response_rows = [row for row in rows if bool(row.get("response_window"))]
    non_response_rows = [row for row in rows if not bool(row.get("response_window"))]
    detection_rate = _detection_rate(rows)
    diagnostic_status = (
        "rock_false_trigger_confirmed"
        if gate.get("expected_actual_gesture") == "rock"
        and gate.get("passed") is False
        and first_false.get("decision_state") in {"paper", "scissors"}
        else "not_confirmed"
    )
    summary: dict[str, object] = {
        "diagnostic_status": diagnostic_status,
        "archive_run_id": config.archive_run_id,
        "frame_log": {
            "path": config.frame_log.as_posix(),
            "record_count": len(rows),
            "detection_rate": detection_rate,
            "raw_prediction_counts": _counts(row.get("prediction") for row in rows),
        },
        "response_window": _window_summary(response_rows),
        "non_response_window": _window_summary(non_response_rows),
        "first_false_trigger": first_false,
        "gate_failure_reasons": _list_value(gate, "failure_reasons"),
        "outputs": {
            "summary_json": (config.output_root / "live_rock_false_trigger_diagnostics.json").as_posix(),
            "summary_md": (config.output_root / "live_rock_false_trigger_diagnostics.md").as_posix(),
        },
        "claim_scope": "diagnostics over existing frame-log/postcapture artifacts; does not run inference or training",
    }
    (config.output_root / "live_rock_false_trigger_diagnostics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (config.output_root / "live_rock_false_trigger_diagnostics.md").write_text(
        _diagnostics_markdown(summary),
        encoding="utf-8",
    )
    return summary


def build_overlay_derived_rock_seed_package(
    *,
    review_json: Path,
    output_root: Path,
    segment_length: int = 72,
    stride: int = 18,
    min_detection_coverage: float = 0.95,
) -> dict[str, object]:
    """Convert overlay-derived MediaPipe review landmarks into rock hard-negative seed windows."""

    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    payload = _read_json(review_json)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{review_json} does not contain a frames list")

    detected_frames: list[dict[str, object]] = [
        frame for frame in frames if isinstance(frame, dict) and bool(frame.get("detected")) and isinstance(frame.get("landmarks"), list)
    ]
    detection_coverage = len(detected_frames) / max(1, len(frames))
    canonical = np.stack(
        [
            canonicalize_mediapipe_landmarks(_landmark_array(cast(list[object], frame["landmarks"])))
            for frame in detected_frames
        ]
    ).astype(np.float32) if detected_frames else np.zeros((0, 21, 3), dtype=np.float32)
    severe_jumps = _severe_jump_count(canonical)
    segments = _segment_canonical(canonical, segment_length=segment_length, stride=stride)
    status = "passed" if detection_coverage >= min_detection_coverage and severe_jumps == 0 and len(segments) > 0 else "blocked"
    training_use = "accepted_overlay_derived_hard_negative" if status == "passed" else "diagnostic_only"
    if segments:
        landmarks = np.stack(segments).astype(np.float32)
    else:
        landmarks = np.zeros((0, segment_length, 21, 3), dtype=np.float32)
    count = int(landmarks.shape[0])
    lengths = np.full((count,), segment_length, dtype=np.int64)
    mask = np.ones((count, segment_length), dtype=np.bool_)
    progress = np.tile(np.linspace(0.0, 1.0, segment_length, dtype=np.float32), (count, 1))
    target_names = np.full((count,), "rock", dtype="<U16")
    label_names = np.full((count,), "rock", dtype="<U24")
    labels = np.zeros((count,), dtype=np.int64)
    sample_ids = np.asarray([f"live_rock_false_trigger_{index:06d}" for index in range(count)], dtype="<U80")
    split_names = np.asarray([_split_for_index(index, count) for index in range(count)], dtype="<U5")
    source_names = np.full((count,), "overlay_derived_live_rock_false_trigger", dtype="<U64")
    hard_flags = np.ones((count,), dtype=np.bool_)
    features = landmark_velocity_features(landmarks, mask=mask, lengths=lengths) if count else np.zeros((0, segment_length, 126), dtype=np.float32)

    np.savez_compressed(
        output_root / "live_rock_false_trigger_seed_dataset.npz",
        sample_ids=sample_ids,
        labels=labels,
        label_names=label_names,
        target_names=target_names,
        split_names=split_names,
        lengths=lengths,
        mask=mask,
        progress=progress,
        canonical_landmarks=landmarks,
        features=features,
        source_names=source_names,
        hard_example_flags=hard_flags,
    )
    _write_segment_metadata(output_root / "segment_metadata.jsonl", sample_ids=sample_ids, split_names=split_names, review_json=review_json)
    summary: dict[str, object] = {
        "status": status,
        "training_use": training_use,
        "review_json": review_json.as_posix(),
        "segment_count": count,
        "segment_length": segment_length,
        "stride": stride,
        "frame_count": len(frames),
        "detected_frame_count": len(detected_frames),
        "detection_coverage": detection_coverage,
        "severe_landmark_jump_count": severe_jumps,
        "outputs": {
            "seed_npz": (output_root / "live_rock_false_trigger_seed_dataset.npz").as_posix(),
            "summary_json": (output_root / "seed_package_summary.json").as_posix(),
            "segment_metadata_jsonl": (output_root / "segment_metadata.jsonl").as_posix(),
        },
        "claim_scope": "overlay-derived rock seed package; accepted only after detection and landmark-jump quality checks",
    }
    (output_root / "seed_package_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_quality_csv(output_root / "quality_summary.csv", summary)
    return summary


def _window_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "frame_count": len(rows),
        "raw_prediction_counts": _counts(row.get("prediction") for row in rows),
        "decision_state_counts": _counts(row.get("decision_state") for row in rows),
        "confirmed_binary_count": sum(
            1
            for row in rows
            if bool(row.get("confirmed_decision")) and row.get("decision_state") in {"paper", "scissors"}
        ),
        "mean_p_rock": _mean(row.get("p_rock") for row in rows),
        "mean_p_paper": _mean(row.get("p_paper") for row in rows),
        "mean_p_scissors": _mean(row.get("p_scissors") for row in rows),
        "mean_transition_mass": _mean(row.get("transition_mass") for row in rows),
    }


def _first_false_trigger(gate: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    first = _dict_value(gate, "first_response_prompt_binary_decision")
    if first:
        return dict(first)
    for row in rows:
        if bool(row.get("response_window")) and row.get("decision_state") in {"paper", "scissors"}:
            return {
                "frame_index": row.get("frame_index"),
                "time_s": row.get("time_s"),
                "decision_state": row.get("decision_state"),
                "robot_action": row.get("robot_action"),
                "confidence": row.get("confidence"),
                "margin": row.get("margin"),
            }
    return {}


def _segment_canonical(canonical: NDArray[np.float32], *, segment_length: int, stride: int) -> list[NDArray[np.float32]]:
    if canonical.shape[0] < segment_length:
        return []
    return [canonical[start : start + segment_length].copy() for start in range(0, canonical.shape[0] - segment_length + 1, stride)]


def _severe_jump_count(canonical: NDArray[np.float32]) -> int:
    if canonical.shape[0] < 3:
        return 0
    deltas = np.linalg.norm(np.diff(canonical, axis=0), axis=2).mean(axis=1)
    median = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - median)))
    threshold = max(1.5, median + 12.0 * mad)
    return int(np.count_nonzero(deltas > threshold))


def _landmark_array(landmarks: Sequence[object]) -> NDArray[np.float32]:
    if len(landmarks) != 21:
        raise ValueError("landmarks must contain 21 points")
    rows: list[tuple[float, float, float]] = []
    for point in landmarks:
        if isinstance(point, Mapping):
            rows.append((float(point.get("x", 0.0)), float(point.get("y", 0.0)), float(point.get("z", 0.0))))
        elif isinstance(point, Sequence) and len(point) >= 3 and not isinstance(point, (str, bytes)):
            rows.append((float(point[0]), float(point[1]), float(point[2])))
        else:
            raise ValueError("each landmark must be a mapping with x/y/z or a sequence of at least three numbers")
    return np.asarray(rows, dtype=np.float32)


def _detection_rate(rows: Sequence[Mapping[str, object]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row.get("detected"))) / len(rows)


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                rows.append(loaded)
    return rows


def _dict_value(payload: Mapping[str, object] | None, key: str) -> dict[str, Any]:
    value = payload.get(key) if payload else None
    return dict(value) if isinstance(value, dict) else {}


def _list_value(payload: Mapping[str, object] | None, key: str) -> list[str]:
    value = payload.get(key) if payload else None
    return [str(item) for item in value] if isinstance(value, list) else []


def _counts(values: Sequence[object] | Any) -> dict[str, int]:
    counter = Counter(str(value) for value in values if value is not None and str(value) != "")
    return dict(sorted(counter.items()))


def _mean(values: Any) -> float | None:
    parsed = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return float(np.mean(parsed)) if parsed else None


def _split_for_index(index: int, count: int) -> str:
    if count <= 0:
        return "test"
    ratio = (index + 0.5) / count
    if ratio < 0.70:
        return "train"
    if ratio < 0.85:
        return "val"
    return "test"


def _write_segment_metadata(path: Path, *, sample_ids: NDArray[np.str_], split_names: NDArray[np.str_], review_json: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample_id, split in zip(sample_ids.tolist(), split_names.tolist(), strict=True):
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "target_name": "rock",
                        "source_name": "overlay_derived_live_rock_false_trigger",
                        "review_json": review_json.as_posix(),
                        "training_role": "rock_wait_front_fist_false_trigger_hard_negative",
                    }
                )
                + "\n"
            )


def _write_quality_csv(path: Path, summary: Mapping[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["status", "training_use", "segment_count", "detection_coverage", "severe_landmark_jump_count"],
        )
        writer.writeheader()
        writer.writerow({key: summary.get(key) for key in writer.fieldnames})


def _diagnostics_markdown(summary: Mapping[str, object]) -> str:
    first_false = summary.get("first_false_trigger")
    response = summary.get("response_window")
    return "\n".join(
        [
            "# Live Rock False-Trigger Diagnostics",
            "",
            f"- Diagnostic status: `{summary.get('diagnostic_status')}`",
            f"- Archive run id: `{summary.get('archive_run_id')}`",
            f"- First false trigger: `{first_false}`",
            f"- Response-window summary: `{response}`",
            "",
        ]
    )


__all__ = [
    "LiveRockDiagnosticsConfig",
    "build_live_rock_false_trigger_diagnostics",
    "build_overlay_derived_rock_seed_package",
]
