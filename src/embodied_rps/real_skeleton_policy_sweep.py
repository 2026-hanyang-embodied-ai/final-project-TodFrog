"""Decision-policy sweeps for saved real-skeleton video validation outputs."""

from __future__ import annotations

import csv
import itertools
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

from embodied_rps.real_skeleton_video_eval import (
    EvaluationGesture,
    StrictDecisionConfig,
    build_validation_summary,
    summarize_clip_decision,
    write_clip_metrics_csv,
)

T = TypeVar("T")


@dataclass(frozen=True)
class SavedValidationClip:
    """A saved clip validation artifact that can be re-scored without inference."""

    clip_id: str
    transition_label: str
    true_gesture: EvaluationGesture
    source_path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    rows: tuple[Mapping[str, object], ...]
    overlay_path: Path | None = None
    frame_csv_path: Path | None = None
    frame_jsonl_path: Path | None = None


def load_saved_validation_clips(validation_root: Path) -> list[SavedValidationClip]:
    """Load saved clip metrics and frame rows from a validation artifact root."""

    if not validation_root.exists():
        raise FileNotFoundError(f"Validation root does not exist: {validation_root}")
    metrics_paths = sorted((validation_root / "clips").glob("*/*/metrics.json"))
    if not metrics_paths:
        raise FileNotFoundError(f"No saved clip metrics found under: {validation_root / 'clips'}")
    return [_load_saved_validation_clip(path) for path in metrics_paths]


def sweep_decision_policies(
    *,
    validation_root: Path,
    output_root: Path | None,
    confidence_thresholds: Sequence[float],
    margin_thresholds: Sequence[float],
    confirmation_counts: Sequence[int],
    max_decision_progress_values: Sequence[float],
    transition_mass_thresholds: Sequence[float],
    paper_wait_terminal_for_transition_values: Sequence[bool] = (True,),
    binary_transition_mass_thresholds: Sequence[float] = (0.0,),
) -> dict[str, object]:
    """Re-score saved validation clips across strict-decision policy combinations."""

    clips = load_saved_validation_clips(validation_root)
    discovery_summary = _load_discovery_summary(validation_root, clips)
    policy_records: list[dict[str, object]] = []
    best_summary: dict[str, object] | None = None
    best_policy: dict[str, object] | None = None
    best_clip_metrics: list[dict[str, object]] = []
    best_rank: tuple[object, ...] | None = None

    combinations = list(
        itertools.product(
            _require_values(confidence_thresholds, "confidence_thresholds"),
            _require_values(margin_thresholds, "margin_thresholds"),
            _require_values(confirmation_counts, "confirmation_counts"),
            _require_values(max_decision_progress_values, "max_decision_progress_values"),
            _require_values(transition_mass_thresholds, "transition_mass_thresholds"),
            _require_values(
                paper_wait_terminal_for_transition_values,
                "paper_wait_terminal_for_transition_values",
            ),
            _require_values(binary_transition_mass_thresholds, "binary_transition_mass_thresholds"),
        )
    )
    for index, (
        confidence_threshold,
        margin_threshold,
        confirmation_count,
        max_decision_progress,
        transition_mass_threshold,
        paper_wait_is_terminal_for_transitions,
        binary_transition_mass_threshold,
    ) in enumerate(combinations, start=1):
        config = StrictDecisionConfig(
            confidence_threshold=float(confidence_threshold),
            margin_threshold=float(margin_threshold),
            confirmation_count=int(confirmation_count),
            max_decision_progress=float(max_decision_progress),
            transition_mass_threshold=float(transition_mass_threshold),
            paper_wait_is_terminal_for_transitions=bool(paper_wait_is_terminal_for_transitions),
            binary_transition_mass_threshold=float(binary_transition_mass_threshold),
        )
        clip_metrics = [_summarize_saved_clip(clip, config=config) for clip in clips]
        summary = build_validation_summary(
            clip_metrics=clip_metrics,
            discovery_summary=discovery_summary,
            config=config,
            event_manifest_path=None,
        )
        policy_id = f"policy_{index:06d}"
        policy = _policy_dict(policy_id=policy_id, config=config)
        record = _policy_record(policy=policy, summary=summary)
        policy_records.append(record)
        rank = _ranking_key(summary=summary, config=config)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_summary = summary
            best_policy = policy
            best_clip_metrics = clip_metrics

    if best_policy is None or best_summary is None:
        raise RuntimeError("No decision policies were evaluated")

    result: dict[str, object] = {
        "validation_root": validation_root.as_posix(),
        "policy_count": len(policy_records),
        "clip_count": len(clips),
        "best_policy": best_policy,
        "best_summary": best_summary,
        "policy_records": policy_records,
    }
    if output_root is not None:
        written = write_policy_sweep_artifacts(
            output_root=output_root,
            result=result,
            policy_records=policy_records,
            best_clip_metrics=best_clip_metrics,
        )
        result.update(written)
    return result


def write_policy_sweep_artifacts(
    *,
    output_root: Path,
    result: Mapping[str, object],
    policy_records: Sequence[Mapping[str, object]],
    best_clip_metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Write machine-readable policy sweep summaries."""

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "policy_sweep_summary.json"
    results_csv_path = output_root / "policy_sweep_results.csv"
    best_clip_metrics_path = output_root / "best_policy_clip_metrics.csv"
    best_clip_metrics_json_path = output_root / "best_policy_clip_metrics.json"
    payload = {
        key: value
        for key, value in result.items()
        if key != "policy_records"
    }
    payload["policy_sweep_results_csv"] = results_csv_path.as_posix()
    payload["best_policy_clip_metrics_csv"] = best_clip_metrics_path.as_posix()
    payload["best_policy_clip_metrics_json"] = best_clip_metrics_json_path.as_posix()
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_policy_records_csv(results_csv_path, policy_records)
    write_clip_metrics_csv(best_clip_metrics_path, best_clip_metrics)
    best_clip_metrics_json_path.write_text(
        json.dumps(list(best_clip_metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "policy_sweep_summary_path": summary_path.as_posix(),
        "policy_sweep_results_csv": results_csv_path.as_posix(),
        "best_policy_clip_metrics_csv": best_clip_metrics_path.as_posix(),
        "best_policy_clip_metrics_json": best_clip_metrics_json_path.as_posix(),
    }


def _load_saved_validation_clip(metrics_path: Path) -> SavedValidationClip:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    frame_jsonl_path = _artifact_path(metrics.get("frame_jsonl_path"), fallback=metrics_path.parent / "frames.jsonl")
    rows = tuple(_read_jsonl(frame_jsonl_path))
    true_gesture = str(metrics.get("true_gesture"))
    if true_gesture not in {"rock", "paper", "scissors"}:
        raise ValueError(f"Unsupported true_gesture in {metrics_path}: {true_gesture}")
    return SavedValidationClip(
        clip_id=str(metrics["clip_id"]),
        transition_label=str(metrics["transition_label"]),
        true_gesture=cast(EvaluationGesture, true_gesture),
        source_path=Path(str(metrics.get("source_path", metrics_path.parent.name))),
        frame_count=int(metrics.get("frame_count", len(rows))),
        fps=float(metrics.get("fps", 30.0)),
        width=int(metrics.get("width", 0)),
        height=int(metrics.get("height", 0)),
        rows=rows,
        overlay_path=_optional_artifact_path(metrics.get("overlay_path")),
        frame_csv_path=_optional_artifact_path(metrics.get("frame_csv_path")),
        frame_jsonl_path=frame_jsonl_path,
    )


def _summarize_saved_clip(
    clip: SavedValidationClip,
    *,
    config: StrictDecisionConfig,
) -> dict[str, object]:
    return summarize_clip_decision(
        clip.rows,
        true_gesture=clip.true_gesture,
        transition_label=clip.transition_label,
        source_path=clip.source_path,
        clip_id=clip.clip_id,
        frame_count=clip.frame_count,
        fps=clip.fps,
        width=clip.width,
        height=clip.height,
        config=config,
        overlay_path=clip.overlay_path,
        frame_csv_path=clip.frame_csv_path,
        frame_jsonl_path=clip.frame_jsonl_path,
    )


def _load_discovery_summary(
    validation_root: Path,
    clips: Sequence[SavedValidationClip],
) -> dict[str, object]:
    summary_path = validation_root / "validation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        discovery = summary.get("discovery")
        if isinstance(discovery, dict):
            return dict(discovery)
    label_counts = Counter(clip.true_gesture for clip in clips)
    transition_counts = Counter(clip.transition_label for clip in clips)
    return {
        "passed": True,
        "video_count": len(clips),
        "expected_count": len(clips),
        "duplicate_count": 0,
        "label_counts": dict(sorted(label_counts.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
        "label_mode": "saved-validation",
    }


def _policy_dict(*, policy_id: str, config: StrictDecisionConfig) -> dict[str, object]:
    return {
        "policy_id": policy_id,
        "confidence_threshold": config.confidence_threshold,
        "margin_threshold": config.margin_threshold,
        "confirmation_count": config.confirmation_count,
        "max_decision_progress": config.max_decision_progress,
        "transition_mass_threshold": config.transition_mass_threshold,
        "paper_wait_is_terminal_for_transitions": config.paper_wait_is_terminal_for_transitions,
        "binary_transition_mass_threshold": config.binary_transition_mass_threshold,
    }


def _policy_record(
    *,
    policy: Mapping[str, object],
    summary: Mapping[str, object],
) -> dict[str, object]:
    decision_progress = summary.get("decision_progress")
    mean_progress = None
    if isinstance(decision_progress, Mapping):
        mean_progress = decision_progress.get("mean")
    return {
        **dict(policy),
        "passed": summary.get("passed"),
        "clip_count": summary.get("clip_count"),
        "passed_clip_count": summary.get("passed_clip_count"),
        "failed_clip_count": summary.get("failed_clip_count"),
        "accuracy": summary.get("accuracy"),
        "paper_scissors_accuracy": summary.get("paper_scissors_accuracy"),
        "rock_wait_success_count": summary.get("rock_wait_success_count"),
        "rock_false_trigger_count": summary.get("rock_false_trigger_count"),
        "decision_progress_mean": mean_progress,
        "failure_reason_counts": json.dumps(summary.get("failure_reason_counts", {}), sort_keys=True),
    }


def _ranking_key(
    *,
    summary: Mapping[str, object],
    config: StrictDecisionConfig,
) -> tuple[object, ...]:
    progress = summary.get("decision_progress")
    mean_progress = 1.0
    if isinstance(progress, Mapping) and progress.get("mean") is not None:
        mean_progress = float(progress["mean"])
    return (
        int(summary.get("passed_clip_count", 0)),
        -int(summary.get("rock_false_trigger_count", 0)),
        float(summary.get("paper_scissors_accuracy", 0.0)),
        int(summary.get("rock_wait_success_count", 0)),
        -int(summary.get("failed_clip_count", 0)),
        -mean_progress,
        config.confidence_threshold,
        config.margin_threshold,
        config.confirmation_count,
        -config.max_decision_progress,
        -config.transition_mass_threshold,
        -config.binary_transition_mass_threshold,
    )


def _write_policy_records_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy_id",
        "confidence_threshold",
        "margin_threshold",
        "confirmation_count",
        "max_decision_progress",
        "transition_mass_threshold",
        "paper_wait_is_terminal_for_transitions",
        "binary_transition_mass_threshold",
        "passed",
        "clip_count",
        "passed_clip_count",
        "failed_clip_count",
        "accuracy",
        "paper_scissors_accuracy",
        "rock_wait_success_count",
        "rock_false_trigger_count",
        "decision_progress_mean",
        "failure_reason_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Frame JSONL does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield parsed


def _optional_artifact_path(value: object) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _artifact_path(value: object, *, fallback: Path) -> Path:
    if value is None:
        return fallback
    path = Path(str(value))
    if path.exists():
        return path
    if fallback.exists():
        return fallback
    return path


def _require_values(values: Sequence[T], name: str) -> tuple[T, ...]:
    parsed = tuple(values)
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed
