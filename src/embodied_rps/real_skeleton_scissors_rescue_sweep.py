"""Saved-output sweeps for conditional scissors rescue selectors."""

from __future__ import annotations

import csv
import itertools
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

from embodied_rps.real_skeleton_policy_sweep import SavedValidationClip, load_saved_validation_clips
from embodied_rps.real_skeleton_open_set_guard import (
    OpenSetGuardConfig,
    annotate_rows_with_open_set_guard,
)
from embodied_rps.real_skeleton_video_eval import (
    EvaluationGesture,
    StrictDecisionConfig,
    build_validation_summary,
    summarize_clip_decision,
    write_clip_metrics_csv,
)

T = TypeVar("T")
GESTURE_LABELS: tuple[str, str, str] = ("rock", "paper", "scissors")


@dataclass(frozen=True)
class ScissorsRescueSelectorConfig:
    """Conditional selector that lets a candidate model rescue decisive scissors frames."""

    candidate_confidence_threshold: float = 0.90
    candidate_margin_threshold: float = 0.90
    baseline_transition_mass_threshold: float = 0.60
    baseline_rock_max: float | None = 0.40

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate_confidence_threshold", self.candidate_confidence_threshold),
            ("candidate_margin_threshold", self.candidate_margin_threshold),
            ("baseline_transition_mass_threshold", self.baseline_transition_mass_threshold),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.baseline_rock_max is not None and not 0.0 <= float(self.baseline_rock_max) <= 1.0:
            raise ValueError("baseline_rock_max must be None or in [0, 1]")


def apply_conditional_scissors_rescue(
    baseline_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    candidate_confidence_threshold: float,
    candidate_margin_threshold: float,
    baseline_transition_mass_threshold: float,
    baseline_rock_max: float | None,
) -> list[dict[str, object]]:
    """Return baseline rows with candidate scissors applied only when guard conditions pass."""

    config = ScissorsRescueSelectorConfig(
        candidate_confidence_threshold=candidate_confidence_threshold,
        candidate_margin_threshold=candidate_margin_threshold,
        baseline_transition_mass_threshold=baseline_transition_mass_threshold,
        baseline_rock_max=baseline_rock_max,
    )
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError("baseline_rows and candidate_rows must have the same length")
    combined: list[dict[str, object]] = []
    for baseline_row, candidate_row in zip(baseline_rows, candidate_rows, strict=True):
        parsed = dict(baseline_row)
        baseline_probs = _probabilities_from_row(baseline_row)
        candidate_probs = _probabilities_from_row(candidate_row)
        candidate_prediction = _top_label(candidate_probs)
        candidate_margin = _margin(candidate_probs)
        baseline_transition_mass = baseline_probs["paper"] + baseline_probs["scissors"]
        baseline_rock_probability = baseline_probs["rock"]
        rescue_allowed = (
            bool(baseline_row.get("detected"))
            and bool(candidate_row.get("detected"))
            and candidate_prediction == "scissors"
            and candidate_probs["scissors"] >= config.candidate_confidence_threshold
            and candidate_margin >= config.candidate_margin_threshold
            and baseline_transition_mass >= config.baseline_transition_mass_threshold
            and (
                config.baseline_rock_max is None
                or baseline_rock_probability <= config.baseline_rock_max
            )
        )
        parsed["baseline_prediction_before_rescue"] = baseline_row.get("prediction")
        parsed["baseline_rock_probability_before_rescue"] = float(baseline_rock_probability)
        parsed["baseline_transition_mass_before_rescue"] = float(baseline_transition_mass)
        parsed["candidate_prediction"] = candidate_prediction
        parsed["candidate_rock_probability"] = float(candidate_probs["rock"])
        parsed["candidate_paper_probability"] = float(candidate_probs["paper"])
        parsed["candidate_scissors_probability"] = float(candidate_probs["scissors"])
        parsed["candidate_confidence"] = float(candidate_probs[candidate_prediction])
        parsed["candidate_confidence_margin"] = float(candidate_margin)
        parsed["rescue_applied"] = bool(rescue_allowed)
        if rescue_allowed:
            parsed["prediction"] = "scissors"
            parsed["rock_probability"] = 0.0
            parsed["paper_probability"] = 0.0
            parsed["scissors_probability"] = 1.0
            parsed["transition_mass"] = 1.0
            parsed["confidence"] = 1.0
            parsed["confidence_margin"] = 1.0
        combined.append(parsed)
    return combined


def sweep_scissors_rescue_selectors(
    *,
    baseline_root: Path,
    candidate_root: Path,
    output_root: Path | None,
    candidate_confidence_thresholds: Sequence[float],
    candidate_margin_thresholds: Sequence[float],
    baseline_transition_mass_thresholds: Sequence[float],
    baseline_rock_max_values: Sequence[float | None],
    decision_config: StrictDecisionConfig | None = None,
    min_binary_decision_progress: float = 0.0,
) -> dict[str, object]:
    """Sweep conditional scissors-rescue selectors over two saved validation roots."""

    baseline_clips = load_saved_validation_clips(baseline_root)
    candidate_clips = load_saved_validation_clips(candidate_root)
    candidate_by_clip_id = _clip_map(candidate_clips, name="candidate")
    config = decision_config or StrictDecisionConfig()
    open_set_config = (
        OpenSetGuardConfig(decision=config, min_binary_decision_progress=float(min_binary_decision_progress))
        if min_binary_decision_progress > 0.0
        else None
    )
    discovery_summary = _load_discovery_summary(baseline_root, baseline_clips)
    baseline_metrics = [_summarize_clip(clip, rows=clip.rows, config=config, open_set_config=open_set_config) for clip in baseline_clips]
    candidate_metrics = [
        _summarize_clip(
            candidate_by_clip_id[clip.clip_id],
            rows=candidate_by_clip_id[clip.clip_id].rows,
            config=config,
            open_set_config=open_set_config,
        )
        for clip in baseline_clips
    ]
    baseline_passed_by_clip_id = {str(metric["clip_id"]): bool(metric.get("passed")) for metric in baseline_metrics}
    candidate_passed_by_clip_id = {str(metric["clip_id"]): bool(metric.get("passed")) for metric in candidate_metrics}

    policy_records: list[dict[str, object]] = []
    best_summary: dict[str, object] | None = None
    best_policy: dict[str, object] | None = None
    best_clip_metrics: list[dict[str, object]] = []
    best_rank: tuple[object, ...] | None = None

    combinations = list(
        itertools.product(
            _require_values(candidate_confidence_thresholds, "candidate_confidence_thresholds"),
            _require_values(candidate_margin_thresholds, "candidate_margin_thresholds"),
            _require_values(baseline_transition_mass_thresholds, "baseline_transition_mass_thresholds"),
            _require_values(baseline_rock_max_values, "baseline_rock_max_values"),
        )
    )
    for index, (
        candidate_confidence_threshold,
        candidate_margin_threshold,
        baseline_transition_mass_threshold,
        baseline_rock_max,
    ) in enumerate(combinations, start=1):
        selector = ScissorsRescueSelectorConfig(
            candidate_confidence_threshold=float(candidate_confidence_threshold),
            candidate_margin_threshold=float(candidate_margin_threshold),
            baseline_transition_mass_threshold=float(baseline_transition_mass_threshold),
            baseline_rock_max=None if baseline_rock_max is None else float(baseline_rock_max),
        )
        clip_metrics: list[dict[str, object]] = []
        for baseline_clip in baseline_clips:
            candidate_clip = candidate_by_clip_id[baseline_clip.clip_id]
            _validate_aligned_clip_pair(baseline_clip, candidate_clip)
            combined_rows = apply_conditional_scissors_rescue(
                baseline_clip.rows,
                candidate_clip.rows,
                candidate_confidence_threshold=selector.candidate_confidence_threshold,
                candidate_margin_threshold=selector.candidate_margin_threshold,
                baseline_transition_mass_threshold=selector.baseline_transition_mass_threshold,
                baseline_rock_max=selector.baseline_rock_max,
            )
            metric = _summarize_clip(
                baseline_clip,
                rows=combined_rows,
                config=config,
                open_set_config=open_set_config,
            )
            metric["rescue_frame_count"] = sum(1 for row in combined_rows if bool(row.get("rescue_applied")))
            metric["rescue_applied"] = bool(metric["rescue_frame_count"])
            metric["rescue_first_frame"] = _first_rescue_value(combined_rows, "frame_index")
            metric["rescue_first_progress"] = _first_rescue_value(combined_rows, config.progress_key)
            metric["baseline_passed"] = baseline_passed_by_clip_id.get(baseline_clip.clip_id, False)
            metric["candidate_passed"] = candidate_passed_by_clip_id.get(baseline_clip.clip_id, False)
            clip_metrics.append(metric)
        summary = build_validation_summary(
            clip_metrics=clip_metrics,
            discovery_summary=discovery_summary,
            config=config,
            event_manifest_path=None,
        )
        policy_id = f"scissors_rescue_{index:06d}"
        policy = _policy_dict(policy_id=policy_id, selector=selector, config=config)
        record = _policy_record(policy=policy, summary=summary, clip_metrics=clip_metrics)
        policy_records.append(record)
        rank = _ranking_key(summary=summary, clip_metrics=clip_metrics, selector=selector)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_summary = summary
            best_policy = policy
            best_clip_metrics = clip_metrics

    if best_policy is None or best_summary is None:
        raise RuntimeError("No scissors rescue selectors were evaluated")

    result: dict[str, object] = {
        "baseline_root": baseline_root.as_posix(),
        "candidate_root": candidate_root.as_posix(),
        "open_set_guard": {
            "min_binary_decision_progress": float(min_binary_decision_progress),
            "early_binary_action": "wait_counter_paper" if min_binary_decision_progress > 0.0 else None,
        },
        "policy_count": len(policy_records),
        "clip_count": len(baseline_clips),
        "baseline_summary": build_validation_summary(
            clip_metrics=baseline_metrics,
            discovery_summary=discovery_summary,
            config=config,
            event_manifest_path=None,
        ),
        "candidate_summary": build_validation_summary(
            clip_metrics=candidate_metrics,
            discovery_summary=discovery_summary,
            config=config,
            event_manifest_path=None,
        ),
        "best_policy": best_policy,
        "best_summary": best_summary,
        "policy_records": policy_records,
    }
    if output_root is not None:
        result.update(
            write_scissors_rescue_sweep_artifacts(
                output_root=output_root,
                result=result,
                policy_records=policy_records,
                best_clip_metrics=best_clip_metrics,
            )
        )
    return result


def write_scissors_rescue_sweep_artifacts(
    *,
    output_root: Path,
    result: Mapping[str, object],
    policy_records: Sequence[Mapping[str, object]],
    best_clip_metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Write machine-readable artifacts for a scissors-rescue selector sweep."""

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "scissors_rescue_sweep_summary.json"
    results_csv_path = output_root / "scissors_rescue_sweep_results.csv"
    best_clip_metrics_path = output_root / "best_selector_clip_metrics.csv"
    best_clip_metrics_json_path = output_root / "best_selector_clip_metrics.json"
    payload = {key: value for key, value in result.items() if key != "policy_records"}
    payload["scissors_rescue_sweep_results_csv"] = results_csv_path.as_posix()
    payload["best_selector_clip_metrics_csv"] = best_clip_metrics_path.as_posix()
    payload["best_selector_clip_metrics_json"] = best_clip_metrics_json_path.as_posix()
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_policy_records_csv(results_csv_path, policy_records)
    write_clip_metrics_csv(best_clip_metrics_path, best_clip_metrics)
    best_clip_metrics_json_path.write_text(
        json.dumps(list(best_clip_metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "scissors_rescue_sweep_summary_path": summary_path.as_posix(),
        "scissors_rescue_sweep_results_csv": results_csv_path.as_posix(),
        "best_selector_clip_metrics_csv": best_clip_metrics_path.as_posix(),
        "best_selector_clip_metrics_json": best_clip_metrics_json_path.as_posix(),
    }


def _summarize_clip(
    clip: SavedValidationClip,
    *,
    rows: Sequence[Mapping[str, object]],
    config: StrictDecisionConfig,
    open_set_config: OpenSetGuardConfig | None,
) -> dict[str, object]:
    annotated_rows = (
        annotate_rows_with_open_set_guard(rows, config=open_set_config)
        if open_set_config is not None
        else None
    )
    return summarize_clip_decision(
        rows,
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
        annotated_rows=annotated_rows,
    )


def _clip_map(clips: Sequence[SavedValidationClip], *, name: str) -> dict[str, SavedValidationClip]:
    by_id = {clip.clip_id: clip for clip in clips}
    if len(by_id) != len(clips):
        duplicates = [clip_id for clip_id, count in Counter(clip.clip_id for clip in clips).items() if count > 1]
        raise ValueError(f"Duplicate {name} clip ids: {duplicates}")
    return by_id


def _validate_aligned_clip_pair(baseline_clip: SavedValidationClip, candidate_clip: SavedValidationClip) -> None:
    if baseline_clip.clip_id != candidate_clip.clip_id:
        raise ValueError("clip ids must match")
    if baseline_clip.true_gesture != candidate_clip.true_gesture:
        raise ValueError(f"true_gesture differs for {baseline_clip.clip_id}")
    if len(baseline_clip.rows) != len(candidate_clip.rows):
        raise ValueError(f"frame row count differs for {baseline_clip.clip_id}")


def _probabilities_from_row(row: Mapping[str, object]) -> dict[str, float]:
    return {
        "rock": _number(row.get("rock_probability")),
        "paper": _number(row.get("paper_probability")),
        "scissors": _number(row.get("scissors_probability")),
    }


def _top_label(probabilities: Mapping[str, float]) -> str:
    return max(GESTURE_LABELS, key=lambda label: probabilities[label])


def _margin(probabilities: Mapping[str, float]) -> float:
    ordered = sorted((float(probabilities[label]) for label in GESTURE_LABELS), reverse=True)
    return ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]


def _first_rescue_value(rows: Sequence[Mapping[str, object]], key: str) -> object | None:
    for row in rows:
        if bool(row.get("rescue_applied")):
            return row.get(key)
    return None


def _load_discovery_summary(
    validation_root: Path,
    clips: Sequence[SavedValidationClip],
) -> dict[str, object]:
    summary_path = validation_root / "validation_summary.json"
    if summary_path.exists():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        discovery = loaded.get("discovery")
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


def _policy_dict(
    *,
    policy_id: str,
    selector: ScissorsRescueSelectorConfig,
    config: StrictDecisionConfig,
) -> dict[str, object]:
    return {
        "policy_id": policy_id,
        "candidate_confidence_threshold": selector.candidate_confidence_threshold,
        "candidate_margin_threshold": selector.candidate_margin_threshold,
        "baseline_transition_mass_threshold": selector.baseline_transition_mass_threshold,
        "baseline_rock_max": selector.baseline_rock_max,
        "confidence_threshold": config.confidence_threshold,
        "margin_threshold": config.margin_threshold,
        "confirmation_count": config.confirmation_count,
        "max_decision_progress": config.max_decision_progress,
        "transition_mass_threshold": config.transition_mass_threshold,
        "paper_wait_is_terminal_for_transitions": config.paper_wait_is_terminal_for_transitions,
        "binary_transition_mass_threshold": config.binary_transition_mass_threshold,
        "progress_key": config.progress_key,
    }


def _policy_record(
    *,
    policy: Mapping[str, object],
    summary: Mapping[str, object],
    clip_metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    decision_progress = summary.get("decision_progress")
    mean_progress = None
    if isinstance(decision_progress, Mapping):
        mean_progress = decision_progress.get("mean")
    baseline_fix_count = sum(
        1
        for metric in clip_metrics
        if not bool(metric.get("baseline_passed")) and bool(metric.get("passed"))
    )
    baseline_regression_count = sum(
        1
        for metric in clip_metrics
        if bool(metric.get("baseline_passed")) and not bool(metric.get("passed"))
    )
    candidate_rescue_clip_count = sum(1 for metric in clip_metrics if bool(metric.get("rescue_applied")))
    candidate_rescue_frame_count = sum(int(metric.get("rescue_frame_count", 0)) for metric in clip_metrics)
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
        "candidate_fix_count": baseline_fix_count,
        "baseline_regression_count": baseline_regression_count,
        "candidate_rescue_clip_count": candidate_rescue_clip_count,
        "candidate_rescue_frame_count": candidate_rescue_frame_count,
    }


def _ranking_key(
    *,
    summary: Mapping[str, object],
    clip_metrics: Sequence[Mapping[str, object]],
    selector: ScissorsRescueSelectorConfig,
) -> tuple[object, ...]:
    progress = summary.get("decision_progress")
    mean_progress = 1.0
    if isinstance(progress, Mapping) and progress.get("mean") is not None:
        mean_progress = float(progress["mean"])
    baseline_regression_count = sum(
        1
        for metric in clip_metrics
        if bool(metric.get("baseline_passed")) and not bool(metric.get("passed"))
    )
    candidate_rescue_clip_count = sum(1 for metric in clip_metrics if bool(metric.get("rescue_applied")))
    candidate_rescue_frame_count = sum(int(metric.get("rescue_frame_count", 0)) for metric in clip_metrics)
    return (
        int(summary.get("passed_clip_count", 0)),
        -int(summary.get("rock_false_trigger_count", 0)),
        -baseline_regression_count,
        float(summary.get("paper_scissors_accuracy", 0.0)),
        int(summary.get("rock_wait_success_count", 0)),
        -int(summary.get("failed_clip_count", 0)),
        -candidate_rescue_clip_count,
        -candidate_rescue_frame_count,
        -mean_progress,
        selector.candidate_confidence_threshold,
        selector.candidate_margin_threshold,
        selector.baseline_transition_mass_threshold,
        selector.baseline_rock_max if selector.baseline_rock_max is not None else 1.0,
    )


def _write_policy_records_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy_id",
        "candidate_confidence_threshold",
        "candidate_margin_threshold",
        "baseline_transition_mass_threshold",
        "baseline_rock_max",
        "confidence_threshold",
        "margin_threshold",
        "confirmation_count",
        "max_decision_progress",
        "transition_mass_threshold",
        "paper_wait_is_terminal_for_transitions",
        "binary_transition_mass_threshold",
        "progress_key",
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
        "candidate_fix_count",
        "baseline_regression_count",
        "candidate_rescue_clip_count",
        "candidate_rescue_frame_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})


def _require_values(values: Sequence[T], name: str) -> tuple[T, ...]:
    parsed = tuple(values)
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed


def _number(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _evaluation_gesture(value: object) -> EvaluationGesture:
    parsed = str(value)
    if parsed not in {"rock", "paper", "scissors"}:
        raise ValueError(f"Unsupported evaluation gesture: {parsed}")
    return cast(EvaluationGesture, parsed)
