"""Open-set guard sweeps for saved real-skeleton validation outputs."""

from __future__ import annotations

import csv
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from embodied_rps.real_skeleton_policy_sweep import (
    SavedValidationClip,
    load_saved_validation_clips,
)
from embodied_rps.real_skeleton_video_eval import (
    TRANSITION_GESTURES,
    WAIT_COUNTER_PAPER_STATE,
    StrictDecisionConfig,
    build_validation_summary,
    robot_action_for_decision_state,
    summarize_clip_decision,
    write_clip_metrics_csv,
)

T = TypeVar("T")


@dataclass(frozen=True)
class OpenSetGuardConfig:
    """Decision configuration with a provisional wait guard for early binary spikes."""

    decision: StrictDecisionConfig
    min_binary_decision_progress: float = 0.0
    early_binary_action: str = WAIT_COUNTER_PAPER_STATE

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_binary_decision_progress <= self.decision.max_decision_progress:
            raise ValueError("min_binary_decision_progress must be in [0, max_decision_progress]")
        if self.early_binary_action not in {WAIT_COUNTER_PAPER_STATE, "suppress"}:
            raise ValueError("early_binary_action must be wait_counter_paper or suppress")


def annotate_rows_with_open_set_guard(
    rows: Sequence[Mapping[str, object]],
    *,
    config: OpenSetGuardConfig,
) -> list[dict[str, object]]:
    """Annotate rows with strict decisions plus a provisional early-binary guard."""

    annotated: list[dict[str, object]] = []
    rolling_label: str | None = None
    rolling_count = 0
    decision_seen = False
    decision = config.decision
    for row in rows:
        parsed = dict(row)
        prediction = _optional_string(parsed.get("prediction"))
        detected = bool(parsed.get("detected", False))
        confidence = _number(parsed.get("confidence"))
        margin = _number(parsed.get("confidence_margin"))
        rock_probability = _number(parsed.get("rock_probability"))
        paper_probability = _number(parsed.get("paper_probability"))
        scissors_probability = _number(parsed.get("scissors_probability"))
        observed_progress = _progress_value(parsed, decision)
        transition_mass = _number(parsed.get("transition_mass"))
        if "transition_mass" not in parsed:
            transition_mass = paper_probability + scissors_probability
            parsed["transition_mass"] = transition_mass

        transition_gate_waits = transition_mass < decision.binary_transition_mass_threshold
        binary_candidate = (
            detected
            and prediction in TRANSITION_GESTURES
            and transition_mass >= decision.binary_transition_mass_threshold
            and confidence >= decision.confidence_threshold
            and margin >= decision.margin_threshold
        )
        binary_blocked_by_progress = bool(
            binary_candidate and observed_progress < config.min_binary_decision_progress
        )
        wait_qualifies = detected and (
            rock_probability >= decision.confidence_threshold
            or transition_mass <= decision.transition_mass_threshold
            or transition_gate_waits
            or (
                binary_blocked_by_progress
                and config.early_binary_action == WAIT_COUNTER_PAPER_STATE
            )
        )
        binary_qualifies = bool(binary_candidate and not binary_blocked_by_progress)
        decision_state = WAIT_COUNTER_PAPER_STATE if wait_qualifies else prediction if binary_qualifies else None
        qualifies = decision_state is not None
        if qualifies:
            if decision_state == rolling_label:
                rolling_count += 1
            else:
                rolling_label = decision_state
                rolling_count = 1
        else:
            rolling_label = None
            rolling_count = 0

        is_decision_frame = qualifies and rolling_count >= decision.confirmation_count and not decision_seen
        if is_decision_frame:
            decision_seen = True
        parsed["open_set_binary_blocked_by_progress"] = bool(binary_blocked_by_progress)
        parsed["open_set_guard_action"] = config.early_binary_action if binary_blocked_by_progress else None
        parsed["decision_state"] = decision_state
        parsed["selected_robot_action"] = robot_action_for_decision_state(decision_state)
        parsed["qualifies_strict_gate"] = bool(qualifies)
        parsed["rolling_prediction"] = rolling_label
        parsed["rolling_confirmation_count"] = int(rolling_count)
        parsed["is_decision_frame"] = bool(is_decision_frame)
        parsed["decision_already_reached"] = bool(decision_seen)
        annotated.append(parsed)
    return annotated


def summarize_open_set_guard_clip(
    clip: SavedValidationClip,
    *,
    config: OpenSetGuardConfig,
) -> dict[str, object]:
    """Summarize one saved clip after open-set guard annotation."""

    annotated_rows = annotate_rows_with_open_set_guard(clip.rows, config=config)
    summary = summarize_clip_decision(
        clip.rows,
        true_gesture=clip.true_gesture,
        transition_label=clip.transition_label,
        source_path=clip.source_path,
        clip_id=clip.clip_id,
        frame_count=clip.frame_count,
        fps=clip.fps,
        width=clip.width,
        height=clip.height,
        config=config.decision,
        overlay_path=clip.overlay_path,
        frame_csv_path=clip.frame_csv_path,
        frame_jsonl_path=clip.frame_jsonl_path,
        annotated_rows=annotated_rows,
    )
    summary["open_set_guard"] = {
        "min_binary_decision_progress": config.min_binary_decision_progress,
        "early_binary_action": config.early_binary_action,
    }
    summary["open_set_guarded_binary_frame_count"] = sum(
        1 for row in annotated_rows if bool(row.get("open_set_binary_blocked_by_progress"))
    )
    return summary


def sweep_open_set_guard_policies(
    *,
    validation_root: Path,
    output_root: Path | None,
    confidence_thresholds: Sequence[float],
    margin_thresholds: Sequence[float],
    confirmation_counts: Sequence[int],
    max_decision_progress_values: Sequence[float],
    transition_mass_thresholds: Sequence[float],
    paper_wait_terminal_for_transition_values: Sequence[bool],
    binary_transition_mass_thresholds: Sequence[float],
    min_binary_decision_progress_values: Sequence[float],
    early_binary_actions: Sequence[str] = (WAIT_COUNTER_PAPER_STATE,),
) -> dict[str, object]:
    """Re-score saved validation clips across open-set guard policy combinations."""

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
            _require_values(min_binary_decision_progress_values, "min_binary_decision_progress_values"),
            _require_values(early_binary_actions, "early_binary_actions"),
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
        min_binary_decision_progress,
        early_binary_action,
    ) in enumerate(combinations, start=1):
        decision = StrictDecisionConfig(
            confidence_threshold=float(confidence_threshold),
            margin_threshold=float(margin_threshold),
            confirmation_count=int(confirmation_count),
            max_decision_progress=float(max_decision_progress),
            transition_mass_threshold=float(transition_mass_threshold),
            paper_wait_is_terminal_for_transitions=bool(paper_wait_is_terminal_for_transitions),
            binary_transition_mass_threshold=float(binary_transition_mass_threshold),
        )
        config = OpenSetGuardConfig(
            decision=decision,
            min_binary_decision_progress=float(min_binary_decision_progress),
            early_binary_action=str(early_binary_action),
        )
        clip_metrics = [summarize_open_set_guard_clip(clip, config=config) for clip in clips]
        summary = build_validation_summary(
            clip_metrics=clip_metrics,
            discovery_summary=discovery_summary,
            config=decision,
            event_manifest_path=None,
        )
        policy_id = f"open_set_guard_{index:06d}"
        policy = _policy_dict(policy_id=policy_id, config=config)
        record = _policy_record(policy=policy, summary=summary, clip_metrics=clip_metrics)
        policy_records.append(record)
        rank = _ranking_key(summary=summary, config=config)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_summary = summary
            best_policy = policy
            best_clip_metrics = clip_metrics

    if best_policy is None or best_summary is None:
        raise RuntimeError("No open-set guard policies were evaluated")

    result: dict[str, object] = {
        "validation_root": validation_root.as_posix(),
        "policy_count": len(policy_records),
        "clip_count": len(clips),
        "best_policy": best_policy,
        "best_summary": best_summary,
        "policy_records": policy_records,
    }
    if output_root is not None:
        result.update(
            write_open_set_guard_sweep_artifacts(
                output_root=output_root,
                result=result,
                policy_records=policy_records,
                best_clip_metrics=best_clip_metrics,
            )
        )
    return result


def write_open_set_guard_sweep_artifacts(
    *,
    output_root: Path,
    result: Mapping[str, object],
    policy_records: Sequence[Mapping[str, object]],
    best_clip_metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Write machine-readable open-set guard sweep summaries."""

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "open_set_guard_sweep_summary.json"
    results_csv_path = output_root / "open_set_guard_sweep_results.csv"
    best_clip_metrics_path = output_root / "best_policy_clip_metrics.csv"
    best_clip_metrics_json_path = output_root / "best_policy_clip_metrics.json"
    payload = {key: value for key, value in result.items() if key != "policy_records"}
    payload["open_set_guard_sweep_results_csv"] = results_csv_path.as_posix()
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
        "open_set_guard_sweep_summary_path": summary_path.as_posix(),
        "open_set_guard_sweep_results_csv": results_csv_path.as_posix(),
        "best_policy_clip_metrics_csv": best_clip_metrics_path.as_posix(),
        "best_policy_clip_metrics_json": best_clip_metrics_json_path.as_posix(),
    }


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
    return {
        "passed": True,
        "video_count": len(clips),
        "expected_count": len(clips),
        "duplicate_count": 0,
        "label_counts": {},
        "transition_counts": {},
        "label_mode": "saved-validation",
    }


def _policy_dict(*, policy_id: str, config: OpenSetGuardConfig) -> dict[str, object]:
    decision = config.decision
    return {
        "policy_id": policy_id,
        "confidence_threshold": decision.confidence_threshold,
        "margin_threshold": decision.margin_threshold,
        "confirmation_count": decision.confirmation_count,
        "max_decision_progress": decision.max_decision_progress,
        "transition_mass_threshold": decision.transition_mass_threshold,
        "paper_wait_is_terminal_for_transitions": decision.paper_wait_is_terminal_for_transitions,
        "binary_transition_mass_threshold": decision.binary_transition_mass_threshold,
        "min_binary_decision_progress": config.min_binary_decision_progress,
        "early_binary_action": config.early_binary_action,
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
        "guarded_binary_frame_count": sum(
            int(metric.get("open_set_guarded_binary_frame_count", 0))
            for metric in clip_metrics
        ),
        "decision_progress_mean": mean_progress,
        "failure_reason_counts": json.dumps(summary.get("failure_reason_counts", {}), sort_keys=True),
    }


def _ranking_key(
    *,
    summary: Mapping[str, object],
    config: OpenSetGuardConfig,
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
        -config.min_binary_decision_progress,
        config.decision.confidence_threshold,
        config.decision.margin_threshold,
        config.decision.confirmation_count,
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
        "min_binary_decision_progress",
        "early_binary_action",
        "passed",
        "clip_count",
        "passed_clip_count",
        "failed_clip_count",
        "accuracy",
        "paper_scissors_accuracy",
        "rock_wait_success_count",
        "rock_false_trigger_count",
        "guarded_binary_frame_count",
        "decision_progress_mean",
        "failure_reason_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _number(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _progress_value(row: Mapping[str, object], decision: StrictDecisionConfig) -> float:
    if decision.progress_key == "model_progress":
        return _number(row.get("model_progress"))
    if decision.progress_key == "observed_progress":
        return _number(row.get("observed_progress"))
    if decision.progress_key == "motion_progress":
        return _number(row.get("motion_progress"))
    return _number(row.get("clip_progress"))


def _require_values(values: Sequence[T], name: str) -> tuple[T, ...]:
    parsed = tuple(values)
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed
