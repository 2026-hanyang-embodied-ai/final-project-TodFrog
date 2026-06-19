"""Batch validation helpers for real MP4 skeleton final-gesture prediction."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np

FinalGesture: TypeAlias = Literal["paper", "scissors"]
EvaluationGesture: TypeAlias = Literal["paper", "scissors", "rock"]
OpponentGesture: TypeAlias = Literal["rock", "paper", "scissors"]
CounterGesture: TypeAlias = Literal["rock", "paper", "scissors"]
DecisionState: TypeAlias = Literal["paper", "scissors", "wait_counter_paper"]
ProgressKey: TypeAlias = Literal["clip_progress", "motion_progress", "observed_progress", "model_progress"]

FOLDER_LABELS: Mapping[str, tuple[str, FinalGesture]] = {
    "바위→가위": ("rock_to_scissors", "scissors"),
    "바위->가위": ("rock_to_scissors", "scissors"),
    "rock_to_scissors": ("rock_to_scissors", "scissors"),
    "바위→보": ("rock_to_paper", "paper"),
    "바위->보": ("rock_to_paper", "paper"),
    "rock_to_paper": ("rock_to_paper", "paper"),
}
TRANSITION_GESTURES: tuple[FinalGesture, ...] = ("paper", "scissors")
WAIT_COUNTER_PAPER_STATE = "wait_counter_paper"
COUNTER_MOVES: Mapping[OpponentGesture, CounterGesture] = {
    "rock": "paper",
    "paper": "scissors",
    "scissors": "rock",
}
FINAL_LABEL_FOLDERS: Mapping[str, tuple[str, EvaluationGesture]] = {
    "paper": ("test_paper", "paper"),
    "scissors": ("test_scissors", "scissors"),
    "rock": ("test_rock_ood", "rock"),
}


@dataclass(frozen=True)
class LabeledVideo:
    """One discovered MP4 with a transition label and final gesture target."""

    path: Path
    transition_label: str
    true_gesture: EvaluationGesture

    @property
    def clip_id(self) -> str:
        return f"{self.transition_label}_{_natural_key(self.path.stem)}"


@dataclass(frozen=True)
class StrictDecisionConfig:
    """Strict validation gate for real-video final gesture prediction."""

    confidence_threshold: float = 0.85
    margin_threshold: float = 0.20
    confirmation_count: int = 3
    max_decision_progress: float = 0.50
    transition_mass_threshold: float = 0.15
    paper_wait_is_terminal_for_transitions: bool = True
    binary_transition_mass_threshold: float = 0.0
    progress_key: ProgressKey = "clip_progress"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.margin_threshold <= 1.0:
            raise ValueError("margin_threshold must be in [0, 1]")
        if self.confirmation_count <= 0:
            raise ValueError("confirmation_count must be positive")
        if not 0.0 < self.max_decision_progress <= 1.0:
            raise ValueError("max_decision_progress must be in (0, 1]")
        if not 0.0 <= self.transition_mass_threshold <= 1.0:
            raise ValueError("transition_mass_threshold must be in [0, 1]")
        if not 0.0 <= self.binary_transition_mass_threshold <= 1.0:
            raise ValueError("binary_transition_mass_threshold must be in [0, 1]")
        if self.progress_key not in {"clip_progress", "motion_progress", "observed_progress", "model_progress"}:
            raise ValueError("progress_key must be clip_progress, motion_progress, observed_progress, or model_progress")


def attach_motion_progress(
    rows: Sequence[Mapping[str, object]],
    canonical_landmarks_by_frame: Sequence[np.ndarray | None],
) -> list[dict[str, object]]:
    """Attach movement-energy progress derived from canonical hand landmarks."""

    if len(rows) != len(canonical_landmarks_by_frame):
        raise ValueError("rows and canonical_landmarks_by_frame must have the same length")
    energies = np.zeros((len(rows),), dtype=np.float64)
    previous: np.ndarray | None = None
    for index, landmarks in enumerate(canonical_landmarks_by_frame):
        if landmarks is None:
            continue
        current = np.asarray(landmarks, dtype=np.float32)
        if current.shape != (21, 3):
            raise ValueError("canonical landmarks must have shape (21, 3)")
        if previous is not None:
            deltas = np.linalg.norm(current - previous, axis=1)
            energies[index] = float(np.mean(deltas))
        previous = current

    effective = _effective_motion_energy(energies)
    total_motion = float(np.sum(effective))
    cumulative = np.cumsum(effective) if total_motion > 0.0 else np.zeros_like(effective)
    annotated: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        parsed = dict(row)
        parsed["motion_energy"] = float(effective[index])
        parsed["motion_progress"] = float(cumulative[index] / total_motion) if total_motion > 0.0 else 0.0
        parsed["observed_progress"] = min(_number(parsed.get("clip_progress")), _number(parsed.get("motion_progress")))
        annotated.append(parsed)
    return annotated


def discover_labeled_videos(root: Path) -> list[LabeledVideo]:
    """Discover labeled MP4s under the expected Korean transition folders."""

    if not root.exists():
        raise FileNotFoundError(f"MP4 root does not exist: {root}")
    videos: list[LabeledVideo] = []
    for path in sorted(root.rglob("*"), key=lambda item: _natural_key(item.as_posix())):
        if not path.is_file() or path.suffix.lower() != ".mp4":
            continue
        parsed = label_for_video_path(path)
        if parsed is None:
            continue
        transition_label, final_gesture = parsed
        videos.append(LabeledVideo(path=path, transition_label=transition_label, true_gesture=final_gesture))
    return videos


def discover_final_label_videos(root: Path) -> list[LabeledVideo]:
    """Discover MP4s under final-label folders: paper, scissors, and rock."""

    if not root.exists():
        raise FileNotFoundError(f"MP4 root does not exist: {root}")
    videos: list[LabeledVideo] = []
    for path in sorted(root.rglob("*"), key=lambda item: _natural_key(item.as_posix())):
        if not path.is_file() or path.suffix.lower() != ".mp4":
            continue
        parsed = final_label_for_video_path(path)
        if parsed is None:
            continue
        transition_label, final_gesture = parsed
        videos.append(LabeledVideo(path=path, transition_label=transition_label, true_gesture=final_gesture))
    return videos


def label_for_video_path(path: Path) -> tuple[str, FinalGesture] | None:
    """Return transition/final labels from any parent folder in a video path."""

    for parent in (path.parent, *path.parents):
        label = FOLDER_LABELS.get(parent.name)
        if label is not None:
            return label
    return None


def final_label_for_video_path(path: Path) -> tuple[str, EvaluationGesture] | None:
    """Return final-label evaluation labels from parent folders."""

    for parent in (path.parent, *path.parents):
        label = FINAL_LABEL_FOLDERS.get(parent.name.lower())
        if label is not None:
            return label
    return None


def validate_discovered_videos(videos: Sequence[LabeledVideo], *, expected_count: int = 20) -> dict[str, object]:
    """Validate count, duplicate paths, and class balance for discovered videos."""

    paths = [video.path.resolve() for video in videos]
    duplicate_count = len(paths) - len(set(paths))
    label_counts = Counter(video.true_gesture for video in videos)
    transition_counts = Counter(video.transition_label for video in videos)
    passed = (
        len(videos) == expected_count
        and duplicate_count == 0
        and label_counts.get("paper", 0) == expected_count // 2
        and label_counts.get("scissors", 0) == expected_count // 2
    )
    return {
        "passed": passed,
        "video_count": len(videos),
        "expected_count": expected_count,
        "duplicate_count": duplicate_count,
        "label_counts": dict(sorted(label_counts.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
    }


def validate_final_label_videos(videos: Sequence[LabeledVideo], *, expected_count: int = 15) -> dict[str, object]:
    """Validate count, duplicate paths, and 5/5/5 final-label balance."""

    paths = [video.path.resolve() for video in videos]
    duplicate_count = len(paths) - len(set(paths))
    label_counts = Counter(video.true_gesture for video in videos)
    transition_counts = Counter(video.transition_label for video in videos)
    expected_per_label = expected_count // 3
    passed = (
        len(videos) == expected_count
        and expected_count % 3 == 0
        and duplicate_count == 0
        and label_counts.get("paper", 0) == expected_per_label
        and label_counts.get("scissors", 0) == expected_per_label
        and label_counts.get("rock", 0) == expected_per_label
    )
    return {
        "passed": passed,
        "video_count": len(videos),
        "expected_count": expected_count,
        "duplicate_count": duplicate_count,
        "label_counts": dict(sorted(label_counts.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
        "label_mode": "final-label",
    }


def annotate_rows_with_strict_decision(
    rows: Sequence[Mapping[str, object]],
    *,
    config: StrictDecisionConfig,
) -> list[dict[str, object]]:
    """Add rolling confirmation state to per-frame prediction rows."""

    annotated: list[dict[str, object]] = []
    rolling_label: str | None = None
    rolling_count = 0
    decision_seen = False
    for row in rows:
        parsed = dict(row)
        prediction = _optional_string(parsed.get("prediction"))
        detected = bool(parsed.get("detected", False))
        confidence = _number(parsed.get("confidence"))
        margin = _number(parsed.get("confidence_margin"))
        rock_probability = _number(parsed.get("rock_probability"))
        paper_probability = _number(parsed.get("paper_probability"))
        scissors_probability = _number(parsed.get("scissors_probability"))
        transition_mass = _number(parsed.get("transition_mass"))
        if "transition_mass" not in parsed:
            transition_mass = paper_probability + scissors_probability
            parsed["transition_mass"] = transition_mass
        transition_gate_waits = transition_mass < config.binary_transition_mass_threshold
        wait_qualifies = detected and (
            rock_probability >= config.confidence_threshold
            or transition_mass <= config.transition_mass_threshold
            or transition_gate_waits
        )
        binary_qualifies = (
            detected
            and prediction in TRANSITION_GESTURES
            and transition_mass >= config.binary_transition_mass_threshold
            and confidence >= config.confidence_threshold
            and margin >= config.margin_threshold
        )
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

        is_decision_frame = qualifies and rolling_count >= config.confirmation_count and not decision_seen
        if is_decision_frame:
            decision_seen = True
        parsed["decision_state"] = decision_state
        parsed["selected_robot_action"] = robot_action_for_decision_state(decision_state)
        parsed["qualifies_strict_gate"] = bool(qualifies)
        parsed["rolling_prediction"] = rolling_label
        parsed["rolling_confirmation_count"] = int(rolling_count)
        parsed["is_decision_frame"] = bool(is_decision_frame)
        parsed["decision_already_reached"] = bool(decision_seen)
        annotated.append(parsed)
    return annotated


def summarize_clip_decision(
    rows: Sequence[Mapping[str, object]],
    *,
    true_gesture: EvaluationGesture,
    transition_label: str,
    source_path: Path,
    clip_id: str,
    frame_count: int,
    fps: float,
    width: int,
    height: int,
    config: StrictDecisionConfig,
    overlay_path: Path | None = None,
    frame_csv_path: Path | None = None,
    frame_jsonl_path: Path | None = None,
    annotated_rows: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Summarize one clip against the strict pass gate."""

    annotated = [dict(row) for row in annotated_rows] if annotated_rows is not None else annotate_rows_with_strict_decision(rows, config=config)
    detected_count = sum(1 for row in annotated if bool(row.get("detected")))
    qualified_count = sum(1 for row in annotated if bool(row.get("qualifies_strict_gate")))
    decision_row = _terminal_decision_row(
        annotated,
        true_gesture=true_gesture,
        config=config,
    )
    first_correct_stable_row = (
        _first_correct_stable_row(
            annotated,
            true_gesture=_final_gesture(true_gesture),
            confirmation_count=config.confirmation_count,
        )
        if true_gesture in TRANSITION_GESTURES
        else None
    )
    prediction_counts = Counter(
        str(row.get("prediction"))
        for row in annotated
        if bool(row.get("detected")) and _optional_string(row.get("prediction")) in COUNTER_MOVES
    )
    base: dict[str, object] = {
        "clip_id": clip_id,
        "source_path": source_path.as_posix(),
        "transition_label": transition_label,
        "true_gesture": true_gesture,
        "frame_count": int(frame_count),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "detected_frame_count": int(detected_count),
        "detection_coverage": float(detected_count / max(1, len(annotated))),
        "qualified_frame_count": int(qualified_count),
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "confidence_threshold": config.confidence_threshold,
        "margin_threshold": config.margin_threshold,
        "confirmation_count": config.confirmation_count,
        "max_decision_progress": config.max_decision_progress,
        "progress_key": config.progress_key,
        "paper_wait_is_terminal_for_transitions": config.paper_wait_is_terminal_for_transitions,
        "binary_transition_mass_threshold": config.binary_transition_mass_threshold,
        "decision_policy": None,
        "overlay_path": overlay_path.as_posix() if overlay_path is not None else None,
        "frame_csv_path": frame_csv_path.as_posix() if frame_csv_path is not None else None,
        "frame_jsonl_path": frame_jsonl_path.as_posix() if frame_jsonl_path is not None else None,
        "first_correct_stable_frame": int(_number(first_correct_stable_row.get("frame_index"))) if first_correct_stable_row is not None else None,
        "first_correct_stable_progress": _row_progress(first_correct_stable_row, config=config) if first_correct_stable_row is not None else None,
        "first_correct_stable_confidence": _number(first_correct_stable_row.get("confidence")) if first_correct_stable_row is not None else None,
    }
    if true_gesture == "rock":
        return _summarize_rock_wait_decision(
            base,
            decision_row=decision_row,
            detected_count=detected_count,
            qualified_count=qualified_count,
            config=config,
        )

    if decision_row is None:
        failure_reason = _no_decision_failure_reason(
            detected_count=detected_count,
            qualified_count=qualified_count,
            prediction_counts=prediction_counts,
        )
        base.update(
            {
                "passed": False,
                "failure_reason": failure_reason,
                "predicted_gesture": None,
                "decision_state": None,
                "decision_frame": None,
                "decision_time_s": None,
                "decision_progress": None,
                "decision_model_progress": None,
                "decision_confidence": None,
                "decision_confidence_margin": None,
                "counter_move": None,
                "selected_robot_action": None,
                "decision_policy": None,
            }
        )
        return base

    decision_state = str(decision_row.get("decision_state"))
    if decision_state == WAIT_COUNTER_PAPER_STATE:
        predicted: EvaluationGesture = "rock"
        counter_move: CounterGesture = "paper"
    else:
        predicted = _final_gesture(str(decision_row["prediction"]))
        counter_move = counter_move_for_prediction(predicted)
    decision_progress = _row_progress(decision_row, config=config)
    correct = predicted == true_gesture
    early_enough = decision_progress <= config.max_decision_progress
    failure_reason = None
    if not correct:
        failure_reason = "wrong_prediction"
    elif not early_enough:
        failure_reason = "late_decision"
    base.update(
        {
            "passed": correct and early_enough,
            "failure_reason": failure_reason,
            "predicted_gesture": predicted,
            "decision_state": decision_state,
            "decision_frame": int(_number(decision_row.get("frame_index"))),
            "decision_time_s": _number(decision_row.get("time_s")),
            "decision_progress": decision_progress,
            "decision_model_progress": _number(decision_row.get("model_progress")),
            "decision_confidence": _number(decision_row.get("confidence")),
            "decision_confidence_margin": _number(decision_row.get("confidence_margin")),
            "counter_move": counter_move,
            "selected_robot_action": counter_move,
            "decision_policy": str(decision_row.get("decision_policy") or "strict_first_stable"),
        }
    )
    return base


def _terminal_decision_row(
    rows: Sequence[Mapping[str, object]],
    *,
    true_gesture: EvaluationGesture,
    config: StrictDecisionConfig,
) -> Mapping[str, object] | None:
    """Return the first terminal decision row for the current evaluation policy."""

    if true_gesture in TRANSITION_GESTURES and not config.paper_wait_is_terminal_for_transitions:
        late_geometry_paper = _late_geometry_paper_decision_row(rows, config=config)
        if late_geometry_paper is not None:
            parsed = dict(late_geometry_paper)
            parsed["decision_policy"] = "late_geometry_paper_override"
            return parsed
        return next(
            (
                row
                for row in rows
                if row.get("decision_state") in TRANSITION_GESTURES
                and bool(row.get("qualifies_strict_gate"))
                and int(_number(row.get("rolling_confirmation_count"))) >= config.confirmation_count
            ),
            None,
        )
    return next((row for row in rows if bool(row.get("is_decision_frame"))), None)


def _late_geometry_paper_decision_row(
    rows: Sequence[Mapping[str, object]],
    *,
    config: StrictDecisionConfig,
) -> Mapping[str, object] | None:
    return next(
        (
            row
            for row in rows
            if bool(row.get("late_geometry_paper_detected"))
            and row.get("decision_state") == "paper"
            and bool(row.get("qualifies_strict_gate"))
            and int(_number(row.get("rolling_confirmation_count"))) >= config.confirmation_count
            and _row_progress(row, config=config) <= config.max_decision_progress
        ),
        None,
    )


def _summarize_rock_wait_decision(
    base: dict[str, object],
    *,
    decision_row: Mapping[str, object] | None,
    detected_count: int,
    qualified_count: int,
    config: StrictDecisionConfig,
) -> dict[str, object]:
    """Summarize a rock clip against the explicit paper-wait policy."""

    if decision_row is None:
        base.update(
            {
                "passed": False,
                "failure_reason": _no_decision_failure_reason(
                    detected_count=detected_count,
                    qualified_count=qualified_count,
                    prediction_counts=Counter(),
                ),
                "predicted_gesture": None,
                "decision_state": None,
                "decision_frame": None,
                "decision_time_s": None,
                "decision_progress": None,
                "decision_model_progress": None,
                "decision_confidence": None,
                "decision_confidence_margin": None,
                "counter_move": None,
                "selected_robot_action": None,
                "ood_status": "no_wait_decision",
            }
        )
        return base

    decision_state = str(decision_row.get("decision_state"))
    predicted = "rock" if decision_state == WAIT_COUNTER_PAPER_STATE else str(decision_row["prediction"])
    decision_progress = _row_progress(decision_row, config=config)
    early_enough = decision_progress <= config.max_decision_progress
    waited = decision_state == WAIT_COUNTER_PAPER_STATE
    robot_action = robot_action_for_decision_state(decision_state)
    false_trigger = not waited and early_enough
    failure_reason = None
    if false_trigger:
        failure_reason = "false_trigger"
    elif waited and not early_enough:
        failure_reason = "late_wait_decision"
    elif not waited:
        failure_reason = "late_binary_decision_after_deadline"
    base.update(
        {
            "passed": waited and early_enough,
            "failure_reason": failure_reason,
            "predicted_gesture": predicted,
            "decision_state": decision_state,
            "decision_frame": int(_number(decision_row.get("frame_index"))),
            "decision_time_s": _number(decision_row.get("time_s")),
            "decision_progress": decision_progress,
            "decision_model_progress": _number(decision_row.get("model_progress")),
            "decision_confidence": _number(decision_row.get("confidence")),
            "decision_confidence_margin": _number(decision_row.get("confidence_margin")),
            "counter_move": robot_action,
            "selected_robot_action": robot_action,
            "ood_status": "wait_counter_paper" if waited and early_enough else failure_reason,
        }
    )
    return base


def counter_move_for_prediction(prediction: str) -> CounterGesture:
    """Return the SCHUNK response gesture for a predicted final opponent gesture."""

    if prediction not in COUNTER_MOVES:
        raise ValueError(f"Unsupported final gesture for event bridge: {prediction}")
    return COUNTER_MOVES[cast(OpponentGesture, prediction)]


def robot_action_for_decision_state(decision_state: object) -> CounterGesture | None:
    """Return the robot action selected by a strict decision state."""

    if decision_state == WAIT_COUNTER_PAPER_STATE:
        return "paper"
    if decision_state in COUNTER_MOVES:
        return counter_move_for_prediction(str(decision_state))
    return None


def build_validation_summary(
    *,
    clip_metrics: Sequence[Mapping[str, object]],
    discovery_summary: Mapping[str, object],
    config: StrictDecisionConfig,
    event_manifest_path: Path | None,
) -> dict[str, object]:
    """Build the global validation summary for the strict gate."""

    total = len(clip_metrics)
    passed_clips = [metric for metric in clip_metrics if bool(metric.get("passed"))]
    failed_clips = [metric for metric in clip_metrics if not bool(metric.get("passed"))]
    by_class: dict[str, dict[str, object]] = {}
    for label in ("paper", "scissors", "rock"):
        class_metrics = [metric for metric in clip_metrics if metric.get("true_gesture") == label]
        class_passed = [metric for metric in class_metrics if bool(metric.get("passed"))]
        by_class[label] = {
            "clip_count": len(class_metrics),
            "passed_count": len(class_passed),
            "accuracy": _safe_rate(len(class_passed), len(class_metrics)),
        }
    binary_metrics = [metric for metric in clip_metrics if metric.get("true_gesture") in ("paper", "scissors")]
    binary_passed = [metric for metric in binary_metrics if bool(metric.get("passed"))]
    rock_metrics = [metric for metric in clip_metrics if metric.get("true_gesture") == "rock"]
    rock_false_triggers = [metric for metric in rock_metrics if metric.get("failure_reason") == "false_trigger"]
    decision_progress_values = [
        _number(metric.get("decision_progress"))
        for metric in passed_clips
        if metric.get("decision_progress") is not None
    ]
    confidence_values = [
        _number(metric.get("decision_confidence"))
        for metric in passed_clips
        if metric.get("decision_confidence") is not None
    ]
    margin_values = [
        _number(metric.get("decision_confidence_margin"))
        for metric in passed_clips
        if metric.get("decision_confidence_margin") is not None
    ]
    passed = total > 0 and len(failed_clips) == 0 and bool(discovery_summary.get("passed"))
    return {
        "passed": passed,
        "strict_gate": {
            "confidence_threshold": config.confidence_threshold,
            "margin_threshold": config.margin_threshold,
            "confirmation_count": config.confirmation_count,
            "max_decision_progress": config.max_decision_progress,
            "progress_key": config.progress_key,
            "transition_mass_threshold": config.transition_mass_threshold,
            "paper_wait_is_terminal_for_transitions": config.paper_wait_is_terminal_for_transitions,
            "binary_transition_mass_threshold": config.binary_transition_mass_threshold,
        },
        "discovery": dict(discovery_summary),
        "clip_count": total,
        "passed_clip_count": len(passed_clips),
        "failed_clip_count": len(failed_clips),
        "accuracy": _safe_rate(len(passed_clips), total),
        "per_class": by_class,
        "paper_scissors_accuracy": _safe_rate(len(binary_passed), len(binary_metrics)),
        "rock_false_trigger_count": len(rock_false_triggers),
        "rock_wait_success_count": sum(1 for metric in rock_metrics if metric.get("ood_status") == "wait_counter_paper"),
        "rock_clip_count": len(rock_metrics),
        "decision_progress": _distribution(decision_progress_values),
        "decision_confidence": _distribution(confidence_values),
        "decision_confidence_margin": _distribution(margin_values),
        "failure_reason_counts": dict(sorted(Counter(str(metric.get("failure_reason")) for metric in failed_clips).items())),
        "failed_clips": [
            {
                "clip_id": str(metric.get("clip_id")),
                "source_path": str(metric.get("source_path")),
                "true_gesture": str(metric.get("true_gesture")),
                "predicted_gesture": metric.get("predicted_gesture"),
                "failure_reason": metric.get("failure_reason"),
            }
            for metric in failed_clips
        ],
        "event_manifest_path": event_manifest_path.as_posix() if passed and event_manifest_path is not None else None,
        "event_manifest_written": passed and event_manifest_path is not None,
    }


def write_clip_metrics_csv(path: Path, clip_metrics: Sequence[Mapping[str, object]]) -> None:
    """Write compact clip-level metrics CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "source_path",
        "transition_label",
        "true_gesture",
        "passed",
        "failure_reason",
        "predicted_gesture",
        "decision_state",
        "counter_move",
        "selected_robot_action",
        "decision_policy",
        "decision_frame",
        "decision_time_s",
        "decision_progress",
        "progress_key",
        "decision_confidence",
        "decision_confidence_margin",
        "first_correct_stable_frame",
        "first_correct_stable_progress",
        "first_correct_stable_confidence",
        "detected_frame_count",
        "detection_coverage",
        "qualified_frame_count",
        "frame_count",
        "fps",
        "width",
        "height",
        "overlay_path",
        "frame_csv_path",
        "frame_jsonl_path",
        "ood_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in clip_metrics:
            writer.writerow({field: metric.get(field) for field in fieldnames})


def write_frame_rows(path_csv: Path, path_jsonl: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write per-frame CSV and JSONL rows."""

    path_csv.parent.mkdir(parents=True, exist_ok=True)
    path_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "time_s",
        "detected",
        "prediction",
        "paper_probability",
        "scissors_probability",
        "rock_probability",
        "transition_mass",
        "confidence",
        "confidence_margin",
        "clip_progress",
        "motion_progress",
        "observed_progress",
        "motion_energy",
        "model_progress",
        "late_geometry_paper_detected",
        "decision_state",
        "selected_robot_action",
        "qualifies_strict_gate",
        "rolling_prediction",
        "rolling_confirmation_count",
        "is_decision_frame",
        "decision_already_reached",
    ]
    with path_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    with path_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_schunk_event_manifest(
    path: Path,
    clip_metrics: Sequence[Mapping[str, object]],
    *,
    response_delay_s: float = 0.0,
) -> list[dict[str, object]]:
    """Write SCHUNK response events for passed clips only."""

    failed = [metric for metric in clip_metrics if not bool(metric.get("passed"))]
    if len(failed) > 0:
        raise ValueError("Refusing to write SCHUNK events while failed clips exist")
    events: list[dict[str, object]] = []
    for metric in sorted(clip_metrics, key=lambda item: str(item.get("clip_id"))):
        predicted = _opponent_gesture(str(metric["predicted_gesture"]))
        decision_time_s = _number(metric.get("decision_time_s"))
        event = {
            "clip_id": str(metric["clip_id"]),
            "source_path": str(metric["source_path"]),
            "transition_label": str(metric["transition_label"]),
            "true_final_gesture": str(metric["true_gesture"]),
            "predicted_final_gesture": predicted,
            "confidence": _number(metric.get("decision_confidence")),
            "confidence_margin": _number(metric.get("decision_confidence_margin")),
            "decision_frame": int(_number(metric.get("decision_frame"))),
            "decision_time_s": decision_time_s,
            "decision_progress": _number(metric.get("decision_progress")),
            "selected_counter_move": counter_move_for_prediction(predicted),
            "decision_state": metric.get("decision_state"),
            "selected_robot_action": metric.get("selected_robot_action") or counter_move_for_prediction(predicted),
            "recommended_response_start_time_s": decision_time_s + float(response_delay_s),
            "overlay_path": metric.get("overlay_path"),
        }
        events.append(event)
    validation = validate_schunk_event_manifest(events, expected_count=len(clip_metrics))
    if not validation["passed"]:
        raise ValueError(f"Invalid SCHUNK event manifest: {validation}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    (path.parent / "validation_summary.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return events


def validate_schunk_event_manifest(events: Sequence[Mapping[str, object]], *, expected_count: int) -> dict[str, object]:
    """Validate SCHUNK event rows before render/response integration."""

    invalid: list[dict[str, object]] = []
    for event in events:
        predicted = str(event.get("predicted_final_gesture"))
        counter = str(event.get("selected_counter_move"))
        expected_counter = COUNTER_MOVES.get(cast(OpponentGesture, predicted))
        if expected_counter is None or counter != expected_counter:
            invalid.append({"clip_id": event.get("clip_id"), "reason": "invalid_counter_move"})
        if event.get("decision_frame") is None or event.get("decision_time_s") is None:
            invalid.append({"clip_id": event.get("clip_id"), "reason": "missing_decision_time"})
    return {
        "passed": len(events) == expected_count and len(invalid) == 0,
        "event_count": len(events),
        "expected_count": expected_count,
        "invalid_events": invalid,
    }


def build_rock_false_trigger_report(clip_metrics: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Build a compact OOD report for rock clips."""

    rock_metrics = [metric for metric in clip_metrics if metric.get("true_gesture") == "rock"]
    false_triggers = [metric for metric in rock_metrics if metric.get("failure_reason") == "false_trigger"]
    return {
        "passed": len(false_triggers) == 0 and len(rock_metrics) > 0 and all(bool(metric.get("passed")) for metric in rock_metrics),
        "rock_clip_count": len(rock_metrics),
        "false_trigger_count": len(false_triggers),
        "false_triggers": [
            {
                "clip_id": metric.get("clip_id"),
                "source_path": metric.get("source_path"),
                "predicted_gesture": metric.get("predicted_gesture"),
                "decision_frame": metric.get("decision_frame"),
                "decision_progress": metric.get("decision_progress"),
                "decision_confidence": metric.get("decision_confidence"),
                "decision_confidence_margin": metric.get("decision_confidence_margin"),
                "overlay_path": metric.get("overlay_path"),
            }
            for metric in false_triggers
        ],
        "rock_clip_status": [
            {
                "clip_id": metric.get("clip_id"),
                "passed": metric.get("passed"),
                "ood_status": metric.get("ood_status"),
                "predicted_gesture": metric.get("predicted_gesture"),
                "decision_state": metric.get("decision_state"),
                "selected_robot_action": metric.get("selected_robot_action"),
                "decision_progress": metric.get("decision_progress"),
                "decision_confidence": metric.get("decision_confidence"),
            }
            for metric in rock_metrics
        ],
    }


def build_dataset_expansion_plan(summary: Mapping[str, object], clip_metrics: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Create targeted dataset expansion guidance when strict validation fails."""

    failed = [metric for metric in clip_metrics if not bool(metric.get("passed"))]
    recommendations = []
    for metric in failed:
        reason = str(metric.get("failure_reason"))
        recommendations.append(
            {
                "clip_id": metric.get("clip_id"),
                "source_path": metric.get("source_path"),
                "true_gesture": metric.get("true_gesture"),
                "predicted_gesture": metric.get("predicted_gesture"),
                "failure_reason": reason,
                "recommended_data_action": _recommended_data_action(reason, metric),
            }
        )
    return {
        "status": "blocked_schunk_event_manifest",
        "reason": "strict_real_mp4_prediction_gate_failed",
        "summary_passed": bool(summary.get("passed")),
        "failed_clip_count": len(failed),
        "recommendations": recommendations,
    }


def write_dataset_expansion_plan(path_json: Path, path_md: Path, plan: Mapping[str, object]) -> None:
    """Write failure-driven skeleton dataset expansion guidance."""

    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(dict(plan), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Real Skeleton Dataset Expansion Plan",
        "",
        f"Status: `{plan.get('status')}`",
        f"Failed clip count: `{plan.get('failed_clip_count')}`",
        "",
        "## Recommendations",
        "",
    ]
    recommendations = plan.get("recommendations")
    if isinstance(recommendations, Sequence) and not isinstance(recommendations, (str, bytes)):
        for item in recommendations:
            if not isinstance(item, Mapping):
                continue
            lines.extend(
                [
                    f"- `{item.get('clip_id')}`: `{item.get('failure_reason')}`",
                    f"  - Source: `{item.get('source_path')}`",
                    f"  - Action: {item.get('recommended_data_action')}",
                ]
            )
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _no_decision_failure_reason(*, detected_count: int, qualified_count: int, prediction_counts: Counter[str]) -> str:
    if detected_count == 0:
        return "no_detection"
    if qualified_count == 0:
        return "low_confidence_or_margin"
    if len(prediction_counts) > 1:
        return "unstable_prediction"
    return "no_stable_decision"


def _recommended_data_action(reason: str, metric: Mapping[str, object]) -> str:
    first_correct_progress = metric.get("first_correct_stable_progress")
    first_correct_note = ""
    if isinstance(first_correct_progress, (int, float)) and not isinstance(first_correct_progress, bool):
        first_correct_note = f" First stable correct prediction appears at clip progress {float(first_correct_progress):.3f}."
    if reason == "no_detection":
        return "Record more real MP4s for the same transition with clearer framing, lighting, and hand scale; inspect MediaPipe detector thresholds before synthetic augmentation."
    if reason == "low_confidence_or_margin":
        return "Add more real skeleton examples near the early decision window and augment with small scale/rotation/speed perturbations."
    if reason == "unstable_prediction":
        return "Add contrastive real skeleton examples around the unstable frames and keep the rolling confirmation gate enabled."
    if reason == "wrong_prediction":
        return (
            "Prioritize new real recordings for the confused class pair, especially slow or fist-like early "
            f"`{metric.get('transition_label')}` openings, then retrain GRU/TCN with class-balanced hard examples."
            + first_correct_note
        )
    if reason == "late_decision":
        return "Add faster early-motion examples and preserve early-frame labels so the model learns before 50 percent clip progress." + first_correct_note
    if reason in {"no_wait_decision", "late_wait_decision"}:
        return "Add explicit rock-hold examples and calibrate the paper-wait policy so rock selects `wait_counter_paper` before 50 percent clip progress."
    if reason == "false_trigger":
        return "Add explicit rock/OOD examples and either train a 3-class predictor or add an abstention head/threshold before SCHUNK control."
    return "Inspect the clip manually, categorize the failure, then add targeted real skeleton examples before retraining."


def _first_correct_stable_row(
    rows: Sequence[Mapping[str, object]],
    *,
    true_gesture: FinalGesture,
    confirmation_count: int,
) -> Mapping[str, object] | None:
    for row in rows:
        if (
            row.get("prediction") == true_gesture
            and bool(row.get("qualifies_strict_gate"))
            and int(_number(row.get("rolling_confirmation_count"))) >= confirmation_count
        ):
            return row
    return None


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if len(values) == 0:
        return {"min": None, "max": None, "mean": None, "median": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _number(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _row_progress(row: Mapping[str, object], *, config: StrictDecisionConfig) -> float:
    if config.progress_key == "model_progress":
        return _number(row.get("model_progress"))
    if config.progress_key == "observed_progress":
        return _number(row.get("observed_progress"))
    if config.progress_key == "motion_progress":
        return _number(row.get("motion_progress"))
    return _number(row.get("clip_progress"))


def _effective_motion_energy(energies: np.ndarray) -> np.ndarray:
    if energies.ndim != 1:
        raise ValueError("energies must be one-dimensional")
    positive = energies[energies > 0.0]
    if positive.size == 0:
        return np.zeros_like(energies, dtype=np.float64)
    median = float(np.median(positive))
    mad = float(np.median(np.abs(positive - median)))
    threshold = max(1e-5, median + 3.0 * mad)
    effective = np.maximum(0.0, energies.astype(np.float64, copy=False) - threshold)
    if float(np.sum(effective)) <= 0.0:
        effective = energies.astype(np.float64, copy=True)
    return effective


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value != "":
        return value
    return None


def _final_gesture(value: str) -> FinalGesture:
    if value not in TRANSITION_GESTURES:
        raise ValueError(f"Expected paper/scissors gesture, got: {value}")
    return cast(FinalGesture, value)


def _opponent_gesture(value: str) -> OpponentGesture:
    if value not in COUNTER_MOVES:
        raise ValueError(f"Expected rock/paper/scissors gesture, got: {value}")
    return cast(OpponentGesture, value)


def _natural_key(value: str) -> str:
    return "".join(part.zfill(8) if part.isdigit() else part for part in re.split(r"(\d+)", value))
