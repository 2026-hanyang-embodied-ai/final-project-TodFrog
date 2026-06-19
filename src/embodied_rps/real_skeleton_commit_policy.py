"""Commit-at-deadline scoring for realtime skeleton RPS predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from embodied_rps.real_skeleton_video_eval import (
    COUNTER_MOVES,
    TRANSITION_GESTURES,
    WAIT_COUNTER_PAPER_STATE,
    EvaluationGesture,
    ProgressKey,
    StrictDecisionConfig,
    annotate_rows_with_strict_decision,
    counter_move_for_prediction,
    robot_action_for_decision_state,
)


@dataclass(frozen=True)
class CommitDecisionConfig:
    """Policy that allows provisional predictions to change until a commit deadline."""

    confidence_threshold: float = 0.85
    margin_threshold: float = 0.20
    confirmation_count: int = 3
    max_commit_progress: float = 0.50
    transition_mass_threshold: float = 0.15
    binary_transition_mass_threshold: float = 0.0
    progress_key: ProgressKey = "clip_progress"
    transition_selection_mode: Literal["latest", "highest_confidence"] = "latest"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.margin_threshold <= 1.0:
            raise ValueError("margin_threshold must be in [0, 1]")
        if self.confirmation_count <= 0:
            raise ValueError("confirmation_count must be positive")
        if not 0.0 < self.max_commit_progress <= 1.0:
            raise ValueError("max_commit_progress must be in (0, 1]")
        if not 0.0 <= self.transition_mass_threshold <= 1.0:
            raise ValueError("transition_mass_threshold must be in [0, 1]")
        if not 0.0 <= self.binary_transition_mass_threshold <= 1.0:
            raise ValueError("binary_transition_mass_threshold must be in [0, 1]")
        if self.progress_key not in {"clip_progress", "motion_progress", "observed_progress", "model_progress"}:
            raise ValueError("progress_key must be clip_progress, motion_progress, observed_progress, or model_progress")
        if self.transition_selection_mode not in {"latest", "highest_confidence"}:
            raise ValueError("transition_selection_mode must be latest or highest_confidence")

    def to_strict_config(self) -> StrictDecisionConfig:
        """Return the per-frame qualifier config used before commit scoring."""

        return StrictDecisionConfig(
            confidence_threshold=self.confidence_threshold,
            margin_threshold=self.margin_threshold,
            confirmation_count=self.confirmation_count,
            max_decision_progress=self.max_commit_progress,
            transition_mass_threshold=self.transition_mass_threshold,
            paper_wait_is_terminal_for_transitions=False,
            binary_transition_mass_threshold=self.binary_transition_mass_threshold,
            progress_key=self.progress_key,
        )


def summarize_commit_clip_decision(
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
    config: CommitDecisionConfig,
    overlay_path: Path | None = None,
    frame_csv_path: Path | None = None,
    frame_jsonl_path: Path | None = None,
) -> dict[str, object]:
    """Score one clip by committing to the latest stable state before deadline."""

    strict_config = config.to_strict_config()
    annotated = annotate_rows_with_strict_decision(rows, config=strict_config)
    detected_count = sum(1 for row in annotated if bool(row.get("detected")))
    stable_transition_rows = [
        row
        for row in annotated
        if row.get("decision_state") in TRANSITION_GESTURES
        and bool(row.get("qualifies_strict_gate"))
        and int(_number(row.get("rolling_confirmation_count"))) >= config.confirmation_count
    ]
    stable_wait_rows = [
        row
        for row in annotated
        if row.get("decision_state") == WAIT_COUNTER_PAPER_STATE
        and bool(row.get("qualifies_strict_gate"))
        and int(_number(row.get("rolling_confirmation_count"))) >= config.confirmation_count
    ]
    transition_rows_before_deadline = [
        row for row in stable_transition_rows if _row_progress(row, progress_key=config.progress_key) <= config.max_commit_progress
    ]
    wait_rows_before_deadline = [
        row for row in stable_wait_rows if _row_progress(row, progress_key=config.progress_key) <= config.max_commit_progress
    ]
    first_stable_transition_row = stable_transition_rows[0] if stable_transition_rows else None
    first_correct_stable_row = next(
        (row for row in stable_transition_rows if str(row.get("decision_state")) == true_gesture),
        None,
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
        "confidence_threshold": config.confidence_threshold,
        "margin_threshold": config.margin_threshold,
        "confirmation_count": config.confirmation_count,
        "max_commit_progress": config.max_commit_progress,
        "progress_key": config.progress_key,
        "binary_transition_mass_threshold": config.binary_transition_mass_threshold,
        "commit_policy": "latest_stable_before_deadline",
        "transition_selection_mode": config.transition_selection_mode,
        "overlay_path": overlay_path.as_posix() if overlay_path is not None else None,
        "frame_csv_path": frame_csv_path.as_posix() if frame_csv_path is not None else None,
        "frame_jsonl_path": frame_jsonl_path.as_posix() if frame_jsonl_path is not None else None,
        "first_stable_transition_state": (
            str(first_stable_transition_row.get("decision_state")) if first_stable_transition_row is not None else None
        ),
        "first_stable_transition_progress": (
            _row_progress(first_stable_transition_row, progress_key=config.progress_key)
            if first_stable_transition_row is not None
            else None
        ),
        "first_correct_stable_progress": (
            _row_progress(first_correct_stable_row, progress_key=config.progress_key)
            if first_correct_stable_row is not None
            else None
        ),
    }

    if true_gesture == "rock":
        return _summarize_rock_commit(
            base,
            transition_rows_before_deadline=transition_rows_before_deadline,
            wait_rows_before_deadline=wait_rows_before_deadline,
            config=config,
        )
    return _summarize_transition_commit(
        base,
        true_gesture=true_gesture,
        transition_rows_before_deadline=transition_rows_before_deadline,
        first_correct_stable_row=first_correct_stable_row,
        detected_count=detected_count,
        config=config,
    )


def _summarize_transition_commit(
    base: dict[str, object],
    *,
    true_gesture: EvaluationGesture,
    transition_rows_before_deadline: Sequence[Mapping[str, object]],
    first_correct_stable_row: Mapping[str, object] | None,
    detected_count: int,
    config: CommitDecisionConfig,
) -> dict[str, object]:
    commit_row = _select_transition_commit_row(transition_rows_before_deadline, config=config)
    if commit_row is None:
        late_correct = (
            first_correct_stable_row is not None
            and _row_progress(first_correct_stable_row, progress_key=config.progress_key) > config.max_commit_progress
        )
        base.update(
            {
                "passed": False,
                "failure_reason": "late_decision" if late_correct else ("no_detection" if detected_count == 0 else "no_stable_decision"),
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
            }
        )
        return base

    decision_state = str(commit_row.get("decision_state"))
    predicted = _final_gesture(decision_state)
    correct = predicted == true_gesture
    counter_move = counter_move_for_prediction(predicted)
    base.update(
        {
            "passed": correct,
            "failure_reason": None if correct else "wrong_commit_prediction",
            "predicted_gesture": predicted,
            "decision_state": decision_state,
            "decision_frame": int(_number(commit_row.get("frame_index"))),
            "decision_time_s": _number(commit_row.get("time_s")),
            "decision_progress": _row_progress(commit_row, progress_key=config.progress_key),
            "decision_model_progress": _number(commit_row.get("model_progress")),
            "decision_confidence": _number(commit_row.get("confidence")),
            "decision_confidence_margin": _number(commit_row.get("confidence_margin")),
            "counter_move": counter_move,
            "selected_robot_action": counter_move,
        }
    )
    return base


def _select_transition_commit_row(
    rows: Sequence[Mapping[str, object]],
    *,
    config: CommitDecisionConfig,
) -> Mapping[str, object] | None:
    if not rows:
        return None
    if config.transition_selection_mode == "highest_confidence":
        return max(
            rows,
            key=lambda row: (
                _number(row.get("confidence")),
                _number(row.get("confidence_margin")),
                _row_progress(row, progress_key=config.progress_key),
            ),
        )
    return rows[-1]


def _summarize_rock_commit(
    base: dict[str, object],
    *,
    transition_rows_before_deadline: Sequence[Mapping[str, object]],
    wait_rows_before_deadline: Sequence[Mapping[str, object]],
    config: CommitDecisionConfig,
) -> dict[str, object]:
    false_trigger_row = transition_rows_before_deadline[0] if transition_rows_before_deadline else None
    if false_trigger_row is not None:
        predicted = _final_gesture(str(false_trigger_row.get("decision_state")))
        base.update(
            {
                "passed": False,
                "failure_reason": "false_trigger",
                "predicted_gesture": predicted,
                "decision_state": str(false_trigger_row.get("decision_state")),
                "decision_frame": int(_number(false_trigger_row.get("frame_index"))),
                "decision_time_s": _number(false_trigger_row.get("time_s")),
                "decision_progress": _row_progress(false_trigger_row, progress_key=config.progress_key),
                "decision_model_progress": _number(false_trigger_row.get("model_progress")),
                "decision_confidence": _number(false_trigger_row.get("confidence")),
                "decision_confidence_margin": _number(false_trigger_row.get("confidence_margin")),
                "counter_move": counter_move_for_prediction(predicted),
                "selected_robot_action": counter_move_for_prediction(predicted),
                "ood_status": "false_trigger",
            }
        )
        return base

    commit_row = wait_rows_before_deadline[-1] if wait_rows_before_deadline else None
    if commit_row is None:
        base.update(
            {
                "passed": False,
                "failure_reason": "no_stable_wait",
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

    robot_action = robot_action_for_decision_state(WAIT_COUNTER_PAPER_STATE)
    base.update(
        {
            "passed": True,
            "failure_reason": None,
            "predicted_gesture": "rock",
            "decision_state": WAIT_COUNTER_PAPER_STATE,
            "decision_frame": int(_number(commit_row.get("frame_index"))),
            "decision_time_s": _number(commit_row.get("time_s")),
            "decision_progress": _row_progress(commit_row, progress_key=config.progress_key),
            "decision_model_progress": _number(commit_row.get("model_progress")),
            "decision_confidence": _number(commit_row.get("confidence")),
            "decision_confidence_margin": _number(commit_row.get("confidence_margin")),
            "counter_move": robot_action,
            "selected_robot_action": robot_action,
            "ood_status": "wait_counter_paper",
        }
    )
    return base


def _final_gesture(value: str) -> EvaluationGesture:
    if value not in COUNTER_MOVES:
        raise ValueError(f"unsupported gesture: {value}")
    return value  # type: ignore[return-value]


def _row_progress(row: Mapping[str, object], *, progress_key: ProgressKey) -> float:
    if progress_key == "model_progress":
        return _number(row.get("model_progress"))
    if progress_key == "observed_progress":
        return _number(row.get("observed_progress"))
    if progress_key == "motion_progress":
        return _number(row.get("motion_progress"))
    return _number(row.get("clip_progress"))


def _number(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


__all__ = ["CommitDecisionConfig", "summarize_commit_clip_decision"]
