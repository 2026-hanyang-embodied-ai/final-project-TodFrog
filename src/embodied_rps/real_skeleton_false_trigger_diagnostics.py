"""Diagnostics for stable binary false-trigger episodes in saved RPS rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class StableStateEpisode:
    """One consecutive stable decision-state episode."""

    state: str
    start_frame: int
    end_frame: int
    start_progress: float
    end_progress: float
    duration_frames: int
    max_confidence: float
    max_margin: float
    max_transition_mass: float
    max_motion_progress: float

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_progress": self.start_progress,
            "end_progress": self.end_progress,
            "duration_frames": self.duration_frames,
            "max_confidence": self.max_confidence,
            "max_margin": self.max_margin,
            "max_transition_mass": self.max_transition_mass,
            "max_motion_progress": self.max_motion_progress,
        }


def extract_stable_state_episodes(
    rows: Sequence[Mapping[str, object]],
    *,
    confirmation_count: int,
    max_progress: float,
    progress_key: str = "clip_progress",
) -> list[StableStateEpisode]:
    """Group consecutive rows whose decision state has reached confirmation."""

    stable_rows = [
        row
        for row in rows
        if _state(row) is not None
        and int(_number(row.get("rolling_confirmation_count"))) >= confirmation_count
        and _progress(row, progress_key=progress_key) <= max_progress
    ]
    episodes: list[StableStateEpisode] = []
    current: list[Mapping[str, object]] = []
    current_state: str | None = None
    previous_frame: int | None = None
    for row in stable_rows:
        row_state = _state(row)
        frame_index = int(_number(row.get("frame_index")))
        if row_state != current_state or (previous_frame is not None and frame_index != previous_frame + 1):
            if current:
                episodes.append(_episode_from_rows(current, progress_key=progress_key))
            current = [row]
            current_state = row_state
        else:
            current.append(row)
        previous_frame = frame_index
    if current:
        episodes.append(_episode_from_rows(current, progress_key=progress_key))
    return episodes


def summarize_rock_false_trigger_diagnostics(
    *,
    clip_id: str,
    true_gesture: str,
    rows: Sequence[Mapping[str, object]],
    confirmation_count: int,
    max_progress: float,
    progress_key: str = "clip_progress",
) -> dict[str, object]:
    """Summarize stable binary episodes that would false-trigger a rock clip."""

    episodes = extract_stable_state_episodes(
        rows,
        confirmation_count=confirmation_count,
        max_progress=max_progress,
        progress_key=progress_key,
    )
    binary_episodes = [episode for episode in episodes if episode.state in {"paper", "scissors"}]
    wait_episodes = [episode for episode in episodes if episode.state == "wait_counter_paper"]
    first_binary = binary_episodes[0] if binary_episodes else None
    stable_wait_before = bool(
        first_binary is not None
        and any(wait.end_progress <= first_binary.start_progress for wait in wait_episodes)
    )
    return {
        "clip_id": clip_id,
        "true_gesture": true_gesture,
        "episode_count": len(episodes),
        "wait_episode_count": len(wait_episodes),
        "false_trigger_episode_count": len(binary_episodes),
        "stable_wait_before_false_trigger": stable_wait_before,
        "first_false_trigger_state": first_binary.state if first_binary is not None else None,
        "first_false_trigger_frame": first_binary.start_frame if first_binary is not None else None,
        "first_false_trigger_progress": first_binary.start_progress if first_binary is not None else None,
        "max_false_trigger_confidence": (
            max(episode.max_confidence for episode in binary_episodes) if binary_episodes else None
        ),
        "max_false_trigger_motion_progress": (
            max(episode.max_motion_progress for episode in binary_episodes) if binary_episodes else None
        ),
        "episodes": [episode.to_dict() for episode in episodes],
    }


def _episode_from_rows(rows: Sequence[Mapping[str, object]], *, progress_key: str) -> StableStateEpisode:
    first = rows[0]
    last = rows[-1]
    row_state = _state(first)
    if row_state is None:
        raise ValueError("episode rows must have a decision_state")
    return StableStateEpisode(
        state=row_state,
        start_frame=int(_number(first.get("frame_index"))),
        end_frame=int(_number(last.get("frame_index"))),
        start_progress=_progress(first, progress_key=progress_key),
        end_progress=_progress(last, progress_key=progress_key),
        duration_frames=len(rows),
        max_confidence=max(_number(row.get("confidence")) for row in rows),
        max_margin=max(_number(row.get("confidence_margin")) for row in rows),
        max_transition_mass=max(_number(row.get("transition_mass")) for row in rows),
        max_motion_progress=max(_number(row.get("motion_progress")) for row in rows),
    )


def _state(row: Mapping[str, object]) -> str | None:
    value = row.get("decision_state")
    return value if isinstance(value, str) and value else None


def _progress(row: Mapping[str, object], *, progress_key: str) -> float:
    return _number(row.get(progress_key))


def _number(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


__all__ = [
    "StableStateEpisode",
    "extract_stable_state_episodes",
    "summarize_rock_false_trigger_diagnostics",
]
