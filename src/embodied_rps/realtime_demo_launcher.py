"""Launcher helpers for the current-best realtime RPS skeleton demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class RealtimeDemoConfig:
    """Current-best realtime demo policy and prompt-cycle configuration."""

    profiles: tuple[Path, ...]
    profile_weights: str | None = None
    scissors_rescue_profile_index: int | None = None
    scissors_rescue_confidence_threshold: float = 0.90
    scissors_rescue_margin_threshold: float = 0.98
    scissors_rescue_min_blended_transition_mass: float = 0.0
    scissors_rescue_max_blended_rock_probability: float | None = None
    conditional_scissors_rescue_profile_index: int | None = None
    conditional_scissors_rescue_confidence_threshold: float = 0.99
    conditional_scissors_rescue_margin_threshold: float = 0.98
    conditional_scissors_rescue_min_blended_transition_mass: float = 0.80
    conditional_scissors_rescue_max_blended_rock_probability: float | None = None
    paper_rescue_min_history_frames: int = 0
    paper_rescue_min_observed_progress: float = 0.35
    paper_rescue_min_scissors_confidence: float = 0.70
    paper_rescue_min_scissors_margin: float = 0.40
    paper_rescue_min_ring_pinky_extension_delta: float = 0.08
    paper_rescue_min_latest_ring_pinky_extension: float = 0.60
    paper_rescue_max_index_middle_minus_ring_pinky: float = 0.25
    paper_rescue_max_rock_probability: float | None = 0.40
    paper_rescue_min_transition_mass: float = 0.60
    late_geometry_paper_min_history_frames: int = 0
    late_geometry_paper_min_observed_progress: float = 0.35
    late_geometry_paper_max_observed_progress: float = 0.50
    late_geometry_paper_min_ring_pinky_extension_delta: float = 0.04
    late_geometry_paper_min_latest_ring_pinky_extension: float = 0.60
    late_geometry_paper_max_index_middle_minus_ring_pinky: float = 0.25
    late_geometry_paper_max_index_middle_delta_minus_ring_pinky_delta: float = 0.12
    rock_hold_guard_min_history_frames: int = 0
    rock_hold_guard_max_latest_finger_extension: float = 0.55
    rock_hold_guard_max_extension_delta: float = 0.08
    gesture_verifier_min_history_frames: int = 0
    gesture_verifier_rock_max_ring_pinky_extension: float = 0.75
    gesture_verifier_rock_max_index_middle_extension: float = 1.08
    gesture_verifier_rock_max_index_middle_minus_ring_pinky: float = 0.38
    gesture_verifier_rock_max_extension_delta: float = 0.20
    gesture_verifier_scissors_min_index_middle_extension: float = 0.85
    gesture_verifier_scissors_min_index_middle_delta: float = 0.04
    gesture_verifier_scissors_min_index_middle_minus_ring_pinky: float = 0.25
    gesture_verifier_scissors_max_ring_pinky_extension: float = 0.62
    gesture_verifier_paper_min_ring_pinky_extension: float = 0.60
    gesture_verifier_paper_min_ring_pinky_delta: float = 0.04
    gesture_verifier_paper_max_index_middle_minus_ring_pinky: float = 0.25
    confidence_threshold: float = 0.70
    margin_threshold: float = 0.10
    confirmation_count: int = 2
    transition_mass_threshold: float = 0.05
    binary_transition_mass_threshold: float = 0.60
    min_binary_decision_progress: float = 0.05
    prompt_cycle: bool = True
    prompt_cycle_s: float = 1.0
    prompt_sequence: str = "rock,paper,scissors"
    response_prompt: str | None = None
    hold_response_prompt_until_decision: bool = False
    response_hold_max_frames: int = 0
    stop_after_confirmed_response_decision: bool = False
    post_decision_hold_frames: int = 0
    reset_on_prompt_cycle: bool = True
    reset_on_prompt_change: bool = False
    display_window: bool | None = None
    device: str = "auto"


def load_realtime_demo_config(path: Path) -> RealtimeDemoConfig:
    """Load a realtime demo launcher YAML file."""

    with path.open("r", encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ValueError("Realtime demo config must be a mapping")
    profiles_value = loaded.get("profiles")
    if not isinstance(profiles_value, list) or len(profiles_value) == 0:
        raise ValueError("Realtime demo config must include at least one profile")
    profiles = tuple(Path(str(item)) for item in profiles_value)
    return RealtimeDemoConfig(
        profiles=profiles,
        profile_weights=_optional_string(loaded.get("profile_weights")),
        scissors_rescue_profile_index=_optional_int(loaded.get("scissors_rescue_profile_index")),
        scissors_rescue_confidence_threshold=_float_value(loaded, "scissors_rescue_confidence_threshold", 0.90),
        scissors_rescue_margin_threshold=_float_value(loaded, "scissors_rescue_margin_threshold", 0.98),
        scissors_rescue_min_blended_transition_mass=_float_value(loaded, "scissors_rescue_min_blended_transition_mass", 0.0),
        scissors_rescue_max_blended_rock_probability=_optional_float(loaded.get("scissors_rescue_max_blended_rock_probability")),
        conditional_scissors_rescue_profile_index=_optional_int(loaded.get("conditional_scissors_rescue_profile_index")),
        conditional_scissors_rescue_confidence_threshold=_float_value(loaded, "conditional_scissors_rescue_confidence_threshold", 0.99),
        conditional_scissors_rescue_margin_threshold=_float_value(loaded, "conditional_scissors_rescue_margin_threshold", 0.98),
        conditional_scissors_rescue_min_blended_transition_mass=_float_value(loaded, "conditional_scissors_rescue_min_blended_transition_mass", 0.80),
        conditional_scissors_rescue_max_blended_rock_probability=_optional_float(loaded.get("conditional_scissors_rescue_max_blended_rock_probability")),
        paper_rescue_min_history_frames=_int_value(loaded, "paper_rescue_min_history_frames", 0),
        paper_rescue_min_observed_progress=_float_value(loaded, "paper_rescue_min_observed_progress", 0.35),
        paper_rescue_min_scissors_confidence=_float_value(loaded, "paper_rescue_min_scissors_confidence", 0.70),
        paper_rescue_min_scissors_margin=_float_value(loaded, "paper_rescue_min_scissors_margin", 0.40),
        paper_rescue_min_ring_pinky_extension_delta=_float_value(loaded, "paper_rescue_min_ring_pinky_extension_delta", 0.08),
        paper_rescue_min_latest_ring_pinky_extension=_float_value(loaded, "paper_rescue_min_latest_ring_pinky_extension", 0.60),
        paper_rescue_max_index_middle_minus_ring_pinky=_float_value(loaded, "paper_rescue_max_index_middle_minus_ring_pinky", 0.25),
        paper_rescue_max_rock_probability=_optional_float(loaded.get("paper_rescue_max_rock_probability")),
        paper_rescue_min_transition_mass=_float_value(loaded, "paper_rescue_min_transition_mass", 0.60),
        late_geometry_paper_min_history_frames=_int_value(loaded, "late_geometry_paper_min_history_frames", 0),
        late_geometry_paper_min_observed_progress=_float_value(loaded, "late_geometry_paper_min_observed_progress", 0.35),
        late_geometry_paper_max_observed_progress=_float_value(loaded, "late_geometry_paper_max_observed_progress", 0.50),
        late_geometry_paper_min_ring_pinky_extension_delta=_float_value(loaded, "late_geometry_paper_min_ring_pinky_extension_delta", 0.04),
        late_geometry_paper_min_latest_ring_pinky_extension=_float_value(loaded, "late_geometry_paper_min_latest_ring_pinky_extension", 0.60),
        late_geometry_paper_max_index_middle_minus_ring_pinky=_float_value(loaded, "late_geometry_paper_max_index_middle_minus_ring_pinky", 0.25),
        late_geometry_paper_max_index_middle_delta_minus_ring_pinky_delta=_float_value(
            loaded,
            "late_geometry_paper_max_index_middle_delta_minus_ring_pinky_delta",
            0.12,
        ),
        rock_hold_guard_min_history_frames=_int_value(loaded, "rock_hold_guard_min_history_frames", 0),
        rock_hold_guard_max_latest_finger_extension=_float_value(loaded, "rock_hold_guard_max_latest_finger_extension", 0.55),
        rock_hold_guard_max_extension_delta=_float_value(loaded, "rock_hold_guard_max_extension_delta", 0.08),
        gesture_verifier_min_history_frames=_int_value(loaded, "gesture_verifier_min_history_frames", 0),
        gesture_verifier_rock_max_ring_pinky_extension=_float_value(loaded, "gesture_verifier_rock_max_ring_pinky_extension", 0.75),
        gesture_verifier_rock_max_index_middle_extension=_float_value(loaded, "gesture_verifier_rock_max_index_middle_extension", 1.08),
        gesture_verifier_rock_max_index_middle_minus_ring_pinky=_float_value(
            loaded,
            "gesture_verifier_rock_max_index_middle_minus_ring_pinky",
            0.38,
        ),
        gesture_verifier_rock_max_extension_delta=_float_value(loaded, "gesture_verifier_rock_max_extension_delta", 0.20),
        gesture_verifier_scissors_min_index_middle_extension=_float_value(
            loaded,
            "gesture_verifier_scissors_min_index_middle_extension",
            0.85,
        ),
        gesture_verifier_scissors_min_index_middle_delta=_float_value(
            loaded,
            "gesture_verifier_scissors_min_index_middle_delta",
            0.04,
        ),
        gesture_verifier_scissors_min_index_middle_minus_ring_pinky=_float_value(
            loaded,
            "gesture_verifier_scissors_min_index_middle_minus_ring_pinky",
            0.25,
        ),
        gesture_verifier_scissors_max_ring_pinky_extension=_float_value(
            loaded,
            "gesture_verifier_scissors_max_ring_pinky_extension",
            0.62,
        ),
        gesture_verifier_paper_min_ring_pinky_extension=_float_value(
            loaded,
            "gesture_verifier_paper_min_ring_pinky_extension",
            0.60,
        ),
        gesture_verifier_paper_min_ring_pinky_delta=_float_value(loaded, "gesture_verifier_paper_min_ring_pinky_delta", 0.04),
        gesture_verifier_paper_max_index_middle_minus_ring_pinky=_float_value(
            loaded,
            "gesture_verifier_paper_max_index_middle_minus_ring_pinky",
            0.25,
        ),
        confidence_threshold=_float_value(loaded, "confidence_threshold", 0.70),
        margin_threshold=_float_value(loaded, "margin_threshold", 0.10),
        confirmation_count=_int_value(loaded, "confirmation_count", 2),
        transition_mass_threshold=_float_value(loaded, "transition_mass_threshold", 0.05),
        binary_transition_mass_threshold=_float_value(loaded, "binary_transition_mass_threshold", 0.60),
        min_binary_decision_progress=_float_value(loaded, "min_binary_decision_progress", 0.05),
        prompt_cycle=bool(loaded.get("prompt_cycle", True)),
        prompt_cycle_s=_float_value(loaded, "prompt_cycle_s", 1.0),
        prompt_sequence=str(loaded.get("prompt_sequence", "rock,paper,scissors")),
        response_prompt=_optional_string(loaded.get("response_prompt")),
        hold_response_prompt_until_decision=bool(loaded.get("hold_response_prompt_until_decision", False)),
        response_hold_max_frames=_int_value(loaded, "response_hold_max_frames", 0),
        stop_after_confirmed_response_decision=bool(loaded.get("stop_after_confirmed_response_decision", False)),
        post_decision_hold_frames=_int_value(loaded, "post_decision_hold_frames", 0),
        reset_on_prompt_cycle=bool(loaded.get("reset_on_prompt_cycle", True)),
        reset_on_prompt_change=bool(loaded.get("reset_on_prompt_change", False)),
        display_window=_optional_bool(loaded.get("display_window")),
        device=str(loaded.get("device", "auto")),
    )


def build_realtime_demo_argv(
    config: RealtimeDemoConfig,
    *,
    video: Path | None,
    camera: int | None,
    output: Path | None,
    max_frames: int | None,
    frame_log_jsonl: Path | None = None,
    skeleton_npz: Path | None = None,
    expected_actual_gesture: str | None = None,
    collection_mode: bool = False,
) -> list[str]:
    """Build argv for `run_realtime_skeleton_predictor` from a demo config."""

    if (video is None) == (camera is None):
        raise ValueError("Provide either video or camera, but not both")
    argv: list[str] = []
    for profile in config.profiles:
        argv.extend(["--profile", str(profile)])
    if config.profile_weights:
        argv.extend(["--profile-weights", config.profile_weights])
    if config.scissors_rescue_profile_index is not None:
        argv.extend(["--scissors-rescue-profile-index", str(config.scissors_rescue_profile_index)])
        argv.extend(["--scissors-rescue-confidence-threshold", str(config.scissors_rescue_confidence_threshold)])
        argv.extend(["--scissors-rescue-margin-threshold", str(config.scissors_rescue_margin_threshold)])
        argv.extend(["--scissors-rescue-min-blended-transition-mass", str(config.scissors_rescue_min_blended_transition_mass)])
        if config.scissors_rescue_max_blended_rock_probability is not None:
            argv.extend(
                [
                    "--scissors-rescue-max-blended-rock-probability",
                    str(config.scissors_rescue_max_blended_rock_probability),
                ]
            )
    if config.conditional_scissors_rescue_profile_index is not None:
        argv.extend(["--conditional-scissors-rescue-profile-index", str(config.conditional_scissors_rescue_profile_index)])
        argv.extend(["--conditional-scissors-rescue-confidence-threshold", str(config.conditional_scissors_rescue_confidence_threshold)])
        argv.extend(["--conditional-scissors-rescue-margin-threshold", str(config.conditional_scissors_rescue_margin_threshold)])
        argv.extend(["--conditional-scissors-rescue-min-blended-transition-mass", str(config.conditional_scissors_rescue_min_blended_transition_mass)])
        if config.conditional_scissors_rescue_max_blended_rock_probability is not None:
            argv.extend(
                [
                    "--conditional-scissors-rescue-max-blended-rock-probability",
                    str(config.conditional_scissors_rescue_max_blended_rock_probability),
                ]
            )
    if config.paper_rescue_min_history_frames > 0:
        argv.extend(["--paper-rescue-min-history-frames", str(config.paper_rescue_min_history_frames)])
        argv.extend(["--paper-rescue-min-observed-progress", str(config.paper_rescue_min_observed_progress)])
        argv.extend(["--paper-rescue-min-scissors-confidence", str(config.paper_rescue_min_scissors_confidence)])
        argv.extend(["--paper-rescue-min-scissors-margin", str(config.paper_rescue_min_scissors_margin)])
        argv.extend(["--paper-rescue-min-ring-pinky-extension-delta", str(config.paper_rescue_min_ring_pinky_extension_delta)])
        argv.extend(["--paper-rescue-min-latest-ring-pinky-extension", str(config.paper_rescue_min_latest_ring_pinky_extension)])
        argv.extend(["--paper-rescue-max-index-middle-minus-ring-pinky", str(config.paper_rescue_max_index_middle_minus_ring_pinky)])
        if config.paper_rescue_max_rock_probability is not None:
            argv.extend(["--paper-rescue-max-rock-probability", str(config.paper_rescue_max_rock_probability)])
        argv.extend(["--paper-rescue-min-transition-mass", str(config.paper_rescue_min_transition_mass)])
    if config.late_geometry_paper_min_history_frames > 0:
        argv.extend(["--late-geometry-paper-min-history-frames", str(config.late_geometry_paper_min_history_frames)])
        argv.extend(["--late-geometry-paper-min-observed-progress", str(config.late_geometry_paper_min_observed_progress)])
        argv.extend(["--late-geometry-paper-max-observed-progress", str(config.late_geometry_paper_max_observed_progress)])
        argv.extend(
            [
                "--late-geometry-paper-min-ring-pinky-extension-delta",
                str(config.late_geometry_paper_min_ring_pinky_extension_delta),
            ]
        )
        argv.extend(
            [
                "--late-geometry-paper-min-latest-ring-pinky-extension",
                str(config.late_geometry_paper_min_latest_ring_pinky_extension),
            ]
        )
        argv.extend(
            [
                "--late-geometry-paper-max-index-middle-minus-ring-pinky",
                str(config.late_geometry_paper_max_index_middle_minus_ring_pinky),
            ]
        )
        argv.extend(
            [
                "--late-geometry-paper-max-index-middle-delta-minus-ring-pinky-delta",
                str(config.late_geometry_paper_max_index_middle_delta_minus_ring_pinky_delta),
            ]
        )
    if config.rock_hold_guard_min_history_frames > 0:
        argv.extend(["--rock-hold-guard-min-history-frames", str(config.rock_hold_guard_min_history_frames)])
        argv.extend(["--rock-hold-guard-max-latest-finger-extension", str(config.rock_hold_guard_max_latest_finger_extension)])
        argv.extend(["--rock-hold-guard-max-extension-delta", str(config.rock_hold_guard_max_extension_delta)])
    if config.gesture_verifier_min_history_frames > 0:
        argv.extend(["--gesture-verifier-min-history-frames", str(config.gesture_verifier_min_history_frames)])
        argv.extend(["--gesture-verifier-rock-max-ring-pinky-extension", str(config.gesture_verifier_rock_max_ring_pinky_extension)])
        argv.extend(["--gesture-verifier-rock-max-index-middle-extension", str(config.gesture_verifier_rock_max_index_middle_extension)])
        argv.extend(
            [
                "--gesture-verifier-rock-max-index-middle-minus-ring-pinky",
                str(config.gesture_verifier_rock_max_index_middle_minus_ring_pinky),
            ]
        )
        argv.extend(["--gesture-verifier-rock-max-extension-delta", str(config.gesture_verifier_rock_max_extension_delta)])
        argv.extend(
            [
                "--gesture-verifier-scissors-min-index-middle-extension",
                str(config.gesture_verifier_scissors_min_index_middle_extension),
            ]
        )
        argv.extend(
            [
                "--gesture-verifier-scissors-min-index-middle-delta",
                str(config.gesture_verifier_scissors_min_index_middle_delta),
            ]
        )
        argv.extend(
            [
                "--gesture-verifier-scissors-min-index-middle-minus-ring-pinky",
                str(config.gesture_verifier_scissors_min_index_middle_minus_ring_pinky),
            ]
        )
        argv.extend(
            [
                "--gesture-verifier-scissors-max-ring-pinky-extension",
                str(config.gesture_verifier_scissors_max_ring_pinky_extension),
            ]
        )
        argv.extend(["--gesture-verifier-paper-min-ring-pinky-extension", str(config.gesture_verifier_paper_min_ring_pinky_extension)])
        argv.extend(["--gesture-verifier-paper-min-ring-pinky-delta", str(config.gesture_verifier_paper_min_ring_pinky_delta)])
        argv.extend(
            [
                "--gesture-verifier-paper-max-index-middle-minus-ring-pinky",
                str(config.gesture_verifier_paper_max_index_middle_minus_ring_pinky),
            ]
        )
    if video is not None:
        argv.extend(["--video", str(video)])
    if camera is not None:
        argv.extend(["--camera", str(camera)])
    if output is not None:
        argv.extend(["--output", str(output)])
    if frame_log_jsonl is not None:
        argv.extend(["--frame-log-jsonl", str(frame_log_jsonl)])
    if skeleton_npz is not None:
        argv.extend(["--skeleton-npz", str(skeleton_npz)])
    if max_frames is not None:
        argv.extend(["--max-frames", str(max_frames)])
    argv.extend(["--device", config.device])
    argv.extend(["--confidence-threshold", str(config.confidence_threshold)])
    argv.extend(["--margin-threshold", str(config.margin_threshold)])
    argv.extend(["--confirmation-count", str(config.confirmation_count)])
    argv.extend(["--transition-mass-threshold", str(config.transition_mass_threshold)])
    argv.extend(["--binary-transition-mass-threshold", str(config.binary_transition_mass_threshold)])
    argv.extend(["--min-binary-decision-progress", str(config.min_binary_decision_progress)])
    if config.prompt_cycle:
        argv.append("--prompt-cycle")
        argv.extend(["--prompt-cycle-s", str(config.prompt_cycle_s)])
        argv.extend(["--prompt-sequence", config.prompt_sequence])
        if config.response_prompt:
            argv.extend(["--response-prompt", config.response_prompt])
        if config.hold_response_prompt_until_decision:
            argv.append("--hold-response-prompt-until-decision")
            argv.extend(["--response-hold-max-frames", str(config.response_hold_max_frames)])
        if config.stop_after_confirmed_response_decision and not collection_mode:
            argv.append("--stop-after-confirmed-response-decision")
            argv.extend(["--post-decision-hold-frames", str(config.post_decision_hold_frames)])
        if expected_actual_gesture:
            argv.extend(["--expected-actual-gesture", expected_actual_gesture])
        if config.reset_on_prompt_cycle:
            argv.append("--reset-on-prompt-cycle")
        if config.reset_on_prompt_change:
            argv.append("--reset-on-prompt-change")
    display_window = config.display_window if config.display_window is not None else camera is not None
    argv.append("--display-window" if display_window else "--no-display-window")
    return argv


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip()
    return parsed or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    parsed = str(value).strip().lower()
    if parsed in {"1", "true", "yes", "on"}:
        return True
    if parsed in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def _float_value(mapping: Mapping[str, Any], key: str, default: float) -> float:
    return float(mapping.get(key, default))


def _int_value(mapping: Mapping[str, Any], key: str, default: int) -> int:
    return int(mapping.get(key, default))


__all__ = ["RealtimeDemoConfig", "build_realtime_demo_argv", "load_realtime_demo_config"]
