"""Run a trained real-skeleton final-gesture predictor on video or webcam input."""

from __future__ import annotations

import argparse
import json
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from embodied_rps.models import build_classifier
from embodied_rps.real_skeleton_training import build_observed_batch_by_lengths, landmark_velocity_features
from embodied_rps.real_skeleton_video_eval import WAIT_COUNTER_PAPER_STATE, robot_action_for_decision_state
from embodied_rps.training_types import ModelRunConfig

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def canonicalize_mediapipe_landmarks(normalized_landmarks: NDArray[np.float32]) -> NDArray[np.float32]:
    """Canonicalize one MediaPipe 21x3 hand landmark frame."""

    points = np.asarray(normalized_landmarks, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError("normalized_landmarks must have shape (21,3)")
    wrist = points[0]
    middle_mcp = points[9]
    index_mcp = points[5]
    pinky_mcp = points[17]
    scale = float(np.linalg.norm(middle_mcp[:2] - wrist[:2]))
    if scale < 1e-6:
        scale = float(np.linalg.norm(pinky_mcp[:2] - index_mcp[:2]))
    if scale < 1e-6:
        scale = 1.0

    x_axis_2d = pinky_mcp[:2] - index_mcp[:2]
    x_norm = float(np.linalg.norm(x_axis_2d))
    if x_norm < 1e-6:
        x_axis_2d = np.array([1.0, 0.0], dtype=np.float32)
    else:
        x_axis_2d = x_axis_2d / x_norm
    y_axis_2d = middle_mcp[:2] - wrist[:2]
    y_axis_2d = y_axis_2d - float(np.dot(y_axis_2d, x_axis_2d)) * x_axis_2d
    y_norm = float(np.linalg.norm(y_axis_2d))
    if y_norm < 1e-6:
        y_axis_2d = np.array([-x_axis_2d[1], x_axis_2d[0]], dtype=np.float32)
    else:
        y_axis_2d = y_axis_2d / y_norm

    centered_xy = points[:, :2] - wrist[:2]
    canonical = np.zeros((21, 3), dtype=np.float32)
    canonical[:, 0] = centered_xy @ x_axis_2d / scale
    canonical[:, 1] = centered_xy @ y_axis_2d / scale
    canonical[:, 2] = (points[:, 2] - wrist[2]) / scale
    return canonical


def features_from_canonical_history(
    canonical_history: Sequence[NDArray[np.float32]],
    *,
    sequence_length: int,
) -> NDArray[np.float32]:
    """Create one padded model input from observed canonical landmark frames."""

    if len(canonical_history) == 0:
        raise ValueError("canonical_history must not be empty")
    observed_count = min(len(canonical_history), sequence_length)
    landmarks = np.zeros((1, sequence_length, 21, 3), dtype=np.float32)
    for frame_index in range(observed_count):
        landmarks[0, frame_index] = canonical_history[-observed_count + frame_index]
    if observed_count < sequence_length:
        landmarks[0, observed_count:] = landmarks[0, observed_count - 1]
    mask = np.ones((1, sequence_length), dtype=np.bool_)
    lengths = np.asarray([observed_count], dtype=np.int64)
    features = landmark_velocity_features(landmarks, mask=mask, lengths=lengths)
    return build_observed_batch_by_lengths(features, lengths, ratio=1.0)


def main(argv: Sequence[str] | None = None) -> int:
    """Run realtime or dry-run video inference."""

    parser = argparse.ArgumentParser(description="Run real skeleton final-gesture prediction.")
    parser.add_argument("--profile", required=True, type=Path, action="append", help="Path to exported profile JSON. Repeat for a weighted ensemble.")
    parser.add_argument("--profile-weights", default=None, help="Comma-separated ensemble weights aligned with repeated --profile values.")
    parser.add_argument("--scissors-rescue-profile-index", type=int, default=None, help="Optional zero-based profile index whose high-margin scissors output overrides the blended probabilities.")
    parser.add_argument("--scissors-rescue-confidence-threshold", type=float, default=0.90)
    parser.add_argument("--scissors-rescue-margin-threshold", type=float, default=0.98)
    parser.add_argument("--scissors-rescue-min-blended-transition-mass", type=float, default=0.0)
    parser.add_argument("--scissors-rescue-max-blended-rock-probability", type=float, default=None)
    parser.add_argument("--conditional-scissors-rescue-profile-index", type=int, default=None)
    parser.add_argument("--conditional-scissors-rescue-confidence-threshold", type=float, default=0.99)
    parser.add_argument("--conditional-scissors-rescue-margin-threshold", type=float, default=0.98)
    parser.add_argument("--conditional-scissors-rescue-min-blended-transition-mass", type=float, default=0.80)
    parser.add_argument("--conditional-scissors-rescue-max-blended-rock-probability", type=float, default=None)
    parser.add_argument("--paper-rescue-min-history-frames", type=int, default=0)
    parser.add_argument("--paper-rescue-min-observed-progress", type=float, default=0.35)
    parser.add_argument("--paper-rescue-min-scissors-confidence", type=float, default=0.70)
    parser.add_argument("--paper-rescue-min-scissors-margin", type=float, default=0.40)
    parser.add_argument("--paper-rescue-min-ring-pinky-extension-delta", type=float, default=0.08)
    parser.add_argument("--paper-rescue-min-latest-ring-pinky-extension", type=float, default=0.60)
    parser.add_argument("--paper-rescue-max-index-middle-minus-ring-pinky", type=float, default=0.25)
    parser.add_argument("--paper-rescue-max-rock-probability", type=float, default=0.40)
    parser.add_argument("--paper-rescue-min-transition-mass", type=float, default=0.60)
    parser.add_argument("--late-geometry-paper-min-history-frames", type=int, default=0)
    parser.add_argument("--late-geometry-paper-min-observed-progress", type=float, default=0.35)
    parser.add_argument("--late-geometry-paper-max-observed-progress", type=float, default=0.50)
    parser.add_argument("--late-geometry-paper-min-ring-pinky-extension-delta", type=float, default=0.04)
    parser.add_argument("--late-geometry-paper-min-latest-ring-pinky-extension", type=float, default=0.60)
    parser.add_argument("--late-geometry-paper-max-index-middle-minus-ring-pinky", type=float, default=0.25)
    parser.add_argument("--late-geometry-paper-max-index-middle-delta-minus-ring-pinky-delta", type=float, default=0.12)
    parser.add_argument("--rock-hold-guard-min-history-frames", type=int, default=0)
    parser.add_argument("--rock-hold-guard-max-latest-finger-extension", type=float, default=0.55)
    parser.add_argument("--rock-hold-guard-max-extension-delta", type=float, default=0.08)
    parser.add_argument("--gesture-verifier-min-history-frames", type=int, default=0)
    parser.add_argument("--gesture-verifier-rock-max-ring-pinky-extension", type=float, default=0.75)
    parser.add_argument("--gesture-verifier-rock-max-index-middle-extension", type=float, default=1.08)
    parser.add_argument("--gesture-verifier-rock-max-index-middle-minus-ring-pinky", type=float, default=0.38)
    parser.add_argument("--gesture-verifier-rock-max-extension-delta", type=float, default=0.20)
    parser.add_argument("--gesture-verifier-scissors-min-index-middle-extension", type=float, default=0.85)
    parser.add_argument("--gesture-verifier-scissors-min-index-middle-delta", type=float, default=0.04)
    parser.add_argument("--gesture-verifier-scissors-min-index-middle-minus-ring-pinky", type=float, default=0.25)
    parser.add_argument("--gesture-verifier-scissors-max-ring-pinky-extension", type=float, default=0.62)
    parser.add_argument("--gesture-verifier-paper-min-ring-pinky-extension", type=float, default=0.60)
    parser.add_argument("--gesture-verifier-paper-min-ring-pinky-delta", type=float, default=0.04)
    parser.add_argument("--gesture-verifier-paper-max-index-middle-minus-ring-pinky", type=float, default=0.25)
    parser.add_argument("--video", type=Path, default=None, help="Input MP4 path for dry-run video mode.")
    parser.add_argument("--camera", type=int, default=None, help="Camera index for live webcam mode.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output overlay MP4 path.")
    parser.add_argument("--frame-log-jsonl", type=Path, default=None, help="Optional per-frame inference log JSONL path.")
    parser.add_argument("--skeleton-npz", type=Path, default=None, help="Optional per-frame canonical skeleton sidecar NPZ path.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--margin-threshold", type=float, default=0.20)
    parser.add_argument("--transition-mass-threshold", type=float, default=0.15)
    parser.add_argument(
        "--binary-transition-mass-threshold",
        type=float,
        default=0.0,
        help="Minimum P(paper)+P(scissors) required before a binary paper/scissors decision; lower mass waits with robot paper.",
    )
    parser.add_argument(
        "--min-binary-decision-progress",
        type=float,
        default=0.0,
        help="Observed progress before which high-confidence paper/scissors spikes remain provisional wait.",
    )
    parser.add_argument("--confirmation-count", type=int, default=3)
    parser.add_argument("--prompt-cycle", action="store_true", help="Overlay a timed rock/paper/scissors prompt cycle for demo runs.")
    parser.add_argument("--prompt-cycle-s", type=float, default=1.0, help="Seconds per prompt when --prompt-cycle is enabled.")
    parser.add_argument("--prompt-sequence", default="rock,paper,scissors", help="Comma-separated prompt sequence for --prompt-cycle.")
    parser.add_argument(
        "--response-prompt",
        default=None,
        help="When prompt cycle is enabled, binary robot actions are provisional wait until this prompt is active.",
    )
    parser.add_argument(
        "--hold-response-prompt-until-decision",
        action="store_true",
        help="Keep the response prompt active after it appears so late paper/scissors evidence is not suppressed by the next prompt.",
    )
    parser.add_argument(
        "--response-hold-max-frames",
        type=int,
        default=0,
        help="Maximum frames to keep the response prompt latched; 0 disables the timeout.",
    )
    parser.add_argument(
        "--expected-actual-gesture",
        default=None,
        help="Operator ground-truth gesture for validation metadata only; not used for model inference.",
    )
    parser.add_argument(
        "--stop-after-confirmed-response-decision",
        action="store_true",
        help="Stop capture after a confirmed decision during the configured response prompt.",
    )
    parser.add_argument(
        "--post-decision-hold-frames",
        type=int,
        default=0,
        help="Frames to keep recording after the first confirmed response-window decision.",
    )
    parser.add_argument(
        "--reset-on-prompt-cycle",
        action="store_true",
        help="Clear skeleton history, rolling confirmation, and decision text when the prompt cycle starts a new trial.",
    )
    parser.add_argument(
        "--reset-on-prompt-change",
        action="store_true",
        help="Clear skeleton history, rolling confirmation, and decision text whenever the active prompt changes.",
    )
    parser.add_argument(
        "--display-window",
        dest="display_window",
        action="store_true",
        default=None,
        help="Show the live OpenCV preview window. Defaults to on for camera mode and off for video mode.",
    )
    parser.add_argument(
        "--no-display-window",
        dest="display_window",
        action="store_false",
        help="Disable the live OpenCV preview window, useful for headless capture.",
    )
    args = parser.parse_args(argv)

    if args.video is None and args.camera is None:
        raise ValueError("Provide either --video or --camera")
    if args.video is not None and args.camera is not None:
        raise ValueError("Use only one of --video or --camera")

    cv2, mp = _load_realtime_dependencies()
    profile_paths = list(args.profile)
    profiles = [_load_profile(path) for path in profile_paths]
    profile_weights = _parse_profile_weights(args.profile_weights, profile_count=len(profiles))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    models = [_load_model(profile, profile_path, device) for profile, profile_path in zip(profiles, profile_paths, strict=True)]
    sequence_length = _shared_int_value(profiles, "sequence_length")
    label_names = _shared_label_names(profiles)
    prompt_sequence = _parse_prompt_sequence(str(args.prompt_sequence))
    response_prompt = _parse_optional_prompt(args.response_prompt)
    expected_actual_gesture = _parse_optional_prompt(args.expected_actual_gesture)
    show_display_window = _should_display_window(camera=args.camera, display_window=args.display_window)

    source: int | str = int(args.camera) if args.camera is not None else str(args.video)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input source: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer: Any = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
    frame_log_handle: Any = None
    if args.frame_log_jsonl is not None:
        args.frame_log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        frame_log_handle = args.frame_log_jsonl.open("w", encoding="utf-8")
    skeleton_canonical_landmarks: list[NDArray[np.float32]] = []
    skeleton_detected: list[bool] = []
    skeleton_frame_indices: list[int] = []
    skeleton_times_s: list[float] = []
    skeleton_active_prompts: list[str | None] = []

    history: deque[NDArray[np.float32]] = deque(maxlen=sequence_length)
    frame_count = 0
    rolling_state: str | None = None
    rolling_count = 0
    decision_text = ""
    confirmed_response_decision_frame: int | None = None
    stopped_after_confirmed_response_decision = False
    previous_prompt_index: int | None = None
    previous_prompt_cycle_index: int | None = None
    response_latch_active = False
    response_latch_start_frame: int | None = None
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            elapsed_s = max(0.0, float(frame_count - 1) / max(fps, 1.0))
            prompt_text: str | None = None
            prompt_index: int | None = None
            prompt_cycle_index: int | None = None
            raw_prompt_text: str | None = None
            if bool(args.prompt_cycle):
                raw_prompt_text, prompt_index, prompt_cycle_index = prompt_cycle_state_for_time_s(
                    elapsed_s,
                    prompt_sequence=prompt_sequence,
                    prompt_cycle_s=float(args.prompt_cycle_s),
                )
                prompt_text, response_latch_active, response_latch_start_frame = _update_response_prompt_latch(
                    raw_prompt=raw_prompt_text,
                    response_prompt=response_prompt,
                    latch_active=response_latch_active,
                    latch_start_frame=response_latch_start_frame,
                    frame_index=frame_count,
                    hold_response_prompt_until_decision=bool(args.hold_response_prompt_until_decision),
                    response_hold_max_frames=int(args.response_hold_max_frames),
                )
                effective_prompt_index = prompt_sequence.index(prompt_text) if prompt_text is not None else prompt_index
                effective_prompt_cycle_index = prompt_cycle_index
                if (
                    response_latch_active
                    and response_prompt is not None
                    and prompt_text == response_prompt
                    and previous_prompt_index == effective_prompt_index
                    and previous_prompt_cycle_index is not None
                ):
                    effective_prompt_cycle_index = previous_prompt_cycle_index
                if (
                    previous_prompt_index is not None
                    and previous_prompt_cycle_index is not None
                    and _should_reset_prompt_state(
                        previous_prompt_index=previous_prompt_index,
                        previous_prompt_cycle_index=previous_prompt_cycle_index,
                        prompt_index=int(effective_prompt_index),
                        prompt_cycle_index=int(effective_prompt_cycle_index),
                        reset_on_prompt_cycle=bool(args.reset_on_prompt_cycle),
                        reset_on_prompt_change=bool(args.reset_on_prompt_change),
                    )
                ):
                    history.clear()
                    rolling_state = None
                    rolling_count = 0
                    decision_text = ""
                previous_prompt_index = int(effective_prompt_index)
                previous_prompt_cycle_index = int(effective_prompt_cycle_index)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            detected = bool(result.multi_hand_landmarks)
            prediction_text = "no hand"
            confidence = 0.0
            margin = 0.0
            probabilities_by_label = {"rock": 0.0, "paper": 0.0, "scissors": 0.0}
            raw_probabilities_by_label = {"rock": 0.0, "paper": 0.0, "scissors": 0.0}
            rock_hold_guard_diagnostics: dict[str, object] = {
                "enabled": bool(int(args.rock_hold_guard_min_history_frames) > 0),
                "applied": False,
            }
            gesture_verifier_diagnostics: dict[str, object] = {
                "enabled": bool(int(args.gesture_verifier_min_history_frames) > 0),
                "verified_gesture": None,
                "override_reason": None,
            }
            decision_state: str | None = None
            robot_action: str | None = None
            canonical_for_sidecar: NDArray[np.float32] | None = None
            if detected:
                hand = result.multi_hand_landmarks[0]
                normalized = np.asarray([(lm.x, lm.y, lm.z) for lm in hand.landmark], dtype=np.float32)
                canonical = canonicalize_mediapipe_landmarks(normalized)
                canonical_for_sidecar = canonical
                history.append(canonical)
                features = features_from_canonical_history(tuple(history), sequence_length=sequence_length)
                with torch.no_grad():
                    feature_tensor = torch.from_numpy(features).to(device)
                    probability_arrays = [
                        torch.softmax(model(feature_tensor), dim=1).detach().cpu().numpy()[0]
                        for model in models
                    ]
                    probabilities = _blend_probabilities_with_optional_scissors_rescue(
                        probability_arrays,
                        weights=profile_weights,
                        label_names=label_names,
                        scissors_rescue_profile_index=args.scissors_rescue_profile_index,
                        scissors_rescue_confidence_threshold=float(args.scissors_rescue_confidence_threshold),
                        scissors_rescue_margin_threshold=float(args.scissors_rescue_margin_threshold),
                        scissors_rescue_min_blended_transition_mass=float(args.scissors_rescue_min_blended_transition_mass),
                        scissors_rescue_max_blended_rock_probability=(
                            None
                            if args.scissors_rescue_max_blended_rock_probability is None
                            else float(args.scissors_rescue_max_blended_rock_probability)
                        ),
                        conditional_scissors_rescue_profile_index=args.conditional_scissors_rescue_profile_index,
                        conditional_scissors_rescue_confidence_threshold=float(args.conditional_scissors_rescue_confidence_threshold),
                        conditional_scissors_rescue_margin_threshold=float(args.conditional_scissors_rescue_margin_threshold),
                        conditional_scissors_rescue_min_blended_transition_mass=float(args.conditional_scissors_rescue_min_blended_transition_mass),
                        conditional_scissors_rescue_max_blended_rock_probability=(
                            None
                            if args.conditional_scissors_rescue_max_blended_rock_probability is None
                            else float(args.conditional_scissors_rescue_max_blended_rock_probability)
                        ),
                    )
                    for label, probability in zip(label_names, probabilities.tolist(), strict=True):
                        raw_probabilities_by_label[str(label)] = float(probability)
                    observed_progress = min(1.0, float(len(history)) / float(sequence_length))
                    probabilities = _apply_skeleton_paper_rescue(
                        probabilities,
                        labels=label_names,
                        canonical_history=tuple(history),
                        observed_progress=observed_progress,
                        min_history_frames=int(args.paper_rescue_min_history_frames),
                        min_observed_progress=float(args.paper_rescue_min_observed_progress),
                        min_scissors_confidence=float(args.paper_rescue_min_scissors_confidence),
                        min_scissors_margin=float(args.paper_rescue_min_scissors_margin),
                        min_ring_pinky_extension_delta=float(args.paper_rescue_min_ring_pinky_extension_delta),
                        min_latest_ring_pinky_extension=float(args.paper_rescue_min_latest_ring_pinky_extension),
                        max_index_middle_minus_ring_pinky=float(args.paper_rescue_max_index_middle_minus_ring_pinky),
                        max_rock_probability=(
                            None
                            if args.paper_rescue_max_rock_probability is None
                            else float(args.paper_rescue_max_rock_probability)
                        ),
                        min_transition_mass=float(args.paper_rescue_min_transition_mass),
                    )
                    probabilities = _apply_late_geometry_paper_detector(
                        probabilities,
                        labels=label_names,
                        canonical_history=tuple(history),
                        observed_progress=float(min(1.0, frame_count / max(1, sequence_length))),
                        min_history_frames=int(args.late_geometry_paper_min_history_frames),
                        min_observed_progress=float(args.late_geometry_paper_min_observed_progress),
                        max_observed_progress=float(args.late_geometry_paper_max_observed_progress),
                        min_ring_pinky_extension_delta=float(args.late_geometry_paper_min_ring_pinky_extension_delta),
                        min_latest_ring_pinky_extension=float(args.late_geometry_paper_min_latest_ring_pinky_extension),
                        max_index_middle_minus_ring_pinky=float(args.late_geometry_paper_max_index_middle_minus_ring_pinky),
                        max_index_middle_delta_minus_ring_pinky_delta=float(
                            args.late_geometry_paper_max_index_middle_delta_minus_ring_pinky_delta
                        ),
                    )
                    rock_hold_guard_diagnostics = _skeleton_rock_hold_guard_diagnostics(
                        probabilities,
                        labels=label_names,
                        canonical_history=tuple(history),
                        min_history_frames=int(args.rock_hold_guard_min_history_frames),
                        max_latest_finger_extension=float(args.rock_hold_guard_max_latest_finger_extension),
                        max_extension_delta=float(args.rock_hold_guard_max_extension_delta),
                    )
                    probabilities = _apply_skeleton_rock_hold_guard(
                        probabilities,
                        labels=label_names,
                        canonical_history=tuple(history),
                        min_history_frames=int(args.rock_hold_guard_min_history_frames),
                        max_latest_finger_extension=float(args.rock_hold_guard_max_latest_finger_extension),
                        max_extension_delta=float(args.rock_hold_guard_max_extension_delta),
                    )
                predicted_index = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_index])
                prediction_text = label_names[predicted_index]
                order = np.argsort(probabilities)[::-1]
                margin = confidence - float(probabilities[int(order[1])]) if len(order) > 1 else confidence
                for label, probability in zip(label_names, probabilities.tolist(), strict=True):
                    probabilities_by_label[str(label)] = float(probability)
                transition_mass = probabilities_by_label["paper"] + probabilities_by_label["scissors"]
                decision_state = _decision_state_for_probabilities(
                    prediction=prediction_text,
                    confidence=confidence,
                    margin=margin,
                    rock_probability=probabilities_by_label["rock"],
                    transition_mass=transition_mass,
                    confidence_threshold=float(args.confidence_threshold),
                    margin_threshold=float(args.margin_threshold),
                    transition_mass_threshold=float(args.transition_mass_threshold),
                    binary_transition_mass_threshold=float(args.binary_transition_mass_threshold),
                    observed_progress=observed_progress,
                    min_binary_decision_progress=float(args.min_binary_decision_progress),
                )
                decision_state = _apply_prompt_response_gate(
                    decision_state=decision_state,
                    active_prompt=prompt_text,
                    response_prompt=response_prompt,
                )
                decision_state, gesture_verifier_diagnostics = _apply_gesture_verifier_decision(
                    decision_state=decision_state,
                    probabilities=probabilities,
                    labels=label_names,
                    canonical_history=tuple(history),
                    min_history_frames=int(args.gesture_verifier_min_history_frames),
                    rock_max_ring_pinky_extension=float(args.gesture_verifier_rock_max_ring_pinky_extension),
                    rock_max_index_middle_extension=float(args.gesture_verifier_rock_max_index_middle_extension),
                    rock_max_index_middle_minus_ring_pinky=float(
                        args.gesture_verifier_rock_max_index_middle_minus_ring_pinky
                    ),
                    rock_max_extension_delta=float(args.gesture_verifier_rock_max_extension_delta),
                    scissors_min_index_middle_extension=float(args.gesture_verifier_scissors_min_index_middle_extension),
                    scissors_min_index_middle_delta=float(args.gesture_verifier_scissors_min_index_middle_delta),
                    scissors_min_index_middle_minus_ring_pinky=float(
                        args.gesture_verifier_scissors_min_index_middle_minus_ring_pinky
                    ),
                    scissors_max_ring_pinky_extension=float(args.gesture_verifier_scissors_max_ring_pinky_extension),
                    paper_min_ring_pinky_extension=float(args.gesture_verifier_paper_min_ring_pinky_extension),
                    paper_min_ring_pinky_delta=float(args.gesture_verifier_paper_min_ring_pinky_delta),
                    paper_max_index_middle_minus_ring_pinky=float(
                        args.gesture_verifier_paper_max_index_middle_minus_ring_pinky
                    ),
                )
                robot_action = robot_action_for_decision_state(decision_state)
                rolling_state, rolling_count, decision_text = _update_rolling_decision_text(
                    decision_state=decision_state,
                    robot_action=robot_action,
                    rolling_state=rolling_state,
                    rolling_count=rolling_count,
                    decision_text=decision_text,
                    confirmation_count=int(args.confirmation_count),
                )
                _draw_landmarks(cv2, frame, normalized)
            else:
                decision_state = _decision_state_for_no_hand_prompt(
                    active_prompt=prompt_text,
                    response_prompt=response_prompt,
                )
                robot_action = robot_action_for_decision_state(decision_state)
                if decision_state is None:
                    rolling_state = None
                    rolling_count = 0
                    decision_text = ""
                else:
                    rolling_state, rolling_count, decision_text = _update_rolling_decision_text(
                        decision_state=decision_state,
                        robot_action=robot_action,
                        rolling_state=rolling_state,
                        rolling_count=rolling_count,
                        decision_text=decision_text,
                        confirmation_count=int(args.confirmation_count),
                    )

            progress = min(1.0, float(len(history)) / float(sequence_length))
            transition_mass = probabilities_by_label["paper"] + probabilities_by_label["scissors"]
            status = (
                f"model {prediction_text} {confidence:.2f} R/P/S "
                f"{probabilities_by_label['rock']:.2f}/{probabilities_by_label['paper']:.2f}/{probabilities_by_label['scissors']:.2f} "
                f"tm {transition_mass:.2f} progress {progress:.2f}"
            )
            policy = f"state {decision_state or 'none'} robot {robot_action or 'none'} margin {margin:.2f}"
            verified_gesture = gesture_verifier_diagnostics.get("verified_gesture")
            if verified_gesture is not None:
                policy += f" verify {verified_gesture}"
            if bool(args.prompt_cycle):
                prompt = prompt_text or prompt_for_time_s(elapsed_s, prompt_sequence=prompt_sequence, prompt_cycle_s=float(args.prompt_cycle_s))
                prompt_progress_fraction = None
                if response_latch_active and response_latch_start_frame is not None and prompt == response_prompt:
                    latch_elapsed_s = max(0.0, float(frame_count - int(response_latch_start_frame) + 1) / max(fps, 1.0))
                    prompt_progress_fraction = min(1.0, latch_elapsed_s / max(float(args.prompt_cycle_s), 1e-6))
                _draw_prompt_banner(
                    cv2,
                    frame,
                    prompt=prompt,
                    elapsed_s=elapsed_s,
                    prompt_cycle_s=float(args.prompt_cycle_s),
                    response_prompt=response_prompt,
                    progress_fraction=prompt_progress_fraction,
                )
            status_y, policy_y, decision_y = _overlay_text_y_positions(
                frame_height=int(frame.shape[0]),
                prompt_cycle=bool(args.prompt_cycle),
            )
            text_style = _overlay_text_style(frame_width=int(frame.shape[1]))
            thickness = int(text_style["thickness"])
            cv2.putText(
                frame,
                status,
                (20, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                float(text_style["status_scale"]),
                (20, 240, 20),
                thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                policy,
                (20, policy_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                float(text_style["policy_scale"]),
                (20, 220, 255),
                thickness,
                cv2.LINE_AA,
            )
            if decision_text:
                cv2.putText(
                    frame,
                    decision_text,
                    (20, decision_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    float(text_style["decision_scale"]),
                    (40, 220, 255),
                    thickness,
                    cv2.LINE_AA,
                )
            if frame_log_handle is not None:
                frame_log_handle.write(
                    json.dumps(
                        _build_frame_log_record(
                            frame_index=frame_count,
                            time_s=elapsed_s,
                            active_prompt=prompt_text,
                            raw_prompt=raw_prompt_text,
                            prompt_index=prompt_index,
                            prompt_cycle_index=prompt_cycle_index,
                            response_prompt=response_prompt,
                            response_window_latched=response_latch_active,
                            expected_actual_gesture=expected_actual_gesture,
                            detected=detected,
                            prediction=prediction_text,
                            confidence=confidence,
                            margin=margin,
                            probabilities_by_label=probabilities_by_label,
                            transition_mass=transition_mass,
                            decision_state=decision_state,
                            robot_action=robot_action,
                            rolling_state=rolling_state,
                            rolling_count=rolling_count,
                            decision_text=decision_text,
                            history_frame_count=len(history),
                            raw_probabilities_by_label=raw_probabilities_by_label,
                            rock_hold_guard_diagnostics=rock_hold_guard_diagnostics,
                            gesture_verifier_diagnostics=gesture_verifier_diagnostics,
                        ),
                        ensure_ascii=True,
                    )
                    + "\n"
                )
            if args.skeleton_npz is not None:
                skeleton_frame_indices.append(frame_count)
                skeleton_times_s.append(elapsed_s)
                skeleton_active_prompts.append(prompt_text)
                skeleton_detected.append(bool(canonical_for_sidecar is not None))
                if canonical_for_sidecar is None:
                    skeleton_canonical_landmarks.append(np.zeros((21, 3), dtype=np.float32))
                else:
                    skeleton_canonical_landmarks.append(canonical_for_sidecar.astype(np.float32, copy=True))
            if writer is not None:
                writer.write(frame)
            if show_display_window:
                cv2.imshow("real skeleton final gesture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            confirmed_response_decision_frame, should_stop_after_decision = _update_confirmed_response_stop_state(
                enabled=bool(args.stop_after_confirmed_response_decision),
                active_prompt=prompt_text,
                response_prompt=response_prompt,
                expected_actual_gesture=expected_actual_gesture,
                decision_state=decision_state,
                decision_text=decision_text,
                frame_index=frame_count,
                confirmed_response_decision_frame=confirmed_response_decision_frame,
                post_decision_hold_frames=int(args.post_decision_hold_frames),
            )
            if should_stop_after_decision:
                stopped_after_confirmed_response_decision = True
                break
            if args.max_frames is not None and frame_count >= args.max_frames:
                break
    finally:
        hands.close()
        capture.release()
        if writer is not None:
            writer.release()
        if frame_log_handle is not None:
            frame_log_handle.close()
        if show_display_window:
            cv2.destroyAllWindows()

    skeleton_sidecar_summary: dict[str, object] | None = None
    if args.skeleton_npz is not None:
        skeleton_sidecar_summary = _write_skeleton_sidecar_npz(
            args.skeleton_npz,
            canonical_landmarks=skeleton_canonical_landmarks,
            detected=skeleton_detected,
            frame_indices=skeleton_frame_indices,
            times_s=skeleton_times_s,
            active_prompts=skeleton_active_prompts,
            expected_actual_gesture=expected_actual_gesture,
            source=f"camera:{args.camera}" if args.camera is not None else str(args.video),
        )

    print(
        json.dumps(
            {
                "frames": frame_count,
                "output": str(args.output) if args.output is not None else None,
                "frame_log_jsonl": str(args.frame_log_jsonl) if args.frame_log_jsonl is not None else None,
                "skeleton_npz": str(args.skeleton_npz) if args.skeleton_npz is not None else None,
                "skeleton_sidecar": skeleton_sidecar_summary,
                "confirmed_response_decision_frame": confirmed_response_decision_frame,
                "stopped_after_confirmed_response_decision": stopped_after_confirmed_response_decision,
            },
            indent=2,
        )
    )
    return 0


def _write_skeleton_sidecar_npz(
    path: Path,
    *,
    canonical_landmarks: Sequence[NDArray[np.float32]],
    detected: Sequence[bool],
    frame_indices: Sequence[int],
    times_s: Sequence[float],
    active_prompts: Sequence[str | None],
    expected_actual_gesture: str | None,
    source: str,
) -> dict[str, object]:
    """Write per-frame canonical skeletons for later crop/seed extraction."""

    frame_count = len(frame_indices)
    if not (
        len(canonical_landmarks)
        == len(detected)
        == len(times_s)
        == len(active_prompts)
        == frame_count
    ):
        raise ValueError("skeleton sidecar inputs must have matching frame counts")
    if frame_count == 0:
        landmarks = np.zeros((0, 21, 3), dtype=np.float32)
    else:
        landmarks = np.stack([np.asarray(frame, dtype=np.float32) for frame in canonical_landmarks], axis=0)
    if landmarks.shape[1:] != (21, 3):
        raise ValueError("canonical_landmarks must contain frames with shape (21,3)")
    detected_array = np.asarray(detected, dtype=np.bool_)
    metadata = {
        "expected_actual_gesture": expected_actual_gesture,
        "source": source,
        "contract": "per-frame canonical MediaPipe skeleton sidecar for scissors pose collection",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        canonical_landmarks=landmarks.astype(np.float32, copy=False),
        detected=detected_array,
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        times_s=np.asarray(times_s, dtype=np.float32),
        active_prompts=np.asarray(["" if prompt is None else str(prompt) for prompt in active_prompts]),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=True)),
    )
    detected_count = int(np.count_nonzero(detected_array))
    return {
        "skeleton_npz": path.as_posix(),
        "frame_count": frame_count,
        "detected_frame_count": detected_count,
        "detection_rate": float(detected_count / frame_count) if frame_count else 0.0,
    }


def prompt_for_time_s(
    elapsed_s: float,
    *,
    prompt_sequence: Sequence[str],
    prompt_cycle_s: float,
) -> str:
    """Return the active demo prompt for a timestamp."""

    return prompt_cycle_state_for_time_s(
        elapsed_s,
        prompt_sequence=prompt_sequence,
        prompt_cycle_s=prompt_cycle_s,
    )[0]


def _should_display_window(*, camera: int | None, display_window: bool | None) -> bool:
    """Return whether the realtime preview window should be shown."""

    if display_window is not None:
        return bool(display_window)
    return camera is not None


def _build_frame_log_record(
    *,
    frame_index: int,
    time_s: float,
    active_prompt: str | None,
    prompt_index: int | None,
    prompt_cycle_index: int | None,
    response_prompt: str | None,
    expected_actual_gesture: str | None,
    detected: bool,
    prediction: str,
    confidence: float,
    margin: float,
    probabilities_by_label: Mapping[str, float],
    transition_mass: float,
    decision_state: str | None,
    robot_action: str | None,
    rolling_state: str | None,
    rolling_count: int,
    decision_text: str,
    history_frame_count: int,
    raw_prompt: str | None = None,
    response_window_latched: bool = False,
    raw_probabilities_by_label: Mapping[str, float] | None = None,
    rock_hold_guard_diagnostics: Mapping[str, object] | None = None,
    gesture_verifier_diagnostics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one compact per-frame record for demo rehearsal analysis."""

    response_window = bool(response_prompt is not None and active_prompt == response_prompt)
    expected = _parse_optional_prompt(expected_actual_gesture)
    expected_decisions = _expected_decisions_for_actual_gesture(expected)
    expected_robot_action = _expected_robot_action_for_actual_gesture(expected)
    ground_truth_match = (
        decision_state in expected_decisions
        if response_window and expected_decisions is not None and decision_state is not None
        else None
    )
    robot_action_match = (
        robot_action == expected_robot_action
        if response_window and expected_robot_action is not None and robot_action is not None
        else None
    )
    return {
        "frame_index": int(frame_index),
        "time_s": float(time_s),
        "active_prompt": active_prompt,
        "raw_prompt": raw_prompt,
        "prompt_index": prompt_index,
        "prompt_cycle_index": prompt_cycle_index,
        "response_prompt": response_prompt,
        "response_window": response_window,
        "response_window_latched": bool(response_window_latched),
        "expected_actual_gesture": expected,
        "detected": bool(detected),
        "prediction": str(prediction),
        "confidence": float(confidence),
        "margin": float(margin),
        "p_rock": float(probabilities_by_label.get("rock", 0.0)),
        "p_paper": float(probabilities_by_label.get("paper", 0.0)),
        "p_scissors": float(probabilities_by_label.get("scissors", 0.0)),
        "raw_p_rock": float((raw_probabilities_by_label or probabilities_by_label).get("rock", 0.0)),
        "raw_p_paper": float((raw_probabilities_by_label or probabilities_by_label).get("paper", 0.0)),
        "raw_p_scissors": float((raw_probabilities_by_label or probabilities_by_label).get("scissors", 0.0)),
        "transition_mass": float(transition_mass),
        "decision_state": decision_state,
        "robot_action": robot_action,
        "rolling_state": rolling_state,
        "rolling_count": int(rolling_count),
        "decision_text": str(decision_text),
        "confirmed_decision": bool(decision_text),
        "ground_truth_match": ground_truth_match,
        "robot_action_match": robot_action_match,
        "history_frame_count": int(history_frame_count),
        "rock_hold_guard": dict(rock_hold_guard_diagnostics or {}),
        "gesture_verifier": dict(gesture_verifier_diagnostics or {}),
    }


def _draw_prompt_banner(
    cv2: Any,
    frame: Any,
    *,
    prompt: str,
    elapsed_s: float,
    prompt_cycle_s: float,
    response_prompt: str | None = None,
    progress_fraction: float | None = None,
) -> None:
    """Draw a large prompt cue and within-prompt progress bar."""

    height, width = frame.shape[:2]
    banner_h = _prompt_banner_height(int(height))
    prompt_key = str(prompt).lower()
    colors = {
        "rock": ((38, 61, 94), (111, 164, 255)),
        "paper": ((35, 89, 62), (98, 220, 146)),
        "scissors": ((96, 58, 38), (255, 181, 92)),
    }
    background, accent = colors.get(prompt_key, ((45, 50, 60), (230, 230, 230)))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_h), background, -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0.0, frame)

    prompt_text = f"PROMPT {prompt_key.upper()}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1.0, min(2.2, width / 520.0))
    thickness = max(2, int(round(font_scale * 2.0)))
    (text_w, text_h), baseline = cv2.getTextSize(prompt_text, font, font_scale, thickness)
    text_x = max(16, (width - text_w) // 2)
    text_y = max(text_h + 12, banner_h // 2 + text_h // 2 - 4)
    cv2.putText(frame, prompt_text, (text_x + 2, text_y + 2), font, font_scale, (10, 15, 20), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, prompt_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    if response_prompt is not None and prompt_key == str(response_prompt).lower():
        label = "RESPONSE WINDOW"
        label_scale = max(0.55, min(0.95, width / 760.0))
        label_thickness = max(1, int(round(label_scale * 2.0)))
        (label_w, label_h), _ = cv2.getTextSize(label, font, label_scale, label_thickness)
        label_x = max(14, (width - label_w) // 2)
        label_y = min(banner_h - 18, text_y + label_h + 12)
        cv2.putText(
            frame,
            label,
            (label_x + 1, label_y + 1),
            font,
            label_scale,
            (10, 15, 20),
            label_thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(frame, label, (label_x, label_y), font, label_scale, accent, label_thickness, cv2.LINE_AA)

    if progress_fraction is None:
        progress_fraction = _prompt_progress_fraction(elapsed_s=elapsed_s, prompt_cycle_s=prompt_cycle_s)
    progress_fraction = min(1.0, max(0.0, float(progress_fraction)))
    bar_margin = max(18, width // 28)
    bar_h = max(8, banner_h // 10)
    bar_top = banner_h - bar_h - max(8, banner_h // 12)
    bar_left = bar_margin
    bar_right = width - bar_margin
    cv2.rectangle(frame, (bar_left, bar_top), (bar_right, bar_top + bar_h), (20, 25, 32), -1)
    fill_right = bar_left + int(round((bar_right - bar_left) * progress_fraction))
    cv2.rectangle(frame, (bar_left, bar_top), (fill_right, bar_top + bar_h), accent, -1)
    cv2.rectangle(frame, (bar_left, bar_top), (bar_right, bar_top + bar_h), (235, 240, 245), 1)


def _prompt_banner_height(frame_height: int) -> int:
    """Return the prompt banner height used by the realtime overlay."""

    return min(max(80, int(frame_height) // 7), max(80, int(frame_height) // 3))


def _overlay_text_y_positions(*, frame_height: int, prompt_cycle: bool) -> tuple[int, int, int]:
    """Return status, policy, and decision text baselines without prompt overlap."""

    if not prompt_cycle:
        return 36, 70, 104
    banner_bottom = _prompt_banner_height(frame_height)
    status_y = banner_bottom + 36
    return status_y, status_y + 34, status_y + 68


def _overlay_text_style(*, frame_width: int) -> dict[str, float | int]:
    """Return resolution-aware text scales for the realtime overlay."""

    width = max(1, int(frame_width))
    if width <= 800:
        return {
            "status_scale": 0.48,
            "policy_scale": 0.46,
            "decision_scale": 0.50,
            "thickness": 1,
        }
    if width <= 1280:
        return {
            "status_scale": 0.62,
            "policy_scale": 0.58,
            "decision_scale": 0.60,
            "thickness": 2,
        }
    return {
        "status_scale": 0.80,
        "policy_scale": 0.70,
        "decision_scale": 0.70,
        "thickness": 2,
    }


def _prompt_progress_fraction(*, elapsed_s: float, prompt_cycle_s: float) -> float:
    if prompt_cycle_s <= 0.0:
        return 1.0
    progress = (max(0.0, float(elapsed_s)) % float(prompt_cycle_s)) / float(prompt_cycle_s)
    return min(1.0, max(0.0, progress))


def prompt_cycle_state_for_time_s(
    elapsed_s: float,
    *,
    prompt_sequence: Sequence[str],
    prompt_cycle_s: float,
) -> tuple[str, int, int]:
    """Return the active prompt, prompt index, and full-cycle index for a timestamp."""

    if prompt_cycle_s <= 0.0:
        raise ValueError("prompt_cycle_s must be positive")
    if len(prompt_sequence) == 0:
        raise ValueError("prompt_sequence must not be empty")
    absolute_prompt_index = int(max(0.0, elapsed_s) // prompt_cycle_s)
    prompt_index = absolute_prompt_index % len(prompt_sequence)
    cycle_index = absolute_prompt_index // len(prompt_sequence)
    return str(prompt_sequence[prompt_index]), prompt_index, cycle_index


def _should_reset_prompt_state(
    *,
    previous_prompt_index: int,
    previous_prompt_cycle_index: int,
    prompt_index: int,
    prompt_cycle_index: int,
    reset_on_prompt_cycle: bool,
    reset_on_prompt_change: bool,
) -> bool:
    """Return whether prompt state should reset between adjacent frames."""

    if reset_on_prompt_change and (
        prompt_index != previous_prompt_index
        or prompt_cycle_index != previous_prompt_cycle_index
    ):
        return True
    if reset_on_prompt_cycle and prompt_cycle_index > previous_prompt_cycle_index:
        return True
    return False


def _update_response_prompt_latch(
    *,
    raw_prompt: str | None,
    response_prompt: str | None,
    latch_active: bool,
    latch_start_frame: int | None,
    frame_index: int,
    hold_response_prompt_until_decision: bool,
    response_hold_max_frames: int,
) -> tuple[str | None, bool, int | None]:
    """Keep the response prompt active for late but valid transition evidence."""

    if not hold_response_prompt_until_decision or response_prompt is None:
        return raw_prompt, False, None
    if not latch_active and raw_prompt == response_prompt:
        latch_active = True
        latch_start_frame = int(frame_index)
    if not latch_active:
        return raw_prompt, False, None
    start = int(latch_start_frame if latch_start_frame is not None else frame_index)
    max_frames = int(response_hold_max_frames)
    if max_frames > 0 and int(frame_index) - start > max_frames:
        return raw_prompt, False, None
    return response_prompt, True, start


def _parse_prompt_sequence(value: str) -> tuple[str, ...]:
    prompts = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    allowed = {"rock", "paper", "scissors"}
    if len(prompts) == 0:
        raise ValueError("prompt_sequence must contain at least one prompt")
    unknown = sorted(set(prompts) - allowed)
    if unknown:
        raise ValueError(f"Unsupported prompt(s): {unknown}")
    return prompts


def _parse_optional_prompt(value: object) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip().lower()
    if parsed == "":
        return None
    if parsed not in {"rock", "paper", "scissors"}:
        raise ValueError(f"Unsupported response prompt: {parsed}")
    return parsed


def _expected_decisions_for_actual_gesture(value: str | None) -> set[str] | None:
    if value is None:
        return None
    if value == "rock":
        return {"rock", WAIT_COUNTER_PAPER_STATE}
    return {value}


def _expected_robot_action_for_actual_gesture(value: str | None) -> str | None:
    if value == "rock":
        return "paper"
    if value == "paper":
        return "scissors"
    if value == "scissors":
        return "rock"
    return None


def _apply_prompt_response_gate(
    *,
    decision_state: str | None,
    active_prompt: str | None,
    response_prompt: str | None,
) -> str | None:
    """Suppress binary robot actions until the configured response prompt is active."""

    if response_prompt is None or active_prompt is None:
        return decision_state
    if active_prompt == response_prompt:
        return decision_state
    return WAIT_COUNTER_PAPER_STATE


def _decision_state_for_no_hand_prompt(
    *,
    active_prompt: str | None,
    response_prompt: str | None,
) -> str | None:
    """Return the prompt-context wait state when no hand is detected off target."""

    return _apply_prompt_response_gate(
        decision_state=None,
        active_prompt=active_prompt,
        response_prompt=response_prompt,
    )


def _update_rolling_decision_text(
    *,
    decision_state: str | None,
    robot_action: str | None,
    rolling_state: str | None,
    rolling_count: int,
    decision_text: str,
    confirmation_count: int,
) -> tuple[str | None, int, str]:
    """Update rolling confirmation and clear stale decision text on state changes."""

    if decision_state is None:
        return None, 0, ""
    if decision_state == rolling_state:
        next_count = rolling_count + 1
        next_text = decision_text
    else:
        next_count = 1
        next_text = ""
    if next_count >= confirmation_count:
        next_text = _format_decision_text(decision_state=decision_state, robot_action=robot_action)
    return decision_state, next_count, next_text


def _format_decision_text(*, decision_state: str, robot_action: str | None) -> str:
    """Return the operator-facing decision overlay string."""

    if decision_state == WAIT_COUNTER_PAPER_STATE:
        return f"ROCK HOLD -> ROBOT {(robot_action or 'none').upper()}"
    return f"DECISION {decision_state} robot {robot_action or 'none'}"


def _update_confirmed_response_stop_state(
    *,
    enabled: bool,
    active_prompt: str | None,
    response_prompt: str | None,
    expected_actual_gesture: str | None,
    decision_state: str | None,
    decision_text: str,
    frame_index: int,
    confirmed_response_decision_frame: int | None,
    post_decision_hold_frames: int,
) -> tuple[int | None, bool]:
    """Track whether capture should stop after a confirmed response-window decision."""

    if not enabled:
        return confirmed_response_decision_frame, False
    first_confirmed_frame = confirmed_response_decision_frame
    if first_confirmed_frame is None:
        expected_decisions = _expected_decisions_for_actual_gesture(_parse_optional_prompt(expected_actual_gesture))
        if _parse_optional_prompt(expected_actual_gesture) == "rock" and decision_state in {"rock", WAIT_COUNTER_PAPER_STATE}:
            return None, False
        expected_match = expected_decisions is None or decision_state in expected_decisions
        if response_prompt is None or active_prompt != response_prompt or decision_text == "" or not expected_match:
            return None, False
        first_confirmed_frame = int(frame_index)
    hold_frames = max(0, int(post_decision_hold_frames))
    should_stop = int(frame_index) - int(first_confirmed_frame) >= hold_frames
    return first_confirmed_frame, bool(should_stop)


def _parse_profile_weights(value: str | None, *, profile_count: int) -> list[float]:
    """Parse and normalize ensemble weights for one or more profiles."""

    if profile_count <= 0:
        raise ValueError("profile_count must be positive")
    if value is None or value.strip() == "":
        return [1.0 / profile_count for _ in range(profile_count)]
    weights = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(weights) != profile_count:
        raise ValueError("profile weights must match the number of --profile values")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("profile weights must sum to a positive value")
    return [weight / total for weight in weights]


def _blend_probabilities(
    probability_arrays: Sequence[NDArray[np.float32]],
    *,
    weights: Sequence[float],
) -> NDArray[np.float32]:
    """Blend same-label probability arrays with normalized ensemble weights."""

    if len(probability_arrays) == 0:
        raise ValueError("probability_arrays must not be empty")
    if len(probability_arrays) != len(weights):
        raise ValueError("probability array count must match weight count")
    blended = np.zeros_like(np.asarray(probability_arrays[0], dtype=np.float32))
    for probabilities, weight in zip(probability_arrays, weights, strict=True):
        parsed = np.asarray(probabilities, dtype=np.float32)
        if parsed.shape != blended.shape:
            raise ValueError("all probability arrays must have the same shape")
        blended += parsed * np.float32(weight)
    total = float(np.sum(blended))
    if total <= 0.0:
        raise ValueError("blended probabilities must have positive mass")
    return (blended / np.float32(total)).astype(np.float32)


def _blend_probabilities_with_optional_scissors_rescue(
    probability_arrays: Sequence[NDArray[np.float32]],
    *,
    weights: Sequence[float],
    label_names: Sequence[str],
    scissors_rescue_profile_index: int | None,
    scissors_rescue_confidence_threshold: float,
    scissors_rescue_margin_threshold: float,
    scissors_rescue_min_blended_transition_mass: float = 0.0,
    scissors_rescue_max_blended_rock_probability: float | None = None,
    conditional_scissors_rescue_profile_index: int | None = None,
    conditional_scissors_rescue_confidence_threshold: float = 0.99,
    conditional_scissors_rescue_margin_threshold: float = 0.98,
    conditional_scissors_rescue_min_blended_transition_mass: float = 0.80,
    conditional_scissors_rescue_max_blended_rock_probability: float | None = None,
) -> NDArray[np.float32]:
    """Blend probabilities and optionally rescue decisive scissors from one profile."""

    blended = _blend_probabilities(probability_arrays, weights=weights)
    labels = [str(label) for label in label_names]
    if "scissors" not in labels:
        raise ValueError("scissors rescue requires a scissors label")
    rescued = _apply_scissors_rescue(
        blended,
        probability_arrays=probability_arrays,
        labels=labels,
        rescue_profile_index=scissors_rescue_profile_index,
        confidence_threshold=scissors_rescue_confidence_threshold,
        margin_threshold=scissors_rescue_margin_threshold,
        min_blended_transition_mass=scissors_rescue_min_blended_transition_mass,
        max_blended_rock_probability=scissors_rescue_max_blended_rock_probability,
    )
    return _apply_scissors_rescue(
        rescued,
        probability_arrays=probability_arrays,
        labels=labels,
        rescue_profile_index=conditional_scissors_rescue_profile_index,
        confidence_threshold=conditional_scissors_rescue_confidence_threshold,
        margin_threshold=conditional_scissors_rescue_margin_threshold,
        min_blended_transition_mass=conditional_scissors_rescue_min_blended_transition_mass,
        max_blended_rock_probability=conditional_scissors_rescue_max_blended_rock_probability,
    )


def _apply_scissors_rescue(
    current: NDArray[np.float32],
    *,
    probability_arrays: Sequence[NDArray[np.float32]],
    labels: Sequence[str],
    rescue_profile_index: int | None,
    confidence_threshold: float,
    margin_threshold: float,
    min_blended_transition_mass: float,
    max_blended_rock_probability: float | None,
) -> NDArray[np.float32]:
    """Apply one optional scissors rescue profile to current probabilities."""

    if rescue_profile_index is None:
        return current
    if not 0 <= rescue_profile_index < len(probability_arrays):
        raise ValueError("scissors rescue profile index is out of range")
    scissors_index = labels.index("scissors")
    rock_index = labels.index("rock") if "rock" in labels else None
    paper_index = labels.index("paper") if "paper" in labels else None
    rescue = np.asarray(probability_arrays[rescue_profile_index], dtype=np.float32)
    if rescue.shape != current.shape:
        raise ValueError("rescue probabilities must match blended probability shape")
    ordered = np.sort(rescue)[::-1]
    rescue_confidence = float(rescue[scissors_index])
    rescue_margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])
    current_transition_mass = (
        float(current[scissors_index]) + float(current[paper_index])
        if paper_index is not None
        else float(current[scissors_index])
    )
    current_rock_probability = float(current[rock_index]) if rock_index is not None else 0.0
    guard_passed = (
        current_transition_mass >= min_blended_transition_mass
        and (
            max_blended_rock_probability is None
            or current_rock_probability <= max_blended_rock_probability
        )
    )
    if (
        int(np.argmax(rescue)) == scissors_index
        and rescue_confidence >= confidence_threshold
        and rescue_margin >= margin_threshold
        and guard_passed
    ):
        rescued = np.zeros_like(current)
        rescued[scissors_index] = np.float32(1.0)
        return rescued
    return current


def _apply_skeleton_paper_rescue(
    current: NDArray[np.float32],
    *,
    labels: Sequence[str],
    canonical_history: Sequence[NDArray[np.float32]],
    observed_progress: float,
    min_history_frames: int,
    min_observed_progress: float,
    min_scissors_confidence: float,
    min_scissors_margin: float,
    min_ring_pinky_extension_delta: float,
    min_latest_ring_pinky_extension: float,
    max_index_middle_minus_ring_pinky: float,
    max_rock_probability: float | None,
    min_transition_mass: float,
) -> NDArray[np.float32]:
    """Rescue ambiguous high-scissors predictions when ring/pinky open like paper."""

    probabilities = np.asarray(current, dtype=np.float32)
    if min_history_frames <= 0:
        return probabilities
    if len(canonical_history) < min_history_frames:
        return probabilities
    if observed_progress < min_observed_progress:
        return probabilities

    label_list = [str(label) for label in labels]
    if "paper" not in label_list or "scissors" not in label_list:
        raise ValueError("skeleton paper rescue requires paper and scissors labels")
    paper_index = label_list.index("paper")
    scissors_index = label_list.index("scissors")
    rock_index = label_list.index("rock") if "rock" in label_list else None

    ordered = np.sort(probabilities)[::-1]
    prediction_index = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction_index])
    margin = confidence - float(ordered[1]) if len(ordered) > 1 else confidence
    transition_mass = float(probabilities[paper_index]) + float(probabilities[scissors_index])
    rock_probability = float(probabilities[rock_index]) if rock_index is not None else 0.0
    if prediction_index != scissors_index:
        return probabilities
    if confidence < min_scissors_confidence or margin < min_scissors_margin:
        return probabilities
    if transition_mass < min_transition_mass:
        return probabilities
    if max_rock_probability is not None and rock_probability > max_rock_probability:
        return probabilities

    stats = _finger_extension_transition_stats(canonical_history)
    if (
        stats["ring_pinky_extension_delta"] >= min_ring_pinky_extension_delta
        and stats["latest_ring_pinky_extension"] >= min_latest_ring_pinky_extension
        and stats["latest_index_middle_minus_ring_pinky"] <= max_index_middle_minus_ring_pinky
    ):
        rescued = np.zeros_like(probabilities)
        rescued[paper_index] = np.float32(1.0)
        return rescued
    return probabilities


def _apply_skeleton_rock_hold_guard(
    current: NDArray[np.float32],
    *,
    labels: Sequence[str],
    canonical_history: Sequence[NDArray[np.float32]],
    min_history_frames: int,
    max_latest_finger_extension: float,
    max_extension_delta: float,
) -> NDArray[np.float32]:
    """Convert binary spikes to rock/wait when the hand is still fist-like."""

    probabilities = np.asarray(current, dtype=np.float32)
    if min_history_frames <= 0:
        return probabilities
    if len(canonical_history) < min_history_frames:
        return probabilities
    label_list = [str(label) for label in labels]
    if not {"rock", "paper", "scissors"}.issubset(set(label_list)):
        raise ValueError("skeleton rock-hold guard requires rock, paper, and scissors labels")
    rock_index = label_list.index("rock")
    paper_index = label_list.index("paper")
    scissors_index = label_list.index("scissors")
    prediction_index = int(np.argmax(probabilities))
    if prediction_index not in {paper_index, scissors_index}:
        return probabilities

    stats = _finger_extension_transition_stats(canonical_history)
    latest_max_extension = max(
        stats["latest_index_middle_extension"],
        stats["latest_ring_pinky_extension"],
    )
    max_delta = max(
        stats["index_middle_extension_delta"],
        stats["ring_pinky_extension_delta"],
    )
    if latest_max_extension <= max_latest_finger_extension and max_delta <= max_extension_delta:
        guarded = np.zeros_like(probabilities)
        guarded[rock_index] = np.float32(1.0)
        return guarded
    return probabilities


def _skeleton_rock_hold_guard_diagnostics(
    current: NDArray[np.float32],
    *,
    labels: Sequence[str],
    canonical_history: Sequence[NDArray[np.float32]],
    min_history_frames: int,
    max_latest_finger_extension: float,
    max_extension_delta: float,
) -> dict[str, object]:
    """Return the rock-hold guard inputs and decision for per-frame diagnostics."""

    probabilities = np.asarray(current, dtype=np.float32)
    label_list = [str(label) for label in labels]
    enabled = min_history_frames > 0
    diagnostics: dict[str, object] = {
        "enabled": bool(enabled),
        "applied": False,
        "history_frame_count": int(len(canonical_history)),
        "min_history_frames": int(min_history_frames),
        "max_latest_finger_extension_threshold": float(max_latest_finger_extension),
        "max_extension_delta_threshold": float(max_extension_delta),
    }
    if not enabled:
        diagnostics["reason"] = "disabled"
        return diagnostics
    if len(canonical_history) < min_history_frames:
        diagnostics["reason"] = "insufficient_history"
        return diagnostics
    if not {"rock", "paper", "scissors"}.issubset(set(label_list)):
        diagnostics["reason"] = "missing_required_labels"
        return diagnostics

    rock_index = label_list.index("rock")
    paper_index = label_list.index("paper")
    scissors_index = label_list.index("scissors")
    prediction_index = int(np.argmax(probabilities))
    prediction = label_list[prediction_index]
    diagnostics["input_prediction"] = prediction
    diagnostics["input_p_rock"] = float(probabilities[rock_index])
    diagnostics["input_p_paper"] = float(probabilities[paper_index])
    diagnostics["input_p_scissors"] = float(probabilities[scissors_index])
    if prediction_index not in {paper_index, scissors_index}:
        diagnostics["reason"] = "input_prediction_not_binary_transition"
        return diagnostics

    stats = _finger_extension_transition_stats(canonical_history)
    latest_max_extension = max(
        stats["latest_index_middle_extension"],
        stats["latest_ring_pinky_extension"],
    )
    observed_max_delta = max(
        stats["index_middle_extension_delta"],
        stats["ring_pinky_extension_delta"],
    )
    diagnostics.update(
        {
            "latest_index_middle_extension": float(stats["latest_index_middle_extension"]),
            "latest_ring_pinky_extension": float(stats["latest_ring_pinky_extension"]),
            "latest_max_finger_extension": float(latest_max_extension),
            "index_middle_extension_delta": float(stats["index_middle_extension_delta"]),
            "ring_pinky_extension_delta": float(stats["ring_pinky_extension_delta"]),
            "max_extension_delta": float(observed_max_delta),
        }
    )
    applied = latest_max_extension <= max_latest_finger_extension and observed_max_delta <= max_extension_delta
    diagnostics["applied"] = bool(applied)
    diagnostics["reason"] = "guard_applied" if applied else "fist_geometry_threshold_not_met"
    return diagnostics


def _apply_gesture_verifier_decision(
    *,
    decision_state: str | None,
    probabilities: NDArray[np.float32],
    labels: Sequence[str],
    canonical_history: Sequence[NDArray[np.float32]],
    min_history_frames: int,
    rock_max_ring_pinky_extension: float = 0.75,
    rock_max_index_middle_extension: float = 1.08,
    rock_max_index_middle_minus_ring_pinky: float = 0.38,
    rock_max_extension_delta: float = 0.20,
    scissors_min_index_middle_extension: float = 0.85,
    scissors_min_index_middle_delta: float = 0.04,
    scissors_min_index_middle_minus_ring_pinky: float = 0.25,
    scissors_max_ring_pinky_extension: float = 0.62,
    paper_min_ring_pinky_extension: float = 0.60,
    paper_min_ring_pinky_delta: float = 0.04,
    paper_max_index_middle_minus_ring_pinky: float = 0.25,
) -> tuple[str | None, dict[str, object]]:
    """Apply an explicit geometry verifier after model-policy decision state selection."""

    diagnostics = _gesture_verifier_diagnostics(
        probabilities,
        labels=labels,
        canonical_history=canonical_history,
        min_history_frames=min_history_frames,
        rock_max_ring_pinky_extension=rock_max_ring_pinky_extension,
        rock_max_index_middle_extension=rock_max_index_middle_extension,
        rock_max_index_middle_minus_ring_pinky=rock_max_index_middle_minus_ring_pinky,
        rock_max_extension_delta=rock_max_extension_delta,
        scissors_min_index_middle_extension=scissors_min_index_middle_extension,
        scissors_min_index_middle_delta=scissors_min_index_middle_delta,
        scissors_min_index_middle_minus_ring_pinky=scissors_min_index_middle_minus_ring_pinky,
        scissors_max_ring_pinky_extension=scissors_max_ring_pinky_extension,
        paper_min_ring_pinky_extension=paper_min_ring_pinky_extension,
        paper_min_ring_pinky_delta=paper_min_ring_pinky_delta,
        paper_max_index_middle_minus_ring_pinky=paper_max_index_middle_minus_ring_pinky,
    )
    diagnostics["input_decision_state"] = decision_state
    if (
        decision_state in {"paper", "scissors"}
        and diagnostics.get("verified_gesture") == "rock_hold"
        and diagnostics.get("rock_hold_passed") is True
    ):
        diagnostics["override_reason"] = "rock_hold_verifier_blocked_binary_transition"
        return WAIT_COUNTER_PAPER_STATE, diagnostics
    diagnostics["override_reason"] = None
    return decision_state, diagnostics


def _gesture_verifier_diagnostics(
    current: NDArray[np.float32],
    *,
    labels: Sequence[str],
    canonical_history: Sequence[NDArray[np.float32]],
    min_history_frames: int,
    rock_max_ring_pinky_extension: float = 0.75,
    rock_max_index_middle_extension: float = 1.08,
    rock_max_index_middle_minus_ring_pinky: float = 0.38,
    rock_max_extension_delta: float = 0.20,
    scissors_min_index_middle_extension: float = 0.85,
    scissors_min_index_middle_delta: float = 0.04,
    scissors_min_index_middle_minus_ring_pinky: float = 0.25,
    scissors_max_ring_pinky_extension: float = 0.62,
    paper_min_ring_pinky_extension: float = 0.60,
    paper_min_ring_pinky_delta: float = 0.04,
    paper_max_index_middle_minus_ring_pinky: float = 0.25,
) -> dict[str, object]:
    """Classify observed geometry as rock-hold, paper, scissors, or unknown."""

    probabilities = np.asarray(current, dtype=np.float32)
    label_list = [str(label) for label in labels]
    diagnostics: dict[str, object] = {
        "enabled": bool(min_history_frames > 0),
        "history_frame_count": int(len(canonical_history)),
        "min_history_frames": int(min_history_frames),
        "verified_gesture": None,
        "rock_hold_passed": False,
        "paper_passed": False,
        "scissors_passed": False,
    }
    if min_history_frames <= 0:
        diagnostics["reason"] = "disabled"
        return diagnostics
    if len(canonical_history) < min_history_frames:
        diagnostics["reason"] = "insufficient_history"
        return diagnostics
    if not {"rock", "paper", "scissors"}.issubset(set(label_list)):
        diagnostics["reason"] = "missing_required_labels"
        return diagnostics

    rock_index = label_list.index("rock")
    paper_index = label_list.index("paper")
    scissors_index = label_list.index("scissors")
    prediction_index = int(np.argmax(probabilities))
    prediction = label_list[prediction_index]
    stats = _finger_extension_transition_stats(canonical_history)
    latest_index_middle = float(stats["latest_index_middle_extension"])
    latest_ring_pinky = float(stats["latest_ring_pinky_extension"])
    index_middle_delta = float(stats["index_middle_extension_delta"])
    ring_pinky_delta = float(stats["ring_pinky_extension_delta"])
    index_middle_minus_ring_pinky = float(stats["latest_index_middle_minus_ring_pinky"])
    max_extension_delta = max(index_middle_delta, ring_pinky_delta)

    rock_hold_passed = (
        latest_ring_pinky <= rock_max_ring_pinky_extension
        and latest_index_middle <= rock_max_index_middle_extension
        and index_middle_minus_ring_pinky <= rock_max_index_middle_minus_ring_pinky
        and max_extension_delta <= rock_max_extension_delta
    )
    scissors_passed = (
        latest_index_middle >= scissors_min_index_middle_extension
        and index_middle_delta >= scissors_min_index_middle_delta
        and index_middle_minus_ring_pinky >= scissors_min_index_middle_minus_ring_pinky
        and latest_ring_pinky <= scissors_max_ring_pinky_extension
    )
    paper_passed = (
        latest_ring_pinky >= paper_min_ring_pinky_extension
        and ring_pinky_delta >= paper_min_ring_pinky_delta
        and index_middle_minus_ring_pinky <= paper_max_index_middle_minus_ring_pinky
    )

    verified_gesture: str | None = None
    if rock_hold_passed:
        verified_gesture = "rock_hold"
    elif prediction == "scissors" and scissors_passed:
        verified_gesture = "scissors"
    elif prediction == "paper" and paper_passed:
        verified_gesture = "paper"
    elif scissors_passed and not paper_passed:
        verified_gesture = "scissors"
    elif paper_passed and not scissors_passed:
        verified_gesture = "paper"

    diagnostics.update(
        {
            "input_prediction": prediction,
            "input_p_rock": float(probabilities[rock_index]),
            "input_p_paper": float(probabilities[paper_index]),
            "input_p_scissors": float(probabilities[scissors_index]),
            "latest_index_middle_extension": latest_index_middle,
            "latest_ring_pinky_extension": latest_ring_pinky,
            "latest_index_middle_minus_ring_pinky": index_middle_minus_ring_pinky,
            "index_middle_extension_delta": index_middle_delta,
            "ring_pinky_extension_delta": ring_pinky_delta,
            "max_extension_delta": float(max_extension_delta),
            "rock_hold_passed": bool(rock_hold_passed),
            "paper_passed": bool(paper_passed),
            "scissors_passed": bool(scissors_passed),
            "verified_gesture": verified_gesture,
            "reason": "verified" if verified_gesture is not None else "geometry_unverified",
        }
    )
    return diagnostics


def _apply_late_geometry_paper_detector(
    current: NDArray[np.float32],
    *,
    labels: Sequence[str],
    canonical_history: Sequence[NDArray[np.float32]],
    observed_progress: float,
    min_history_frames: int,
    min_observed_progress: float,
    max_observed_progress: float,
    min_ring_pinky_extension_delta: float,
    min_latest_ring_pinky_extension: float,
    max_index_middle_minus_ring_pinky: float,
    max_index_middle_delta_minus_ring_pinky_delta: float,
) -> NDArray[np.float32]:
    """Promote late paper when all fingers open together despite weak model transition mass."""

    probabilities = np.asarray(current, dtype=np.float32)
    if min_history_frames <= 0:
        return probabilities
    if len(canonical_history) < min_history_frames:
        return probabilities
    if observed_progress < min_observed_progress or observed_progress > max_observed_progress:
        return probabilities

    label_list = [str(label) for label in labels]
    if "paper" not in label_list:
        raise ValueError("late geometry paper detector requires a paper label")
    paper_index = label_list.index("paper")
    stats = _finger_extension_transition_stats(canonical_history)
    delta_gap = stats["index_middle_extension_delta"] - stats["ring_pinky_extension_delta"]
    if (
        stats["latest_ring_pinky_extension"] >= min_latest_ring_pinky_extension
        and stats["ring_pinky_extension_delta"] >= min_ring_pinky_extension_delta
        and stats["latest_index_middle_minus_ring_pinky"] <= max_index_middle_minus_ring_pinky
        and delta_gap <= max_index_middle_delta_minus_ring_pinky_delta
    ):
        detected = np.zeros_like(probabilities)
        detected[paper_index] = np.float32(1.0)
        return detected
    return probabilities


def _finger_extension_transition_stats(
    canonical_history: Sequence[NDArray[np.float32]],
    *,
    window_frames: int = 4,
) -> dict[str, float]:
    frames = np.asarray(canonical_history, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (21, 3):
        raise ValueError("canonical_history must contain frames with shape (21,3)")
    frame_count = int(frames.shape[0])
    if frame_count == 0:
        raise ValueError("canonical_history must not be empty")
    window = max(1, min(int(window_frames), frame_count))

    denominator = np.maximum(
        np.linalg.norm(frames[:, 9, :] - frames[:, 0, :], axis=1).astype(np.float64),
        0.75,
    )

    def extension(mcp_index: int, tip_index: int) -> NDArray[np.float64]:
        distances = np.linalg.norm(frames[:, tip_index, :] - frames[:, mcp_index, :], axis=1).astype(np.float64)
        return distances / denominator

    index_extension = extension(5, 8)
    middle_extension = extension(9, 12)
    ring_extension = extension(13, 16)
    pinky_extension = extension(17, 20)
    first_ring_pinky = float(np.mean((ring_extension[:window] + pinky_extension[:window]) * 0.5))
    latest_ring_pinky = float(np.mean((ring_extension[-window:] + pinky_extension[-window:]) * 0.5))
    latest_index_middle = float(np.mean((index_extension[-window:] + middle_extension[-window:]) * 0.5))
    first_index_middle = float(np.mean((index_extension[:window] + middle_extension[:window]) * 0.5))
    return {
        "latest_ring_pinky_extension": latest_ring_pinky,
        "ring_pinky_extension_delta": latest_ring_pinky - first_ring_pinky,
        "latest_index_middle_extension": latest_index_middle,
        "index_middle_extension_delta": latest_index_middle - first_index_middle,
        "latest_index_middle_minus_ring_pinky": latest_index_middle - latest_ring_pinky,
    }


def _decision_state_for_probabilities(
    *,
    prediction: str,
    confidence: float,
    margin: float,
    rock_probability: float,
    transition_mass: float,
    confidence_threshold: float,
    margin_threshold: float,
    transition_mass_threshold: float,
    binary_transition_mass_threshold: float,
    observed_progress: float = 1.0,
    min_binary_decision_progress: float = 0.0,
) -> str | None:
    if (
        rock_probability >= confidence_threshold
        or transition_mass <= transition_mass_threshold
        or transition_mass < binary_transition_mass_threshold
    ):
        return WAIT_COUNTER_PAPER_STATE
    binary_candidate = (
        prediction in {"paper", "scissors"}
        and transition_mass >= binary_transition_mass_threshold
        and confidence >= confidence_threshold
        and margin >= margin_threshold
    )
    if binary_candidate and observed_progress < min_binary_decision_progress:
        return WAIT_COUNTER_PAPER_STATE
    if binary_candidate:
        return prediction
    return None


def _shared_int_value(profiles: Sequence[Mapping[str, object]], key: str) -> int:
    values = [_int_value(profile, key) for profile in profiles]
    first = values[0]
    if any(value != first for value in values):
        raise ValueError(f"All ensemble profiles must share {key}")
    return first


def _shared_label_names(profiles: Sequence[Mapping[str, object]]) -> list[str]:
    values = [_string_list(profile["label_names"]) for profile in profiles]
    first = values[0]
    if any(value != first for value in values):
        raise ValueError("All ensemble profiles must share label_names")
    return first


def _load_model(profile: Mapping[str, object], profile_path: Path, device: torch.device) -> torch.nn.Module:
    config_mapping = _mapping(profile["config"], "config")
    run_config = ModelRunConfig(
        model=_model_name(_string_value(profile, "model")),
        seed=_int_value(config_mapping, "seed"),
        hidden_size=_int_value(config_mapping, "hidden_size"),
        dropout=_float_value(config_mapping, "dropout"),
        layers=_int_value(config_mapping, "layers"),
        heads=_int_value(config_mapping, "heads"),
        kernel_size=_int_value(config_mapping, "kernel_size"),
    )
    model = build_classifier(
        run_config,
        input_dim=_int_value(profile, "feature_dim"),
        sequence_length=_int_value(profile, "sequence_length"),
        num_classes=len(_string_list(profile["label_names"])),
    ).to(device)
    state_path = Path(_string_value(profile, "model_state_path"))
    if not state_path.is_absolute() and not state_path.exists():
        state_path = profile_path.parent / state_path
    checkpoint = torch.load(state_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _load_profile(path: Path) -> Mapping[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(loaded, "profile")


def _draw_landmarks(cv2: Any, frame: Any, normalized: NDArray[np.float32]) -> None:
    height, width = frame.shape[:2]
    points = [(int(float(x) * width), int(float(y) * height)) for x, y, _ in normalized.tolist()]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 180, 40), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 3, (40, 240, 255), -1, cv2.LINE_AA)


def _load_realtime_dependencies() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import mediapipe as mp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Realtime mode requires optional dependencies: mediapipe and opencv-python. "
            "Install the realtime extra or install those packages in the active environment."
        ) from exc
    return cv2, mp


def _model_name(value: str) -> str:
    if value not in {"mlp", "gru", "tcn", "transformer", "stgcn"}:
        raise ValueError(f"Unsupported model name in profile: {value}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    parsed: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} must use string keys")
        parsed[key] = item
    return parsed


def _string_value(mapping: Mapping[str, object], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _int_value(mapping: Mapping[str, object], key: str) -> int:
    value = mapping[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _float_value(mapping: Mapping[str, object], key: str) -> float:
    value = mapping[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("value must be a sequence of strings")
    return [str(item) for item in value]


if __name__ == "__main__":
    raise SystemExit(main())
