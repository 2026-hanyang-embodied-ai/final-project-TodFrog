"""Evaluate a real-skeleton final gesture profile on labeled MP4 clips."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import deque
from pathlib import Path
from shutil import copy2
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from embodied_rps.real_skeleton_video_eval import (
    StrictDecisionConfig,
    attach_motion_progress,
    annotate_rows_with_strict_decision,
    build_dataset_expansion_plan,
    build_rock_false_trigger_report,
    build_validation_summary,
    discover_final_label_videos,
    discover_labeled_videos,
    summarize_clip_decision,
    validate_discovered_videos,
    validate_final_label_videos,
    write_clip_metrics_csv,
    write_dataset_expansion_plan,
    write_frame_rows,
    write_schunk_event_manifest,
)
from embodied_rps.real_skeleton_open_set_guard import (
    OpenSetGuardConfig,
    annotate_rows_with_open_set_guard,
)
from embodied_rps.tools.run_realtime_skeleton_predictor import (
    _apply_late_geometry_paper_detector,
    _apply_skeleton_paper_rescue,
    _apply_skeleton_rock_hold_guard,
    _blend_probabilities_with_optional_scissors_rescue,
    _blend_probabilities,
    _draw_landmarks,
    _load_model,
    _load_profile,
    _parse_profile_weights,
    _shared_int_value,
    _shared_label_names,
    canonicalize_mediapipe_landmarks,
    features_from_canonical_history,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate the exported real skeleton predictor on all labeled MP4s."""

    parser = argparse.ArgumentParser(description="Evaluate real MP4 skeleton final gesture predictions.")
    parser.add_argument("--profile", required=True, type=Path, action="append", help="Path to exported real skeleton profile JSON. Repeat for a weighted ensemble.")
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
    parser.add_argument("--input-root", required=True, type=Path, help="Root containing labeled transition MP4 folders.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output artifact root.")
    parser.add_argument(
        "--event-output",
        type=Path,
        default=Path("artifacts/real_skeleton_schunk_events_20260611/events.jsonl"),
        help="SCHUNK event JSONL path written only when the strict gate passes.",
    )
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument(
        "--label-mode",
        choices=("transition", "final-label"),
        default="transition",
        help="transition uses rock_to_paper/scissors folders; final-label uses paper/scissors/rock test folders.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--margin-threshold", type=float, default=0.20)
    parser.add_argument("--confirmation-count", type=int, default=3)
    parser.add_argument("--max-decision-progress", type=float, default=0.50)
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
        help="Clip progress before which high-confidence paper/scissors spikes remain provisional wait.",
    )
    parser.add_argument(
        "--progress-mode",
        choices=("clip", "motion", "observed", "model"),
        default="clip",
        help="Progress source for strict decision deadlines.",
    )
    parser.add_argument(
        "--paper-wait-nonterminal-for-transitions",
        action="store_true",
        help="Treat wait_counter_paper as provisional robot behavior, not a terminal paper/scissors clip prediction.",
    )
    parser.add_argument("--response-delay-s", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    cv2, mp = _load_video_dependencies()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    config = StrictDecisionConfig(
        confidence_threshold=float(args.confidence_threshold),
        margin_threshold=float(args.margin_threshold),
        confirmation_count=int(args.confirmation_count),
        max_decision_progress=float(args.max_decision_progress),
        transition_mass_threshold=float(args.transition_mass_threshold),
        paper_wait_is_terminal_for_transitions=not bool(args.paper_wait_nonterminal_for_transitions),
        binary_transition_mass_threshold=float(args.binary_transition_mass_threshold),
        progress_key=_progress_key_for_mode(str(args.progress_mode)),
    )
    if args.label_mode == "final-label":
        videos = discover_final_label_videos(args.input_root)
        discovery = validate_final_label_videos(videos, expected_count=int(args.expected_count))
    else:
        videos = discover_labeled_videos(args.input_root)
        discovery = validate_discovered_videos(videos, expected_count=int(args.expected_count))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    profile_paths = list(args.profile)
    profiles = [_load_profile(path) for path in profile_paths]
    profile_weights = _parse_profile_weights(args.profile_weights, profile_count=len(profiles))
    models = [_load_model(profile, profile_path, device) for profile, profile_path in zip(profiles, profile_paths, strict=True)]
    sequence_length = _shared_int_value(profiles, "sequence_length")
    label_names = _shared_label_names(profiles)

    clip_metrics: list[dict[str, object]] = []
    contact_frames: list[tuple[str, NDArray[np.uint8]]] = []
    for video in videos:
        metrics, contact = _evaluate_one_video(
            cv2=cv2,
            mp=mp,
            models=models,
            device=device,
            label_names=label_names,
            profile_weights=profile_weights,
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
            paper_rescue_min_history_frames=int(args.paper_rescue_min_history_frames),
            paper_rescue_min_observed_progress=float(args.paper_rescue_min_observed_progress),
            paper_rescue_min_scissors_confidence=float(args.paper_rescue_min_scissors_confidence),
            paper_rescue_min_scissors_margin=float(args.paper_rescue_min_scissors_margin),
            paper_rescue_min_ring_pinky_extension_delta=float(args.paper_rescue_min_ring_pinky_extension_delta),
            paper_rescue_min_latest_ring_pinky_extension=float(args.paper_rescue_min_latest_ring_pinky_extension),
            paper_rescue_max_index_middle_minus_ring_pinky=float(args.paper_rescue_max_index_middle_minus_ring_pinky),
            paper_rescue_max_rock_probability=(
                None
                if args.paper_rescue_max_rock_probability is None
                else float(args.paper_rescue_max_rock_probability)
            ),
            paper_rescue_min_transition_mass=float(args.paper_rescue_min_transition_mass),
            late_geometry_paper_min_history_frames=int(args.late_geometry_paper_min_history_frames),
            late_geometry_paper_min_observed_progress=float(args.late_geometry_paper_min_observed_progress),
            late_geometry_paper_max_observed_progress=float(args.late_geometry_paper_max_observed_progress),
            late_geometry_paper_min_ring_pinky_extension_delta=float(args.late_geometry_paper_min_ring_pinky_extension_delta),
            late_geometry_paper_min_latest_ring_pinky_extension=float(args.late_geometry_paper_min_latest_ring_pinky_extension),
            late_geometry_paper_max_index_middle_minus_ring_pinky=float(args.late_geometry_paper_max_index_middle_minus_ring_pinky),
            late_geometry_paper_max_index_middle_delta_minus_ring_pinky_delta=float(
                args.late_geometry_paper_max_index_middle_delta_minus_ring_pinky_delta
            ),
            rock_hold_guard_min_history_frames=int(args.rock_hold_guard_min_history_frames),
            rock_hold_guard_max_latest_finger_extension=float(args.rock_hold_guard_max_latest_finger_extension),
            rock_hold_guard_max_extension_delta=float(args.rock_hold_guard_max_extension_delta),
            sequence_length=sequence_length,
            video_path=video.path,
            transition_label=video.transition_label,
            true_gesture=video.true_gesture,
            clip_id=video.clip_id,
            output_root=output_root,
            config=config,
            min_binary_decision_progress=float(args.min_binary_decision_progress),
        )
        clip_metrics.append(metrics)
        if contact is not None:
            contact_frames.append((video.clip_id, contact))

    clip_metrics_csv = output_root / "clip_metrics.csv"
    write_clip_metrics_csv(clip_metrics_csv, clip_metrics)
    summary = build_validation_summary(
        clip_metrics=clip_metrics,
        discovery_summary=discovery,
        config=config,
        event_manifest_path=args.event_output,
    )
    summary["open_set_guard"] = {
        "min_binary_decision_progress": float(args.min_binary_decision_progress),
        "early_binary_action": "wait_counter_paper",
        "progress_mode": str(args.progress_mode),
    }
    summary["late_geometry_paper_detector"] = {
        "min_history_frames": int(args.late_geometry_paper_min_history_frames),
        "min_observed_progress": float(args.late_geometry_paper_min_observed_progress),
        "max_observed_progress": float(args.late_geometry_paper_max_observed_progress),
        "min_ring_pinky_extension_delta": float(args.late_geometry_paper_min_ring_pinky_extension_delta),
        "min_latest_ring_pinky_extension": float(args.late_geometry_paper_min_latest_ring_pinky_extension),
        "max_index_middle_minus_ring_pinky": float(args.late_geometry_paper_max_index_middle_minus_ring_pinky),
        "max_index_middle_delta_minus_ring_pinky_delta": float(
            args.late_geometry_paper_max_index_middle_delta_minus_ring_pinky_delta
        ),
    }
    summary["clip_metrics_csv"] = clip_metrics_csv.as_posix()
    summary_path = output_root / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_contact_sheet(cv2, output_root / "review_contact_sheet.png", contact_frames)
    if args.label_mode == "final-label":
        rock_report = build_rock_false_trigger_report(clip_metrics)
        rock_report_path = output_root / "rock_false_trigger_report.json"
        rock_report_path.write_text(json.dumps(rock_report, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["rock_false_trigger_report"] = rock_report_path.as_posix()
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if summary["passed"]:
        events = write_schunk_event_manifest(
            args.event_output,
            clip_metrics,
            response_delay_s=float(args.response_delay_s),
        )
        summary["event_count"] = len(events)
        summary["event_manifest_path"] = args.event_output.as_posix()
        summary["event_manifest_written"] = True
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    elif not summary["passed"]:
        expansion_plan = build_dataset_expansion_plan(summary, clip_metrics)
        write_dataset_expansion_plan(
            output_root / "dataset_expansion_plan.json",
            output_root / "dataset_expansion_plan.md",
            expansion_plan,
        )

    print(json.dumps({"summary_path": summary_path.as_posix(), "passed": summary["passed"]}, indent=2, ensure_ascii=False))
    return 0


def _evaluate_one_video(
    *,
    cv2: Any,
    mp: Any,
    models: Sequence[torch.nn.Module],
    device: torch.device,
    label_names: Sequence[str],
    profile_weights: Sequence[float],
    scissors_rescue_profile_index: int | None,
    scissors_rescue_confidence_threshold: float,
    scissors_rescue_margin_threshold: float,
    scissors_rescue_min_blended_transition_mass: float,
    scissors_rescue_max_blended_rock_probability: float | None,
    conditional_scissors_rescue_profile_index: int | None,
    conditional_scissors_rescue_confidence_threshold: float,
    conditional_scissors_rescue_margin_threshold: float,
    conditional_scissors_rescue_min_blended_transition_mass: float,
    conditional_scissors_rescue_max_blended_rock_probability: float | None,
    paper_rescue_min_history_frames: int,
    paper_rescue_min_observed_progress: float,
    paper_rescue_min_scissors_confidence: float,
    paper_rescue_min_scissors_margin: float,
    paper_rescue_min_ring_pinky_extension_delta: float,
    paper_rescue_min_latest_ring_pinky_extension: float,
    paper_rescue_max_index_middle_minus_ring_pinky: float,
    paper_rescue_max_rock_probability: float | None,
    paper_rescue_min_transition_mass: float,
    late_geometry_paper_min_history_frames: int,
    late_geometry_paper_min_observed_progress: float,
    late_geometry_paper_max_observed_progress: float,
    late_geometry_paper_min_ring_pinky_extension_delta: float,
    late_geometry_paper_min_latest_ring_pinky_extension: float,
    late_geometry_paper_max_index_middle_minus_ring_pinky: float,
    late_geometry_paper_max_index_middle_delta_minus_ring_pinky_delta: float,
    rock_hold_guard_min_history_frames: int,
    rock_hold_guard_max_latest_finger_extension: float,
    rock_hold_guard_max_extension_delta: float,
    sequence_length: int,
    video_path: Path,
    transition_label: str,
    true_gesture: str,
    clip_id: str,
    output_root: Path,
    config: StrictDecisionConfig,
    min_binary_decision_progress: float,
) -> tuple[dict[str, object], NDArray[np.uint8] | None]:
    clip_dir = output_root / "clips" / transition_label / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = clip_dir / "overlay.mp4"
    frame_csv_path = clip_dir / "frames.csv"
    frame_jsonl_path = clip_dir / "frames.jsonl"
    metrics_path = clip_dir / "metrics.json"

    capture, temp_dir = _open_capture_with_optional_ascii_copy(cv2, video_path, clip_dir)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open input video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count_hint = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames: list[NDArray[np.uint8]] = []
        landmarks_by_frame: list[NDArray[np.float32] | None] = []
        canonical_by_frame: list[NDArray[np.float32] | None] = []
        rows: list[dict[str, object]] = []
        history: deque[NDArray[np.float32]] = deque(maxlen=sequence_length)
        hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        try:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame.copy())
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                detected = bool(result.multi_hand_landmarks)
                normalized: NDArray[np.float32] | None = None
                canonical_for_progress: NDArray[np.float32] | None = None
                probabilities = {label: 0.0 for label in label_names}
                prediction: str | None = None
                confidence = 0.0
                margin = 0.0
                late_geometry_paper_detected = False
                if detected:
                    hand = result.multi_hand_landmarks[0]
                    normalized = np.asarray([(lm.x, lm.y, lm.z) for lm in hand.landmark], dtype=np.float32)
                    canonical = canonicalize_mediapipe_landmarks(normalized)
                    canonical_for_progress = canonical
                    history.append(canonical)
                    features = features_from_canonical_history(tuple(history), sequence_length=sequence_length)
                    with torch.no_grad():
                        feature_tensor = torch.from_numpy(features).to(device)
                        probability_arrays = [
                            torch.softmax(model(feature_tensor), dim=1).detach().cpu().numpy()[0]
                            for model in models
                        ]
                        probability_array = _blend_probabilities_with_optional_scissors_rescue(
                            probability_arrays,
                            weights=profile_weights,
                            label_names=label_names,
                            scissors_rescue_profile_index=scissors_rescue_profile_index,
                            scissors_rescue_confidence_threshold=scissors_rescue_confidence_threshold,
                            scissors_rescue_margin_threshold=scissors_rescue_margin_threshold,
                            scissors_rescue_min_blended_transition_mass=scissors_rescue_min_blended_transition_mass,
                            scissors_rescue_max_blended_rock_probability=scissors_rescue_max_blended_rock_probability,
                            conditional_scissors_rescue_profile_index=conditional_scissors_rescue_profile_index,
                            conditional_scissors_rescue_confidence_threshold=conditional_scissors_rescue_confidence_threshold,
                            conditional_scissors_rescue_margin_threshold=conditional_scissors_rescue_margin_threshold,
                            conditional_scissors_rescue_min_blended_transition_mass=conditional_scissors_rescue_min_blended_transition_mass,
                            conditional_scissors_rescue_max_blended_rock_probability=conditional_scissors_rescue_max_blended_rock_probability,
                        )
                        probability_array = _apply_skeleton_paper_rescue(
                            probability_array,
                            labels=label_names,
                            canonical_history=tuple(history),
                            observed_progress=float((frame_index + 1) / max(1, frame_count_hint)),
                            min_history_frames=paper_rescue_min_history_frames,
                            min_observed_progress=paper_rescue_min_observed_progress,
                            min_scissors_confidence=paper_rescue_min_scissors_confidence,
                            min_scissors_margin=paper_rescue_min_scissors_margin,
                            min_ring_pinky_extension_delta=paper_rescue_min_ring_pinky_extension_delta,
                            min_latest_ring_pinky_extension=paper_rescue_min_latest_ring_pinky_extension,
                            max_index_middle_minus_ring_pinky=paper_rescue_max_index_middle_minus_ring_pinky,
                            max_rock_probability=paper_rescue_max_rock_probability,
                            min_transition_mass=paper_rescue_min_transition_mass,
                        )
                        before_late_geometry = probability_array.copy()
                        probability_array = _apply_late_geometry_paper_detector(
                            probability_array,
                            labels=label_names,
                            canonical_history=tuple(history),
                            observed_progress=float((frame_index + 1) / max(1, frame_count_hint)),
                            min_history_frames=late_geometry_paper_min_history_frames,
                            min_observed_progress=late_geometry_paper_min_observed_progress,
                            max_observed_progress=late_geometry_paper_max_observed_progress,
                            min_ring_pinky_extension_delta=late_geometry_paper_min_ring_pinky_extension_delta,
                            min_latest_ring_pinky_extension=late_geometry_paper_min_latest_ring_pinky_extension,
                            max_index_middle_minus_ring_pinky=late_geometry_paper_max_index_middle_minus_ring_pinky,
                            max_index_middle_delta_minus_ring_pinky_delta=(
                                late_geometry_paper_max_index_middle_delta_minus_ring_pinky_delta
                            ),
                        )
                        late_geometry_paper_detected = bool(not np.allclose(before_late_geometry, probability_array))
                        probability_array = _apply_skeleton_rock_hold_guard(
                            probability_array,
                            labels=label_names,
                            canonical_history=tuple(history),
                            min_history_frames=rock_hold_guard_min_history_frames,
                            max_latest_finger_extension=rock_hold_guard_max_latest_finger_extension,
                            max_extension_delta=rock_hold_guard_max_extension_delta,
                        )
                    for label, probability in zip(label_names, probability_array.tolist(), strict=True):
                        probabilities[str(label)] = float(probability)
                    order = np.argsort(probability_array)[::-1]
                    prediction = str(label_names[int(order[0])])
                    confidence = float(probability_array[int(order[0])])
                    margin = confidence - float(probability_array[int(order[1])]) if len(order) > 1 else confidence
                frame_count_for_progress = max(1, frame_count_hint)
                rows.append(
                    {
                        "frame_index": frame_index,
                        "time_s": float(frame_index / fps) if fps > 0.0 else 0.0,
                        "detected": detected,
                        "prediction": prediction,
                        "rock_probability": probabilities.get("rock", 0.0),
                        "paper_probability": probabilities.get("paper", 0.0),
                        "scissors_probability": probabilities.get("scissors", 0.0),
                        "transition_mass": probabilities.get("paper", 0.0) + probabilities.get("scissors", 0.0),
                        "confidence": confidence,
                        "confidence_margin": margin,
                        "late_geometry_paper_detected": late_geometry_paper_detected,
                        "clip_progress": float((frame_index + 1) / frame_count_for_progress),
                        "model_progress": float(min(1.0, len(history) / max(1, sequence_length))),
                    }
                )
                landmarks_by_frame.append(normalized)
                canonical_by_frame.append(canonical_for_progress.copy() if canonical_for_progress is not None else None)
                frame_index += 1
        finally:
            hands.close()

        actual_frame_count = len(frames)
        if actual_frame_count > 0 and frame_count_hint != actual_frame_count:
            for row in rows:
                row["clip_progress"] = float((int(row["frame_index"]) + 1) / actual_frame_count)
        rows = attach_motion_progress(rows, canonical_by_frame)
        if min_binary_decision_progress > 0.0:
            guard_config = OpenSetGuardConfig(
                decision=config,
                min_binary_decision_progress=float(min_binary_decision_progress),
            )
            annotated_rows = annotate_rows_with_open_set_guard(rows, config=guard_config)
        else:
            annotated_rows = annotate_rows_with_strict_decision(rows, config=config)
        _write_overlay(
            cv2=cv2,
            overlay_path=overlay_path,
            frames=frames,
            landmarks_by_frame=landmarks_by_frame,
            rows=annotated_rows,
            fps=fps,
            width=width,
            height=height,
        )
        write_frame_rows(frame_csv_path, frame_jsonl_path, annotated_rows)
        metrics = summarize_clip_decision(
            rows,
            true_gesture=true_gesture,
            transition_label=transition_label,
            source_path=video_path,
            clip_id=clip_id,
            frame_count=actual_frame_count,
            fps=fps,
            width=width,
            height=height,
            config=config,
            overlay_path=overlay_path,
            frame_csv_path=frame_csv_path,
            frame_jsonl_path=frame_jsonl_path,
            annotated_rows=annotated_rows,
        )
        if min_binary_decision_progress > 0.0:
            metrics["open_set_guard"] = {
                "min_binary_decision_progress": float(min_binary_decision_progress),
                "early_binary_action": "wait_counter_paper",
            }
            metrics["open_set_guarded_binary_frame_count"] = sum(
                1 for row in annotated_rows if bool(row.get("open_set_binary_blocked_by_progress"))
            )
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        contact = _contact_frame(frames, metrics)
        return metrics, contact
    finally:
        capture.release()
        if temp_dir is not None:
            temp_dir.cleanup()


def _write_overlay(
    *,
    cv2: Any,
    overlay_path: Path,
    frames: Sequence[NDArray[np.uint8]],
    landmarks_by_frame: Sequence[NDArray[np.float32] | None],
    rows: Sequence[dict[str, object]],
    fps: float,
    width: int,
    height: int,
) -> None:
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps or 30.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open overlay writer: {overlay_path}")
    try:
        decision_seen = False
        decision_text = ""
        for frame, landmarks, row in zip(frames, landmarks_by_frame, rows, strict=True):
            annotated = frame.copy()
            if landmarks is not None:
                _draw_landmarks(cv2, annotated, landmarks)
            if row.get("is_decision_frame"):
                decision_seen = True
                decision_text = (
                    f"DECISION {row.get('decision_state')} robot {row.get('selected_robot_action')} "
                    f"{float(row.get('confidence', 0.0)):.2f}"
                )
            prediction = row.get("prediction") if row.get("prediction") is not None else "no hand"
            decision_state = row.get("decision_state") or "none"
            robot_action = row.get("selected_robot_action") or "none"
            status = (
                f"{prediction} conf {float(row.get('confidence', 0.0)):.2f} "
                f"R/P/S {float(row.get('rock_probability', 0.0)):.2f}/"
                f"{float(row.get('paper_probability', 0.0)):.2f}/"
                f"{float(row.get('scissors_probability', 0.0)):.2f} "
                f"tm {float(row.get('transition_mass', 0.0)):.2f} "
                f"progress {float(row.get('clip_progress', 0.0)):.2f}"
            )
            policy_status = (
                f"state {decision_state} robot {robot_action} "
                f"margin {float(row.get('confidence_margin', 0.0)):.2f}"
            )
            cv2.putText(annotated, status, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 240, 20), 2, cv2.LINE_AA)
            cv2.putText(annotated, policy_status, (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 220, 255), 2, cv2.LINE_AA)
            if decision_seen:
                cv2.putText(annotated, decision_text, (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 255), 2, cv2.LINE_AA)
            writer.write(annotated)
    finally:
        writer.release()


def _write_contact_sheet(cv2: Any, path: Path, contact_frames: Sequence[tuple[str, NDArray[np.uint8]]]) -> None:
    if len(contact_frames) == 0:
        return
    tile_width = 240
    tile_height = 180
    columns = 5
    rows = int(np.ceil(len(contact_frames) / columns))
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, (clip_id, frame) in enumerate(contact_frames):
        resized = cv2.resize(frame, (tile_width, tile_height))
        row = index // columns
        col = index % columns
        y0 = row * tile_height
        x0 = col * tile_width
        sheet[y0 : y0 + tile_height, x0 : x0 + tile_width] = resized
        cv2.putText(sheet, clip_id[:28], (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 240, 20), 1, cv2.LINE_AA)
    _imwrite_unicode(cv2, path, sheet)


def _contact_frame(frames: Sequence[NDArray[np.uint8]], metrics: Mapping[str, object]) -> NDArray[np.uint8] | None:
    if len(frames) == 0:
        return None
    decision_frame = metrics.get("decision_frame")
    if isinstance(decision_frame, int):
        index = max(0, min(len(frames) - 1, decision_frame))
    else:
        index = len(frames) // 2
    return frames[index].copy()


def _open_capture_with_optional_ascii_copy(cv2: Any, path: Path, clip_dir: Path) -> tuple[Any, tempfile.TemporaryDirectory[str] | None]:
    capture = cv2.VideoCapture(str(path))
    if capture.isOpened():
        return capture, None
    capture.release()
    temp_dir = tempfile.TemporaryDirectory(prefix="rps_video_eval_")
    temp_path = Path(temp_dir.name) / f"{clip_dir.name}.mp4"
    copy2(path, temp_path)
    capture = cv2.VideoCapture(str(temp_path))
    return capture, temp_dir


def _imwrite_unicode(cv2: Any, path: Path, image: NDArray[np.uint8]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def _load_video_dependencies() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import mediapipe as mp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Video evaluation requires mediapipe and opencv-python.") from exc
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        raise RuntimeError("Video evaluation requires a MediaPipe build with mp.solutions.hands, such as mediapipe==0.10.21.")
    return cv2, mp


def _progress_key_for_mode(mode: str) -> str:
    if mode == "clip":
        return "clip_progress"
    if mode == "motion":
        return "motion_progress"
    if mode == "observed":
        return "observed_progress"
    if mode == "model":
        return "model_progress"
    raise ValueError(f"Unsupported progress mode: {mode}")


def _int_value(mapping: dict[str, object] | Any, key: str) -> int:
    value = mapping[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("label_names must be a sequence")
    return [str(item) for item in value]


if __name__ == "__main__":
    raise SystemExit(main())
