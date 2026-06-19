"""Evaluate a two-stage skeleton predictor on labeled MP4 clips."""

from __future__ import annotations

import argparse
import json
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from embodied_rps.real_skeleton_two_stage import two_stage_frame_row, validate_two_stage_label_names
from embodied_rps.real_skeleton_video_eval import (
    StrictDecisionConfig,
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
from embodied_rps.tools.evaluate_real_skeleton_video_predictions import (
    _contact_frame,
    _load_video_dependencies,
    _open_capture_with_optional_ascii_copy,
    _write_contact_sheet,
    _write_overlay,
)
from embodied_rps.tools.run_realtime_skeleton_predictor import (
    _load_model,
    _load_profile,
    _shared_int_value,
    _shared_label_names,
    canonicalize_mediapipe_landmarks,
    features_from_canonical_history,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate stage1 rock/transition and stage2 paper/scissors profiles."""

    parser = argparse.ArgumentParser(description="Evaluate two-stage real skeleton predictions on MP4 clips.")
    parser.add_argument("--stage1-profile", required=True, type=Path, help="Exported rock/transition profile JSON.")
    parser.add_argument("--stage2-profile", required=True, type=Path, help="Exported paper/scissors profile JSON.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--event-output", type=Path, default=Path("artifacts/real_skeleton_schunk_events_20260611/two_stage_events.jsonl"))
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument("--label-mode", choices=("transition", "final-label"), default="transition")
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--margin-threshold", type=float, default=0.10)
    parser.add_argument("--confirmation-count", type=int, default=2)
    parser.add_argument("--max-decision-progress", type=float, default=0.50)
    parser.add_argument("--transition-mass-threshold", type=float, default=0.05)
    parser.add_argument("--binary-transition-mass-threshold", type=float, default=0.60)
    parser.add_argument("--paper-wait-nonterminal-for-transitions", action="store_true")
    parser.add_argument("--response-delay-s", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Validate profile labels and print the plan without loading weights or videos.")
    args = parser.parse_args(argv)

    stage1_profile = _load_profile(args.stage1_profile)
    stage2_profile = _load_profile(args.stage2_profile)
    stage1_labels = _shared_label_names([stage1_profile])
    stage2_labels = _shared_label_names([stage2_profile])
    validate_two_stage_label_names(stage1_labels, stage2_labels)
    stage1_sequence_length = _shared_int_value([stage1_profile], "sequence_length")
    stage2_sequence_length = _shared_int_value([stage2_profile], "sequence_length")
    if stage1_sequence_length != stage2_sequence_length:
        raise ValueError("stage1 and stage2 profiles must share sequence_length")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage1_profile": args.stage1_profile.as_posix(),
                    "stage2_profile": args.stage2_profile.as_posix(),
                    "stage1_labels": stage1_labels,
                    "stage2_labels": stage2_labels,
                    "sequence_length": stage1_sequence_length,
                    "input_root": args.input_root.as_posix(),
                    "output_root": args.output_root.as_posix(),
                    "label_mode": args.label_mode,
                    "expected_count": int(args.expected_count),
                },
                indent=2,
            )
        )
        return 0

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
    )
    if args.label_mode == "final-label":
        videos = discover_final_label_videos(args.input_root)
        discovery = validate_final_label_videos(videos, expected_count=int(args.expected_count))
    else:
        videos = discover_labeled_videos(args.input_root)
        discovery = validate_discovered_videos(videos, expected_count=int(args.expected_count))

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    stage1_model = _load_model(stage1_profile, args.stage1_profile, device)
    stage2_model = _load_model(stage2_profile, args.stage2_profile, device)
    clip_metrics: list[dict[str, object]] = []
    contact_frames: list[tuple[str, NDArray[np.uint8]]] = []
    for video in videos:
        metrics, contact = _evaluate_one_video(
            cv2=cv2,
            mp=mp,
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            device=device,
            stage1_labels=stage1_labels,
            stage2_labels=stage2_labels,
            sequence_length=stage1_sequence_length,
            video_path=video.path,
            transition_label=video.transition_label,
            true_gesture=video.true_gesture,
            clip_id=video.clip_id,
            output_root=output_root,
            config=config,
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
    summary["stage1_profile"] = args.stage1_profile.as_posix()
    summary["stage2_profile"] = args.stage2_profile.as_posix()
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
        events = write_schunk_event_manifest(args.event_output, clip_metrics, response_delay_s=float(args.response_delay_s))
        summary["event_count"] = len(events)
        summary["event_manifest_path"] = args.event_output.as_posix()
        summary["event_manifest_written"] = True
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        expansion_plan = build_dataset_expansion_plan(summary, clip_metrics)
        write_dataset_expansion_plan(output_root / "dataset_expansion_plan.json", output_root / "dataset_expansion_plan.md", expansion_plan)

    print(json.dumps({"summary_path": summary_path.as_posix(), "passed": summary["passed"]}, indent=2, ensure_ascii=False))
    return 0


def _evaluate_one_video(
    *,
    cv2: Any,
    mp: Any,
    stage1_model: torch.nn.Module,
    stage2_model: torch.nn.Module,
    device: torch.device,
    stage1_labels: Sequence[str],
    stage2_labels: Sequence[str],
    sequence_length: int,
    video_path: Path,
    transition_label: str,
    true_gesture: str,
    clip_id: str,
    output_root: Path,
    config: StrictDecisionConfig,
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
                stage1_probabilities = np.zeros((len(stage1_labels),), dtype=np.float32)
                stage2_probabilities = np.zeros((len(stage2_labels),), dtype=np.float32)
                if detected:
                    hand = result.multi_hand_landmarks[0]
                    normalized = np.asarray([(lm.x, lm.y, lm.z) for lm in hand.landmark], dtype=np.float32)
                    canonical = canonicalize_mediapipe_landmarks(normalized)
                    history.append(canonical)
                    features = features_from_canonical_history(tuple(history), sequence_length=sequence_length)
                    with torch.no_grad():
                        feature_tensor = torch.from_numpy(features).to(device)
                        stage1_probabilities = torch.softmax(stage1_model(feature_tensor), dim=1).detach().cpu().numpy()[0].astype(np.float32)
                        stage2_probabilities = torch.softmax(stage2_model(feature_tensor), dim=1).detach().cpu().numpy()[0].astype(np.float32)
                frame_count_for_progress = max(1, frame_count_hint)
                row = two_stage_frame_row(
                    frame_index=frame_index,
                    time_s=float(frame_index / fps) if fps > 0.0 else 0.0,
                    clip_progress=float((frame_index + 1) / frame_count_for_progress),
                    model_progress=float(min(1.0, len(history) / max(1, sequence_length))),
                    detected=detected,
                    stage1_labels=stage1_labels,
                    stage1_probabilities=stage1_probabilities.tolist() if detected else [1.0, 0.0],
                    stage2_labels=stage2_labels,
                    stage2_probabilities=stage2_probabilities.tolist() if detected else [0.5, 0.5],
                )
                row["stage1_rock_probability"] = float(stage1_probabilities[stage1_labels.index("rock")]) if detected else 0.0
                row["stage1_transition_probability"] = float(stage1_probabilities[stage1_labels.index("transition")]) if detected else 0.0
                row["stage2_paper_probability"] = float(stage2_probabilities[stage2_labels.index("paper")]) if detected else 0.0
                row["stage2_scissors_probability"] = float(stage2_probabilities[stage2_labels.index("scissors")]) if detected else 0.0
                rows.append(row)
                landmarks_by_frame.append(normalized)
                frame_index += 1
        finally:
            hands.close()

        actual_frame_count = len(frames)
        if actual_frame_count > 0 and frame_count_hint != actual_frame_count:
            for row in rows:
                row["clip_progress"] = float((int(cast(int, row["frame_index"])) + 1) / actual_frame_count)
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
            annotated_rows,
            true_gesture=cast(Any, true_gesture),
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
        )
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return metrics, _contact_frame(frames, metrics)
    finally:
        capture.release()
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
