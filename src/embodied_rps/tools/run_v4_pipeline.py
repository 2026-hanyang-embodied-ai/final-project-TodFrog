"""Advance v4 calibration pipeline metadata gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v4_pipeline_runner import V4PipelineRunConfig, run_v4_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Advance v4 calibration pipeline metadata gates.")
    parser.add_argument("--calibration-input-root", required=True, type=Path)
    parser.add_argument("--heldout-root", action="append", type=Path, default=[])
    parser.add_argument("--expected-min-per-label", type=int, default=20)
    parser.add_argument("--intake-output-root", type=Path, default=Path("artifacts/real_skeleton_v4_calibration_intake_20260611"))
    parser.add_argument("--skeleton-review-plan-output-root", type=Path, default=Path("artifacts/real_skeleton_v4_calibration_review_plan_20260611"))
    parser.add_argument("--skeleton-review-output-root", type=Path, default=Path("artifacts/real_hand_skeleton_review_v4_calibration_20260611"))
    parser.add_argument("--seed-package-root", type=Path, default=Path("artifacts/real_skeleton_v4_calibration_seed_package_20260612"))
    parser.add_argument("--recording-slot-audit-output-root", type=Path, default=Path("artifacts/real_skeleton_v4_recording_slot_audit_20260612"))
    parser.add_argument("--mp4-preflight-output-root", type=Path, default=Path("artifacts/real_skeleton_v4_mp4_preflight_20260612"))
    parser.add_argument("--dataset-generation-plan-output-root", type=Path, default=Path("artifacts/real_skeleton_v4_dataset_generation_plan_20260611"))
    parser.add_argument("--dataset-output-root", type=Path, default=Path("artifacts/real_guided_three_class_wait_expanded_v4_20260611"))
    parser.add_argument("--base-dataset-root", type=Path, default=Path("artifacts/real_guided_large_sharded_20260610"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/real_skeleton_three_class_wait_prediction_v4.yaml"))
    parser.add_argument("--pipeline-output-root", type=Path, default=Path("artifacts/real_skeleton_v4_pipeline_run_20260612"))
    parser.add_argument("--v3-summary", type=Path, default=Path("artifacts/real_guided_three_class_wait_expanded_v3_20260611/training_and_validation_summary.json"))
    parser.add_argument("--recording-ingest-summary", type=Path, default=Path("artifacts/real_skeleton_v4_recording_ingest_20260612/recording_ingest_summary.json"))
    parser.add_argument("--min-detection-coverage", type=float, default=0.98)
    parser.add_argument("--min-frame-count", type=int, default=5)
    parser.add_argument("--min-fps", type=float, default=1.0)
    parser.add_argument("--min-width", type=int, default=1)
    parser.add_argument("--min-height", type=int, default=1)
    args = parser.parse_args(argv)

    summary = run_v4_pipeline(
        V4PipelineRunConfig(
            calibration_input_root=args.calibration_input_root,
            heldout_roots=tuple(args.heldout_root),
            expected_min_per_label=int(args.expected_min_per_label),
            intake_output_root=args.intake_output_root,
            skeleton_review_plan_output_root=args.skeleton_review_plan_output_root,
            skeleton_review_output_root=args.skeleton_review_output_root,
            seed_package_root=args.seed_package_root,
            recording_slot_audit_output_root=args.recording_slot_audit_output_root,
            mp4_preflight_output_root=args.mp4_preflight_output_root,
            dataset_generation_plan_output_root=args.dataset_generation_plan_output_root,
            dataset_output_root=args.dataset_output_root,
            base_dataset_root=args.base_dataset_root,
            training_config_path=args.training_config,
            pipeline_output_root=args.pipeline_output_root,
            v3_summary_path=args.v3_summary if args.v3_summary.exists() else None,
            recording_ingest_summary_path=args.recording_ingest_summary,
            min_detection_coverage=float(args.min_detection_coverage),
            min_frame_count=int(args.min_frame_count),
            min_fps=float(args.min_fps),
            min_width=int(args.min_width),
            min_height=int(args.min_height),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
