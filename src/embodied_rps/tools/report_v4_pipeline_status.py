"""Write a current status report for the v4 skeleton-calibration pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v4_pipeline_status import V4PipelineStatusConfig, build_v4_pipeline_status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report the current v4 skeleton-calibration pipeline gate.")
    parser.add_argument("--calibration-input-root", required=True, type=Path)
    parser.add_argument("--heldout-root", action="append", type=Path, default=[])
    parser.add_argument("--expected-min-per-label", type=int, default=20)
    parser.add_argument("--intake-manifest", type=Path, default=Path("artifacts/real_skeleton_v4_calibration_intake_20260611/intake_manifest.json"))
    parser.add_argument("--skeleton-review-plan", type=Path, default=Path("artifacts/real_skeleton_v4_calibration_review_plan_20260611/skeleton_review_plan.json"))
    parser.add_argument("--skeleton-review-manifest", type=Path, default=Path("artifacts/real_hand_skeleton_review_v4_calibration_20260611/manifest.json"))
    parser.add_argument("--seed-package-root", type=Path, default=Path("artifacts/real_skeleton_v4_calibration_seed_package_20260612"))
    parser.add_argument("--dataset-generation-plan", type=Path, default=Path("artifacts/real_skeleton_v4_dataset_generation_plan_20260611/dataset_generation_plan.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/real_guided_three_class_wait_expanded_v4_20260611"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/real_skeleton_three_class_wait_prediction_v4.yaml"))
    parser.add_argument("--recording-ingest-summary", type=Path, default=Path("artifacts/real_skeleton_v4_recording_ingest_20260612/recording_ingest_summary.json"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_pipeline_status_20260612"))
    args = parser.parse_args(argv)

    status = build_v4_pipeline_status(
        V4PipelineStatusConfig(
            calibration_input_root=args.calibration_input_root,
            heldout_roots=tuple(args.heldout_root),
            expected_min_per_label=int(args.expected_min_per_label),
            intake_manifest_path=args.intake_manifest,
            skeleton_review_plan_path=args.skeleton_review_plan,
            skeleton_review_manifest_path=args.skeleton_review_manifest,
            seed_package_root=args.seed_package_root,
            dataset_generation_plan_path=args.dataset_generation_plan,
            dataset_root=args.dataset_root,
            training_config_path=args.training_config,
            recording_ingest_summary_path=args.recording_ingest_summary,
            output_root=args.output_root,
        )
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
