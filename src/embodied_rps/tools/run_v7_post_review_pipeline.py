"""Run the gated v7 post-review seed and dataset pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v7_post_review_runner import V7PostReviewRunConfig, run_v7_post_review_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run v7 post-review seed-package and dataset-generation gates.")
    parser.add_argument("--review-root", type=Path, default=Path("artifacts/real_skeleton_v7_rps_seed_package_20260617"))
    parser.add_argument("--dataset-output-root", type=Path, default=Path("artifacts/real_guided_three_class_wait_expanded_v7_rps_pose_20260617"))
    parser.add_argument("--base-dataset-root", type=Path, default=Path("artifacts/real_guided_large_sharded_20260610"))
    parser.add_argument("--pipeline-output-root", type=Path, default=Path("artifacts/real_skeleton_v7_post_review_run_20260617"))
    parser.add_argument("--calibration-seed-package-root", type=Path, default=Path("artifacts/real_skeleton_v4_calibration_seed_package_fewshot_20260615"))
    parser.add_argument("--live-rock-seed-package-root", type=Path, default=Path("artifacts/live_rock_false_trigger_overlay_seed_20260616"))
    parser.add_argument("--generated-per-target", type=int, default=10000)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--min-length", type=int, default=48)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--base-rock-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--overwrite-dataset", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the status-only post-review audit. This is the default unless --execute-dataset-generation is set.",
    )
    parser.add_argument(
        "--execute-dataset-generation",
        action="store_true",
        help="Build the approved seed package and generate the v7 dataset after review readiness passes.",
    )
    args = parser.parse_args(argv)
    if args.dry_run and args.execute_dataset_generation:
        parser.error("--dry-run cannot be combined with --execute-dataset-generation")

    summary = run_v7_post_review_pipeline(
        V7PostReviewRunConfig(
            review_root=args.review_root,
            dataset_output_root=args.dataset_output_root,
            base_dataset_root=args.base_dataset_root,
            pipeline_output_root=args.pipeline_output_root,
            calibration_seed_package_root=args.calibration_seed_package_root,
            live_rock_seed_package_root=args.live_rock_seed_package_root,
            generated_per_target=int(args.generated_per_target),
            sequence_length=int(args.sequence_length),
            min_length=int(args.min_length),
            shard_size=int(args.shard_size),
            base_rock_stride=int(args.base_rock_stride),
            seed=int(args.seed),
            overwrite_dataset=bool(args.overwrite_dataset),
            dry_run=not bool(args.execute_dataset_generation),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] in {"awaiting_manual_segment_approval", "ready_for_v7_dataset_generation", "v7_dataset_generated"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
