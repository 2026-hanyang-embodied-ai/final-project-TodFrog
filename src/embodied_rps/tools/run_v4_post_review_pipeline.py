"""Advance v4 after skeleton-review approval toward dataset generation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_post_review_runner import V4PostReviewRunConfig, run_v4_post_review_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    """Run the v4 post-review pipeline."""

    parser = argparse.ArgumentParser(description="Run v4 post-review seed-package and dataset-generation gates.")
    parser.add_argument("--skeleton-review-plan", type=Path, default=Path("artifacts/real_skeleton_v4_calibration_review_plan_20260611/skeleton_review_plan.json"))
    parser.add_argument("--review-manifest", type=Path, default=Path("artifacts/real_hand_skeleton_review_v4_calibration_20260611/manifest.json"))
    parser.add_argument("--seed-package-root", type=Path, default=Path("artifacts/real_skeleton_v4_calibration_seed_package_20260612"))
    parser.add_argument("--dataset-plan-output-root", type=Path, default=Path("artifacts/real_skeleton_v4_dataset_generation_plan_20260611"))
    parser.add_argument("--dataset-output-root", type=Path, default=Path("artifacts/real_guided_three_class_wait_expanded_v4_20260611"))
    parser.add_argument("--base-dataset-root", type=Path, default=Path("artifacts/real_guided_large_sharded_20260610"))
    parser.add_argument("--pipeline-output-root", type=Path, default=Path("artifacts/real_skeleton_v4_post_review_run_20260612"))
    parser.add_argument("--min-detection-coverage", type=float, default=0.98)
    parser.add_argument("--generated-per-target", type=int, default=10000)
    parser.add_argument("--augmentation-profile", choices=("baseline", "v2_targeted", "v3_targeted"), default="v3_targeted")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--overwrite-dataset", action="store_true")
    parser.add_argument("--execute-dataset-generation", action="store_true", help="Actually write the v4 dataset after the seed package passes.")
    args = parser.parse_args(argv)

    summary = run_v4_post_review_pipeline(
        V4PostReviewRunConfig(
            skeleton_review_plan_path=args.skeleton_review_plan,
            review_manifest_path=args.review_manifest,
            seed_package_root=args.seed_package_root,
            dataset_plan_output_root=args.dataset_plan_output_root,
            dataset_output_root=args.dataset_output_root,
            base_dataset_root=args.base_dataset_root,
            pipeline_output_root=args.pipeline_output_root,
            min_detection_coverage=float(args.min_detection_coverage),
            generated_per_target=int(args.generated_per_target),
            augmentation_profile=str(args.augmentation_profile),
            seed=int(args.seed),
            overwrite_dataset=bool(args.overwrite_dataset),
            dry_run=not bool(args.execute_dataset_generation),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] in {"awaiting_approved_skeleton_review", "ready_for_v4_dataset_generation", "v4_dataset_generated"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
