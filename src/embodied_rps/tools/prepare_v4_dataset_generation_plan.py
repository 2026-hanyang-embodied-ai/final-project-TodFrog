"""Prepare the v4 dataset-generation readiness plan after skeleton review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v4_calibration_intake import build_v4_dataset_generation_plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate v4 skeleton-review readiness before dataset generation.")
    parser.add_argument("--skeleton-review-plan", required=True, type=Path, help="Path to skeleton_review_plan.json.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output root for dataset-generation readiness artifacts.")
    parser.add_argument("--dataset-output-root", required=True, type=Path, help="Planned v4 dataset output root.")
    parser.add_argument("--base-dataset-root", required=True, type=Path, help="Existing base skeleton shard root.")
    parser.add_argument("--calibration-seed-package-root", type=Path, default=None, help="Planned v4 calibration seed package root.")
    parser.add_argument("--review-manifest", type=Path, default=None, help="Optional explicit skeleton review manifest path.")
    parser.add_argument("--min-detection-coverage", type=float, default=0.98)
    args = parser.parse_args(argv)

    summary = build_v4_dataset_generation_plan(
        skeleton_review_plan_path=args.skeleton_review_plan,
        output_root=args.output_root,
        dataset_output_root=args.dataset_output_root,
        base_dataset_root=args.base_dataset_root,
        calibration_seed_package_root=args.calibration_seed_package_root,
        review_manifest_path=args.review_manifest,
        min_detection_coverage=float(args.min_detection_coverage),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] in {"ready_for_v4_dataset_generation", "awaiting_skeleton_review"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
