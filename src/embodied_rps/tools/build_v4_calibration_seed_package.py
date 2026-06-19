"""Build canonical v4 calibration seed package from approved review landmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v4_calibration_seed_package import V4SeedPackageConfig, build_v4_calibration_seed_package


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build v4 calibration canonical seed package.")
    parser.add_argument("--review-manifest", required=True, type=Path, help="Approved v4 skeleton review manifest.")
    parser.add_argument("--skeleton-review-plan", required=True, type=Path, help="v4 skeleton_review_plan.json.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output root for seed package artifacts.")
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--min-detection-coverage", type=float, default=0.98)
    parser.add_argument(
        "--allow-missing-review",
        action="store_true",
        help="Write an awaiting-review summary when the review manifest does not exist yet.",
    )
    args = parser.parse_args(argv)

    summary = build_v4_calibration_seed_package(
        V4SeedPackageConfig(
            review_manifest_path=args.review_manifest,
            skeleton_review_plan_path=args.skeleton_review_plan,
            output_root=args.output_root,
            sequence_length=int(args.sequence_length),
            min_detection_coverage=float(args.min_detection_coverage),
            allow_missing_review=bool(args.allow_missing_review),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] in {"passed", "awaiting_skeleton_review"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
