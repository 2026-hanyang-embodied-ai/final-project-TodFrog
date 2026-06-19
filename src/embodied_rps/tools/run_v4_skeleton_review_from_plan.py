"""Run a prepared v4 skeleton-review plan."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_skeleton_review_executor import V4SkeletonReviewExecutionConfig, run_v4_skeleton_review_from_plan


def main(argv: Sequence[str] | None = None) -> int:
    """Execute v4 skeleton review artifacts from a prepared plan."""

    parser = argparse.ArgumentParser(description="Run a prepared v4 MediaPipe skeleton-review plan.")
    parser.add_argument(
        "--skeleton-review-plan",
        type=Path,
        default=Path("artifacts/real_skeleton_v4_calibration_review_plan_20260611/skeleton_review_plan.json"),
        help="Path to skeleton_review_plan.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/real_skeleton_v4_review_execution_20260612"),
        help="Execution summary output root.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate the plan and input set without running MediaPipe.")
    args = parser.parse_args(argv)

    summary = run_v4_skeleton_review_from_plan(
        V4SkeletonReviewExecutionConfig(
            skeleton_review_plan_path=args.skeleton_review_plan,
            output_root=args.output_root,
            dry_run=bool(args.dry_run),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] in {"ready_for_review_execution", "skeleton_review_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
