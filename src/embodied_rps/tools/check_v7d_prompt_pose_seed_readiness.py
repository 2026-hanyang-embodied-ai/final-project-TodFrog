"""Check whether v7d prompt-pose seed packaging is allowed."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_seed_package import (
    DEFAULT_READINESS_ROOT,
    V7DSeedReadinessConfig,
    check_v7d_prompt_pose_seed_readiness,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check v7d prompt-pose seed-package readiness.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--review-root",
        type=Path,
        default=Path("artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_READINESS_ROOT)
    args = parser.parse_args(argv)

    summary = check_v7d_prompt_pose_seed_readiness(
        V7DSeedReadinessConfig(
            project_root=args.project_root,
            review_root=args.review_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "ready_for_v7d_seed_package" else 2


if __name__ == "__main__":
    raise SystemExit(main())
