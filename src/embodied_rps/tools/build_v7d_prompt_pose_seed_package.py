"""Build the approved v7d prompt-pose seed package."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_seed_package import (
    DEFAULT_OUTPUT_ROOT,
    V7DSeedPackageConfig,
    build_v7d_prompt_pose_seed_package,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build v7d prompt-pose seed package after manual approval.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--review-root",
        type=Path,
        default=Path("artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = build_v7d_prompt_pose_seed_package(
            V7DSeedPackageConfig(
                project_root=args.project_root,
                review_root=args.review_root,
                output_root=args.output_root,
                sequence_length=int(args.sequence_length),
                overwrite=bool(args.overwrite),
            )
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        summary = {
            "status": "blocked_v7d_seed_package_not_built",
            "branch_label": "v7d_real_seeded_two_stage_prompt_window_guard",
            "review_root": args.review_root.as_posix(),
            "output_root": args.output_root.as_posix(),
            "error": str(exc),
            "training_started": False,
            "dataset_generated": False,
            "seed_package_created": False,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
