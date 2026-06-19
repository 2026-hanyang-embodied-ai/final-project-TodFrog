"""Write a non-mutating v7e paper seed review approval patch plan."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7e_paper_seed_review_apply_plan import (
    V7EPaperSeedReviewApplyPlanConfig,
    write_v7e_paper_seed_review_apply_plan,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a reviewable v7e paper decision-template patch without applying it.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan-root", type=Path, default=V7EPaperSeedReviewApplyPlanConfig.plan_root)
    parser.add_argument("--fill-guide-root", type=Path, default=V7EPaperSeedReviewApplyPlanConfig.fill_guide_root)
    parser.add_argument("--output-root", type=Path, default=V7EPaperSeedReviewApplyPlanConfig.output_root)
    args = parser.parse_args(argv)

    summary = write_v7e_paper_seed_review_apply_plan(
        V7EPaperSeedReviewApplyPlanConfig(
            project_root=args.project_root,
            plan_root=args.plan_root,
            fill_guide_root=args.fill_guide_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "ready_for_manual_v7e_paper_seed_patch_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
