"""Write a non-mutating v7e paper seed review fill guide."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7e_paper_seed_review_fill_guide import (
    V7EPaperSeedReviewFillGuideConfig,
    write_v7e_paper_seed_review_fill_guide,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a reviewer-facing v7e paper seed fill guide without approving seeds.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan-root", type=Path, default=V7EPaperSeedReviewFillGuideConfig.plan_root)
    parser.add_argument("--output-root", type=Path, default=V7EPaperSeedReviewFillGuideConfig.output_root)
    args = parser.parse_args(argv)

    summary = write_v7e_paper_seed_review_fill_guide(
        V7EPaperSeedReviewFillGuideConfig(
            project_root=args.project_root,
            plan_root=args.plan_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "ready_for_manual_v7e_paper_seed_approval" else 2


if __name__ == "__main__":
    raise SystemExit(main())
