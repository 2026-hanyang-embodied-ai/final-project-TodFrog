"""Validate v7e paper seed review decisions without applying them."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7e_paper_seed_review_validator import (
    DEFAULT_MINIMUM_APPROVED_PAPER_SEED_COUNT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PLAN_ROOT,
    V7EPaperSeedReviewValidatorConfig,
    validate_v7e_paper_seed_review,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate v7e paper seed review decisions without mutation.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--decisions-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--minimum-approved-paper-seed-count", type=int, default=DEFAULT_MINIMUM_APPROVED_PAPER_SEED_COUNT)
    args = parser.parse_args(argv)

    summary = validate_v7e_paper_seed_review(
        V7EPaperSeedReviewValidatorConfig(
            project_root=args.project_root,
            plan_root=args.plan_root,
            decisions_csv=args.decisions_csv,
            output_root=args.output_root,
            minimum_approved_paper_seed_count=int(args.minimum_approved_paper_seed_count),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    status = str(summary.get("status", ""))
    if status == "ready_for_v7e_seed_package_inputs":
        return 0
    if status in {"no_review_decisions", "approval_notes_missing", "insufficient_approved_paper_seeds"}:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
