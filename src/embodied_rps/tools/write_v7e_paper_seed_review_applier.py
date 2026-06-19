"""Dry-run or explicitly apply v7e paper seed review decisions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7e_paper_seed_review_applier import (
    V7EPaperSeedReviewApplierConfig,
    write_v7e_paper_seed_review_applier,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Guarded v7e paper seed review applier.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan-root", type=Path, default=V7EPaperSeedReviewApplierConfig.plan_root)
    parser.add_argument("--fill-guide-root", type=Path, default=V7EPaperSeedReviewApplierConfig.fill_guide_root)
    parser.add_argument("--output-root", type=Path, default=V7EPaperSeedReviewApplierConfig.output_root)
    parser.add_argument("--validator-output-root", type=Path, default=V7EPaperSeedReviewApplierConfig.validator_output_root)
    parser.add_argument(
        "--minimum-approved-paper-seed-count",
        type=int,
        default=V7EPaperSeedReviewApplierConfig.minimum_approved_paper_seed_count,
    )
    parser.add_argument("--mode", choices=("dry-run", "apply"), default="dry-run")
    parser.add_argument("--apply-confirmation", default="")
    args = parser.parse_args(argv)

    summary = write_v7e_paper_seed_review_applier(
        V7EPaperSeedReviewApplierConfig(
            project_root=args.project_root,
            plan_root=args.plan_root,
            fill_guide_root=args.fill_guide_root,
            output_root=args.output_root,
            validator_output_root=args.validator_output_root,
            minimum_approved_paper_seed_count=int(args.minimum_approved_paper_seed_count),
            mode=args.mode,
            apply_confirmation=str(args.apply_confirmation),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    status = str(summary.get("status", ""))
    if status in {"dry_run_ready_for_v7e_paper_seed_apply", "applied_ready_for_v7e_seed_package_inputs"}:
        return 0
    if status == "apply_confirmation_required":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
