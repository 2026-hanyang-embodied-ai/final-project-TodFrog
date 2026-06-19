"""Build the v7e stage1 paper-transition rescue seed package."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7e_paper_seed_review_validator import DEFAULT_PLAN_ROOT
from embodied_rps.v7e_seed_package_builder import (
    DEFAULT_PAPER_REVIEW_VALIDATION_ROOT,
    DEFAULT_V7D_REVIEW_ROOT,
    V7ESeedPackageBuilderConfig,
    blocked_v7e_seed_package_summary,
    build_v7e_stage1_paper_transition_rescue_seed_package,
)
from embodied_rps.v7e_seed_package_preflight import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_PREFLIGHT_ROOT,
    DEFAULT_V7D_SEED_PACKAGE_ROOT,
    DEFAULT_V7E_SEED_PACKAGE_ROOT,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the approved v7e paper-expanded seed package.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--v7d-seed-package-root", type=Path, default=DEFAULT_V7D_SEED_PACKAGE_ROOT)
    parser.add_argument("--v7d-review-root", type=Path, default=DEFAULT_V7D_REVIEW_ROOT)
    parser.add_argument("--v7e-plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--paper-review-validation-root", type=Path, default=DEFAULT_PAPER_REVIEW_VALIDATION_ROOT)
    parser.add_argument("--preflight-root", type=Path, default=DEFAULT_PREFLIGHT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_V7E_SEED_PACKAGE_ROOT)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--minimum-approved-paper-seed-count", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    config = V7ESeedPackageBuilderConfig(
        project_root=args.project_root,
        v7d_seed_package_root=args.v7d_seed_package_root,
        v7d_review_root=args.v7d_review_root,
        v7e_plan_root=args.v7e_plan_root,
        paper_review_validation_root=args.paper_review_validation_root,
        output_root=args.output_root,
        preflight_root=args.preflight_root,
        sequence_length=int(args.sequence_length),
        minimum_approved_paper_seed_count=int(args.minimum_approved_paper_seed_count),
        overwrite=bool(args.overwrite),
    )
    try:
        summary = build_v7e_stage1_paper_transition_rescue_seed_package(config)
    except ValueError as exc:
        if "v7e seed-package preflight is not ready" not in str(exc):
            raise
        summary = blocked_v7e_seed_package_summary(config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
