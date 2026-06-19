"""Validate v7d manual review decisions without applying them."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_review_decision_validator import (
    DEFAULT_DECISIONS_CSV,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REVIEW_ROOT,
    V7DReviewDecisionValidatorConfig,
    validate_v7d_review_decisions,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate v7d review decisions without mutation.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--decisions-csv", type=Path, default=DEFAULT_DECISIONS_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    summary = validate_v7d_review_decisions(
        V7DReviewDecisionValidatorConfig(
            project_root=args.project_root,
            review_root=args.review_root,
            decisions_csv=args.decisions_csv,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    status = str(summary.get("status", ""))
    if status == "ready_for_apply":
        return 0
    if status in {"no_review_decisions", "missing_required_roles"}:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
