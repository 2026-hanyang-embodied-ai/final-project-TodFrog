"""Dry-run or apply explicit v7 segment review decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v7_rps_seed_package import apply_v7_segment_review_decisions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate or apply v7 RPS segment review decisions.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/real_skeleton_v7_rps_seed_package_20260617"),
        help="v7 review artifact root.",
    )
    parser.add_argument(
        "--decisions-csv",
        type=Path,
        default=Path("artifacts/real_skeleton_v7_rps_seed_package_20260617/segment_review_decision_template.csv"),
        help="CSV containing explicit segment review decisions.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually update segment_review_manifest.csv after the dry-run validation is clean.",
    )
    args = parser.parse_args(argv)

    summary = apply_v7_segment_review_decisions(
        output_root=args.output_root,
        decisions_csv=args.decisions_csv,
        apply=bool(args.apply),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] in {"no_review_decisions", "dry_run_ready", "applied"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
