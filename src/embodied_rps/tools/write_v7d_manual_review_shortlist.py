"""Write the v7d prompt-pose manual-review shortlist."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_manual_review_shortlist import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REVIEW_ROOT,
    V7DManualReviewShortlistConfig,
    write_v7d_manual_review_shortlist,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a non-mutating v7d blocker manual-review shortlist.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    summary = write_v7d_manual_review_shortlist(
        V7DManualReviewShortlistConfig(
            project_root=args.project_root,
            review_root=args.review_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "awaiting_manual_decisions" else 2


if __name__ == "__main__":
    raise SystemExit(main())
