"""Record operator manual visual review for an archived realtime demo run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_manual_review import (
    RealtimeDemoManualReviewConfig,
    record_realtime_demo_manual_review,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for recording manual review decisions."""

    parser = argparse.ArgumentParser(description="Record manual review status for an archived realtime demo run.")
    parser.add_argument("--archive-index", type=Path, default=RealtimeDemoManualReviewConfig.archive_index)
    parser.add_argument(
        "--manual-review-decisions",
        type=Path,
        default=RealtimeDemoManualReviewConfig.manual_review_decisions,
    )
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoManualReviewConfig.output_root)
    parser.add_argument("--run-id", default=None, help="Archived run ID. Defaults to the latest complete archived run.")
    parser.add_argument("--status", choices=("approved", "rejected_by_manual_review"), required=True)
    parser.add_argument("--notes", default="", help="Short manual review note.")
    args = parser.parse_args(argv)

    summary = record_realtime_demo_manual_review(
        RealtimeDemoManualReviewConfig(
            archive_index=args.archive_index,
            manual_review_decisions=args.manual_review_decisions,
            output_root=args.output_root,
            run_id=str(args.run_id) if args.run_id else None,
            status=str(args.status),
            notes=str(args.notes),
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
