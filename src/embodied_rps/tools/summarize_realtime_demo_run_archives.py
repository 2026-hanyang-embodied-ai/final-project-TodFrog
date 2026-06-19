"""Build an index over archived realtime demo rehearsal runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_run_archive_index import (
    RealtimeDemoRunArchiveIndexConfig,
    summarize_realtime_demo_run_archives,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo run archive indexing."""

    parser = argparse.ArgumentParser(description="Summarize archived realtime demo runs.")
    parser.add_argument("--archive-root", type=Path, default=RealtimeDemoRunArchiveIndexConfig.archive_root)
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoRunArchiveIndexConfig.output_root)
    parser.add_argument(
        "--manual-review-decisions",
        type=Path,
        default=RealtimeDemoRunArchiveIndexConfig.manual_review_decisions,
        help="Optional JSON file with manual review status per archived run.",
    )
    args = parser.parse_args(argv)

    summary = summarize_realtime_demo_run_archives(
        RealtimeDemoRunArchiveIndexConfig(
            archive_root=args.archive_root,
            output_root=args.output_root,
            manual_review_decisions=args.manual_review_decisions,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
