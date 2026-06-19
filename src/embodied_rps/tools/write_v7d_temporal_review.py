"""Write temporal skeleton review strips for v7d seed-required candidates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_temporal_review import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REVIEW_ROOT,
    DEFAULT_SHORTLIST_ROOT,
    V7DTemporalReviewConfig,
    write_v7d_temporal_review,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write v7d temporal skeleton review strips.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--shortlist-root", type=Path, default=DEFAULT_SHORTLIST_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--frames-per-strip", type=int, default=6)
    args = parser.parse_args(argv)

    summary = write_v7d_temporal_review(
        V7DTemporalReviewConfig(
            project_root=args.project_root,
            review_root=args.review_root,
            shortlist_root=args.shortlist_root,
            output_root=args.output_root,
            max_rows=args.max_rows,
            frames_per_strip=int(args.frames_per_strip),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "awaiting_manual_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
