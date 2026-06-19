"""Prepare a v4 calibration intake manifest without consuming held-out clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v4_calibration_intake import build_v4_calibration_intake_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare v4 calibration intake and recording targets.")
    parser.add_argument("--input-root", required=True, type=Path, help="Non-held-out calibration MP4 root with rock/paper/scissors folders.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output artifact root for intake manifest and recording plan.")
    parser.add_argument(
        "--heldout-root",
        action="append",
        default=[],
        type=Path,
        help="Held-out validation root to exclude. May be passed multiple times.",
    )
    parser.add_argument("--v3-summary", type=Path, default=None, help="Optional v3 training_and_validation_summary.json.")
    parser.add_argument("--expected-min-per-label", type=int, default=10)
    parser.add_argument(
        "--allow-missing-input",
        action="store_true",
        help="Write an awaiting-data recording plan when the calibration root does not yet exist.",
    )
    args = parser.parse_args(argv)

    summary = build_v4_calibration_intake_report(
        input_root=args.input_root,
        output_root=args.output_root,
        heldout_roots=tuple(args.heldout_root),
        v3_summary_path=args.v3_summary,
        expected_min_per_label=int(args.expected_min_per_label),
        allow_missing_input=bool(args.allow_missing_input),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] in {"ready_for_skeleton_review", "awaiting_calibration_videos"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
