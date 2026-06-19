"""Build diagnostics and optional seed package for a live rock false-trigger run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_live_rock_diagnostics import (
    LiveRockDiagnosticsConfig,
    build_live_rock_false_trigger_diagnostics,
    build_overlay_derived_rock_seed_package,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build live rock false-trigger diagnostics.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-log", type=Path, required=True)
    parser.add_argument("--postcapture-summary", type=Path, required=True)
    parser.add_argument("--archive-run-id", default="run_20260616_163603")
    parser.add_argument("--review-json", type=Path, default=None)
    parser.add_argument("--seed-output-root", type=Path, default=None)
    parser.add_argument("--segment-length", type=int, default=72)
    parser.add_argument("--stride", type=int, default=18)
    parser.add_argument("--min-detection-coverage", type=float, default=0.95)
    args = parser.parse_args(argv)

    summary = build_live_rock_false_trigger_diagnostics(
        LiveRockDiagnosticsConfig(
            output_root=args.output_root,
            frame_log=args.frame_log,
            postcapture_summary=args.postcapture_summary,
            archive_run_id=str(args.archive_run_id),
        )
    )
    if args.review_json is not None and args.seed_output_root is not None:
        summary["seed_package"] = build_overlay_derived_rock_seed_package(
            review_json=args.review_json,
            output_root=args.seed_output_root,
            segment_length=int(args.segment_length),
            stride=int(args.stride),
            min_detection_coverage=float(args.min_detection_coverage),
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
