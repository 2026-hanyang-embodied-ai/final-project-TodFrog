"""Build an operator-facing outcome report for the realtime demo workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_operator_report import (
    RealtimeDemoOperatorReportConfig,
    build_realtime_demo_operator_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo operator outcome reporting."""

    parser = argparse.ArgumentParser(description="Build a realtime demo operator outcome report.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/realtime_demo_operator_outcome_20260616"))
    parser.add_argument("--acceptance-report", type=Path, default=RealtimeDemoOperatorReportConfig.acceptance_report)
    parser.add_argument("--triage-summary", type=Path, default=RealtimeDemoOperatorReportConfig.triage_summary)
    parser.add_argument(
        "--review-packet-manifest",
        type=Path,
        default=RealtimeDemoOperatorReportConfig.review_packet_manifest,
    )
    parser.add_argument("--readiness-summary", type=Path, default=RealtimeDemoOperatorReportConfig.readiness_summary)
    parser.add_argument(
        "--strict-exit-code",
        action="store_true",
        help="Return the report's recommended exit code instead of always returning 0.",
    )
    args = parser.parse_args(argv)

    summary = build_realtime_demo_operator_report(
        RealtimeDemoOperatorReportConfig(
            output_root=args.output_root,
            acceptance_report=args.acceptance_report,
            triage_summary=args.triage_summary,
            review_packet_manifest=args.review_packet_manifest,
            readiness_summary=args.readiness_summary,
        )
    )
    print(json.dumps(summary, indent=2))
    if args.strict_exit_code:
        return int(summary.get("recommended_exit_code", 60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
