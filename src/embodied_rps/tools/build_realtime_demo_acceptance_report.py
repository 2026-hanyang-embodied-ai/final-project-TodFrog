"""Build a final acceptance report for realtime RPS demo evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_acceptance_report import (
    RealtimeDemoAcceptanceReportConfig,
    build_realtime_demo_acceptance_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for building the final demo acceptance report."""

    parser = argparse.ArgumentParser(description="Build realtime RPS demo acceptance report.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/realtime_demo_acceptance_report_20260616"))
    parser.add_argument("--evidence-bundle", type=Path, default=RealtimeDemoAcceptanceReportConfig.evidence_bundle)
    parser.add_argument(
        "--review-packet-manifest",
        type=Path,
        default=RealtimeDemoAcceptanceReportConfig.review_packet_manifest,
    )
    parser.add_argument(
        "--dry-run-overlay-contract-summary",
        type=Path,
        default=RealtimeDemoAcceptanceReportConfig.dry_run_overlay_contract_summary,
    )
    parser.add_argument(
        "--live-overlay-contract-summary",
        type=Path,
        default=RealtimeDemoAcceptanceReportConfig.live_overlay_contract_summary,
    )
    args = parser.parse_args(argv)

    summary = build_realtime_demo_acceptance_report(
        RealtimeDemoAcceptanceReportConfig(
            output_root=args.output_root,
            evidence_bundle=args.evidence_bundle,
            review_packet_manifest=args.review_packet_manifest,
            dry_run_overlay_contract_summary=args.dry_run_overlay_contract_summary,
            live_overlay_contract_summary=args.live_overlay_contract_summary,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
