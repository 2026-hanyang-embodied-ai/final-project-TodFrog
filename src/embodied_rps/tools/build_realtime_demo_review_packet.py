"""Build a lightweight review packet for realtime demo artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_review_packet import (
    RealtimeDemoReviewPacketConfig,
    build_realtime_demo_review_packet,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo review packet generation."""

    parser = argparse.ArgumentParser(description="Build a lightweight realtime demo review packet.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/realtime_demo_review_packet_20260616"))
    parser.add_argument("--evidence-bundle", type=Path, default=RealtimeDemoReviewPacketConfig.evidence_bundle)
    parser.add_argument(
        "--dry-run-postcapture-summary",
        type=Path,
        default=RealtimeDemoReviewPacketConfig.dry_run_postcapture_summary,
    )
    parser.add_argument(
        "--dry-run-composite-manifest",
        type=Path,
        default=RealtimeDemoReviewPacketConfig.dry_run_composite_manifest,
    )
    parser.add_argument("--preflight-summary", type=Path, default=RealtimeDemoReviewPacketConfig.preflight_summary)
    args = parser.parse_args(argv)

    summary = build_realtime_demo_review_packet(
        RealtimeDemoReviewPacketConfig(
            output_root=args.output_root,
            evidence_bundle=args.evidence_bundle,
            dry_run_postcapture_summary=args.dry_run_postcapture_summary,
            dry_run_composite_manifest=args.dry_run_composite_manifest,
            preflight_summary=args.preflight_summary,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
