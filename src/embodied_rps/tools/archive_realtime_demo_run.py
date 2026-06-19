"""Archive current realtime demo artifacts into a timestamped run folder."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_run_archive import RealtimeDemoRunArchiveConfig, archive_realtime_demo_run


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo run archiving."""

    parser = argparse.ArgumentParser(description="Archive current realtime demo artifacts.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/realtime_demo_run_archive_20260616"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--live-overlay-video", type=Path, default=RealtimeDemoRunArchiveConfig.live_overlay_video)
    parser.add_argument("--live-frame-log", type=Path, default=RealtimeDemoRunArchiveConfig.live_frame_log)
    parser.add_argument(
        "--live-postcapture-summary",
        type=Path,
        default=RealtimeDemoRunArchiveConfig.live_postcapture_summary,
    )
    parser.add_argument(
        "--live-composite-manifest",
        type=Path,
        default=RealtimeDemoRunArchiveConfig.live_composite_manifest,
    )
    parser.add_argument("--operator-outcome", type=Path, default=RealtimeDemoRunArchiveConfig.operator_outcome)
    parser.add_argument("--triage-summary", type=Path, default=RealtimeDemoRunArchiveConfig.triage_summary)
    parser.add_argument("--acceptance-report", type=Path, default=RealtimeDemoRunArchiveConfig.acceptance_report)
    parser.add_argument("--evidence-bundle", type=Path, default=RealtimeDemoRunArchiveConfig.evidence_bundle)
    parser.add_argument(
        "--review-packet-manifest",
        type=Path,
        default=RealtimeDemoRunArchiveConfig.review_packet_manifest,
    )
    parser.add_argument("--readiness-summary", type=Path, default=RealtimeDemoRunArchiveConfig.readiness_summary)
    parser.add_argument("--preflight-summary", type=Path, default=RealtimeDemoRunArchiveConfig.preflight_summary)
    parser.add_argument("--live-rock-retake-gate", type=Path, default=RealtimeDemoRunArchiveConfig.live_rock_retake_gate)
    args = parser.parse_args(argv)

    summary = archive_realtime_demo_run(
        RealtimeDemoRunArchiveConfig(
            output_root=args.output_root,
            run_id=args.run_id,
            live_overlay_video=args.live_overlay_video,
            live_frame_log=args.live_frame_log,
            live_postcapture_summary=args.live_postcapture_summary,
            live_composite_manifest=args.live_composite_manifest,
            operator_outcome=args.operator_outcome,
            triage_summary=args.triage_summary,
            acceptance_report=args.acceptance_report,
            evidence_bundle=args.evidence_bundle,
            review_packet_manifest=args.review_packet_manifest,
            readiness_summary=args.readiness_summary,
            preflight_summary=args.preflight_summary,
            live_rock_retake_gate=args.live_rock_retake_gate,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
