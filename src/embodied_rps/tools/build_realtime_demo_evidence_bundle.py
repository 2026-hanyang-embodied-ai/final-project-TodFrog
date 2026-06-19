"""Build a submission-facing evidence bundle for the realtime demo."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_evidence_bundle import (
    RealtimeDemoEvidenceBundleConfig,
    build_realtime_demo_evidence_bundle,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the realtime demo evidence bundle."""

    parser = argparse.ArgumentParser(description="Build realtime RPS demo evidence bundle summary.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/realtime_demo_evidence_bundle_20260616"))
    parser.add_argument("--readiness-summary", type=Path, default=RealtimeDemoEvidenceBundleConfig.readiness_summary)
    parser.add_argument("--triage-summary", type=Path, default=RealtimeDemoEvidenceBundleConfig.triage_summary)
    parser.add_argument(
        "--dry-run-postcapture-summary",
        type=Path,
        default=RealtimeDemoEvidenceBundleConfig.dry_run_postcapture_summary,
    )
    parser.add_argument(
        "--dry-run-composite-manifest",
        type=Path,
        default=RealtimeDemoEvidenceBundleConfig.dry_run_composite_manifest,
    )
    parser.add_argument(
        "--dry-run-overlay-contract-summary",
        type=Path,
        default=RealtimeDemoEvidenceBundleConfig.dry_run_overlay_contract_summary,
    )
    parser.add_argument("--live-overlay-video", type=Path, default=RealtimeDemoEvidenceBundleConfig.live_overlay_video)
    parser.add_argument("--live-frame-log", type=Path, default=RealtimeDemoEvidenceBundleConfig.live_frame_log)
    parser.add_argument("--live-postcapture-summary", type=Path, default=RealtimeDemoEvidenceBundleConfig.live_postcapture_summary)
    parser.add_argument("--live-composite-manifest", type=Path, default=RealtimeDemoEvidenceBundleConfig.live_composite_manifest)
    parser.add_argument(
        "--live-overlay-contract-summary",
        type=Path,
        default=RealtimeDemoEvidenceBundleConfig.live_overlay_contract_summary,
    )
    parser.add_argument(
        "--live-artifact-cleanup-summary",
        type=Path,
        default=RealtimeDemoEvidenceBundleConfig.live_artifact_cleanup_summary,
    )
    args = parser.parse_args(argv)

    summary = build_realtime_demo_evidence_bundle(
        RealtimeDemoEvidenceBundleConfig(
            output_root=args.output_root,
            readiness_summary=args.readiness_summary,
            triage_summary=args.triage_summary,
            dry_run_postcapture_summary=args.dry_run_postcapture_summary,
            dry_run_composite_manifest=args.dry_run_composite_manifest,
            dry_run_overlay_contract_summary=args.dry_run_overlay_contract_summary,
            live_overlay_video=args.live_overlay_video,
            live_frame_log=args.live_frame_log,
            live_postcapture_summary=args.live_postcapture_summary,
            live_composite_manifest=args.live_composite_manifest,
            live_overlay_contract_summary=args.live_overlay_contract_summary,
            live_artifact_cleanup_summary=args.live_artifact_cleanup_summary,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
