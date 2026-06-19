"""Summarize current realtime demo readiness from existing artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_readiness import (
    RealtimeDemoReadinessConfig,
    summarize_realtime_demo_readiness,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo readiness summaries."""

    parser = argparse.ArgumentParser(description="Summarize realtime RPS demo readiness.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/realtime_demo_readiness_20260616"))
    parser.add_argument("--original20-validation-summary", type=Path, default=RealtimeDemoReadinessConfig.original20_validation_summary)
    parser.add_argument("--heldout15-validation-summary", type=Path, default=RealtimeDemoReadinessConfig.heldout15_validation_summary)
    parser.add_argument("--launch-summary", type=Path, default=RealtimeDemoReadinessConfig.launch_summary)
    parser.add_argument("--preflight-summary", type=Path, default=RealtimeDemoReadinessConfig.preflight_summary)
    parser.add_argument("--dry-run-postcapture-summary", type=Path, default=RealtimeDemoReadinessConfig.dry_run_postcapture_summary)
    parser.add_argument("--dry-run-composite-manifest", type=Path, default=RealtimeDemoReadinessConfig.dry_run_composite_manifest)
    parser.add_argument("--prelaunch-audit-summary", type=Path, default=RealtimeDemoReadinessConfig.prelaunch_audit_summary)
    parser.add_argument("--wrapper-contract-probe-summary", type=Path, default=RealtimeDemoReadinessConfig.wrapper_contract_probe_summary)
    parser.add_argument("--operator-command-audit-summary", type=Path, default=RealtimeDemoReadinessConfig.operator_command_audit_summary)
    parser.add_argument("--live-overlay-video", type=Path, default=RealtimeDemoReadinessConfig.live_overlay_video)
    parser.add_argument("--live-postcapture-summary", type=Path, default=RealtimeDemoReadinessConfig.live_postcapture_summary)
    parser.add_argument("--live-composite-manifest", type=Path, default=RealtimeDemoReadinessConfig.live_composite_manifest)
    args = parser.parse_args(argv)

    summary = summarize_realtime_demo_readiness(
        RealtimeDemoReadinessConfig(
            output_root=args.output_root,
            original20_validation_summary=args.original20_validation_summary,
            heldout15_validation_summary=args.heldout15_validation_summary,
            launch_summary=args.launch_summary,
            preflight_summary=args.preflight_summary,
            dry_run_postcapture_summary=args.dry_run_postcapture_summary,
            dry_run_composite_manifest=args.dry_run_composite_manifest,
            prelaunch_audit_summary=args.prelaunch_audit_summary,
            wrapper_contract_probe_summary=args.wrapper_contract_probe_summary,
            operator_command_audit_summary=args.operator_command_audit_summary,
            live_overlay_video=args.live_overlay_video,
            live_postcapture_summary=args.live_postcapture_summary,
            live_composite_manifest=args.live_composite_manifest,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0 if bool(summary.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
