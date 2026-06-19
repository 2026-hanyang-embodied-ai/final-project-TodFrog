"""Triage a realtime demo capture from existing verification artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_triage import RealtimeDemoTriageConfig, triage_realtime_demo_capture


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo failure triage."""

    parser = argparse.ArgumentParser(description="Triage realtime RPS demo capture artifacts.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/realtime_demo_triage_20260616"))
    parser.add_argument("--readiness-summary", type=Path, default=RealtimeDemoTriageConfig.readiness_summary)
    parser.add_argument("--preflight-summary", type=Path, default=RealtimeDemoTriageConfig.preflight_summary)
    parser.add_argument("--postcapture-summary", type=Path, default=RealtimeDemoTriageConfig.postcapture_summary)
    parser.add_argument("--composite-manifest", type=Path, default=RealtimeDemoTriageConfig.composite_manifest)
    parser.add_argument("--live-rock-retake-gate", type=Path, default=RealtimeDemoTriageConfig.live_rock_retake_gate)
    args = parser.parse_args(argv)

    summary = triage_realtime_demo_capture(
        RealtimeDemoTriageConfig(
            output_root=args.output_root,
            readiness_summary=args.readiness_summary,
            preflight_summary=args.preflight_summary,
            postcapture_summary=args.postcapture_summary,
            composite_manifest=args.composite_manifest,
            live_rock_retake_gate=args.live_rock_retake_gate,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
