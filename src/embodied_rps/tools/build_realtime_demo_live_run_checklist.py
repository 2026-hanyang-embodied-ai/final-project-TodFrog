"""Build the pre-run realtime demo operator checklist."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_live_run_checklist import (
    RealtimeDemoLiveRunChecklistConfig,
    build_realtime_demo_live_run_checklist,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo live-run checklist generation."""

    parser = argparse.ArgumentParser(description="Build a pre-run realtime demo operator checklist.")
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoLiveRunChecklistConfig.output_root)
    parser.add_argument("--readiness-summary", type=Path, default=RealtimeDemoLiveRunChecklistConfig.readiness_summary)
    parser.add_argument("--operator-outcome", type=Path, default=RealtimeDemoLiveRunChecklistConfig.operator_outcome)
    parser.add_argument("--preflight-summary", type=Path, default=RealtimeDemoLiveRunChecklistConfig.preflight_summary)
    parser.add_argument("--launch-summary", type=Path, default=RealtimeDemoLiveRunChecklistConfig.launch_summary)
    args = parser.parse_args(argv)

    summary = build_realtime_demo_live_run_checklist(
        RealtimeDemoLiveRunChecklistConfig(
            output_root=args.output_root,
            readiness_summary=args.readiness_summary,
            operator_outcome=args.operator_outcome,
            preflight_summary=args.preflight_summary,
            launch_summary=args.launch_summary,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
