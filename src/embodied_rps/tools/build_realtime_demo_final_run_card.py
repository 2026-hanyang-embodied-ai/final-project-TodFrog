"""Build the one-card realtime demo final run summary."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_final_run_card import (
    RealtimeDemoFinalRunCardConfig,
    build_realtime_demo_final_run_card,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo final run card generation."""

    parser = argparse.ArgumentParser(description="Build the realtime demo final run card.")
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoFinalRunCardConfig.output_root)
    parser.add_argument("--live-run-checklist", type=Path, default=RealtimeDemoFinalRunCardConfig.live_run_checklist)
    parser.add_argument("--operator-outcome", type=Path, default=RealtimeDemoFinalRunCardConfig.operator_outcome)
    parser.add_argument("--submission-packet", type=Path, default=RealtimeDemoFinalRunCardConfig.submission_packet)
    parser.add_argument("--launch-summary", type=Path, default=RealtimeDemoFinalRunCardConfig.launch_summary)
    args = parser.parse_args(argv)

    summary = build_realtime_demo_final_run_card(
        RealtimeDemoFinalRunCardConfig(
            output_root=args.output_root,
            live_run_checklist=args.live_run_checklist,
            operator_outcome=args.operator_outcome,
            submission_packet=args.submission_packet,
            launch_summary=args.launch_summary,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
