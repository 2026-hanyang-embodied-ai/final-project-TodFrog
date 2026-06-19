"""Build the operator handoff card for the realtime demo."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_operator_handoff_card import (
    RealtimeDemoOperatorHandoffCardConfig,
    build_realtime_demo_operator_handoff_card,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo operator handoff card generation."""

    parser = argparse.ArgumentParser(description="Build the realtime demo operator handoff card.")
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoOperatorHandoffCardConfig.output_root)
    parser.add_argument(
        "--live-status-snapshot",
        type=Path,
        default=RealtimeDemoOperatorHandoffCardConfig.live_status_snapshot,
    )
    parser.add_argument("--final-run-card", type=Path, default=RealtimeDemoOperatorHandoffCardConfig.final_run_card)
    parser.add_argument("--launch-summary", type=Path, default=RealtimeDemoOperatorHandoffCardConfig.launch_summary)
    args = parser.parse_args(argv)

    card = build_realtime_demo_operator_handoff_card(
        RealtimeDemoOperatorHandoffCardConfig(
            output_root=args.output_root,
            live_status_snapshot=args.live_status_snapshot,
            final_run_card=args.final_run_card,
            launch_summary=args.launch_summary,
        )
    )
    print(json.dumps(card, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
