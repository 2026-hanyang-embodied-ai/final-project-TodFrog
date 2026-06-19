"""Build the strict live-rock retake gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_live_rock_retake_gate import (
    RealtimeDemoLiveRockRetakeGateConfig,
    build_realtime_demo_live_rock_retake_gate,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for strict live-rock retake gate generation."""

    parser = argparse.ArgumentParser(description="Build the live rock-retake false-trigger gate.")
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoLiveRockRetakeGateConfig.output_root)
    parser.add_argument("--frame-log", type=Path, default=RealtimeDemoLiveRockRetakeGateConfig.frame_log)
    parser.add_argument(
        "--postcapture-summary",
        type=Path,
        default=RealtimeDemoLiveRockRetakeGateConfig.postcapture_summary,
    )
    parser.add_argument("--expected-actual-gesture", choices=("rock", "paper", "scissors"), default=None)
    parser.add_argument("--response-prompt", default=RealtimeDemoLiveRockRetakeGateConfig.response_prompt)
    parser.add_argument("--min-detection-rate", type=float, default=RealtimeDemoLiveRockRetakeGateConfig.min_detection_rate)
    parser.add_argument(
        "--strict-exit-code",
        action="store_true",
        help="Return 45 when an expected-rock retake has any binary false-trigger leakage.",
    )
    args = parser.parse_args(argv)

    summary = build_realtime_demo_live_rock_retake_gate(
        RealtimeDemoLiveRockRetakeGateConfig(
            output_root=args.output_root,
            frame_log=args.frame_log,
            postcapture_summary=args.postcapture_summary,
            expected_actual_gesture=args.expected_actual_gesture,
            response_prompt=str(args.response_prompt),
            min_detection_rate=float(args.min_detection_rate),
        )
    )
    print(json.dumps(summary, indent=2))
    if args.strict_exit_code and summary.get("passed") is not True:
        return 45
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
