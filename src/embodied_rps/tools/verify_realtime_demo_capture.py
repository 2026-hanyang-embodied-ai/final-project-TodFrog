"""Verify a captured realtime demo overlay and prepare composite review frames."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_postcapture import (
    RealtimeDemoPostCaptureConfig,
    verify_realtime_demo_capture,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo post-capture verification."""

    parser = argparse.ArgumentParser(description="Verify a prompt-gated realtime demo overlay.")
    parser.add_argument("--overlay-video", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--response-preview-image", type=Path, default=None)
    parser.add_argument("--live-composite-output-root", type=Path, default=None)
    parser.add_argument("--frame-log-jsonl", type=Path, default=None)
    parser.add_argument("--response-prompt", default="scissors")
    parser.add_argument("--expected-actual-gesture", default=None)
    parser.add_argument("--expected-response-decision", default=None)
    parser.add_argument("--expected-robot-action", default=None)
    parser.add_argument("--enforce-demo-success-gate", action="store_true")
    parser.add_argument("--max-response-binary-latency-s", type=float, default=0.50)
    parser.add_argument("--min-detection-rate", type=float, default=0.80)
    parser.add_argument("--prompt-sequence", default="rock,paper,scissors")
    parser.add_argument("--prompt-cycle-s", type=float, default=1.0)
    parser.add_argument("--min-frame-count", type=int, default=30)
    parser.add_argument("--min-duration-s", type=float, default=3.0)
    args = parser.parse_args(argv)

    prompt_sequence = tuple(part.strip().lower() for part in str(args.prompt_sequence).split(",") if part.strip())
    summary = verify_realtime_demo_capture(
        RealtimeDemoPostCaptureConfig(
            overlay_video=args.overlay_video,
            output_root=args.output_root,
            response_preview_image=args.response_preview_image,
            live_composite_output_root=args.live_composite_output_root,
            frame_log_jsonl=args.frame_log_jsonl,
            response_prompt=str(args.response_prompt) if args.response_prompt else None,
            expected_actual_gesture=str(args.expected_actual_gesture) if args.expected_actual_gesture else None,
            expected_response_decision=str(args.expected_response_decision) if args.expected_response_decision else None,
            expected_robot_action=str(args.expected_robot_action) if args.expected_robot_action else None,
            enforce_demo_success_gate=bool(args.enforce_demo_success_gate),
            max_response_binary_latency_s=float(args.max_response_binary_latency_s),
            min_detection_rate=float(args.min_detection_rate),
            prompt_sequence=prompt_sequence,
            prompt_cycle_s=float(args.prompt_cycle_s),
            min_frame_count=int(args.min_frame_count),
            min_duration_s=float(args.min_duration_s),
        )
    )
    print(json.dumps(summary, indent=2))
    return 0 if bool(summary.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
