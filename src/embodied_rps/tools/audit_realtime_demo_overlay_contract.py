"""Audit a realtime demo overlay against the required display contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_overlay_contract import (
    RealtimeDemoOverlayContractConfig,
    audit_realtime_demo_overlay_contract,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo overlay contract auditing."""

    parser = argparse.ArgumentParser(description="Audit realtime RPS demo overlay contract.")
    parser.add_argument("--overlay-video", required=True, type=Path)
    parser.add_argument("--frame-log-jsonl", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--response-prompt", default="scissors")
    parser.add_argument("--required-prompts", default="rock,paper,scissors")
    parser.add_argument("--min-detection-rate", type=float, default=0.80)
    parser.add_argument("--max-binary-latency-s", type=float, default=0.50)
    parser.add_argument("--expected-actual-gesture", default=None)
    args = parser.parse_args(argv)

    required_prompts = tuple(part.strip() for part in str(args.required_prompts).split(",") if part.strip())
    summary = audit_realtime_demo_overlay_contract(
        RealtimeDemoOverlayContractConfig(
            overlay_video=args.overlay_video,
            frame_log_jsonl=args.frame_log_jsonl,
            output_root=args.output_root,
            response_prompt=str(args.response_prompt),
            required_prompts=required_prompts,
            min_detection_rate=float(args.min_detection_rate),
            max_binary_latency_s=float(args.max_binary_latency_s),
            expected_actual_gesture=str(args.expected_actual_gesture) if args.expected_actual_gesture else None,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0 if bool(summary.get("contract_passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
