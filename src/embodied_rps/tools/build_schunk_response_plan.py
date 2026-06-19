"""Build SCHUNK response-plan artifacts from a validated event manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_rps.schunk_event_bridge import write_schunk_response_plan_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SCHUNK response-plan metadata from skeleton prediction events.")
    parser.add_argument("--event-manifest", required=True, type=Path, help="Validated skeleton-to-SCHUNK event JSONL.")
    parser.add_argument("--pose-config", required=True, type=Path, help="SCHUNK RPS pose config YAML.")
    parser.add_argument("--output-root", required=True, type=Path, help="Directory for response-plan artifacts.")
    parser.add_argument("--expected-count", type=int, default=None, help="Optional expected event row count.")
    parser.add_argument("--response-window-s", type=float, default=1.0, help="Prompt-cycle response window in seconds.")
    parser.add_argument("--deadline-progress", type=float, default=0.50, help="Latest allowed response progress in the prompt window.")
    parser.add_argument("--wait-pose", default="paper", choices=("rock", "paper", "scissors", "neutral"), help="SCHUNK pose used before a response command.")
    args = parser.parse_args()

    summary = write_schunk_response_plan_artifacts(
        event_manifest_path=args.event_manifest,
        pose_config_path=args.pose_config,
        output_root=args.output_root,
        expected_count=args.expected_count,
        response_window_s=float(args.response_window_s),
        deadline_progress=float(args.deadline_progress),
        wait_pose=args.wait_pose,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
