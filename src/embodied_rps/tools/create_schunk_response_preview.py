"""Create a visual SCHUNK response preview from response-plan metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_rps.schunk_response_preview import create_schunk_response_preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SCHUNK response preview images from response-plan JSONL.")
    parser.add_argument("--response-plan", required=True, type=Path, help="Response-plan JSONL from build_schunk_response_plan.")
    parser.add_argument("--output-root", required=True, type=Path, help="Directory for preview artifacts.")
    parser.add_argument("--max-events", type=int, default=6, help="Maximum number of events to draw.")
    parser.add_argument("--fps", type=int, default=2, help="GIF frames per second.")
    args = parser.parse_args()

    manifest = create_schunk_response_preview(
        response_plan_jsonl=args.response_plan,
        out_dir=args.output_root,
        max_events=int(args.max_events),
        fps=int(args.fps),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
