"""CLI for summarizing long scissors-pose collection runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.scissors_pose_collection import summarize_scissors_pose_collection


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a long scissors pose collection run.")
    parser.add_argument("--frame-log-jsonl", required=True, type=Path)
    parser.add_argument("--skeleton-npz", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overlay-video", type=Path, default=None)
    parser.add_argument("--collection-label", default="scissors")
    args = parser.parse_args(argv)

    summary = summarize_scissors_pose_collection(
        frame_log_jsonl=args.frame_log_jsonl,
        skeleton_npz=args.skeleton_npz,
        output_root=args.output_root,
        overlay_video=args.overlay_video,
        collection_label=str(args.collection_label),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
