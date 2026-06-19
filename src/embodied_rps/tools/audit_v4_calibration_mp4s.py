"""Audit v4 calibration MP4s before MediaPipe skeleton review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v4_mp4_preflight import V4Mp4PreflightConfig, audit_v4_calibration_mp4s


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit v4 calibration MP4 counts and basic stream metadata.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--heldout-root", action="append", type=Path, default=[])
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_mp4_preflight_20260612"))
    parser.add_argument("--expected-min-per-label", type=int, default=20)
    parser.add_argument("--min-frame-count", type=int, default=5)
    parser.add_argument("--min-fps", type=float, default=1.0)
    parser.add_argument("--min-width", type=int, default=1)
    parser.add_argument("--min-height", type=int, default=1)
    args = parser.parse_args(argv)

    summary = audit_v4_calibration_mp4s(
        V4Mp4PreflightConfig(
            input_root=args.input_root,
            heldout_roots=tuple(args.heldout_root),
            output_root=args.output_root,
            expected_min_per_label=int(args.expected_min_per_label),
            min_frame_count=int(args.min_frame_count),
            min_fps=float(args.min_fps),
            min_width=int(args.min_width),
            min_height=int(args.min_height),
        )
    )
    print(json.dumps({key: summary[key] for key in ("status", "video_count", "label_counts", "failed_video_count")}, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
