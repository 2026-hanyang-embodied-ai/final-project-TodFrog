"""Create the actuator-constrained SCHUNK RPS demo video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.demo_video import create_actuator_demo_video


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create actuator-constrained SCHUNK RPS demo artifacts.")
    parser.add_argument("--motion-dir", required=True, type=Path)
    parser.add_argument("--episode-log", required=True, type=Path)
    parser.add_argument("--max-win-summary", required=True, type=Path)
    parser.add_argument("--loss-free-summary", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--skip-mp4", action="store_true")
    args = parser.parse_args(argv)

    manifest = create_actuator_demo_video(
        motion_dir=args.motion_dir,
        episode_log=args.episode_log,
        max_win_summary=args.max_win_summary,
        loss_free_summary=args.loss_free_summary,
        out_dir=args.out_dir,
        fps=args.fps,
        ffmpeg_path=args.ffmpeg,
        skip_mp4=bool(args.skip_mp4),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
