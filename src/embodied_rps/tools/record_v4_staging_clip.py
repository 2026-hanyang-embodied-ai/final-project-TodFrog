"""Record one non-held-out MP4 into the v4 staging folder."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_staging_recorder import (
    DEFAULT_V4_RECORDING_CAPTURE_ROOT,
    V4StagingRecorderConfig,
    record_v4_staging_clip,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Record or dry-run one v4 staging clip."""

    parser = argparse.ArgumentParser(description="Record one non-held-out MP4 into v4 recording staging.")
    parser.add_argument("--label", required=True, choices=("rock", "paper", "scissors"))
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_V4_RECORDING_CAPTURE_ROOT)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--pre-roll-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--filename", default=None)
    parser.add_argument("--prefix", default="v4")
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    summary = record_v4_staging_clip(
        V4StagingRecorderConfig(
            staging_root=args.staging_root,
            label=str(args.label),
            output_root=args.output_root,
            camera_index=int(args.camera),
            pre_roll_s=float(args.pre_roll_s),
            duration_s=float(args.duration_s),
            fps=float(args.fps),
            width=args.width,
            height=args.height,
            filename=args.filename,
            prefix=str(args.prefix),
            codec=str(args.codec),
            dry_run=bool(args.dry_run),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
