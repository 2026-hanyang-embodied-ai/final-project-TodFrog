"""Run a guided v4 recording session."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER
from embodied_rps.v4_guided_recording_session import V4GuidedRecordingSessionConfig, run_v4_guided_recording_session
from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT


def main(argv: Sequence[str] | None = None) -> int:
    """Plan or execute a guided v4 recording session."""

    parser = argparse.ArgumentParser(description="Guide v4 staging recording one prompt at a time.")
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_guided_recording_session_20260612"))
    parser.add_argument("--label", action="append", choices=REVIEW_LABEL_ORDER, default=[])
    parser.add_argument("--count-per-label", type=int, default=1)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--pre-roll-s", type=float, default=1.5)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--prefix", default="v4")
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--inter-clip-pause-s", type=float, default=0.0)
    parser.add_argument("--slot-manifest", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration" / "recording_slot_manifest.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Do not wait for Enter before each clip.")
    args = parser.parse_args(argv)

    summary = run_v4_guided_recording_session(
        V4GuidedRecordingSessionConfig(
            staging_root=args.staging_root,
            output_root=args.output_root,
            labels=tuple(args.label or REVIEW_LABEL_ORDER),
            count_per_label=int(args.count_per_label),
            camera_index=int(args.camera),
            pre_roll_s=float(args.pre_roll_s),
            duration_s=float(args.duration_s),
            fps=float(args.fps),
            width=args.width,
            height=args.height,
            prefix=str(args.prefix),
            codec=str(args.codec),
            execute=bool(args.execute),
            assume_yes=bool(args.yes),
            inter_clip_pause_s=float(args.inter_clip_pause_s),
            slot_manifest_path=args.slot_manifest,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] not in {"aborted_before_recording"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
