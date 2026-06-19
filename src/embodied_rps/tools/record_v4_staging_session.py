"""Plan or execute a bounded v4 staging recording session."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER
from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_session import V4RecordingSessionConfig, run_v4_recording_session


def main(argv: Sequence[str] | None = None) -> int:
    """Plan or execute a v4 recording session."""

    parser = argparse.ArgumentParser(description="Plan or execute a bounded v4 staging recording session.")
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_recording_session_20260612"))
    parser.add_argument("--label", action="append", choices=REVIEW_LABEL_ORDER, default=[])
    parser.add_argument("--count-per-label", type=int, default=1)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--pre-roll-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--prefix", default="v4")
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--inter-clip-pause-s", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--refresh-ingest", action="store_true")
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration")
    parser.add_argument("--heldout-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "test")
    parser.add_argument("--ingest-output-root", type=Path, default=Path("artifacts/real_skeleton_v4_recording_ingest_20260612"))
    parser.add_argument("--end-to-end-summary", type=Path, default=Path("artifacts/real_skeleton_v4_end_to_end_20260612/end_to_end_summary.json"))
    parser.add_argument("--expected-per-label", type=int, default=20)
    parser.add_argument("--slot-manifest", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration" / "recording_slot_manifest.json")
    args = parser.parse_args(argv)

    summary = run_v4_recording_session(
        V4RecordingSessionConfig(
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
            inter_clip_pause_s=float(args.inter_clip_pause_s),
            refresh_ingest=bool(args.refresh_ingest),
            calibration_root=args.calibration_root,
            heldout_root=args.heldout_root,
            ingest_output_root=args.ingest_output_root,
            end_to_end_summary_path=args.end_to_end_summary,
            expected_per_label=int(args.expected_per_label),
            slot_manifest_path=args.slot_manifest,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
