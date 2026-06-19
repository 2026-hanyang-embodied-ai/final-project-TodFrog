"""Run the v4 recording postcheck."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_postcheck import V4RecordingPostcheckConfig, run_v4_recording_postcheck


def main(argv: Sequence[str] | None = None) -> int:
    """Run the recording postcheck and print its summary."""

    parser = argparse.ArgumentParser(description="Check whether newly staged v4 recordings are ready for assignment review.")
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging")
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration")
    parser.add_argument("--heldout-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "test")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_recording_postcheck_20260612"))
    parser.add_argument("--slot-manifest", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration" / "recording_slot_manifest.json")
    parser.add_argument("--end-to-end-summary", type=Path, default=Path("artifacts/real_skeleton_v4_end_to_end_20260612/end_to_end_summary.json"))
    parser.add_argument("--expected-per-label", type=int, default=20)
    parser.add_argument("--expected-new-per-label", type=int, default=1)
    parser.add_argument("--session-output-root", type=Path, default=Path("artifacts/real_skeleton_v4_recording_session_20260612"))
    parser.add_argument("--pre-roll-s", type=float, default=1.5)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args(argv)

    summary = run_v4_recording_postcheck(
        V4RecordingPostcheckConfig(
            staging_root=args.staging_root,
            calibration_root=args.calibration_root,
            heldout_root=args.heldout_root,
            output_root=args.output_root,
            slot_manifest_path=args.slot_manifest,
            end_to_end_summary_path=args.end_to_end_summary,
            expected_per_label=int(args.expected_per_label),
            expected_new_per_label=int(args.expected_new_per_label),
            session_output_root=args.session_output_root,
            pre_roll_s=float(args.pre_roll_s),
            duration_s=float(args.duration_s),
            fps=float(args.fps),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if summary["status"] == "postcheck_blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
