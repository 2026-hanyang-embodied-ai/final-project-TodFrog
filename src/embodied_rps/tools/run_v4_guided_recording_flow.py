"""Run the full v4 guided recording flow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER
from embodied_rps.v4_guided_recording_flow import V4GuidedRecordingFlowConfig, run_v4_guided_recording_flow
from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT


def main(argv: Sequence[str] | None = None) -> int:
    """Run preflight, guided recording, and postcheck."""

    parser = argparse.ArgumentParser(description="Run the safe v4 guided recording flow.")
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging")
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration")
    parser.add_argument("--heldout-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "test")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_guided_recording_flow_20260612"))
    parser.add_argument("--label", action="append", choices=REVIEW_LABEL_ORDER, default=[])
    parser.add_argument("--count-per-label", type=int, default=1)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--check-camera", action="store_true")
    parser.add_argument("--pre-roll-s", type=float, default=1.5)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--inter-clip-pause-s", type=float, default=0.0)
    parser.add_argument("--slot-manifest", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration" / "recording_slot_manifest.json")
    parser.add_argument("--end-to-end-summary", type=Path, default=Path("artifacts/real_skeleton_v4_end_to_end_20260612/end_to_end_summary.json"))
    parser.add_argument("--expected-per-label", type=int, default=20)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Do not wait for Enter before each clip.")
    args = parser.parse_args(argv)

    summary = run_v4_guided_recording_flow(
        V4GuidedRecordingFlowConfig(
            staging_root=args.staging_root,
            calibration_root=args.calibration_root,
            heldout_root=args.heldout_root,
            output_root=args.output_root,
            labels=tuple(args.label or REVIEW_LABEL_ORDER),
            count_per_label=int(args.count_per_label),
            camera_index=int(args.camera),
            check_camera=bool(args.check_camera),
            pre_roll_s=float(args.pre_roll_s),
            duration_s=float(args.duration_s),
            fps=float(args.fps),
            width=args.width,
            height=args.height,
            execute=bool(args.execute),
            assume_yes=bool(args.yes),
            inter_clip_pause_s=float(args.inter_clip_pause_s),
            slot_manifest_path=args.slot_manifest,
            end_to_end_summary_path=args.end_to_end_summary,
            expected_per_label=int(args.expected_per_label),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if summary["status"] in {"blocked_at_preflight", "postcheck_blocked", "aborted_before_recording"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
