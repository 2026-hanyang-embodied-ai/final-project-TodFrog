"""Run preflight checks before executing a v4 recording session."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER
from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_preflight import V4RecordingPreflightConfig, preflight_v4_recording_session


def main(argv: Sequence[str] | None = None) -> int:
    """Run v4 recording preflight checks."""

    parser = argparse.ArgumentParser(description="Preflight a v4 recording session before opening the camera for capture.")
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging")
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration")
    parser.add_argument("--heldout-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "test")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_recording_preflight_20260612"))
    parser.add_argument("--slot-manifest", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration" / "recording_slot_manifest.json")
    parser.add_argument("--label", action="append", choices=REVIEW_LABEL_ORDER, default=[])
    parser.add_argument("--count-per-label", type=int, default=1)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--check-camera", action="store_true")
    parser.add_argument("--pre-roll-s", type=float, default=1.5)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--expected-per-label", type=int, default=20)
    args = parser.parse_args(argv)

    summary = preflight_v4_recording_session(
        V4RecordingPreflightConfig(
            staging_root=args.staging_root,
            calibration_root=args.calibration_root,
            heldout_root=args.heldout_root,
            output_root=args.output_root,
            slot_manifest_path=args.slot_manifest,
            labels=tuple(args.label or REVIEW_LABEL_ORDER),
            count_per_label=int(args.count_per_label),
            camera_index=int(args.camera),
            check_camera=bool(args.check_camera),
            pre_roll_s=float(args.pre_roll_s),
            duration_s=float(args.duration_s),
            fps=float(args.fps),
            expected_per_label=int(args.expected_per_label),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] in {"ready_for_recording", "ready_without_camera_check"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
