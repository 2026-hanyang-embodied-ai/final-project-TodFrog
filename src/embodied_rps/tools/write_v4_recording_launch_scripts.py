"""Write local PowerShell launch scripts for v4 recording."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_launch_scripts import V4RecordingLaunchScriptsConfig, write_v4_recording_launch_scripts


def main(argv: Sequence[str] | None = None) -> int:
    """Write the local v4 recording launch scripts."""

    parser = argparse.ArgumentParser(description="Write PowerShell launch scripts for the v4 recording workflow.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_recording_launch_20260612"))
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging")
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration")
    parser.add_argument("--heldout-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "test")
    parser.add_argument("--flow-output-root", type=Path, default=Path("artifacts/real_skeleton_v4_guided_recording_flow_20260612_camera_check"))
    parser.add_argument("--count-per-label", type=int, default=1)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--pre-roll-s", type=float, default=1.5)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--expected-per-label", type=int, default=20)
    args = parser.parse_args(argv)

    summary = write_v4_recording_launch_scripts(
        V4RecordingLaunchScriptsConfig(
            output_root=args.output_root,
            project_root=args.project_root,
            staging_root=args.staging_root,
            calibration_root=args.calibration_root,
            heldout_root=args.heldout_root,
            flow_output_root=args.flow_output_root,
            count_per_label=int(args.count_per_label),
            camera_index=int(args.camera),
            pre_roll_s=float(args.pre_roll_s),
            duration_s=float(args.duration_s),
            fps=float(args.fps),
            expected_per_label=int(args.expected_per_label),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
