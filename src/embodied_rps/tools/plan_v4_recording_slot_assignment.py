"""Plan or execute safe copies from a staging folder into v4 recording slots."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_slot_assignment import (
    V4RecordingSlotAssignmentConfig,
    plan_v4_recording_slot_assignment,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the v4 recording slot assignment planner."""

    parser = argparse.ArgumentParser(description="Plan safe staging-MP4 copies into v4 calibration recording slots.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging")
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration")
    parser.add_argument("--slot-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_recording_slot_assignment_20260612"))
    parser.add_argument("--execute-copy", action="store_true", help="Copy files into empty slot targets after reviewing dry-run output.")
    args = parser.parse_args(argv)

    summary = plan_v4_recording_slot_assignment(
        V4RecordingSlotAssignmentConfig(
            source_root=args.source_root,
            calibration_root=args.calibration_root,
            slot_manifest_path=args.slot_manifest,
            output_root=args.output_root,
            execute_copy=bool(args.execute_copy),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] in {"assignment_ready", "partial_assignment_ready", "copy_complete", "partial_copy_complete", "no_staging_sources"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
