"""Audit v4 calibration MP4 coverage against the recording slot manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_slot_audit import V4RecordingSlotAuditConfig, audit_v4_recording_slots


def main(argv: Sequence[str] | None = None) -> int:
    """Run the v4 recording slot audit."""

    parser = argparse.ArgumentParser(description="Audit v4 calibration MP4 coverage against planned recording slots.")
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration")
    parser.add_argument("--slot-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_recording_slot_audit_20260612"))
    args = parser.parse_args(argv)

    summary = audit_v4_recording_slots(
        V4RecordingSlotAuditConfig(
            calibration_root=args.calibration_root,
            slot_manifest_path=args.slot_manifest,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] in {"ready_for_mp4_preflight", "awaiting_recordings"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
