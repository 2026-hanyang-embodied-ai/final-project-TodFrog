"""Write a human-readable v4 readiness dashboard."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_readiness_dashboard import V4ReadinessDashboardConfig, build_v4_readiness_dashboard


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the v4 readiness dashboard."""

    parser = argparse.ArgumentParser(description="Write the v4 calibration and training readiness dashboard.")
    parser.add_argument("--calibration-root", type=Path, default=Path("D:/dataset/텀프영상/v4_calibration"))
    parser.add_argument("--heldout-root", type=Path, default=Path("D:/dataset/텀프영상/test"))
    parser.add_argument("--expected-per-label", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_readiness_dashboard_20260612"))
    parser.add_argument("--end-to-end-summary", type=Path, default=Path("artifacts/real_skeleton_v4_end_to_end_20260612/end_to_end_summary.json"))
    parser.add_argument("--recording-slot-audit-output-root", type=Path, default=Path("artifacts/real_skeleton_v4_recording_slot_audit_20260612"))
    args = parser.parse_args(argv)

    dashboard = build_v4_readiness_dashboard(
        V4ReadinessDashboardConfig(
            calibration_root=args.calibration_root,
            heldout_root=args.heldout_root,
            expected_per_label=int(args.expected_per_label),
            output_root=args.output_root,
            end_to_end_summary_path=args.end_to_end_summary,
            recording_slot_audit_output_root=args.recording_slot_audit_output_root,
        )
    )
    print(json.dumps(dashboard, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
