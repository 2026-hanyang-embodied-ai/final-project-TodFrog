"""Audit raw v4 recording staging MP4s."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_staging_audit import V4RecordingStagingAuditConfig, audit_v4_recording_staging


def main(argv: Sequence[str] | None = None) -> int:
    """Run the v4 recording staging audit."""

    parser = argparse.ArgumentParser(description="Audit raw non-held-out v4 recording staging MP4s.")
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging")
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration")
    parser.add_argument("--heldout-root", action="append", type=Path, default=[])
    parser.add_argument("--expected-per-label", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_recording_staging_audit_20260612"))
    args = parser.parse_args(argv)

    summary = audit_v4_recording_staging(
        V4RecordingStagingAuditConfig(
            staging_root=args.staging_root,
            calibration_root=args.calibration_root,
            heldout_roots=tuple(args.heldout_root or [DEFAULT_LOCAL_DATA_ROOT / "test"]),
            expected_per_label=int(args.expected_per_label),
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] not in {"invalid_roots", "staging_needs_review"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
