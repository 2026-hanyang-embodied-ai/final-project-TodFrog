"""Run the safe v4 recording staging ingest workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_ingest_runner import V4RecordingIngestConfig, run_v4_recording_ingest


def main(argv: Sequence[str] | None = None) -> int:
    """Run assignment, slot audit, and dashboard for v4 recording ingest."""

    parser = argparse.ArgumentParser(description="Run safe v4 recording ingest from staging to calibration readiness.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging")
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration")
    parser.add_argument("--heldout-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "test")
    parser.add_argument("--slot-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_recording_ingest_20260612"))
    parser.add_argument("--end-to-end-summary", type=Path, default=Path("artifacts/real_skeleton_v4_end_to_end_20260612/end_to_end_summary.json"))
    parser.add_argument("--expected-per-label", type=int, default=20)
    parser.add_argument("--execute-copy", action="store_true")
    args = parser.parse_args(argv)

    summary = run_v4_recording_ingest(
        V4RecordingIngestConfig(
            source_root=args.source_root,
            calibration_root=args.calibration_root,
            heldout_root=args.heldout_root,
            slot_manifest_path=args.slot_manifest,
            output_root=args.output_root,
            end_to_end_summary_path=args.end_to_end_summary,
            expected_per_label=int(args.expected_per_label),
            execute_copy=bool(args.execute_copy),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] not in {"recording_ingest_blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
