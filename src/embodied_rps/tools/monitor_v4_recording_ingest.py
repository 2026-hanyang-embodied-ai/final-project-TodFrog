"""Monitor v4 recording staging and refresh ingest status on MP4 changes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_ingest_monitor import V4RecordingIngestMonitorConfig, monitor_v4_recording_ingest


def main(argv: Sequence[str] | None = None) -> int:
    """Run a bounded opt-in monitor for v4 recording ingest."""

    parser = argparse.ArgumentParser(description="Monitor v4 recording staging and refresh ingest status on MP4 changes.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging")
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration")
    parser.add_argument("--heldout-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "test")
    parser.add_argument("--slot-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_recording_ingest_monitor_20260612"))
    parser.add_argument("--ingest-output-root", type=Path, default=Path("artifacts/real_skeleton_v4_recording_ingest_20260612"))
    parser.add_argument("--end-to-end-summary", type=Path, default=Path("artifacts/real_skeleton_v4_end_to_end_20260612/end_to_end_summary.json"))
    parser.add_argument("--expected-per-label", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    args = parser.parse_args(argv)

    summary = monitor_v4_recording_ingest(
        V4RecordingIngestMonitorConfig(
            source_root=args.source_root,
            calibration_root=args.calibration_root,
            heldout_root=args.heldout_root,
            output_root=args.output_root,
            ingest_output_root=args.ingest_output_root,
            end_to_end_summary_path=args.end_to_end_summary,
            expected_per_label=int(args.expected_per_label),
            iterations=int(args.iterations),
            poll_interval_s=float(args.poll_interval_s),
            slot_manifest_path=args.slot_manifest,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] not in {"recording_ingest_blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
