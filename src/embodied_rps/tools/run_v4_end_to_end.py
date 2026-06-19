"""Run a safe v4 end-to-end pipeline sweep."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_end_to_end_orchestrator import V4EndToEndConfig, run_v4_end_to_end


def main(argv: Sequence[str] | None = None) -> int:
    """Run all safe v4 gates and report the current blocker."""

    parser = argparse.ArgumentParser(description="Run a safe v4 end-to-end pipeline sweep.")
    parser.add_argument("--calibration-input-root", type=Path, default=Path("D:/dataset/텀프영상/v4_calibration"))
    parser.add_argument("--heldout-root", type=Path, default=Path("D:/dataset/텀프영상/test"))
    parser.add_argument("--original20-root", type=Path, default=Path("D:/dataset/텀프영상"))
    parser.add_argument("--expected-min-per-label", type=int, default=20)
    parser.add_argument("--base-dataset-root", type=Path, default=Path("artifacts/real_guided_large_sharded_20260610"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/real_skeleton_three_class_wait_prediction_v4.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_end_to_end_20260612"))
    parser.add_argument("--recording-ingest-summary", type=Path, default=Path("artifacts/real_skeleton_v4_recording_ingest_20260612/recording_ingest_summary.json"))
    parser.add_argument("--execute-skeleton-review", action="store_true", help="Run MediaPipe skeleton review when the review plan is ready.")
    parser.add_argument("--execute-dataset-generation", action="store_true", help="Generate the v4 dataset after review approval.")
    parser.add_argument("--overwrite-dataset", action="store_true")
    args = parser.parse_args(argv)

    summary = run_v4_end_to_end(
        V4EndToEndConfig(
            calibration_input_root=args.calibration_input_root,
            heldout_root=args.heldout_root,
            expected_min_per_label=int(args.expected_min_per_label),
            base_dataset_root=args.base_dataset_root,
            training_config_path=args.training_config,
            output_root=args.output_root,
            original20_root=args.original20_root,
            execute_skeleton_review=bool(args.execute_skeleton_review),
            execute_dataset_generation=bool(args.execute_dataset_generation),
            overwrite_dataset=bool(args.overwrite_dataset),
            recording_ingest_summary_path=args.recording_ingest_summary,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] in {"blocked_at_current_gate", "strict_gates_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
