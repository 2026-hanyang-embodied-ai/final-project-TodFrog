"""Report the v4 training and strict-validation gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_training_gate_runner import V4TrainingGateConfig, run_v4_training_gate


def main(argv: Sequence[str] | None = None) -> int:
    """Run the v4 training gate reporter."""

    parser = argparse.ArgumentParser(description="Report v4 training and strict real-video validation readiness.")
    parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/real_guided_three_class_wait_expanded_v4_20260611"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/real_skeleton_three_class_wait_prediction_v4.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_training_gate_20260612"))
    parser.add_argument("--original20-root", type=Path, default=Path("D:/dataset/텀프영상"))
    parser.add_argument("--heldout15-root", type=Path, default=Path("D:/dataset/텀프영상/test"))
    parser.add_argument("--profile-json", type=Path, default=Path("results/model_profiles/real_skeleton_three_class_wait_v4.json"))
    parser.add_argument("--original20-validation-root", type=Path, default=Path("artifacts/real_mp4_prediction_validation_original20_v4_20260612"))
    parser.add_argument("--heldout15-validation-root", type=Path, default=Path("artifacts/real_mp4_prediction_validation_new15_v4_20260612"))
    parser.add_argument("--event-manifest", type=Path, default=Path("artifacts/real_skeleton_schunk_events_v4_20260612/events.jsonl"))
    parser.add_argument("--model", choices=("gru", "tcn", "transformer", "all"), default="all")
    args = parser.parse_args(argv)

    summary = run_v4_training_gate(
        V4TrainingGateConfig(
            dataset_root=args.dataset_root,
            training_config_path=args.training_config,
            output_root=args.output_root,
            original20_root=args.original20_root,
            heldout15_root=args.heldout15_root,
            profile_json_path=args.profile_json,
            original20_validation_root=args.original20_validation_root,
            heldout15_validation_root=args.heldout15_validation_root,
            event_manifest_path=args.event_manifest,
            model=str(args.model),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] in {"awaiting_v4_dataset", "ready_for_v4_training", "ready_for_strict_video_validation", "strict_gates_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
