"""Report the v7 strict training, validation, and promotion gates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7_training_gate_runner import (
    V7TrainingGateConfig,
    discover_v7_validation_roots,
    run_v7_training_gate,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the v7 strict gate reporter."""

    parser = argparse.ArgumentParser(description="Report v7 training, validation, replay, and promotion readiness.")
    parser.add_argument("--seed-package-root", type=Path, default=Path("artifacts/real_skeleton_v7_rps_seed_package_20260617"))
    parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/real_guided_three_class_wait_expanded_v7_rps_pose_20260617"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/real_skeleton_three_class_wait_prediction_v7_rps_pose_tcn_ensemble.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v7_training_gate_20260617"))
    parser.add_argument("--profile-json", type=Path, default=Path("results/model_profiles/real_skeleton_three_class_wait_v7_rps_pose_tcn_ensemble.json"))
    parser.add_argument("--original20-validation-root", type=Path, default=Path("artifacts/real_mp4_prediction_validation_original20_v7_rps_pose_20260617"))
    parser.add_argument("--heldout15-validation-root", type=Path, default=Path("artifacts/real_mp4_prediction_validation_heldout15_v7_rps_pose_20260617"))
    parser.add_argument("--archived-live-replay-root", type=Path, default=Path("artifacts/real_skeleton_v7_archived_live_replay_20260617"))
    parser.add_argument("--approved-segment-replay-root", type=Path, default=Path("artifacts/real_skeleton_v7_approved_segment_replay_20260617"))
    parser.add_argument("--fresh-live-retake-root", type=Path, default=Path("artifacts/real_skeleton_v7_fresh_live_retakes_20260617"))
    parser.add_argument("--event-manifest", type=Path, default=Path("artifacts/real_skeleton_schunk_events_v7_rps_pose_20260617/events.jsonl"))
    parser.add_argument("--dataset-search-root", type=Path, default=Path("D:/dataset"))
    parser.add_argument("--original20-root", type=Path, default=None)
    parser.add_argument("--heldout15-root", type=Path, default=None)
    parser.add_argument("--expected-generated-per-target", type=int, default=10000)
    parser.add_argument("--branch-label", default="v7")
    parser.add_argument("--expected-augmentation-profile", default="v7_rps_pose")
    parser.add_argument("--expected-profile-metadata-key", default="v7_rps_pose_profile")
    args = parser.parse_args(argv)

    discovery = discover_v7_validation_roots(args.dataset_search_root)
    original20_root = args.original20_root
    heldout15_root = args.heldout15_root
    original20_from_discovery = False
    heldout15_from_discovery = False
    if original20_root is None and discovery.get("original20_root"):
        original20_root = Path(str(discovery["original20_root"]))
        original20_from_discovery = True
    if heldout15_root is None and discovery.get("heldout15_root"):
        heldout15_root = Path(str(discovery["heldout15_root"]))
        heldout15_from_discovery = True

    summary = run_v7_training_gate(
        V7TrainingGateConfig(
            seed_package_root=args.seed_package_root,
            dataset_root=args.dataset_root,
            training_config_path=args.training_config,
            output_root=args.output_root,
            profile_json_path=args.profile_json,
            original20_validation_root=args.original20_validation_root,
            heldout15_validation_root=args.heldout15_validation_root,
            archived_live_replay_root=args.archived_live_replay_root,
            approved_segment_replay_root=args.approved_segment_replay_root,
            fresh_live_retake_root=args.fresh_live_retake_root,
            event_manifest_path=args.event_manifest,
            original20_root=original20_root,
            heldout15_root=heldout15_root,
            validation_root_discovery=discovery,
            validation_roots_are_discovered=original20_from_discovery or heldout15_from_discovery,
            expected_generated_per_target=int(args.expected_generated_per_target),
            branch_label=str(args.branch_label),
            expected_augmentation_profile=str(args.expected_augmentation_profile),
            expected_profile_metadata_key=str(args.expected_profile_metadata_key),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] != "v7_strict_gates_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
