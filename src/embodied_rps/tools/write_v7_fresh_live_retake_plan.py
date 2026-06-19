"""Write the status-only v7 fresh live retake plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v7_fresh_live_retake_plan import V7FreshLiveRetakePlanConfig, write_v7_fresh_live_retake_plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write v7 fresh live retake planning artifacts without running capture.")
    parser.add_argument("--seed-package-root", type=Path, default=Path("artifacts/real_skeleton_v7_rps_seed_package_20260617"))
    parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/real_guided_three_class_wait_expanded_v7_rps_pose_20260617"))
    parser.add_argument(
        "--profile-json",
        type=Path,
        default=Path("results/model_profiles/real_skeleton_three_class_wait_v7_rps_pose_tcn_ensemble.json"),
    )
    parser.add_argument("--approved-segment-replay-root", type=Path, default=Path("artifacts/real_skeleton_v7_approved_segment_replay_20260617"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v7_fresh_live_retakes_20260617"))
    parser.add_argument(
        "--strict-live-wrapper",
        type=Path,
        default=Path("artifacts/realtime_demo_launch_20260616/24_run_live_demo_operator_confirmed_strict.ps1"),
    )
    args = parser.parse_args(argv)

    summary = write_v7_fresh_live_retake_plan(
        V7FreshLiveRetakePlanConfig(
            seed_package_root=args.seed_package_root,
            dataset_root=args.dataset_root,
            profile_json_path=args.profile_json,
            approved_segment_replay_root=args.approved_segment_replay_root,
            output_root=args.output_root,
            strict_live_wrapper=args.strict_live_wrapper,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
