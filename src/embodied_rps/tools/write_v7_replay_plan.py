"""Write v7 archived-live and approved-segment replay planning artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7_replay_plan import V7ReplayPlanConfig, write_v7_replay_plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write v7 replay plan manifests without running inference.")
    parser.add_argument("--seed-package-root", type=Path, default=Path("artifacts/real_skeleton_v7_rps_seed_package_20260617"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v7_replay_plan_20260617"))
    parser.add_argument("--profile-json", type=Path, default=Path("results/model_profiles/real_skeleton_three_class_wait_v7_rps_pose_tcn_ensemble.json"))
    args = parser.parse_args(argv)

    summary = write_v7_replay_plan(
        V7ReplayPlanConfig(
            seed_package_root=args.seed_package_root,
            output_root=args.output_root,
            profile_json_path=args.profile_json,
            project_root=Path.cwd(),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["status"] == "invalid_replay_plan" else 0


if __name__ == "__main__":
    raise SystemExit(main())
