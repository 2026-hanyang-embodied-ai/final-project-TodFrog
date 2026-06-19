"""Write the v7 remote TCN training handoff without launching remote commands."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7_remote_training_handoff import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REMOTE_HOST,
    DEFAULT_REMOTE_WORKSPACE,
    DEFAULT_V7_CONFIG,
    DEFAULT_V7_DATASET,
    V7RemoteTrainingHandoffConfig,
    write_v7_remote_training_handoff,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write v7 remote training handoff artifacts.")
    parser.add_argument("--training-config", type=Path, default=DEFAULT_V7_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_V7_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-workspace", default=DEFAULT_REMOTE_WORKSPACE)
    parser.add_argument("--expected-generated-per-target", type=int, default=10000)
    parser.add_argument("--branch-label", default="v7")
    parser.add_argument("--expected-augmentation-profile", default="v7_rps_pose")
    parser.add_argument("--expected-profile-metadata-key", default="v7_rps_pose_profile")
    args = parser.parse_args(argv)

    summary = write_v7_remote_training_handoff(
        V7RemoteTrainingHandoffConfig(
            training_config_path=args.training_config,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            remote_host=str(args.remote_host),
            remote_workspace=str(args.remote_workspace),
            expected_generated_per_target=int(args.expected_generated_per_target),
            branch_label=str(args.branch_label),
            expected_augmentation_profile=str(args.expected_augmentation_profile),
            expected_profile_metadata_key=str(args.expected_profile_metadata_key),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["status"] == "invalid_training_config" else 0


if __name__ == "__main__":
    raise SystemExit(main())
