"""Generate a robust multi-person RPS pose-family dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from embodied_rps.pose_family import generate_pose_family_dataset, load_pose_family_config, save_pose_family_dataset


def main(argv: Sequence[str] | None = None) -> int:
    """Generate and save the pose-family dataset."""

    parser = argparse.ArgumentParser(description="Generate robust RPS pose-family skeleton trajectories.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/rps_pose_families.yaml")
    args = parser.parse_args(argv)

    config = load_pose_family_config(args.config)
    dataset = generate_pose_family_dataset(config)
    save_pose_family_dataset(dataset, config)
    print(
        "Generated pose-family dataset: "
        f"samples={dataset.labels.shape[0]} "
        f"shape={tuple(dataset.sequences.shape)} "
        f"out={config.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
