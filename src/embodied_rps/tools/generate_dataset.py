"""Generate the synthetic skeleton RPS training dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from embodied_rps.config import load_kinematic_config
from embodied_rps.dataset import generate_synthetic_dataset, load_synthetic_dataset_config, save_synthetic_dataset


def main(argv: Sequence[str] | None = None) -> int:
    """Generate and save a synthetic dataset from YAML config."""

    parser = argparse.ArgumentParser(description="Generate synthetic skeleton RPS trajectories.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/dataset_synthetic.yaml")
    parser.add_argument(
        "--hand-config",
        default=Path("configs/kinematic_rps.yaml"),
        type=Path,
        help="Path to the hand pose config.",
    )
    args = parser.parse_args(argv)

    dataset_config = load_synthetic_dataset_config(args.config)
    hand_config = load_kinematic_config(args.hand_config)
    dataset = generate_synthetic_dataset(dataset_config, hand_config)
    save_synthetic_dataset(dataset, dataset_config)
    print(
        "Generated dataset: "
        f"samples={dataset.labels.shape[0]} "
        f"shape={tuple(dataset.sequences.shape)} "
        f"out={dataset_config.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
