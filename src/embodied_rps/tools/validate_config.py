"""Validate the kinematic fallback configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from embodied_rps.config import load_kinematic_config


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a kinematic config and print a concise summary."""

    parser = argparse.ArgumentParser(description="Validate the kinematic RPS config.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/kinematic_rps.yaml")
    args = parser.parse_args(argv)

    config = load_kinematic_config(args.config)
    print(
        "Valid kinematic config: "
        f"{len(config.joint_names)} joints, "
        f"{len(config.gestures)} gestures, "
        f"deadline={config.deadline_s:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
