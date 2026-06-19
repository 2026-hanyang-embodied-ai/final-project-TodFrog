"""Validate semantic RPS pose-family ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.pose_family import load_pose_family_config, validate_pose_family_config


def main(argv: Sequence[str] | None = None) -> int:
    """Validate pose-family YAML without generating a full dataset."""

    parser = argparse.ArgumentParser(description="Validate robust RPS pose-family config.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/rps_pose_families.yaml")
    args = parser.parse_args(argv)

    config = load_pose_family_config(args.config)
    result = validate_pose_family_config(config)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
