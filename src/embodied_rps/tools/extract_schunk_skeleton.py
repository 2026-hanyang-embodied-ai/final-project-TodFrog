"""Extract the robot-native SCHUNK SVH skeleton schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.schunk import load_schunk_asset_config, write_skeleton_schema


def main(argv: Sequence[str] | None = None) -> int:
    """Write the SCHUNK skeleton schema JSON artifact."""

    parser = argparse.ArgumentParser(description="Extract SCHUNK SVH robot-native skeleton schema.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/schunk_svh_asset.yaml")
    args = parser.parse_args(argv)
    schema = write_skeleton_schema(load_schunk_asset_config(args.config))
    print(json.dumps(schema, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
