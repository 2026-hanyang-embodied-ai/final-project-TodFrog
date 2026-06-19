"""Prepare or report the SCHUNK SVH Isaac import smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.schunk import load_schunk_asset_config, write_isaac_import_smoke


def main(argv: Sequence[str] | None = None) -> int:
    """Write Isaac import evidence or the exact user-assisted blocker command."""

    parser = argparse.ArgumentParser(description="Prepare SCHUNK SVH URDF-to-USD Isaac import smoke test.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/schunk_svh_asset.yaml")
    args = parser.parse_args(argv)
    result = write_isaac_import_smoke(load_schunk_asset_config(args.config))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
