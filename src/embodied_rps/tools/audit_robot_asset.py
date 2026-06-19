"""Audit the SCHUNK SVH dex-urdf asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.schunk import load_schunk_asset_config, write_asset_audit_outputs


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SCHUNK asset audit and write JSON evidence."""

    parser = argparse.ArgumentParser(description="Audit SCHUNK SVH URDF asset files.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/schunk_svh_asset.yaml")
    args = parser.parse_args(argv)
    audit = write_asset_audit_outputs(load_schunk_asset_config(args.config))
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
