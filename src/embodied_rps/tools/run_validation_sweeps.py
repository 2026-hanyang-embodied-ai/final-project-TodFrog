"""Run MiniRocket-style and ST-GCN validation sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.special_sweeps import load_validation_sweep_config, run_validation_sweeps


def main(argv: Sequence[str] | None = None) -> int:
    """Run configured validation sweeps and print comparison summary."""

    parser = argparse.ArgumentParser(description="Run special validation sweeps for RPS skeleton classifiers.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/validation_sweep.yaml")
    args = parser.parse_args(argv)

    summary = run_validation_sweeps(load_validation_sweep_config(args.config))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
