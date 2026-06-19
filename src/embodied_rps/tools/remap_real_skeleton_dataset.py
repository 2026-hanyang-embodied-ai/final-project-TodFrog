"""Remap real-skeleton shard labels for two-stage model experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.real_skeleton_dataset_remap import remap_real_skeleton_dataset


def main(argv: Sequence[str] | None = None) -> int:
    """Run a real-skeleton dataset label remap."""

    parser = argparse.ArgumentParser(description="Remap real skeleton dataset labels.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("rock_vs_transition", "paper_vs_scissors"))
    args = parser.parse_args(argv)
    summary = remap_real_skeleton_dataset(args.source_root, args.output_root, mode=args.mode)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
