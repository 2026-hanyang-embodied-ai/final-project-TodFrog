"""Run the bundled GRU episode confidence-threshold sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.episode_sweep import load_threshold_sweep_config, run_episode_threshold_sweep


def main(argv: Sequence[str] | None = None) -> int:
    """Run configured GRU episode thresholds and print aggregate summary."""

    parser = argparse.ArgumentParser(description="Run GRU episode policy confidence-threshold sweep.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/gru_episode_sweep.yaml")
    args = parser.parse_args(argv)

    base_config, thresholds, velocity_scales, confidence_margins, confirmation_counts, output_path, log_dir = (
        load_threshold_sweep_config(args.config)
    )
    summary = run_episode_threshold_sweep(
        base_config=base_config,
        thresholds=thresholds,
        velocity_scales=velocity_scales,
        confidence_margins=confidence_margins,
        confirmation_counts=confirmation_counts,
        output_path=output_path,
        log_dir=log_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
