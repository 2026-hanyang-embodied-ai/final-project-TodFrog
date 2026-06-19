"""Train supervised skeleton RPS classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.training import load_sweep_config, train_model_runs


def main(argv: Sequence[str] | None = None) -> int:
    """Train one model family or the full configured sweep."""

    parser = argparse.ArgumentParser(description="Train supervised skeleton RPS classifiers.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/model_sweep.yaml")
    parser.add_argument("--model", required=True, help="Model name or 'all'.")
    parser.add_argument("--smoke", action="store_true", help="Use smoke_epochs for fast verification.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on expanded runs.")
    args = parser.parse_args(argv)

    sweep_config = load_sweep_config(args.config)
    completed = train_model_runs(
        sweep_config=sweep_config,
        requested_model=args.model,
        smoke=bool(args.smoke),
        max_runs=args.max_runs,
    )
    print(json.dumps({"completed_runs": len(completed), "run_ids": [run["run_id"] for run in completed]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
