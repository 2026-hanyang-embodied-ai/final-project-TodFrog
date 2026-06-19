"""Aggregate supervised classifier run metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.training import write_model_comparison


def main(argv: Sequence[str] | None = None) -> int:
    """Aggregate model run directories into one comparison JSON."""

    parser = argparse.ArgumentParser(description="Evaluate saved skeleton RPS model runs.")
    parser.add_argument("--runs", required=True, type=Path, help="Path to results/model_runs")
    parser.add_argument("--out", required=True, type=Path, help="Output comparison JSON path")
    args = parser.parse_args(argv)

    comparison = write_model_comparison(args.runs, args.out)
    print(json.dumps(comparison["best_for_clear_distinction_50"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
