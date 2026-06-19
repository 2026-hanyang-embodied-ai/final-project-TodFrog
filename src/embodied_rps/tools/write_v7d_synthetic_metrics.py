"""Write v7d synthetic observation-ratio metrics from two-stage TCN comparisons."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_synthetic_metrics import V7DSyntheticMetricsConfig, write_v7d_synthetic_metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write v7d two-stage synthetic observation-ratio metrics.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=V7DSyntheticMetricsConfig.output_root)
    parser.add_argument("--stage1-results-root", type=Path, default=V7DSyntheticMetricsConfig.stage1_results_root)
    parser.add_argument("--stage2-results-root", type=Path, default=V7DSyntheticMetricsConfig.stage2_results_root)
    parser.add_argument("--quality-ratio", default=V7DSyntheticMetricsConfig.quality_ratio)
    parser.add_argument("--min-accuracy", type=float, default=V7DSyntheticMetricsConfig.min_accuracy)
    args = parser.parse_args(argv)

    summary = write_v7d_synthetic_metrics(
        V7DSyntheticMetricsConfig(
            project_root=args.project_root,
            output_root=args.output_root,
            stage1_results_root=args.stage1_results_root,
            stage2_results_root=args.stage2_results_root,
            quality_ratio=str(args.quality_ratio),
            min_accuracy=float(args.min_accuracy),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
