"""Write v7e stage1 synthetic observation-ratio metrics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7e_stage1_synthetic_metrics import (
    V7EStage1SyntheticMetricsConfig,
    write_v7e_stage1_synthetic_metrics,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write v7e stage1 synthetic observation-ratio metrics.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=V7EStage1SyntheticMetricsConfig.output_root)
    parser.add_argument("--stage1-results-root", type=Path, default=V7EStage1SyntheticMetricsConfig.stage1_results_root)
    parser.add_argument("--stage2-reuse-results-root", type=Path, default=V7EStage1SyntheticMetricsConfig.stage2_reuse_results_root)
    parser.add_argument("--quality-ratio", default=V7EStage1SyntheticMetricsConfig.quality_ratio)
    parser.add_argument("--min-accuracy", type=float, default=V7EStage1SyntheticMetricsConfig.min_accuracy)
    args = parser.parse_args(argv)

    summary = write_v7e_stage1_synthetic_metrics(
        V7EStage1SyntheticMetricsConfig(
            project_root=args.project_root,
            output_root=args.output_root,
            stage1_results_root=args.stage1_results_root,
            stage2_reuse_results_root=args.stage2_reuse_results_root,
            quality_ratio=str(args.quality_ratio),
            min_accuracy=float(args.min_accuracy),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
