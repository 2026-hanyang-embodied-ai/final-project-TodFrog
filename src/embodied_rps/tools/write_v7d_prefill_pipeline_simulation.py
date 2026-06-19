"""Write the sandbox v7d prefill pipeline simulation artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_prefill_pipeline_simulation import (
    V7DPrefillPipelineSimulationConfig,
    write_v7d_prefill_pipeline_simulation,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate copied v7d prefill selections in an isolated sandbox.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--review-root", type=Path, default=V7DPrefillPipelineSimulationConfig.review_root)
    parser.add_argument("--shortlist-root", type=Path, default=V7DPrefillPipelineSimulationConfig.shortlist_root)
    parser.add_argument("--selection-root", type=Path, default=V7DPrefillPipelineSimulationConfig.selection_root)
    parser.add_argument("--prefill-root", type=Path, default=V7DPrefillPipelineSimulationConfig.prefill_root)
    parser.add_argument("--output-root", type=Path, default=V7DPrefillPipelineSimulationConfig.output_root)
    args = parser.parse_args(argv)

    summary = write_v7d_prefill_pipeline_simulation(
        V7DPrefillPipelineSimulationConfig(
            project_root=args.project_root,
            review_root=args.review_root,
            shortlist_root=args.shortlist_root,
            selection_root=args.selection_root,
            prefill_root=args.prefill_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "sandbox_ready_for_review_decision_apply" else 2


if __name__ == "__main__":
    raise SystemExit(main())
