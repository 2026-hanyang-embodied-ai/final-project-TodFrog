"""Write the sandbox v7d prefill apply-readiness simulation artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_prefill_apply_readiness_simulation import (
    V7DPrefillApplyReadinessSimulationConfig,
    write_v7d_prefill_apply_readiness_simulation,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rehearse v7d prefill approval application in an isolated sandbox.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--review-root", type=Path, default=V7DPrefillApplyReadinessSimulationConfig.review_root)
    parser.add_argument("--shortlist-root", type=Path, default=V7DPrefillApplyReadinessSimulationConfig.shortlist_root)
    parser.add_argument("--selection-root", type=Path, default=V7DPrefillApplyReadinessSimulationConfig.selection_root)
    parser.add_argument("--prefill-root", type=Path, default=V7DPrefillApplyReadinessSimulationConfig.prefill_root)
    parser.add_argument("--output-root", type=Path, default=V7DPrefillApplyReadinessSimulationConfig.output_root)
    args = parser.parse_args(argv)

    summary = write_v7d_prefill_apply_readiness_simulation(
        V7DPrefillApplyReadinessSimulationConfig(
            project_root=args.project_root,
            review_root=args.review_root,
            shortlist_root=args.shortlist_root,
            selection_root=args.selection_root,
            prefill_root=args.prefill_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "sandbox_ready_for_seed_package_build" else 2


if __name__ == "__main__":
    raise SystemExit(main())
