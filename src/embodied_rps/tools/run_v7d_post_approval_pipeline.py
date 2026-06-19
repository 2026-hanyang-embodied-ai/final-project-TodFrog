"""Run the fail-closed v7d post-approval local dataset pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_post_approval_pipeline import (
    V7DPostApprovalPipelineConfig,
    run_v7d_post_approval_pipeline,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fail-closed v7d post-approval local pipeline.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--review-root", type=Path, default=V7DPostApprovalPipelineConfig.review_root)
    parser.add_argument("--shortlist-root", type=Path, default=V7DPostApprovalPipelineConfig.shortlist_root)
    parser.add_argument("--selection-root", type=Path, default=V7DPostApprovalPipelineConfig.selection_root)
    parser.add_argument(
        "--selection-decision-materialization-root",
        type=Path,
        default=V7DPostApprovalPipelineConfig.selection_decision_materialization_root,
    )
    parser.add_argument("--output-root", type=Path, default=V7DPostApprovalPipelineConfig.output_root)
    parser.add_argument("--preflight-output-root", type=Path, default=V7DPostApprovalPipelineConfig.preflight_output_root)
    parser.add_argument("--decisions-csv", type=Path, default=V7DPostApprovalPipelineConfig.decisions_csv)
    parser.add_argument("--review-decision-mode", choices=("none", "dry-run", "apply"), default="none")
    parser.add_argument("--apply-confirmation", default="")
    parser.add_argument("--materialize-selection-decisions", action="store_true")
    parser.add_argument("--execute-local", action="store_true")
    parser.add_argument("--overwrite-outputs", action="store_true")
    args = parser.parse_args(argv)

    summary = run_v7d_post_approval_pipeline(
        V7DPostApprovalPipelineConfig(
            project_root=args.project_root,
            review_root=args.review_root,
            shortlist_root=args.shortlist_root,
            selection_root=args.selection_root,
            selection_decision_materialization_root=args.selection_decision_materialization_root,
            output_root=args.output_root,
            preflight_output_root=args.preflight_output_root,
            decisions_csv=args.decisions_csv,
            review_decision_mode=args.review_decision_mode,
            apply_confirmation=str(args.apply_confirmation),
            materialize_selection_decisions=bool(args.materialize_selection_decisions),
            execute_local=bool(args.execute_local),
            overwrite_outputs=bool(args.overwrite_outputs),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    status = str(summary.get("status", ""))
    if status in {"ready_for_review_decision_apply", "ready_for_local_pipeline_execution", "local_v7d_datasets_ready"}:
        return 0
    if status in {"blocked_manual_approval_required", "selection_decision_materialization_blocked", "apply_confirmation_required"}:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
