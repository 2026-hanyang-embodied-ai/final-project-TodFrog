"""Write the v7e stage1 paper-transition rescue diagnostic plan."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7e_stage1_paper_transition_rescue import (
    V7EStage1PaperTransitionRescueConfig,
    write_v7e_stage1_paper_transition_rescue_plan,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the v7e stage1 paper-transition rescue plan.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--v7d-original20-validation-root",
        type=Path,
        default=V7EStage1PaperTransitionRescueConfig.v7d_original20_validation_root,
    )
    parser.add_argument(
        "--policy-probe-original20-root",
        type=Path,
        default=V7EStage1PaperTransitionRescueConfig.policy_probe_original20_root,
    )
    parser.add_argument("--temporal-review-root", type=Path, default=V7EStage1PaperTransitionRescueConfig.temporal_review_root)
    parser.add_argument(
        "--prompt-pose-collection-review-root",
        type=Path,
        default=V7EStage1PaperTransitionRescueConfig.prompt_pose_collection_review_root,
    )
    parser.add_argument("--v7d-selection-root", type=Path, default=V7EStage1PaperTransitionRescueConfig.v7d_selection_root)
    parser.add_argument("--output-root", type=Path, default=V7EStage1PaperTransitionRescueConfig.output_root)
    parser.add_argument(
        "--recommended-additional-paper-seed-count",
        type=int,
        default=V7EStage1PaperTransitionRescueConfig.recommended_additional_paper_seed_count,
    )
    args = parser.parse_args(argv)

    summary = write_v7e_stage1_paper_transition_rescue_plan(
        V7EStage1PaperTransitionRescueConfig(
            project_root=args.project_root,
            v7d_original20_validation_root=args.v7d_original20_validation_root,
            policy_probe_original20_root=args.policy_probe_original20_root,
            temporal_review_root=args.temporal_review_root,
            prompt_pose_collection_review_root=args.prompt_pose_collection_review_root,
            v7d_selection_root=args.v7d_selection_root,
            output_root=args.output_root,
            recommended_additional_paper_seed_count=int(args.recommended_additional_paper_seed_count),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") in {"ready_for_v7e_paper_seed_review", "ready_for_heldout_policy_probe_before_training"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
