"""Write the v7d post-approval execution manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_post_approval_execution_manifest import (
    V7DPostApprovalExecutionManifestConfig,
    write_v7d_post_approval_execution_manifest,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a status-only v7d post-approval execution manifest.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--review-root", type=Path, default=V7DPostApprovalExecutionManifestConfig.review_root)
    parser.add_argument("--shortlist-root", type=Path, default=V7DPostApprovalExecutionManifestConfig.shortlist_root)
    parser.add_argument("--selection-root", type=Path, default=V7DPostApprovalExecutionManifestConfig.selection_root)
    parser.add_argument(
        "--selection-decision-materialization-root",
        type=Path,
        default=V7DPostApprovalExecutionManifestConfig.selection_decision_materialization_root,
    )
    parser.add_argument(
        "--evidence-integrity-root",
        type=Path,
        default=V7DPostApprovalExecutionManifestConfig.evidence_integrity_root,
    )
    parser.add_argument("--preflight-output-root", type=Path, default=V7DPostApprovalExecutionManifestConfig.preflight_output_root)
    parser.add_argument("--output-root", type=Path, default=V7DPostApprovalExecutionManifestConfig.output_root)
    args = parser.parse_args(argv)

    summary = write_v7d_post_approval_execution_manifest(
        V7DPostApprovalExecutionManifestConfig(
            project_root=args.project_root,
            review_root=args.review_root,
            shortlist_root=args.shortlist_root,
            selection_root=args.selection_root,
            selection_decision_materialization_root=args.selection_decision_materialization_root,
            evidence_integrity_root=args.evidence_integrity_root,
            preflight_output_root=args.preflight_output_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    status = str(summary.get("status", ""))
    if status in {"ready_for_local_seed_dataset_execution", "ready_for_local_smoke"}:
        return 0
    if status == "awaiting_manual_approval":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
