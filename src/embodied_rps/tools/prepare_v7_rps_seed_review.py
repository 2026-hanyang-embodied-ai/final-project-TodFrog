"""Prepare v7 RPS seed inventory and segment review artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v7_rps_seed_package import (
    V7ArchivedLiveOverlayProposalConfig,
    V7SeedManifestConfig,
    V7SegmentProposalConfig,
    propose_v7_archived_live_overlay_segments,
    propose_v7_rps_segments,
    write_v7_archived_live_candidate_manifest,
    write_v7_segment_review_coverage_report,
    write_v7_seed_manifest,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare v7 RPS seed review artifacts.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root; defaults to current directory.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/real_skeleton_v7_rps_seed_package_20260617"),
        help="Review artifact output root.",
    )
    parser.add_argument("--dataset-search-root", type=Path, default=Path("D:/dataset"), help="Root used only to discover held-out test MP4s.")
    parser.add_argument(
        "--collection-root",
        type=Path,
        default=Path("artifacts/realtime_scissors_pose_collection_20260617"),
        help="Root containing realtime scissors pose collection runs.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("artifacts/realtime_demo_run_archive_20260616"),
        help="Root containing archived live demo runs.",
    )
    parser.add_argument("--transition-run-id", action="append", default=["run_20260617_150616"], help="Collection run ID to treat as rock-standby-to-scissors transition evidence.")
    parser.add_argument("--static-run-id", action="append", default=["run_20260617_150023"], help="Collection run ID to treat as static/varied scissors evidence.")
    parser.add_argument("--prefix-frames", type=int, default=24, help="Frames of standby context to include before scissors prompts.")
    parser.add_argument("--static-stride", type=int, default=144, help="Stride for static scissors proposal windows.")
    args = parser.parse_args(argv)

    project_root = args.project_root
    output_root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    collection_root = args.collection_root if args.collection_root.is_absolute() else project_root / args.collection_root
    run_ids = tuple(dict.fromkeys([*args.transition_run_id, *args.static_run_id]))
    run_roots = tuple(collection_root / run_id for run_id in run_ids if (collection_root / run_id).exists())

    manifest_summary = write_v7_seed_manifest(
        V7SeedManifestConfig(
            project_root=project_root,
            output_root=output_root,
            dataset_search_root=args.dataset_search_root,
            archive_root=args.archive_root,
            scissors_collection_root=args.collection_root,
        )
    )
    archived_summary = write_v7_archived_live_candidate_manifest(
        project_root=project_root,
        output_root=output_root,
        archive_root=args.archive_root,
    )
    proposal_summary = propose_v7_rps_segments(
        V7SegmentProposalConfig(
            run_roots=run_roots,
            output_root=output_root,
            transition_run_ids=tuple(args.transition_run_id),
            static_run_ids=tuple(args.static_run_id),
            prefix_frames=int(args.prefix_frames),
            static_stride=int(args.static_stride),
        )
    )
    archived_overlay_summary = propose_v7_archived_live_overlay_segments(
        V7ArchivedLiveOverlayProposalConfig(
            project_root=project_root,
            output_root=output_root,
            archive_root=args.archive_root,
        )
    )
    coverage_summary = write_v7_segment_review_coverage_report(output_root=output_root)
    print(
        json.dumps(
            {
                "manifest": manifest_summary,
                "archived_live_candidates": archived_summary,
                "proposals": proposal_summary,
                "archived_live_overlay_proposals": archived_overlay_summary,
                "coverage": coverage_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if manifest_summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
