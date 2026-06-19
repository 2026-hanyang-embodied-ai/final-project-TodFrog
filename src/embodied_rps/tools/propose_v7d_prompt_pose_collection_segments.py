"""CLI for v7d prompt-pose collection audit and segment proposal."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_prompt_pose_collection import (
    DEFAULT_AUDIT_ROOT,
    DEFAULT_COLLECTION_ROOT,
    DEFAULT_REVIEW_ROOT,
    DEFAULT_RUN_RELS,
    V7DPromptPoseCollectionConfig,
    V7DPromptPoseSegmentProposalConfig,
    audit_v7d_prompt_pose_collections,
    propose_v7d_prompt_pose_segments,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit completed prompt-pose collections and propose review-gated v7d segments."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--collection-root", type=Path, default=DEFAULT_COLLECTION_ROOT)
    parser.add_argument("--run-rel", type=Path, action="append", default=None)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--target-prompt", default="scissors")
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--prefix-frames", type=int, default=24)
    parser.add_argument("--min-segment-frames", type=int, default=30)
    parser.add_argument("--min-detection-coverage", type=float, default=0.95)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)

    run_rels = tuple(args.run_rel) if args.run_rel is not None else DEFAULT_RUN_RELS
    audit_summary = audit_v7d_prompt_pose_collections(
        V7DPromptPoseCollectionConfig(
            project_root=args.project_root,
            collection_root=args.collection_root,
            run_rels=run_rels,
            output_root=args.audit_root,
            target_prompt=args.target_prompt,
        )
    )
    if args.audit_only:
        print(json.dumps(audit_summary, ensure_ascii=False, indent=2))
        return 0
    proposal_summary = propose_v7d_prompt_pose_segments(
        V7DPromptPoseSegmentProposalConfig(
            project_root=args.project_root,
            collection_root=args.collection_root,
            run_rels=run_rels,
            output_root=args.output_root,
            target_prompt=args.target_prompt,
            sequence_length=args.sequence_length,
            prefix_frames=args.prefix_frames,
            min_segment_frames=args.min_segment_frames,
            min_detection_coverage=args.min_detection_coverage,
        )
    )
    print(json.dumps({"audit": audit_summary, "proposal": proposal_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
