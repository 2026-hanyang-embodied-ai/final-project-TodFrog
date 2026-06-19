"""Propose v7 archived-live overlay skeleton candidates for manual review."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7_rps_seed_package import (
    ARCHIVED_LIVE_EVIDENCE_ROLES,
    DEFAULT_ARCHIVE_ROOT,
    DEFAULT_V7_OUTPUT_ROOT,
    V7ArchivedLiveOverlayProposalConfig,
    propose_v7_archived_live_overlay_segments,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Propose archived live overlay skeleton segments for v7 manual review without approving seeds."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_V7_OUTPUT_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--run-id", action="append", dest="run_ids", default=None, help="Archived run id to process; repeatable.")
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--prefix-frames", type=int, default=24)
    parser.add_argument("--min-segment-frames", type=int, default=30)
    parser.add_argument("--min-detection-coverage", type=float, default=0.95)
    parser.add_argument(
        "--extract-missing-sidecars",
        action="store_true",
        help="Run MediaPipe on archived overlay MP4s when live_camera_skeletons.npz is missing.",
    )
    parser.add_argument(
        "--overwrite-extracted-sidecars",
        action="store_true",
        help="Overwrite existing extracted sidecar NPZ files when --extract-missing-sidecars is set.",
    )
    args = parser.parse_args(argv)

    run_ids = tuple(args.run_ids) if args.run_ids else tuple(ARCHIVED_LIVE_EVIDENCE_ROLES)
    summary = propose_v7_archived_live_overlay_segments(
        V7ArchivedLiveOverlayProposalConfig(
            project_root=args.project_root,
            output_root=args.output_root,
            archive_root=args.archive_root,
            run_ids=run_ids,
            sequence_length=args.sequence_length,
            prefix_frames=args.prefix_frames,
            min_segment_frames=args.min_segment_frames,
            min_detection_coverage=args.min_detection_coverage,
            extract_missing_sidecars=bool(args.extract_missing_sidecars),
            overwrite_extracted_sidecars=bool(args.overwrite_extracted_sidecars),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
