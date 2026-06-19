"""Create inspection-only MediaPipe hand-skeleton review artifacts for MP4 clips."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from embodied_rps.real_skeleton_review import (
    build_review_manifest,
    discover_skeleton_review_videos,
    process_review_video,
    validate_review_outputs,
    validate_skeleton_review_discovery,
    write_contact_sheet,
    write_quality_table,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the skeleton review extraction pipeline."""

    parser = argparse.ArgumentParser(description="Extract MediaPipe 21-landmark skeleton review videos from real MP4s.")
    parser.add_argument("--input-root", required=True, type=Path, help="Root containing rock/paper/scissors MP4 folders.")
    parser.add_argument("--output-root", required=True, type=Path, help="Review artifact output root.")
    parser.add_argument("--expected-count", default=15, type=int)
    parser.add_argument("--expected-per-label", default=5, type=int)
    parser.add_argument("--output-prefix", default="test", help="Stable prefix for per-clip review artifact IDs.")
    parser.add_argument("--review-stage", default="held_out_test_skeleton_review", help="Review-stage name written to manifest.json.")
    parser.add_argument(
        "--training-policy",
        default="These 15 clips remain validation-only and are not training seeds.",
        help="Dataset-use policy written to manifest.json.",
    )
    args = parser.parse_args(argv)

    cv2, mp = _load_review_dependencies()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    videos = discover_skeleton_review_videos(args.input_root, output_prefix=str(args.output_prefix))
    discovery = validate_skeleton_review_discovery(
        videos,
        expected_count=int(args.expected_count),
        expected_per_label=int(args.expected_per_label),
    )
    if not bool(discovery["passed"]):
        raise RuntimeError(f"Unexpected skeleton review input set: {discovery}")

    records: list[dict[str, object]] = []
    contact_frames = []
    for item in videos:
        record = process_review_video(cv2=cv2, mp=mp, item=item, output_root=output_root)
        contact = record.get("contact_frame")
        if contact is not None:
            contact_frames.append((str(record["video_id"]), contact))
        records.append(record)

    write_quality_table(output_root / "quality_table.csv", records)
    write_contact_sheet(cv2, output_root / "review_contact_sheet.png", contact_frames)
    validation = validate_review_outputs(records, output_root)
    manifest = build_review_manifest(
        input_root=args.input_root,
        output_root=output_root,
        records=records,
        discovery=discovery,
        validation=validation,
        review_stage=str(args.review_stage),
        training_policy=str(args.training_policy),
    )
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "video_count": manifest["video_count"], "manifest": manifest_path.as_posix()}, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "passed" else 1


def _load_review_dependencies() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import mediapipe as mp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Skeleton review requires optional dependencies: mediapipe and opencv-python.") from exc
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        raise RuntimeError("Skeleton review requires a MediaPipe build with mp.solutions.hands, such as mediapipe==0.10.21.")
    return cv2, mp


if __name__ == "__main__":
    raise SystemExit(main())
