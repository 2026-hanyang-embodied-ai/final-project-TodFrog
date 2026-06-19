"""Prepare the v4 calibration MediaPipe skeleton-review command and gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v4_calibration_intake import build_v4_skeleton_review_plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a v4 calibration skeleton-review plan.")
    parser.add_argument("--intake-manifest", required=True, type=Path, help="Path to v4 intake_manifest.json.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output root for the review-plan artifact.")
    parser.add_argument("--review-output-root", required=True, type=Path, help="Future MediaPipe review artifact root.")
    args = parser.parse_args(argv)

    summary = build_v4_skeleton_review_plan(
        intake_manifest_path=args.intake_manifest,
        output_root=args.output_root,
        review_output_root=args.review_output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
