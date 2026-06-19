"""Build the v7 RPS seed NPZ after manual segment approval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v7_rps_seed_package import build_v7_rps_seed_package


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build approved v7 RPS seed package.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/real_skeleton_v7_rps_seed_package_20260617"),
        help="Review artifact root containing proposed_segments.jsonl and segment_review_manifest.csv.",
    )
    parser.add_argument("--sequence-length", type=int, default=72, help="Padded seed sequence length.")
    args = parser.parse_args(argv)

    summary = build_v7_rps_seed_package(output_root=args.output_root, sequence_length=int(args.sequence_length))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
