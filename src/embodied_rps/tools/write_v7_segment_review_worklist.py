"""Write the v7 segment review worklist without approving segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v7_rps_seed_package import write_v7_segment_review_worklist


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a v7 RPS segment review worklist.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/real_skeleton_v7_rps_seed_package_20260617"),
        help="v7 review artifact root.",
    )
    args = parser.parse_args(argv)

    summary = write_v7_segment_review_worklist(output_root=args.output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
