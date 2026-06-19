"""Audit v7 segment review readiness without building seed data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v7_rps_seed_package import audit_v7_segment_review_readiness


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit v7 RPS segment review readiness.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/real_skeleton_v7_rps_seed_package_20260617"),
        help="v7 review artifact root.",
    )
    args = parser.parse_args(argv)

    summary = audit_v7_segment_review_readiness(output_root=args.output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] in {"awaiting_manual_segment_approval", "ready_for_seed_package_build"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
