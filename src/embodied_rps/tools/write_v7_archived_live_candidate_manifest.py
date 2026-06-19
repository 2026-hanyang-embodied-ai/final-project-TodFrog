"""Write v7 archived-live seed candidate status artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v7_rps_seed_package import write_v7_archived_live_candidate_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write v7 archived-live seed candidate status artifacts.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root; defaults to current directory.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/real_skeleton_v7_rps_seed_package_20260617"),
        help="v7 review artifact root.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("artifacts/realtime_demo_run_archive_20260616"),
        help="Root containing archived live demo runs.",
    )
    args = parser.parse_args(argv)
    summary = write_v7_archived_live_candidate_manifest(
        project_root=args.project_root,
        output_root=args.output_root,
        archive_root=args.archive_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
