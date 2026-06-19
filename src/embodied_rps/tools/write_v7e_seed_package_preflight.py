"""Write the v7e seed-package preflight summary."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7e_seed_package_preflight import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_V7D_SEED_PACKAGE_ROOT,
    V7ESeedPackagePreflightConfig,
    write_v7e_seed_package_preflight,
)
from embodied_rps.v7e_paper_seed_review_validator import DEFAULT_OUTPUT_ROOT as DEFAULT_PAPER_REVIEW_VALIDATION_ROOT


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a fail-closed v7e seed-package preflight summary.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--v7d-seed-package-root", type=Path, default=DEFAULT_V7D_SEED_PACKAGE_ROOT)
    parser.add_argument("--paper-review-validation-root", type=Path, default=DEFAULT_PAPER_REVIEW_VALIDATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    summary = write_v7e_seed_package_preflight(
        V7ESeedPackagePreflightConfig(
            project_root=args.project_root,
            v7d_seed_package_root=args.v7d_seed_package_root,
            paper_review_validation_root=args.paper_review_validation_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    status = str(summary.get("status", ""))
    if status == "ready_for_v7e_seed_package_build":
        return 0
    if status.startswith("blocked_"):
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
