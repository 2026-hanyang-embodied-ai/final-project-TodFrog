"""Write the v7d hard-paper and rock-wait real-seed review packet."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_real_seed_review import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PREPARATION_ROOT,
    V7DRealSeedReviewConfig,
    write_v7d_real_seed_review_packet,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write v7d blocker-focused real-seed review artifacts without approving rows.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--preparation-root", type=Path, default=DEFAULT_PREPARATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    summary = write_v7d_real_seed_review_packet(
        V7DRealSeedReviewConfig(
            project_root=args.project_root,
            preparation_root=args.preparation_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
