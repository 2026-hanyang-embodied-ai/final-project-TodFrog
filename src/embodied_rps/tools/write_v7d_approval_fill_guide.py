"""Write the non-mutating v7d approval fill guide."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_approval_fill_guide import (
    V7DApprovalFillGuideConfig,
    write_v7d_approval_fill_guide,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a non-mutating v7d approval fill guide.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--shortlist-root", type=Path, default=V7DApprovalFillGuideConfig.shortlist_root)
    parser.add_argument("--temporal-review-root", type=Path, default=V7DApprovalFillGuideConfig.temporal_review_root)
    parser.add_argument("--review-root", type=Path, default=V7DApprovalFillGuideConfig.review_root)
    parser.add_argument("--output-root", type=Path, default=V7DApprovalFillGuideConfig.output_root)
    parser.add_argument("--candidates-per-role", type=int, default=V7DApprovalFillGuideConfig.candidates_per_role)
    args = parser.parse_args(argv)

    summary = write_v7d_approval_fill_guide(
        V7DApprovalFillGuideConfig(
            project_root=args.project_root,
            shortlist_root=args.shortlist_root,
            temporal_review_root=args.temporal_review_root,
            review_root=args.review_root,
            output_root=args.output_root,
            candidates_per_role=int(args.candidates_per_role),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
