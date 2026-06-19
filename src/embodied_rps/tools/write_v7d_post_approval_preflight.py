"""Write the v7d post-approval preflight artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_post_approval_preflight import (
    V7DPostApprovalPreflightConfig,
    write_v7d_post_approval_preflight,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a fail-closed v7d post-approval preflight.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--review-root", type=Path, default=V7DPostApprovalPreflightConfig.review_root)
    parser.add_argument("--shortlist-root", type=Path, default=V7DPostApprovalPreflightConfig.shortlist_root)
    parser.add_argument("--output-root", type=Path, default=V7DPostApprovalPreflightConfig.output_root)
    args = parser.parse_args(argv)

    summary = write_v7d_post_approval_preflight(
        V7DPostApprovalPreflightConfig(
            project_root=args.project_root,
            review_root=args.review_root,
            shortlist_root=args.shortlist_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] != "blocked_manual_approval_required" else 2


if __name__ == "__main__":
    raise SystemExit(main())
