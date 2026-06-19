"""Write the non-mutating v7d required-role approval packet."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_required_role_approval_packet import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REVIEW_ROOT,
    DEFAULT_SHORTLIST_ROOT,
    DEFAULT_TEMPORAL_REVIEW_ROOT,
    V7DRequiredRoleApprovalPacketConfig,
    write_v7d_required_role_approval_packet,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the v7d required-role approval review packet.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--shortlist-root", type=Path, default=DEFAULT_SHORTLIST_ROOT)
    parser.add_argument("--temporal-review-root", type=Path, default=DEFAULT_TEMPORAL_REVIEW_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    summary = write_v7d_required_role_approval_packet(
        V7DRequiredRoleApprovalPacketConfig(
            project_root=args.project_root,
            shortlist_root=args.shortlist_root,
            temporal_review_root=args.temporal_review_root,
            review_root=args.review_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] in {"awaiting_manual_approval", "ready_after_manual_approval"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
