"""Write the non-mutating v7d manual approval selection aid."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_manual_approval_selection import (
    V7DManualApprovalSelectionConfig,
    write_v7d_manual_approval_selection,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a non-mutating v7d manual approval selection aid.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--fill-guide-root", type=Path, default=V7DManualApprovalSelectionConfig.fill_guide_root)
    parser.add_argument("--shortlist-root", type=Path, default=V7DManualApprovalSelectionConfig.shortlist_root)
    parser.add_argument("--output-root", type=Path, default=V7DManualApprovalSelectionConfig.output_root)
    args = parser.parse_args(argv)

    summary = write_v7d_manual_approval_selection(
        V7DManualApprovalSelectionConfig(
            project_root=args.project_root,
            fill_guide_root=args.fill_guide_root,
            shortlist_root=args.shortlist_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
