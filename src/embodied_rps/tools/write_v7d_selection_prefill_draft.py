"""Write the non-mutating v7d approval-selection prefill draft."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_selection_prefill_draft import (
    V7DSelectionPrefillDraftConfig,
    write_v7d_selection_prefill_draft,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a reviewer-only v7d approval-selection prefill draft.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--selection-root", type=Path, default=V7DSelectionPrefillDraftConfig.selection_root)
    parser.add_argument("--output-root", type=Path, default=V7DSelectionPrefillDraftConfig.output_root)
    args = parser.parse_args(argv)

    summary = write_v7d_selection_prefill_draft(
        V7DSelectionPrefillDraftConfig(
            project_root=args.project_root,
            selection_root=args.selection_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "review_required_before_copy" else 2


if __name__ == "__main__":
    raise SystemExit(main())
