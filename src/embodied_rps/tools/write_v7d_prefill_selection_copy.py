"""Write or apply the guarded v7d prefill selection copy plan."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_prefill_selection_copy import (
    V7DPrefillSelectionCopyConfig,
    write_v7d_prefill_selection_copy_plan,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a guarded v7d prefill-to-selection copy plan.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--selection-root", type=Path, default=V7DPrefillSelectionCopyConfig.selection_root)
    parser.add_argument("--prefill-root", type=Path, default=V7DPrefillSelectionCopyConfig.prefill_root)
    parser.add_argument("--output-root", type=Path, default=V7DPrefillSelectionCopyConfig.output_root)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--reviewer-confirmation", default="")
    args = parser.parse_args(argv)

    summary = write_v7d_prefill_selection_copy_plan(
        V7DPrefillSelectionCopyConfig(
            project_root=args.project_root,
            selection_root=args.selection_root,
            prefill_root=args.prefill_root,
            output_root=args.output_root,
            apply=bool(args.apply),
            reviewer_confirmation=str(args.reviewer_confirmation),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") in {"ready_for_manual_copy_or_apply", "applied_to_selection_template"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
