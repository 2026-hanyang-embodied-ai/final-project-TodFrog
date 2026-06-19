"""Write a non-mutating v7d decision CSV from a filled manual selection sheet."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_selection_decision_materializer import (
    V7DSelectionDecisionMaterializerConfig,
    write_v7d_selection_decision_materialization,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize a filled v7d selection sheet into a decision CSV.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--review-root", type=Path, default=V7DSelectionDecisionMaterializerConfig.review_root)
    parser.add_argument("--selection-root", type=Path, default=V7DSelectionDecisionMaterializerConfig.selection_root)
    parser.add_argument("--shortlist-root", type=Path, default=V7DSelectionDecisionMaterializerConfig.shortlist_root)
    parser.add_argument("--output-root", type=Path, default=V7DSelectionDecisionMaterializerConfig.output_root)
    args = parser.parse_args(argv)

    summary = write_v7d_selection_decision_materialization(
        V7DSelectionDecisionMaterializerConfig(
            project_root=args.project_root,
            review_root=args.review_root,
            selection_root=args.selection_root,
            shortlist_root=args.shortlist_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "ready_for_review_decision_apply" else 2


if __name__ == "__main__":
    raise SystemExit(main())
