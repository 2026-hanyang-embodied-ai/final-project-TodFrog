"""Analyze saved-output disagreements between two skeleton prediction policies."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.real_skeleton_model_disagreement import (
    build_meta_selector_target_notes,
    summarize_model_disagreement,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare saved real-skeleton validation roots.")
    parser.add_argument("--baseline-root", required=True, type=Path, help="Saved validation root for the current baseline.")
    parser.add_argument("--candidate-root", required=True, type=Path, help="Saved validation root for the candidate model/policy.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output artifact directory.")
    parser.add_argument("--observation-progress", type=float, default=0.5, help="Probability-history cutoff for features.")
    args = parser.parse_args(argv)

    rows, summary = summarize_model_disagreement(
        baseline_root=args.baseline_root,
        candidate_root=args.candidate_root,
        observation_progress=args.observation_progress,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "disagreement_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_root / "clip_disagreements.csv", rows)
    (args.output_root / "meta_selector_targets.md").write_text(
        build_meta_selector_target_notes(summary, rows),
        encoding="utf-8",
    )
    return 0


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
