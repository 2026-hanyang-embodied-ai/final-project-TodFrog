"""Evaluate a guarded selector between baseline and candidate saved outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.real_skeleton_guarded_selector import (
    GuardedSelectorConfig,
    summarize_guarded_selector_experiment,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate a guarded saved-output model selector.")
    parser.add_argument("--train-baseline-root", action="append", required=True, type=Path)
    parser.add_argument("--train-candidate-root", action="append", required=True, type=Path)
    parser.add_argument("--eval-baseline-root", required=True, type=Path)
    parser.add_argument("--eval-candidate-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--observation-progress", type=float, default=0.5)
    parser.add_argument("--positive-distance-multiplier", type=float, default=1.0)
    parser.add_argument("--min-positive-radius", type=float, default=0.25)
    parser.add_argument("--min-distance-margin", type=float, default=0.0)
    parser.add_argument(
        "--feature-set",
        choices=("basic", "temporal"),
        default="basic",
        help="Feature set for the saved-output guard. temporal adds max/delta probability history features.",
    )
    args = parser.parse_args(argv)

    if len(args.train_baseline_root) != len(args.train_candidate_root):
        raise ValueError("--train-baseline-root and --train-candidate-root counts must match")

    config = GuardedSelectorConfig(
        observation_progress=float(args.observation_progress),
        positive_distance_multiplier=float(args.positive_distance_multiplier),
        min_positive_radius=float(args.min_positive_radius),
        min_distance_margin=float(args.min_distance_margin),
        feature_set=args.feature_set,
    )
    train_pairs = list(zip(args.train_baseline_root, args.train_candidate_root, strict=True))
    selected_rows, summary, selector = summarize_guarded_selector_experiment(
        train_pairs=train_pairs,
        eval_baseline_root=args.eval_baseline_root,
        eval_candidate_root=args.eval_candidate_root,
        config=config,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "guarded_selector_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "guarded_selector_model.json").write_text(
        json.dumps(selector.to_json_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_root / "guarded_clip_selection.csv", selected_rows)
    _write_markdown(args.output_root / "guarded_selector_report.md", summary, selected_rows)
    print(
        json.dumps(
            {
                "summary_path": (args.output_root / "guarded_selector_summary.json").as_posix(),
                "selected_passed_clip_count": summary["selected_passed_clip_count"],
                "eval_clip_count": summary["eval_clip_count"],
            },
            indent=2,
        )
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict[str, object], rows: Sequence[dict[str, object]]) -> None:
    lines = [
        "# Guarded Model Selector Report",
        "",
        f"- Train clips: `{summary['train_clip_count']}`",
        f"- Eval clips: `{summary['eval_clip_count']}`",
        f"- Baseline passed: `{summary['baseline_passed_clip_count']}`",
        f"- Candidate passed: `{summary['candidate_passed_clip_count']}`",
        f"- Guarded selected passed: `{summary['selected_passed_clip_count']}`",
        f"- Candidate selections: `{summary['candidate_selection_count']}`",
        "",
        "## Selected Candidate Clips",
        "",
    ]
    candidate_rows = [row for row in rows if row.get("selected_source") == "candidate"]
    if candidate_rows:
        for row in candidate_rows:
            lines.append(
                f"- `{row.get('clip_id')}`: category `{row.get('disagreement_category')}`, "
                f"selected_passed `{row.get('selected_passed')}`"
            )
    else:
        lines.append("- No candidate outputs selected.")
    lines.extend(["", "## Failed Selected Clips", ""])
    failed_rows = [row for row in rows if not bool(row.get("selected_passed"))]
    if failed_rows:
        for row in failed_rows:
            lines.append(
                f"- `{row.get('clip_id')}`: selected `{row.get('selected_source')}`, "
                f"reason `{row.get('selected_failure_reason')}`"
            )
    else:
        lines.append("- None.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
