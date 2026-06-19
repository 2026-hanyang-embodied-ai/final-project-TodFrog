"""Evaluate a few-shot skeleton-geometry classifier from MediaPipe review JSONs."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.real_skeleton_geometry_classifier import (
    GeometryClassifierConfig,
    summarize_geometry_classifier_experiment,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a few-shot skeleton geometry classifier.")
    parser.add_argument("--train-review-root", action="append", required=True, type=Path)
    parser.add_argument("--eval-review-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--observation-progress", type=float, default=0.5)
    parser.add_argument("--confidence-temperature", type=float, default=0.5)
    args = parser.parse_args(argv)

    config = GeometryClassifierConfig(
        observation_progress=float(args.observation_progress),
        confidence_temperature=float(args.confidence_temperature),
    )
    rows, summary, classifier = summarize_geometry_classifier_experiment(
        train_roots=args.train_review_root,
        eval_root=args.eval_review_root,
        config=config,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "geometry_classifier_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "geometry_classifier_model.json").write_text(
        json.dumps(classifier.to_json_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_root / "geometry_clip_predictions.csv", rows)
    _write_markdown(args.output_root / "geometry_classifier_report.md", summary, rows)
    print(
        json.dumps(
            {
                "summary_path": (args.output_root / "geometry_classifier_summary.json").as_posix(),
                "passed_clip_count": summary["passed_clip_count"],
                "eval_clip_count": summary["eval_clip_count"],
            },
            ensure_ascii=False,
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
        "# Skeleton Geometry Classifier Report",
        "",
        f"- Train clips: `{summary['train_clip_count']}`",
        f"- Eval clips: `{summary['eval_clip_count']}`",
        f"- Passed clips: `{summary['passed_clip_count']}`",
        f"- Accuracy: `{summary['accuracy']}`",
        "",
        "## Per-Class",
        "",
    ]
    per_class = summary.get("per_class", {})
    if isinstance(per_class, dict):
        for label, values in per_class.items():
            if isinstance(values, dict):
                lines.append(
                    f"- `{label}`: `{values.get('passed_count')}/{values.get('clip_count')}` "
                    f"accuracy `{values.get('accuracy')}`"
                )
    lines.extend(["", "## Failed Clips", ""])
    failed = [row for row in rows if not bool(row.get("passed"))]
    if failed:
        for row in failed:
            lines.append(
                f"- `{row.get('clip_id')}`: true `{row.get('true_label')}`, "
                f"predicted `{row.get('predicted_label')}`"
            )
    else:
        lines.append("- None.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
