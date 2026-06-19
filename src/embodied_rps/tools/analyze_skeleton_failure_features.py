"""Analyze skeleton feature signatures for real-video prediction failures."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.real_skeleton_failure_features import summarize_prediction_artifacts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze feature signatures for failed skeleton predictions.")
    parser.add_argument("--review-root", required=True, type=Path, help="Skeleton review artifact root containing landmarks_json.")
    parser.add_argument("--prediction-root", required=True, type=Path, help="Prediction validation artifact root containing clips/*/metrics.json.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output artifact directory.")
    parser.add_argument("--progress-cutoff", type=float, default=0.5, help="Early observation cutoff for feature summaries.")
    args = parser.parse_args(argv)

    rows, summary = summarize_prediction_artifacts(
        review_root=args.review_root,
        prediction_root=args.prediction_root,
        progress_cutoff=args.progress_cutoff,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "failure_feature_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_root / "clip_feature_summary.csv", rows)
    _write_markdown(args.output_root / "augmentation_targets.md", summary)
    return 0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def _write_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Skeleton Failure Feature Augmentation Targets",
        "",
        f"- Clip count: `{summary['clip_count']}`",
        f"- Failed clip count: `{summary['failed_clip_count']}`",
        f"- Missing review count: `{summary['missing_review_count']}`",
        "",
        "## Recommendation Counts",
        "",
    ]
    counts = summary.get("recommendation_counts", {})
    if isinstance(counts, dict) and counts:
        for target, count in counts.items():
            lines.append(f"- `{target}`: `{count}`")
    else:
        lines.append("- No failed clips.")
    lines.extend(["", "## Failed Clip Targets", ""])
    recommendations = summary.get("recommendations", [])
    if isinstance(recommendations, list) and recommendations:
        for item in recommendations:
            if not isinstance(item, dict):
                continue
            lines.extend(
                [
                    f"### {item.get('clip_id')}",
                    "",
                    f"- Label: `{item.get('label')}`",
                    f"- Failure reason: `{item.get('failure_reason')}`",
                    f"- Decision state: `{item.get('decision_state')}`",
                    f"- Target family: `{item.get('target_family')}`",
                    f"- Rationale: {item.get('rationale')}",
                    "",
                ]
            )
    else:
        lines.append("No failed clips.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
