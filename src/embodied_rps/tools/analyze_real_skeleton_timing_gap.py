"""CLI for measuring real-vs-synthetic skeleton timing gaps."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.real_skeleton_timing_analysis import (
    analyze_timing_gap,
    expand_synthetic_metadata_paths,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze real review and synthetic skeleton opening timing.")
    parser.add_argument(
        "--review-root",
        action="append",
        required=True,
        type=Path,
        help="Skeleton review artifact root containing landmarks_json. Can be passed multiple times.",
    )
    parser.add_argument(
        "--synthetic-metadata",
        action="append",
        required=True,
        type=Path,
        help="sample_metadata.jsonl path or dataset root. Can be passed multiple times.",
    )
    parser.add_argument("--output-root", required=True, type=Path, help="Output artifact directory.")
    parser.add_argument("--threshold-fraction", type=float, default=0.5, help="Per-finger opening threshold fraction.")
    args = parser.parse_args(argv)

    metadata_paths = expand_synthetic_metadata_paths(args.synthetic_metadata)
    real_rows, synthetic_rows, summary = analyze_timing_gap(
        review_roots=args.review_root,
        synthetic_metadata_paths=metadata_paths,
        threshold_fraction=args.threshold_fraction,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "timing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_root / "real_clip_timing.csv", real_rows)
    _write_csv(args.output_root / "synthetic_profile_timing.csv", synthetic_rows)
    _write_markdown(args.output_root / "calibration_target.md", summary)
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


def _format_float(value: object) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "n/a"


def _write_markdown(path: Path, summary: dict[str, object]) -> None:
    real = summary.get("real", {})
    synthetic = summary.get("synthetic", {})
    gap = summary.get("gap", {})
    real_by_label = real.get("by_label", {}) if isinstance(real, dict) else {}
    real_paper = real_by_label.get("paper", {}) if isinstance(real_by_label, dict) else {}

    lines = [
        "# Real-Synthetic Skeleton Timing Gap",
        "",
        "## Inputs",
        "",
    ]
    for review_root in summary.get("review_roots", []):
        lines.append(f"- Review root: `{review_root}`")
    for metadata_path in summary.get("synthetic_metadata_paths", []):
        lines.append(f"- Synthetic metadata: `{metadata_path}`")
    lines.extend(
        [
            "",
            "## Real Review Summary",
            "",
            f"- Clip count: `{real.get('clip_count', 0) if isinstance(real, dict) else 0}`",
            f"- Label counts: `{real.get('label_counts', {}) if isinstance(real, dict) else {}}`",
            f"- Paper ring open median: `{_format_float(real_paper.get('ring_open_progress_median'))}`",
            f"- Paper pinky open median: `{_format_float(real_paper.get('pinky_open_progress_median'))}`",
            "",
            "## Synthetic Metadata Summary",
            "",
            f"- Records total: `{synthetic.get('records_total', 0) if isinstance(synthetic, dict) else 0}`",
            f"- Records with onsets: `{synthetic.get('records_with_onsets', 0) if isinstance(synthetic, dict) else 0}`",
            f"- Records without onsets: `{synthetic.get('records_without_onsets', 0) if isinstance(synthetic, dict) else 0}`",
            "",
            "## Gap Targets",
            "",
        ]
    )
    if isinstance(gap, dict) and gap:
        for key, value in sorted(gap.items()):
            lines.append(f"- `{key}`: `{_format_float(value)}`")
    else:
        lines.append("- No paper timing gap could be computed from the provided inputs.")
    lines.extend(
        [
            "",
            "## Generator Implication",
            "",
            (
                "Use non-held-out real timing as the next calibration target. "
                "Do not fit generator thresholds directly from the held-out 15 MP4 validation set."
            ),
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
