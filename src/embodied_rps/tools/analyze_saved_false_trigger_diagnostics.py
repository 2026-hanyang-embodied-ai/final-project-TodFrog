"""Analyze stable false-trigger episodes from saved validation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from embodied_rps.real_skeleton_false_trigger_diagnostics import summarize_rock_false_trigger_diagnostics
from embodied_rps.real_skeleton_policy_sweep import load_saved_validation_clips


def main(argv: Sequence[str] | None = None) -> int:
    """Write rock false-trigger diagnostics for a saved validation root."""

    parser = argparse.ArgumentParser(description="Analyze stable binary false-trigger episodes in saved validation rows.")
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--confirmation-count", type=int, default=2)
    parser.add_argument("--max-progress", type=float, default=0.50)
    parser.add_argument(
        "--progress-mode",
        choices=("clip", "motion", "observed", "model"),
        default="clip",
    )
    args = parser.parse_args(argv)

    progress_key = _progress_key_for_mode(str(args.progress_mode))
    clips = load_saved_validation_clips(args.validation_root)
    rock_rows = [
        summarize_rock_false_trigger_diagnostics(
            clip_id=clip.clip_id,
            true_gesture=clip.true_gesture,
            rows=clip.rows,
            confirmation_count=int(args.confirmation_count),
            max_progress=float(args.max_progress),
            progress_key=progress_key,
        )
        for clip in clips
        if clip.true_gesture == "rock"
    ]
    all_episode_rows = _flatten_episode_rows(rock_rows)
    summary = _build_summary(
        validation_root=args.validation_root,
        rock_rows=rock_rows,
        max_progress=float(args.max_progress),
        confirmation_count=int(args.confirmation_count),
        progress_mode=str(args.progress_mode),
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    diagnostics_json = args.output_root / "false_trigger_diagnostics.json"
    diagnostics_csv = args.output_root / "false_trigger_diagnostics.csv"
    episodes_csv = args.output_root / "false_trigger_episodes.csv"
    summary_path = args.output_root / "false_trigger_diagnostics_summary.json"
    diagnostics_json.write_text(json.dumps(rock_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rows_csv(diagnostics_csv, rock_rows, _DIAGNOSTIC_FIELDS)
    _write_rows_csv(episodes_csv, all_episode_rows, _EPISODE_FIELDS)
    summary["false_trigger_diagnostics_json"] = diagnostics_json.as_posix()
    summary["false_trigger_diagnostics_csv"] = diagnostics_csv.as_posix()
    summary["false_trigger_episodes_csv"] = episodes_csv.as_posix()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": summary_path.as_posix(), "rock_clip_count": summary["rock_clip_count"]}, indent=2))
    return 0


_DIAGNOSTIC_FIELDS = [
    "clip_id",
    "true_gesture",
    "episode_count",
    "wait_episode_count",
    "false_trigger_episode_count",
    "stable_wait_before_false_trigger",
    "first_false_trigger_state",
    "first_false_trigger_frame",
    "first_false_trigger_progress",
    "max_false_trigger_confidence",
    "max_false_trigger_motion_progress",
]
_EPISODE_FIELDS = [
    "clip_id",
    "true_gesture",
    "state",
    "start_frame",
    "end_frame",
    "start_progress",
    "end_progress",
    "duration_frames",
    "max_confidence",
    "max_margin",
    "max_transition_mass",
    "max_motion_progress",
]


def _build_summary(
    *,
    validation_root: Path,
    rock_rows: Sequence[Mapping[str, object]],
    max_progress: float,
    confirmation_count: int,
    progress_mode: str,
) -> dict[str, object]:
    false_trigger_rows = [row for row in rock_rows if int(row.get("false_trigger_episode_count", 0)) > 0]
    wait_before_rows = [row for row in false_trigger_rows if bool(row.get("stable_wait_before_false_trigger"))]
    return {
        "validation_root": validation_root.as_posix(),
        "rock_clip_count": len(rock_rows),
        "rock_false_trigger_clip_count": len(false_trigger_rows),
        "rock_stable_wait_before_false_trigger_count": len(wait_before_rows),
        "confirmation_count": confirmation_count,
        "max_progress": max_progress,
        "progress_mode": progress_mode,
        "first_false_trigger_progress_values": [
            row.get("first_false_trigger_progress")
            for row in false_trigger_rows
            if row.get("first_false_trigger_progress") is not None
        ],
        "max_false_trigger_confidence_values": [
            row.get("max_false_trigger_confidence")
            for row in false_trigger_rows
            if row.get("max_false_trigger_confidence") is not None
        ],
    }


def _flatten_episode_rows(rock_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for clip in rock_rows:
        for episode in clip.get("episodes", []):
            if not isinstance(episode, Mapping):
                continue
            rows.append(
                {
                    "clip_id": clip.get("clip_id"),
                    "true_gesture": clip.get("true_gesture"),
                    **dict(episode),
                }
            )
    return rows


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _progress_key_for_mode(mode: str) -> str:
    if mode == "model":
        return "model_progress"
    if mode == "observed":
        return "observed_progress"
    if mode == "motion":
        return "motion_progress"
    return "clip_progress"


if __name__ == "__main__":
    raise SystemExit(main())
