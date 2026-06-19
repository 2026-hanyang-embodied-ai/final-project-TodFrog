"""Failure-map audit for the v7b prompt-window correction branch."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Mapping

PROGRESS_BINS: tuple[str, ...] = (
    "early_0.00_0.10",
    "prompt_0.10_0.25",
    "mid_0.25_0.50",
    "late_after_0.50",
    "missing",
)
FAILURE_ROW_FIELDS: tuple[str, ...] = (
    "stage",
    "clip_id",
    "true_gesture",
    "predicted_gesture",
    "failure_group",
    "failure_reason",
    "decision_frame",
    "decision_progress",
    "progress_bin",
    "decision_confidence",
)


def build_v7b_failure_audit(
    *,
    original20_validation_root: Path,
    heldout15_validation_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Build a compact v7 failure map to guide the v7b simulation-correction profile."""

    stage_roots = (
        ("original20", original20_validation_root),
        ("heldout15", heldout15_validation_root),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    failure_rows: list[dict[str, object]] = []
    stage_summaries: dict[str, dict[str, object]] = {}
    group_counts: Counter[str] = Counter()
    progress_bin_counts: Counter[str] = Counter({name: 0 for name in PROGRESS_BINS})
    reason_counts: Counter[str] = Counter()

    for stage_name, validation_root in stage_roots:
        validation = _read_json_object(validation_root / "validation_summary.json")
        rows = _read_clip_rows(validation_root / "clip_metrics.csv")
        failed_rows = [row for row in rows if not _passed(row.get("passed", ""))]
        stage_summaries[stage_name] = _stage_summary(
            stage_name=stage_name,
            validation_root=validation_root,
            validation=validation,
            clip_rows=rows,
            failed_rows=failed_rows,
        )
        for row in failed_rows:
            true_gesture = _normalized_label(row.get("true_gesture", ""))
            predicted_gesture = _normalized_label(row.get("predicted_gesture", ""))
            failure_group = f"{true_gesture} -> {predicted_gesture}"
            progress = _optional_float(row.get("decision_progress", ""))
            progress_bin = _progress_bin(progress)
            reason = str(row.get("failure_reason", "")).strip() or "failed"
            audit_row: dict[str, object] = {
                "stage": stage_name,
                "clip_id": str(row.get("clip_id", "")).strip(),
                "true_gesture": true_gesture,
                "predicted_gesture": predicted_gesture,
                "failure_group": failure_group,
                "failure_reason": reason,
                "decision_frame": str(row.get("decision_frame", "")).strip(),
                "decision_progress": "" if progress is None else f"{progress:.6f}",
                "progress_bin": progress_bin,
                "decision_confidence": str(row.get("decision_confidence", "")).strip(),
            }
            failure_rows.append(audit_row)
            group_counts[failure_group] += 1
            progress_bin_counts[progress_bin] += 1
            reason_counts[reason] += 1

    summary: dict[str, object] = {
        "status": "passed",
        "audit_name": "v7b_rps_pose_conservative_scissors_failure_map",
        "output_root": output_root.as_posix(),
        "stages": stage_summaries,
        "failed_clip_count": len(failure_rows),
        "group_counts": dict(group_counts),
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "progress_bin_counts": {name: int(progress_bin_counts[name]) for name in PROGRESS_BINS},
        "correction_targets": [
            "paper -> scissors",
            "rock -> scissors",
            "scissors -> paper",
        ],
        "prompt_window_interpretation": (
            "Failures are interpreted as prompt-conditioned temporal sequences; early or prompt-bin "
            "decisions are not treated as independent final-label thumbnails."
        ),
    }
    _write_outputs(output_root=output_root, summary=summary, rows=failure_rows)
    return summary


def _stage_summary(
    *,
    stage_name: str,
    validation_root: Path,
    validation: Mapping[str, object],
    clip_rows: list[dict[str, str]],
    failed_rows: list[dict[str, str]],
) -> dict[str, object]:
    clip_count = _optional_int(validation.get("clip_count"))
    passed_count = _optional_int(validation.get("passed_clip_count"))
    failed_count = _optional_int(validation.get("failed_clip_count"))
    if clip_count is None:
        clip_count = len(clip_rows)
    if failed_count is None:
        failed_count = len(failed_rows)
    if passed_count is None:
        passed_count = max(0, clip_count - failed_count)
    return {
        "stage": stage_name,
        "validation_root_name": validation_root.name,
        "passed": bool(validation.get("passed")),
        "clip_count": clip_count,
        "passed_clip_count": passed_count,
        "failed_clip_count": failed_count,
        "failure_reason_counts": dict(_mapping(validation.get("failure_reason_counts"))),
    }


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing validation summary: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Validation summary must be a JSON object: {path}")
    return dict(value)


def _read_clip_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing clip metrics CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_outputs(*, output_root: Path, summary: Mapping[str, object], rows: list[dict[str, object]]) -> None:
    (output_root / "failure_map_summary.json").write_text(json.dumps(dict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_root / "failure_map_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FAILURE_ROW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "failure_map_summary.md").write_text(_markdown(summary), encoding="utf-8")


def _markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V7b Failure Map Audit",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Failed clip count: `{summary.get('failed_clip_count')}`",
        "- Recording model: prompt-conditioned temporal response window",
        "",
        "## Failure Groups",
        "",
    ]
    for group, count in _mapping(summary.get("group_counts")).items():
        lines.append(f"- `{group}`: {count}")
    lines.extend(["", "## Prompt Progress Bins", ""])
    for bin_name, count in _mapping(summary.get("progress_bin_counts")).items():
        lines.append(f"- `{bin_name}`: {count}")
    lines.extend(["", "## Stage Summary", ""])
    stages = summary.get("stages")
    if isinstance(stages, Mapping):
        for stage_name, stage_value in stages.items():
            if isinstance(stage_value, Mapping):
                lines.append(
                    f"- `{stage_name}`: passed={stage_value.get('passed')}, "
                    f"passed_clips={stage_value.get('passed_clip_count')}, "
                    f"failed_clips={stage_value.get('failed_clip_count')}"
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(summary.get("prompt_window_interpretation", "")),
            "",
        ]
    )
    return "\n".join(lines)


def _progress_bin(progress: float | None) -> str:
    if progress is None:
        return "missing"
    if progress < 0.10:
        return "early_0.00_0.10"
    if progress < 0.25:
        return "prompt_0.10_0.25"
    if progress <= 0.50:
        return "mid_0.25_0.50"
    return "late_after_0.50"


def _passed(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "passed"}


def _normalized_label(value: str) -> str:
    text = value.strip().lower()
    return text if text else "missing"


def _optional_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


__all__ = ["build_v7b_failure_audit"]
