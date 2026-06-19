"""Post-validation failure audit for the v7d two-stage branch."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"


@dataclass(frozen=True)
class V7DPostValidationFailureAuditConfig:
    project_root: Path = Path.cwd()
    original20_validation_root: Path = Path(
        "artifacts/real_mp4_prediction_validation_original20_v7d_real_seeded_prompt_window_guard_20260618"
    )
    output_root: Path = Path("artifacts/real_skeleton_v7d_post_validation_failure_audit_20260618")


def write_v7d_post_validation_failure_audit(config: V7DPostValidationFailureAuditConfig) -> dict[str, Any]:
    project_root = config.project_root.resolve()
    validation_root = _resolve_path(project_root, config.original20_validation_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = validation_root / "validation_summary.json"
    metrics_path = validation_root / "clip_metrics.csv"
    validation_summary = _read_json(summary_path)
    rows = _read_rows(metrics_path)

    failed_rows = [row for row in rows if str(row.get("passed", "")).lower() != "true"]
    predicted_by_true = Counter((row.get("true_gesture", ""), row.get("predicted_gesture", "")) for row in failed_rows)
    failure_reasons = Counter(row.get("failure_reason", "") for row in failed_rows)
    transition_counts = Counter(row.get("transition_label", "") for row in failed_rows)

    failure_rows_path = output_root / "failed_clips.csv"
    with failure_rows_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "clip_id",
            "transition_label",
            "true_gesture",
            "predicted_gesture",
            "failure_reason",
            "decision_progress",
            "decision_confidence",
            "first_correct_stable_progress",
            "overlay_path",
            "frame_csv_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in failed_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    summary: dict[str, Any] = {
        "status": "failed_original20_gate_preserved",
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "validation_summary": _display_path(summary_path, base=project_root),
        "clip_metrics_csv": _display_path(metrics_path, base=project_root),
        "failed_clips_csv": _display_path(failure_rows_path, base=project_root),
        "original20": {
            "passed": validation_summary.get("passed"),
            "clip_count": validation_summary.get("clip_count"),
            "passed_clip_count": validation_summary.get("passed_clip_count"),
            "failed_clip_count": validation_summary.get("failed_clip_count"),
            "accuracy": validation_summary.get("accuracy"),
            "per_class": validation_summary.get("per_class"),
            "rock_false_trigger_count": validation_summary.get("rock_false_trigger_count"),
            "failure_reason_counts": validation_summary.get("failure_reason_counts"),
        },
        "failure_groups": [
            {"true_gesture": true, "predicted_gesture": pred, "count": count}
            for (true, pred), count in sorted(predicted_by_true.items())
        ],
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "failed_transition_counts": dict(sorted(transition_counts.items())),
        "next_branch_target": (
            "rescue paper transitions from stage1 rock over-abstention without losing the stage2 scissors recovery; "
            "do not promote v7d and keep v4 as the live/demo fallback"
        ),
        "training_started": False,
        "validation_continued_after_failed_original20": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout15 was not run because original20 failed first; heldout */test MP4s remain validation-only",
    }
    (output_root / "failure_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "failure_audit.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# V7d Post-Validation Failure Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Branch: `{summary['branch_label']}`",
        f"- Validation summary: `{summary['validation_summary']}`",
        f"- Clip metrics: `{summary['clip_metrics_csv']}`",
        f"- Failed clips CSV: `{summary['failed_clips_csv']}`",
        "",
        "## Original20",
        "",
    ]
    original = summary["original20"]
    lines.extend(
        [
            f"- Passed: `{original.get('passed')}`",
            f"- Clips: `{original.get('passed_clip_count')} / {original.get('clip_count')}`",
            f"- Accuracy: `{original.get('accuracy')}`",
            f"- Rock false triggers: `{original.get('rock_false_trigger_count')}`",
            "",
            "## Failure Groups",
            "",
        ]
    )
    for group in summary["failure_groups"]:
        lines.append(f"- `{group['true_gesture']} -> {group['predicted_gesture']}`: `{group['count']}`")
    lines.extend(["", "## Next Target", "", summary["next_branch_target"], ""])
    return "\n".join(lines)


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
