"""Post-validation failure audit for the v7e stage1 paper-transition rescue branch."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BRANCH_LABEL = "v7e_stage1_paper_transition_rescue"
STRICT_PREFLIGHT_FILENAME = "v7e_stage1_strict_validation_preflight_summary.json"


@dataclass(frozen=True)
class V7EStage1PostValidationFailureAuditConfig:
    project_root: Path = Path.cwd()
    original20_validation_root: Path = Path(
        "artifacts/real_mp4_prediction_validation_original20_v7e_stage1_paper_transition_rescue_20260619"
    )
    strict_preflight_root: Path = Path("artifacts/real_skeleton_v7e_stage1_strict_validation_preflight_20260619")
    output_root: Path = Path("artifacts/real_skeleton_v7e_stage1_post_validation_failure_audit_20260619")


def write_v7e_stage1_post_validation_failure_audit(
    config: V7EStage1PostValidationFailureAuditConfig,
) -> dict[str, Any]:
    project_root = config.project_root.resolve()
    validation_root = _resolve_path(project_root, config.original20_validation_root)
    strict_preflight_root = _resolve_path(project_root, config.strict_preflight_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = validation_root / "validation_summary.json"
    metrics_path = validation_root / "clip_metrics.csv"
    validation_summary = _read_json(summary_path)
    rows = _read_rows(metrics_path)
    strict_preflight_summary = _read_optional_strict_preflight(strict_preflight_root)

    failed_rows = [row for row in rows if str(row.get("passed", "")).lower() != "true"]
    predicted_by_true = Counter((row.get("true_gesture", ""), row.get("predicted_gesture", "")) for row in failed_rows)
    failure_reasons = Counter(row.get("failure_reason", "") for row in failed_rows)
    transition_counts = Counter(row.get("transition_label", "") for row in failed_rows)
    paper_scissors_confusion_count = sum(
        1
        for row in failed_rows
        if _is_paper_scissors_confusion(row.get("true_gesture", ""), row.get("predicted_gesture", ""))
    )

    failure_rows_path = output_root / "failed_clips.csv"
    _write_failed_clips(failure_rows_path, failed_rows)

    post_original20_statuses = _post_original20_statuses(strict_preflight_summary)
    validation_continued_after_failed_original20 = any(
        status not in {"", "missing", "blocked", "not_run"}
        for status in post_original20_statuses.values()
    )

    summary: dict[str, Any] = {
        "status": (
            "failed_original20_gate_preserved"
            if validation_summary.get("passed") is False
            else "original20_gate_state_preserved"
        ),
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "validation_summary": _display_path(summary_path, base=project_root),
        "clip_metrics_csv": _display_path(metrics_path, base=project_root),
        "failed_clips_csv": _display_path(failure_rows_path, base=project_root),
        "strict_preflight_summary": _display_optional_path(strict_preflight_root / STRICT_PREFLIGHT_FILENAME, base=project_root),
        "original20": {
            "passed": validation_summary.get("passed"),
            "clip_count": validation_summary.get("clip_count"),
            "passed_clip_count": validation_summary.get("passed_clip_count"),
            "failed_clip_count": validation_summary.get("failed_clip_count"),
            "accuracy": validation_summary.get("accuracy"),
            "per_class": validation_summary.get("per_class"),
            "paper_scissors_accuracy": validation_summary.get("paper_scissors_accuracy"),
            "rock_false_trigger_count": validation_summary.get("rock_false_trigger_count"),
            "failure_reason_counts": validation_summary.get("failure_reason_counts"),
        },
        "failure_groups": [
            {"true_gesture": true, "predicted_gesture": pred, "count": count}
            for (true, pred), count in sorted(predicted_by_true.items())
        ],
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "failed_transition_counts": dict(sorted(transition_counts.items())),
        "paper_scissors_confusion_count": paper_scissors_confusion_count,
        "post_original20_stage_statuses": post_original20_statuses,
        "validation_continued_after_failed_original20": validation_continued_after_failed_original20,
        "next_branch_training_started": False,
        "promotion_eligible": False,
        "fallback_policy": "keep_v4_live_demo_fallback",
        "next_branch_target": (
            "rescue scissors-boundary and stage2 paper-vs-scissors separation while preserving v7e stage1 "
            "paper timing gains and zero rock false triggers"
        ),
        "heldout_policy": (
            "heldout15 was not run because original20 failed first; heldout */test MP4s remain validation-only"
        ),
        "notes": [
            "V7e is not promoted because original20 did not reach 20/20.",
            "Heldout15, replay diagnostics, and fresh live retakes cannot override a failed original20 strict gate.",
            "Failed-clips metadata intentionally omits raw validation source paths from dataset roots.",
        ],
    }
    (output_root / "failure_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "failure_audit.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _write_failed_clips(path: Path, failed_rows: list[dict[str, str]]) -> None:
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in failed_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_strict_preflight(root: Path) -> dict[str, Any] | None:
    path = root / STRICT_PREFLIGHT_FILENAME
    if not path.exists():
        return None
    return _read_json(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _post_original20_statuses(strict_preflight_summary: dict[str, Any] | None) -> dict[str, str]:
    if strict_preflight_summary is None:
        return {
            "heldout15_strict_validation": "missing",
            "replay_diagnostics": "missing",
            "fresh_live_retakes": "missing",
        }
    stage_outputs = strict_preflight_summary.get("stage_outputs", {})
    return {
        name: str(stage_outputs.get(name, {}).get("status", "missing"))
        for name in ("heldout15_strict_validation", "replay_diagnostics", "fresh_live_retakes")
    }


def _is_paper_scissors_confusion(true_gesture: str, predicted_gesture: str) -> bool:
    gestures = {true_gesture, predicted_gesture}
    return true_gesture != predicted_gesture and gestures <= {"paper", "scissors"}


def _markdown(summary: dict[str, Any]) -> str:
    original = summary["original20"]
    lines = [
        "# V7e Stage1 Post-Validation Failure Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Branch: `{summary['branch_label']}`",
        f"- Validation summary: `{summary['validation_summary']}`",
        f"- Clip metrics: `{summary['clip_metrics_csv']}`",
        f"- Failed clips CSV: `{summary['failed_clips_csv']}`",
        f"- Strict preflight summary: `{summary['strict_preflight_summary']}`",
        "",
        "## Original20",
        "",
        f"- Passed: `{original.get('passed')}`",
        f"- Clips: `{original.get('passed_clip_count')} / {original.get('clip_count')}`",
        f"- Accuracy: `{original.get('accuracy')}`",
        f"- Paper/scissors accuracy: `{original.get('paper_scissors_accuracy')}`",
        f"- Rock false triggers: `{original.get('rock_false_trigger_count')}`",
        f"- Paper/scissors confusion count: `{summary['paper_scissors_confusion_count']}`",
        "",
        "## Failure Groups",
        "",
    ]
    for group in summary["failure_groups"]:
        lines.append(f"- `{group['true_gesture']} -> {group['predicted_gesture']}`: `{group['count']}`")
    lines.extend(
        [
            "",
            "## Gate Continuation",
            "",
            f"- Validation continued after failed original20: `{summary['validation_continued_after_failed_original20']}`",
        ]
    )
    for name, status in summary["post_original20_stage_statuses"].items():
        lines.append(f"- `{name}`: `{status}`")
    lines.extend(["", "## Next Target", "", summary["next_branch_target"], ""])
    return "\n".join(lines)


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_optional_path(path: Path, *, base: Path) -> str | None:
    if not path.exists():
        return None
    return _display_path(path, base=base)


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
