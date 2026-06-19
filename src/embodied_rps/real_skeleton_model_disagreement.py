"""Compare saved real-skeleton validation outputs from two model policies."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from embodied_rps.real_skeleton_history_policy import feature_vector_from_rows

DisagreementCategory = Literal["both_pass", "candidate_fix", "candidate_regression", "both_fail"]


@dataclass(frozen=True)
class SavedValidationClip:
    """A saved clip-level validation result plus its per-frame probability rows."""

    clip_id: str
    true_gesture: str
    source_path: str
    transition_label: str
    metrics: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]

    @property
    def passed(self) -> bool:
        return bool(self.metrics.get("passed", False))

    @property
    def failure_reason(self) -> str:
        return str(self.metrics.get("failure_reason") or "")

    @property
    def decision_state(self) -> str:
        return str(self.metrics.get("decision_state") or "")


def load_saved_validation_root(root: Path) -> dict[str, SavedValidationClip]:
    """Load all clip metrics and frame JSONL rows from a saved validation artifact."""

    metric_paths = sorted((root / "clips").rglob("metrics.json"))
    if not metric_paths:
        raise ValueError(f"no metrics.json files found under {root / 'clips'}")
    clips: dict[str, SavedValidationClip] = {}
    for metrics_path in metric_paths:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        clip_id = str(metrics.get("clip_id") or metrics_path.parent.name)
        if clip_id in clips:
            raise ValueError(f"duplicate clip_id {clip_id!r} under {root}")
        frame_path = _resolve_frame_jsonl_path(metrics_path, metrics)
        clips[clip_id] = SavedValidationClip(
            clip_id=clip_id,
            true_gesture=str(metrics.get("true_gesture") or ""),
            source_path=str(metrics.get("source_path") or ""),
            transition_label=str(metrics.get("transition_label") or metrics_path.parent.parent.name),
            metrics=metrics,
            rows=tuple(_read_jsonl(frame_path)),
        )
    return clips


def summarize_model_disagreement(
    *,
    baseline_root: Path,
    candidate_root: Path,
    observation_progress: float = 0.5,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Summarize per-clip disagreements between baseline and candidate outputs."""

    baseline_clips = load_saved_validation_root(baseline_root)
    candidate_clips = load_saved_validation_root(candidate_root)
    missing_from_candidate = sorted(set(baseline_clips) - set(candidate_clips))
    missing_from_baseline = sorted(set(candidate_clips) - set(baseline_clips))
    common_clip_ids = sorted(set(baseline_clips) & set(candidate_clips))
    if not common_clip_ids:
        raise ValueError("baseline and candidate validation roots have no clip_id overlap")

    rows = [
        _disagreement_row(
            baseline_clips[clip_id],
            candidate_clips[clip_id],
            observation_progress=observation_progress,
        )
        for clip_id in common_clip_ids
    ]
    category_counts = Counter(str(row["disagreement_category"]) for row in rows)
    label_category_counts = Counter(
        f"{row['true_gesture']}::{row['disagreement_category']}"
        for row in rows
    )
    summary: dict[str, object] = {
        "baseline_root": baseline_root.as_posix(),
        "candidate_root": candidate_root.as_posix(),
        "observation_progress": observation_progress,
        "clip_count": len(rows),
        "baseline_clip_count": len(baseline_clips),
        "candidate_clip_count": len(candidate_clips),
        "baseline_passed_clip_count": sum(1 for row in rows if bool(row["baseline_passed"])),
        "candidate_passed_clip_count": sum(1 for row in rows if bool(row["candidate_passed"])),
        "category_counts": dict(sorted(category_counts.items())),
        "label_category_counts": dict(sorted(label_category_counts.items())),
        "missing_from_candidate": missing_from_candidate,
        "missing_from_baseline": missing_from_baseline,
        "candidate_fix_clip_ids": [str(row["clip_id"]) for row in rows if row["disagreement_category"] == "candidate_fix"],
        "candidate_regression_clip_ids": [
            str(row["clip_id"]) for row in rows if row["disagreement_category"] == "candidate_regression"
        ],
    }
    return rows, summary


def build_meta_selector_target_notes(summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    """Create a markdown note describing which selector targets are supported by the comparison."""

    category_counts = summary.get("category_counts", {})
    lines = [
        "# Model Disagreement Meta-Selector Targets",
        "",
        f"- Baseline root: `{summary.get('baseline_root')}`",
        f"- Candidate root: `{summary.get('candidate_root')}`",
        f"- Observation progress: `{summary.get('observation_progress')}`",
        f"- Common clip count: `{summary.get('clip_count')}`",
        "",
        "## Category Counts",
        "",
    ]
    if isinstance(category_counts, Mapping) and category_counts:
        for category, count in category_counts.items():
            lines.append(f"- `{category}`: `{count}`")
    else:
        lines.append("- No comparable clips.")

    candidate_fixes = [row for row in rows if row.get("disagreement_category") == "candidate_fix"]
    candidate_regressions = [row for row in rows if row.get("disagreement_category") == "candidate_regression"]
    lines.extend(
        [
            "",
            "## Candidate-Trust Gate Targets",
            "",
            "Use these only as diagnostics unless the selector is trained without held-out labels.",
            "",
        ]
    )
    if candidate_fixes:
        for row in candidate_fixes:
            lines.append(_target_line(row, "candidate-trust gate"))
    else:
        lines.append("- No candidate-only fixes were found.")

    lines.extend(["", "## Candidate-Suppression Guard Targets", ""])
    if candidate_regressions:
        for row in candidate_regressions:
            lines.append(_target_line(row, "candidate-suppression guard"))
    else:
        lines.append("- No candidate-only regressions were found.")
    return "\n".join(lines).rstrip() + "\n"


def _disagreement_row(
    baseline: SavedValidationClip,
    candidate: SavedValidationClip,
    *,
    observation_progress: float,
) -> dict[str, object]:
    category = _classify_disagreement(baseline_passed=baseline.passed, candidate_passed=candidate.passed)
    baseline_features = _prefixed_features("baseline", baseline.rows, observation_progress=observation_progress)
    candidate_features = _prefixed_features("candidate", candidate.rows, observation_progress=observation_progress)
    row: dict[str, object] = {
        "clip_id": baseline.clip_id,
        "true_gesture": baseline.true_gesture,
        "transition_label": baseline.transition_label,
        "source_path": baseline.source_path,
        "disagreement_category": category,
        "baseline_passed": baseline.passed,
        "candidate_passed": candidate.passed,
        "baseline_failure_reason": baseline.failure_reason,
        "candidate_failure_reason": candidate.failure_reason,
        "baseline_decision_state": baseline.decision_state,
        "candidate_decision_state": candidate.decision_state,
        "baseline_decision_progress": baseline.metrics.get("decision_progress"),
        "candidate_decision_progress": candidate.metrics.get("decision_progress"),
        "baseline_decision_confidence": baseline.metrics.get("decision_confidence"),
        "candidate_decision_confidence": candidate.metrics.get("decision_confidence"),
    }
    row.update(baseline_features)
    row.update(candidate_features)
    for name in _probability_delta_feature_names():
        row[f"candidate_minus_baseline_{name}"] = float(candidate_features[f"candidate_{name}"]) - float(
            baseline_features[f"baseline_{name}"]
        )
    return row


def _classify_disagreement(*, baseline_passed: bool, candidate_passed: bool) -> DisagreementCategory:
    if baseline_passed and candidate_passed:
        return "both_pass"
    if not baseline_passed and candidate_passed:
        return "candidate_fix"
    if baseline_passed and not candidate_passed:
        return "candidate_regression"
    return "both_fail"


def _prefixed_features(
    prefix: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    observation_progress: float,
) -> dict[str, float]:
    features = feature_vector_from_rows(rows, observation_progress=observation_progress)
    return {f"{prefix}_{name}": float(value) for name, value in features.items()}


def _probability_delta_feature_names() -> tuple[str, ...]:
    return (
        "latest_rock_probability",
        "latest_paper_probability",
        "latest_scissors_probability",
        "mean_rock_probability",
        "mean_paper_probability",
        "mean_scissors_probability",
        "max_rock_probability",
        "max_paper_probability",
        "max_scissors_probability",
        "delta_rock_probability",
        "delta_paper_probability",
        "delta_scissors_probability",
    )


def _resolve_frame_jsonl_path(metrics_path: Path, metrics: Mapping[str, Any]) -> Path:
    raw_path = str(metrics.get("frame_jsonl_path") or "")
    candidates: list[Path] = []
    if raw_path:
        raw = Path(raw_path)
        candidates.append(raw)
        if not raw.is_absolute():
            candidates.append(metrics_path.parents[3] / raw)
            candidates.append(Path.cwd() / raw)
    candidates.append(metrics_path.with_name("frames.jsonl"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not resolve frames.jsonl for {metrics_path}")


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(item)
    return rows


def _target_line(row: Mapping[str, object], target_name: str) -> str:
    label = row.get("true_gesture")
    clip_id = row.get("clip_id")
    baseline_reason = row.get("baseline_failure_reason") or "none"
    candidate_reason = row.get("candidate_failure_reason") or "none"
    paper_delta = row.get("candidate_minus_baseline_latest_paper_probability")
    scissors_delta = row.get("candidate_minus_baseline_latest_scissors_probability")
    return (
        f"- `{clip_id}` (`{label}`): {target_name}; "
        f"baseline reason `{baseline_reason}`, candidate reason `{candidate_reason}`, "
        f"latest paper delta `{_format_float(paper_delta)}`, latest scissors delta `{_format_float(scissors_delta)}`."
    )


def _format_float(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return "n/a"


__all__ = [
    "SavedValidationClip",
    "build_meta_selector_target_notes",
    "load_saved_validation_root",
    "summarize_model_disagreement",
]
