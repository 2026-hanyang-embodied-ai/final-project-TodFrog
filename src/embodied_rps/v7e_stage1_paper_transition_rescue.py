"""V7e stage1 paper-transition rescue diagnostics and review-gated seed plan."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any


BRANCH_LABEL = "v7e_stage1_paper_transition_rescue"


@dataclass(frozen=True)
class V7EStage1PaperTransitionRescueConfig:
    """Inputs for planning the v7e paper-transition rescue branch."""

    project_root: Path = field(default_factory=Path.cwd)
    v7d_original20_validation_root: Path = Path(
        "artifacts/real_mp4_prediction_validation_original20_v7d_real_seeded_prompt_window_guard_20260618"
    )
    policy_probe_original20_root: Path = Path(
        "artifacts/real_mp4_prediction_validation_original20_v7e_stage1_paper_transition_rescue_policy_probe_20260619"
    )
    temporal_review_root: Path = Path("artifacts/real_skeleton_v7d_temporal_review_20260618")
    prompt_pose_collection_review_root: Path = Path("artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618")
    v7d_selection_root: Path = Path("artifacts/real_skeleton_v7d_manual_approval_selection_20260618")
    output_root: Path = Path("artifacts/real_skeleton_v7e_stage1_paper_transition_rescue_plan_20260619")
    recommended_additional_paper_seed_count: int = 5


def write_v7e_stage1_paper_transition_rescue_plan(
    config: V7EStage1PaperTransitionRescueConfig,
) -> dict[str, Any]:
    """Write a non-mutating diagnostic and review-gated seed-expansion plan."""

    project_root = config.project_root.resolve()
    baseline_root = _resolve_path(project_root, config.v7d_original20_validation_root)
    probe_root = _resolve_path(project_root, config.policy_probe_original20_root)
    temporal_review_root = _resolve_path(project_root, config.temporal_review_root)
    prompt_review_root = _resolve_path(project_root, config.prompt_pose_collection_review_root)
    selection_root = _resolve_path(project_root, config.v7d_selection_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_summary = _read_json(baseline_root / "validation_summary.json")
    probe_summary = _read_json(probe_root / "validation_summary.json")
    baseline_rows = _read_rows(baseline_root / "clip_metrics.csv")
    probe_rows = _read_rows(probe_root / "clip_metrics.csv")
    _reject_heldout_rows(baseline_rows, context=baseline_root / "clip_metrics.csv")
    _reject_heldout_rows(probe_rows, context=probe_root / "clip_metrics.csv")

    baseline_analysis = _original20_analysis(
        summary=baseline_summary,
        rows=baseline_rows,
        root=baseline_root,
        project_root=project_root,
        label="v7d_terminal_wait_baseline",
    )
    probe_analysis = _policy_probe_analysis(
        summary=probe_summary,
        rows=probe_rows,
        root=probe_root,
        project_root=project_root,
        baseline=baseline_analysis,
    )
    seed_expansion = _review_gated_seed_expansion(
        temporal_review_root=temporal_review_root,
        prompt_review_root=prompt_review_root,
        selection_root=selection_root,
        output_root=output_root,
        project_root=project_root,
        recommended_count=config.recommended_additional_paper_seed_count,
    )

    status = _status(probe_analysis=probe_analysis, seed_expansion=seed_expansion)
    summary: dict[str, Any] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "baseline_original20": baseline_analysis,
        "policy_probe_original20": probe_analysis,
        "review_gated_seed_expansion": seed_expansion,
        "experiment_matrix": _experiment_matrix(),
        "next_action": _next_action(status),
        "training_started": False,
        "validation_started": False,
        "heldout15_started": False,
        "promotion_eligible": False,
        "fallback_policy": "keep_v4_live_demo_fallback",
        "heldout_policy": "heldout */test MP4s remain validation-only and are rejected from v7e planning metadata",
        "protected_outputs_policy": "proposal.pdf, presentation-slides.pdf, and final packaging remain untouched",
        "config": _config_summary(project_root=project_root, config=config),
    }
    (output_root / "v7e_stage1_paper_transition_rescue_plan_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7e_stage1_paper_transition_rescue_plan.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _original20_analysis(
    *,
    summary: dict[str, Any],
    rows: list[dict[str, str]],
    root: Path,
    project_root: Path,
    label: str,
) -> dict[str, Any]:
    paper_rows = [row for row in rows if _text(row.get("true_gesture")) == "paper"]
    paper_early_rock_locks = [
        _paper_rescue_row(row=row, root=root, project_root=project_root)
        for row in paper_rows
        if _is_paper_early_rock_lock(row, max_progress=_max_decision_progress(summary))
    ]
    first_correct_progresses = [
        _float(row.get("first_correct_stable_progress"))
        for row in paper_rows
        if _float(row.get("first_correct_stable_progress")) is not None
    ]
    return {
        "label": label,
        "summary_path": _display_path(root / "validation_summary.json", base=project_root),
        "clip_metrics_csv": _display_path(root / "clip_metrics.csv", base=project_root),
        "passed": bool(summary.get("passed")),
        "clip_count": _int(summary.get("clip_count")),
        "passed_clip_count": _int(summary.get("passed_clip_count")),
        "failed_clip_count": _int(summary.get("failed_clip_count")),
        "paper_accuracy": _class_accuracy(summary, "paper"),
        "scissors_accuracy": _class_accuracy(summary, "scissors"),
        "rock_false_trigger_count": _int(summary.get("rock_false_trigger_count")),
        "paper_wait_is_terminal_for_transitions": bool(_mapping(summary.get("strict_gate")).get("paper_wait_is_terminal_for_transitions")),
        "max_decision_progress": _max_decision_progress(summary),
        "paper_early_rock_lock_count": len(paper_early_rock_locks),
        "paper_early_rock_lock_clips": paper_early_rock_locks,
        "paper_first_correct_stable_progress_mean": mean(first_correct_progresses) if first_correct_progresses else None,
    }


def _policy_probe_analysis(
    *,
    summary: dict[str, Any],
    rows: list[dict[str, str]],
    root: Path,
    project_root: Path,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    analysis = _original20_analysis(
        summary=summary,
        rows=rows,
        root=root,
        project_root=project_root,
        label="v7e_nonterminal_wait_policy_probe",
    )
    failed_paper = [
        _paper_rescue_row(row=row, root=root, project_root=project_root)
        for row in rows
        if _text(row.get("true_gesture")) == "paper" and _text(row.get("passed")).lower() != "true"
    ]
    remaining_late = [row for row in failed_paper if row.get("failure_reason") == "late_decision"]
    baseline_paper_accuracy = float(baseline.get("paper_accuracy") or 0.0)
    analysis.update(
        {
            "paper_accuracy_delta_vs_baseline": float(analysis.get("paper_accuracy") or 0.0) - baseline_paper_accuracy,
            "remaining_failed_paper_count": len(failed_paper),
            "remaining_late_paper_count": len(remaining_late),
            "remaining_failed_paper_clips": failed_paper,
            "interpretation": (
                "nonterminal early-wait policy rescues most paper clips, but remaining late paper still needs "
                "stage1 paper-transition seed expansion"
            ),
        }
    )
    return analysis


def _review_gated_seed_expansion(
    *,
    temporal_review_root: Path,
    prompt_review_root: Path,
    selection_root: Path,
    output_root: Path,
    project_root: Path,
    recommended_count: int,
) -> dict[str, Any]:
    temporal_rows = _read_rows(temporal_review_root / "temporal_review_manifest.csv")
    selection_rows = _read_rows(selection_root / "approval_selection_template.csv")
    _reject_heldout_rows(temporal_rows, context=temporal_review_root / "temporal_review_manifest.csv")
    _reject_heldout_rows(selection_rows, context=selection_root / "approval_selection_template.csv")
    selected_paper_ids = [
        _text(row.get("selected_segment_id"))
        for row in selection_rows
        if _text(row.get("proposal_role")) == "hard_paper_prompt_window" and _text(row.get("selected_segment_id"))
    ]
    paper_candidates = [
        row
        for row in temporal_rows
        if _text(row.get("target_name")) == "paper" and _text(row.get("proposal_role")) == "hard_paper_prompt_window"
    ]
    paper_candidates = sorted(paper_candidates, key=lambda row: (_int(row.get("rank")) or 0, _text(row.get("segment_id"))))
    recommended = [
        row
        for row in paper_candidates
        if _text(row.get("segment_id")) not in set(selected_paper_ids)
    ][: max(0, int(recommended_count))]

    candidates_csv = output_root / "v7e_paper_seed_review_candidates.csv"
    decision_template_csv = output_root / "v7e_paper_seed_review_decision_template.csv"
    candidate_rows = [
        _candidate_row(
            row,
            project_root=project_root,
            temporal_review_root=temporal_review_root,
            prompt_review_root=prompt_review_root,
        )
        for row in recommended
    ]
    _write_csv(
        candidates_csv,
        [
            "review_priority",
            "segment_id",
            "target_name",
            "proposal_role",
            "frame_count",
            "detection_coverage",
            "temporal_strip",
            "preview_image",
            "skeleton_npz",
            "recommended_review_note",
        ],
        candidate_rows,
    )
    decision_rows = [
        {
            **row,
            "decision": "",
            "review_notes": "",
            "instruction": "Manually inspect temporal_review.html evidence before approving; this file does not auto-approve seeds.",
        }
        for row in candidate_rows
    ]
    _write_csv(
        decision_template_csv,
        [
            "review_priority",
            "segment_id",
            "target_name",
            "proposal_role",
            "decision",
            "review_notes",
            "frame_count",
            "detection_coverage",
            "temporal_strip",
            "preview_image",
            "skeleton_npz",
            "recommended_review_note",
            "instruction",
        ],
        decision_rows,
    )
    return {
        "temporal_review_manifest": _display_path(temporal_review_root / "temporal_review_manifest.csv", base=project_root),
        "temporal_review_html": _display_path(temporal_review_root / "temporal_review.html", base=project_root),
        "already_selected_paper_segment_ids": selected_paper_ids,
        "available_paper_candidate_count": len(paper_candidates),
        "recommended_additional_paper_seed_count": len(recommended),
        "recommended_paper_segment_ids": [_text(row.get("segment_id")) for row in recommended],
        "paper_seed_review_candidates_csv": _display_path(candidates_csv, base=project_root),
        "paper_seed_review_decision_template_csv": _display_path(decision_template_csv, base=project_root),
        "review_required_before_seed_package": True,
        "auto_approved": False,
    }


def _candidate_row(
    row: dict[str, str],
    *,
    project_root: Path,
    temporal_review_root: Path,
    prompt_review_root: Path,
) -> dict[str, Any]:
    priority = _int(row.get("rank")) or 0
    segment_id = _text(row.get("segment_id"))
    return {
        "review_priority": priority,
        "segment_id": segment_id,
        "target_name": _text(row.get("target_name")),
        "proposal_role": _text(row.get("proposal_role")),
        "frame_count": _int(row.get("frame_count")),
        "detection_coverage": _float(row.get("detection_coverage")),
        "temporal_strip": _display_path(
            _resolve_existing_review_link(
                primary_root=temporal_review_root,
                fallback_root=prompt_review_root,
                value=row.get("temporal_strip"),
            ),
            base=project_root,
        ),
        "preview_image": _display_path(
            _resolve_existing_review_link(
                primary_root=prompt_review_root,
                fallback_root=temporal_review_root,
                value=row.get("preview_image"),
            ),
            base=project_root,
        ),
        "skeleton_npz": _display_path(
            _resolve_existing_review_link(
                primary_root=prompt_review_root,
                fallback_root=temporal_review_root,
                value=row.get("skeleton_npz"),
            ),
            base=project_root,
        ),
        "recommended_review_note": (
            "Candidate v7e hard-paper prompt-window rescue seed; approve only after confirming bounded PROMPT SCISSORS "
            f"paper resolution in temporal evidence for {segment_id}."
        ),
    }


def _paper_rescue_row(*, row: dict[str, str], root: Path, project_root: Path) -> dict[str, Any]:
    decision_progress = _float(row.get("decision_progress"))
    first_correct = _float(row.get("first_correct_stable_progress"))
    return {
        "clip_id": _text(row.get("clip_id")),
        "transition_label": _text(row.get("transition_label")),
        "true_gesture": _text(row.get("true_gesture")),
        "predicted_gesture": _text(row.get("predicted_gesture")),
        "decision_state": _text(row.get("decision_state")),
        "failure_reason": _text(row.get("failure_reason")),
        "decision_progress": decision_progress,
        "first_correct_stable_progress": first_correct,
        "rescue_gap_progress": first_correct - decision_progress if first_correct is not None and decision_progress is not None else None,
        "source_path": _display_dataset_path(row.get("source_path")),
        "overlay_path": _display_artifact_link(root=root, project_root=project_root, value=row.get("overlay_path")),
        "frame_csv_path": _display_artifact_link(root=root, project_root=project_root, value=row.get("frame_csv_path")),
    }


def _is_paper_early_rock_lock(row: dict[str, str], *, max_progress: float) -> bool:
    first_correct = _float(row.get("first_correct_stable_progress"))
    return (
        _text(row.get("true_gesture")) == "paper"
        and _text(row.get("predicted_gesture")) == "rock"
        and _text(row.get("decision_state")) == "wait_counter_paper"
        and first_correct is not None
        and first_correct <= max_progress
    )


def _status(*, probe_analysis: dict[str, Any], seed_expansion: dict[str, Any]) -> str:
    if int(seed_expansion.get("recommended_additional_paper_seed_count") or 0) <= 0:
        return "blocked_no_additional_paper_review_candidates"
    if bool(probe_analysis.get("passed")):
        return "ready_for_heldout_policy_probe_before_training"
    if int(probe_analysis.get("remaining_late_paper_count") or 0) > 0:
        return "ready_for_v7e_paper_seed_review"
    return "ready_for_v7e_branch_design_review"


def _next_action(status: str) -> str:
    if status == "ready_for_v7e_paper_seed_review":
        return (
            "review the recommended paper temporal strips, approve additional hard-paper seeds manually, then build a "
            "v7e stage1-only dataset/profile while reusing the v7d stage2 profile"
        )
    if status == "ready_for_heldout_policy_probe_before_training":
        return "run heldout15 only after confirming original20 full pass for the policy probe"
    if status == "blocked_no_additional_paper_review_candidates":
        return "collect or propose more bounded PROMPT SCISSORS paper prompt-window segments before training"
    return "inspect v7e diagnostics before selecting a training or policy-only branch"


def _experiment_matrix() -> list[dict[str, object]]:
    return [
        {
            "run_id": "v7d_baseline",
            "factor": "terminal wait policy",
            "value": "paper_wait_is_terminal_for_transitions=true",
            "fixed_config": "v7d two-stage TCN profiles, original20 MP4s",
            "expected_outcome": "preserve v7d failure surface: paper clips lock as rock early",
        },
        {
            "run_id": "v7e_policy_probe",
            "factor": "terminal wait policy",
            "value": "paper_wait_is_terminal_for_transitions=false",
            "fixed_config": "same v7d profiles and original20 MP4s",
            "expected_outcome": "separate policy terminal-lock failures from model transition-timing failures",
        },
        {
            "run_id": "v7e_stage1_retrain",
            "factor": "stage1 paper transition evidence",
            "value": "add reviewed hard-paper prompt-window seeds only",
            "fixed_config": "reuse v7d stage2 unless diagnostics show paper/scissors regression",
            "expected_outcome": "move the remaining late paper clip before the 0.50 strict deadline without rock false triggers",
        },
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _reject_heldout_rows(rows: list[dict[str, str]], *, context: Path) -> None:
    for row in rows:
        for key, value in row.items():
            if _is_heldout_test_path(value):
                raise ValueError(f"{context} contains heldout test path in {key}: {value}")


def _is_heldout_test_path(value: object) -> bool:
    normalized = str(value or "").replace("\\", "/").lower().rstrip("/")
    return "/test/" in normalized or normalized.endswith("/test")


def _max_decision_progress(summary: dict[str, Any]) -> float:
    value = _float(_mapping(summary.get("strict_gate")).get("max_decision_progress"))
    return value if value is not None else 0.5


def _class_accuracy(summary: dict[str, Any], label: str) -> float:
    per_class = _mapping(summary.get("per_class"))
    label_summary = _mapping(per_class.get(label))
    value = _float(label_summary.get("accuracy"))
    return value if value is not None else 0.0


def _display_artifact_link(*, root: Path, project_root: Path, value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute():
        return _display_path(path, base=project_root)
    candidate = project_root / path
    if candidate.exists() or text.startswith("artifacts/"):
        return _display_path(candidate, base=project_root)
    return _display_path(root / path, base=project_root)


def _display_dataset_path(value: object) -> str:
    normalized = str(value or "").replace("\\", "/")
    lower = normalized.lower()
    prefix = "d:/dataset"
    if lower == prefix:
        return "dataset:/"
    if lower.startswith(f"{prefix}/"):
        return f"dataset:/{normalized[len(prefix) + 1:]}"
    return Path(normalized).name if normalized else ""


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _resolve_existing_review_link(*, primary_root: Path, fallback_root: Path, value: object) -> Path:
    text = _text(value)
    path = Path(text)
    if path.is_absolute():
        return path
    primary = primary_root / path
    if primary.exists():
        return primary
    fallback = fallback_root / path
    return fallback if fallback.exists() else primary


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(base.resolve(strict=False)).as_posix()
    except ValueError:
        dataset_path = _display_dataset_path(path.as_posix())
        return dataset_path or path.name


def _config_summary(*, project_root: Path, config: V7EStage1PaperTransitionRescueConfig) -> dict[str, object]:
    summary = asdict(config)
    summary["project_root"] = "."
    for key, value in list(summary.items()):
        if isinstance(value, Path):
            summary[key] = _display_path(_resolve_path(project_root, value), base=project_root)
    return _json_ready(summary)  # type: ignore[return-value]


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _text(value: object) -> str:
    return str(value or "").strip()


def _float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _int(value: object) -> int | None:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _markdown(summary: dict[str, Any]) -> str:
    baseline = summary["baseline_original20"]
    probe = summary["policy_probe_original20"]
    seed = summary["review_gated_seed_expansion"]
    lines = [
        "# V7e Stage1 Paper Transition Rescue Plan",
        "",
        f"- Status: `{summary['status']}`",
        f"- Branch: `{summary['branch_label']}`",
        f"- Fallback policy: `{summary['fallback_policy']}`",
        f"- Baseline original20: `{baseline['passed_clip_count']} / {baseline['clip_count']}`",
        f"- Baseline paper accuracy: `{baseline['paper_accuracy']}`",
        f"- Baseline early paper->rock locks: `{baseline['paper_early_rock_lock_count']}`",
        f"- Policy probe original20: `{probe['passed_clip_count']} / {probe['clip_count']}`",
        f"- Policy probe paper accuracy: `{probe['paper_accuracy']}`",
        f"- Remaining late paper clips: `{probe['remaining_late_paper_count']}`",
        f"- Recommended additional paper seeds: `{seed['recommended_additional_paper_seed_count']}`",
        f"- Review candidate CSV: `{seed['paper_seed_review_candidates_csv']}`",
        f"- Decision template CSV: `{seed['paper_seed_review_decision_template_csv']}`",
        "",
        "## Recommended Paper Segments",
        "",
    ]
    for segment_id in seed["recommended_paper_segment_ids"]:
        lines.append(f"- `{segment_id}`")
    lines.extend(["", "## Next Action", "", summary["next_action"], ""])
    return "\n".join(lines)


__all__ = ["V7EStage1PaperTransitionRescueConfig", "write_v7e_stage1_paper_transition_rescue_plan"]
