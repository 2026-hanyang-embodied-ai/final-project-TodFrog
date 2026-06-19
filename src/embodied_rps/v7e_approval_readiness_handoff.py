"""Approval readiness handoff for the v7e stage1 branch."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from embodied_rps.v7e_paper_seed_review_applier import APPLY_CONFIRMATION_PHRASE


BRANCH_LABEL = "v7e_stage1_paper_transition_rescue"
DEFAULT_PLAN_ROOT = Path("artifacts/real_skeleton_v7e_stage1_paper_transition_rescue_plan_20260619")
DEFAULT_VALIDATOR_ROOT = Path("artifacts/real_skeleton_v7e_paper_seed_review_validation_20260619")
DEFAULT_APPLIER_ROOT = Path("artifacts/real_skeleton_v7e_paper_seed_review_applier_20260619")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7e_approval_readiness_handoff_20260619")
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = ("temporal_strip", "preview_image", "skeleton_npz", "source_path")
PROMOTION_GATES: tuple[str, ...] = (
    "synthetic_observation_ratio_metrics",
    "original20_strict_mp4_20_of_20",
    "heldout15_strict_mp4_15_of_15",
    "zero_heldout_rock_false_triggers",
    "replay_diagnostics_after_mp4_full_pass",
    "fresh_live_after_replay_full_pass",
)


@dataclass(frozen=True)
class V7EApprovalReadinessHandoffConfig:
    """Inputs for the non-mutating v7e approval readiness handoff."""

    project_root: Path = field(default_factory=Path.cwd)
    plan_root: Path = DEFAULT_PLAN_ROOT
    validator_root: Path = DEFAULT_VALIDATOR_ROOT
    applier_root: Path = DEFAULT_APPLIER_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


def write_v7e_approval_readiness_handoff(config: V7EApprovalReadinessHandoffConfig) -> dict[str, Any]:
    """Write a concise handoff for the current v7e approval gate."""

    project_root = config.project_root.resolve()
    plan_root = _resolve_path(project_root, config.plan_root)
    validator_root = _resolve_path(project_root, config.validator_root)
    applier_root = _resolve_path(project_root, config.applier_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    candidates_csv = plan_root / "v7e_paper_seed_review_candidates.csv"
    decision_template_csv = plan_root / "v7e_paper_seed_review_decision_template.csv"
    if not decision_template_csv.exists():
        raise FileNotFoundError(f"Missing v7e decision template: {decision_template_csv}")
    if not candidates_csv.exists():
        raise FileNotFoundError(f"Missing v7e candidates CSV: {candidates_csv}")

    candidate_rows = _read_rows(candidates_csv)
    decision_rows = _read_rows(decision_template_csv)
    _reject_heldout_rows(candidate_rows, context=candidates_csv)
    _reject_heldout_rows(decision_rows, context=decision_template_csv)

    candidate_segment_ids = _segment_ids(candidate_rows)
    approved_rows = [row for row in decision_rows if _text(row.get("decision")).lower() == "approve"]
    approved_segment_ids = _segment_ids(approved_rows)
    validator_summary = _read_optional_json(validator_root / "v7e_paper_seed_review_validation_summary.json")
    applier_summary = _read_optional_json(applier_root / "v7e_paper_seed_review_applier_summary.json")
    approval_ready = _is_review_ready(
        approval_count=len(approved_segment_ids),
        candidate_count=len(candidate_segment_ids),
        validator_summary=validator_summary,
        applier_summary=applier_summary,
    )
    status = (
        "ready_for_v7e_official_seed_package_sequence"
        if approval_ready
        else "awaiting_v7e_paper_seed_approval"
    )
    summary = _summary(
        status=status,
        project_root=project_root,
        plan_root=plan_root,
        validator_root=validator_root,
        applier_root=applier_root,
        output_root=output_root,
        candidates_csv=candidates_csv,
        decision_template_csv=decision_template_csv,
        candidate_segment_ids=candidate_segment_ids,
        approved_segment_ids=approved_segment_ids,
        validator_summary=validator_summary,
        applier_summary=applier_summary,
        config=config,
    )
    _write_summary(output_root, summary)
    return summary


def _is_review_ready(
    *,
    approval_count: int,
    candidate_count: int,
    validator_summary: Mapping[str, Any] | None,
    applier_summary: Mapping[str, Any] | None,
) -> bool:
    validator_ready = bool(validator_summary and validator_summary.get("status") == "ready_for_v7e_seed_package_inputs")
    applier_ready = bool(applier_summary and applier_summary.get("status") == "applied_ready_for_v7e_seed_package_inputs")
    return candidate_count > 0 and approval_count == candidate_count and validator_ready and applier_ready


def _summary(
    *,
    status: str,
    project_root: Path,
    plan_root: Path,
    validator_root: Path,
    applier_root: Path,
    output_root: Path,
    candidates_csv: Path,
    decision_template_csv: Path,
    candidate_segment_ids: Sequence[str],
    approved_segment_ids: Sequence[str],
    validator_summary: Mapping[str, Any] | None,
    applier_summary: Mapping[str, Any] | None,
    config: V7EApprovalReadinessHandoffConfig,
) -> dict[str, Any]:
    official_pipeline_allowed = status == "ready_for_v7e_official_seed_package_sequence"
    return _json_ready(
        {
            "status": status,
            "branch_label": BRANCH_LABEL,
            "plan_root": _display_path(plan_root, base=project_root),
            "validator_root": _display_path(validator_root, base=project_root),
            "applier_root": _display_path(applier_root, base=project_root),
            "output_root": _display_path(output_root, base=project_root),
            "candidates_csv": _display_path(candidates_csv, base=project_root),
            "decision_template_csv": _display_path(decision_template_csv, base=project_root),
            "candidate_count": len(candidate_segment_ids),
            "candidate_segment_ids": list(candidate_segment_ids),
            "approval_count": len(approved_segment_ids),
            "approved_segment_ids": list(approved_segment_ids),
            "validator_status": _status_or_missing(validator_summary),
            "applier_status": _status_or_missing(applier_summary),
            "official_pipeline_allowed": official_pipeline_allowed,
            "commands_to_run_now": _commands_to_run_now(official_pipeline_allowed=official_pipeline_allowed),
            "commands_after_manual_approval": _commands_after_manual_approval(),
            "safe_next_manual_action": _safe_next_manual_action(
                candidate_segment_ids=candidate_segment_ids,
                approved_segment_ids=approved_segment_ids,
            ),
            "ordered_promotion_gates": list(PROMOTION_GATES),
            "validator_summary": validator_summary,
            "applier_summary": applier_summary,
            "review_manifest_modified": False,
            "seed_package_created": False,
            "dataset_generated": False,
            "training_started": False,
            "validation_started": False,
            "heldout15_started": False,
            "replay_started": False,
            "live_started": False,
            "promotion_eligible": False,
            "fallback_policy": "keep_v4_live_demo_fallback_until_v7e_full_strict_pass",
            "heldout_policy": "heldout */test MP4s remain validation-only and are rejected from approval metadata",
            "protected_outputs": {
                "proposal_pdf_modified": False,
                "presentation_slides_pdf_modified": False,
                "final_packaging_started": False,
            },
            "config": _config_summary(project_root=project_root, config=config),
        }
    )


def _commands_to_run_now(*, official_pipeline_allowed: bool) -> list[str]:
    if not official_pipeline_allowed:
        return []
    return [
        "python -m embodied_rps.tools.build_v7e_stage1_paper_transition_rescue_seed_package",
        "python -m embodied_rps.tools.run_v7e_stage1_local_pipeline --execute-local",
        "python -m embodied_rps.tools.write_v7e_stage1_remote_training_preflight",
        "remote train TCN stage1 seeds [11, 17, 23] only after local smoke passes",
        "python -m embodied_rps.tools.write_v7e_stage1_strict_validation_preflight",
    ]


def _commands_after_manual_approval() -> list[str]:
    return [
        "python -m embodied_rps.tools.write_v7e_paper_seed_review_applier --mode apply "
        f"--apply-confirmation {APPLY_CONFIRMATION_PHRASE}",
        "python -m embodied_rps.tools.validate_v7e_paper_seed_review",
        "python -m embodied_rps.tools.build_v7e_stage1_paper_transition_rescue_seed_package",
        "python -m embodied_rps.tools.run_v7e_stage1_local_pipeline --execute-local",
    ]


def _safe_next_manual_action(*, candidate_segment_ids: Sequence[str], approved_segment_ids: Sequence[str]) -> dict[str, Any]:
    missing = [segment_id for segment_id in candidate_segment_ids if segment_id not in set(approved_segment_ids)]
    return {
        "action": "approve_or_reject_v7e_paper_seed_decision_template",
        "required_segments": missing,
        "decision_template_csv": "artifacts/real_skeleton_v7e_stage1_paper_transition_rescue_plan_20260619/v7e_paper_seed_review_decision_template.csv",
        "review_evidence_html": "artifacts/real_skeleton_v7d_temporal_review_20260618/temporal_review.html",
        "guarded_apply_confirmation": APPLY_CONFIRMATION_PHRASE,
        "note": "Only bounded PROMPT SCISSORS paper-resolution evidence should be approved.",
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _segment_ids(rows: Sequence[Mapping[str, str]]) -> list[str]:
    return [_text(row.get("segment_id")) for row in rows if _text(row.get("segment_id"))]


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _status_or_missing(summary: Mapping[str, Any] | None) -> str:
    if not summary:
        return "missing"
    return _text(summary.get("status")) or "missing"


def _reject_heldout_rows(rows: Sequence[Mapping[str, str]], *, context: Path) -> None:
    for row in rows:
        for field_name in PATH_FIELDS_TO_AUDIT:
            value = _text(row.get(field_name))
            if _is_heldout_test_path(value):
                raise ValueError(f"{context} contains heldout test path in {field_name}: {value}")


def _is_heldout_test_path(value: str) -> bool:
    normalized = value.replace("\\", "/").lower().rstrip("/")
    return "/test/" in normalized or normalized.endswith("/test")


def _resolve_path(project_root: Path, path: Path | None) -> Path:
    parsed = Path() if path is None else path
    return parsed if parsed.is_absolute() else project_root / parsed


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(base.resolve(strict=False)).as_posix()
    except ValueError:
        return path.name


def _config_summary(*, project_root: Path, config: V7EApprovalReadinessHandoffConfig) -> dict[str, object]:
    summary = asdict(config)
    summary["project_root"] = "."
    for key, value in list(summary.items()):
        if isinstance(value, Path):
            summary[key] = _display_path(_resolve_path(project_root, value), base=project_root)
    return _json_ready(summary)  # type: ignore[return-value]


def _write_summary(output_root: Path, summary: Mapping[str, Any]) -> None:
    (output_root / "v7e_approval_readiness_handoff_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7e_approval_readiness_handoff.md").write_text(_markdown(summary), encoding="utf-8")


def _markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# V7e Approval Readiness Handoff",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Candidate count: `{summary.get('candidate_count')}`",
            f"- Approval count: `{summary.get('approval_count')}`",
            f"- Validator status: `{summary.get('validator_status')}`",
            f"- Applier status: `{summary.get('applier_status')}`",
            f"- Official pipeline allowed: `{summary.get('official_pipeline_allowed')}`",
            f"- Commands to run now: `{len(summary.get('commands_to_run_now', []))}`",
            "- Fallback policy: `keep_v4_live_demo_fallback_until_v7e_full_strict_pass`",
            "- Heldout policy: `heldout */test MP4s remain validation-only`",
            "",
        ]
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _text(value: object) -> str:
    return str(value or "").strip()


__all__ = ["V7EApprovalReadinessHandoffConfig", "write_v7e_approval_readiness_handoff"]
