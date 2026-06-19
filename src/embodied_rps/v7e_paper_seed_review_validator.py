"""Non-mutating validator for v7e paper seed review decisions."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


BRANCH_LABEL = "v7e_stage1_paper_transition_rescue"
DEFAULT_PLAN_ROOT = Path("artifacts/real_skeleton_v7e_stage1_paper_transition_rescue_plan_20260619")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7e_paper_seed_review_validation_20260619")
DEFAULT_MINIMUM_APPROVED_PAPER_SEED_COUNT = 5
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = ("temporal_strip", "preview_image", "skeleton_npz", "source_path")


@dataclass(frozen=True)
class V7EPaperSeedReviewValidatorConfig:
    """Inputs for validating v7e paper seed review decisions."""

    project_root: Path = field(default_factory=Path.cwd)
    plan_root: Path = DEFAULT_PLAN_ROOT
    decisions_csv: Path | None = None
    output_root: Path = DEFAULT_OUTPUT_ROOT
    minimum_approved_paper_seed_count: int = DEFAULT_MINIMUM_APPROVED_PAPER_SEED_COUNT


def validate_v7e_paper_seed_review(config: V7EPaperSeedReviewValidatorConfig) -> dict[str, Any]:
    """Validate v7e paper seed approvals without mutating review or training state."""

    project_root = config.project_root.resolve()
    plan_root = _resolve_path(project_root, config.plan_root)
    decisions_csv = (
        _resolve_path(project_root, config.decisions_csv)
        if config.decisions_csv is not None
        else plan_root / "v7e_paper_seed_review_decision_template.csv"
    )
    candidates_csv = plan_root / "v7e_paper_seed_review_candidates.csv"
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not candidates_csv.exists():
        raise FileNotFoundError(f"Missing v7e paper candidates CSV: {candidates_csv}")
    if not decisions_csv.exists():
        raise FileNotFoundError(f"Missing v7e paper decision CSV: {decisions_csv}")

    candidate_rows = _read_rows(candidates_csv)
    decision_rows = _read_rows(decisions_csv)
    _reject_heldout_rows(candidate_rows, context=candidates_csv)
    _reject_heldout_rows(decision_rows, context=decisions_csv)
    candidates_by_id = {
        _text(row.get("segment_id")): row
        for row in candidate_rows
        if _text(row.get("segment_id"))
    }

    decisions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    approval_notes_missing: list[str] = []
    evidence_audit: list[dict[str, Any]] = []

    for row_number, row in enumerate(decision_rows, start=2):
        segment_id = _text(row.get("segment_id"))
        raw_decision = _text(row.get("decision"))
        if not raw_decision:
            continue
        if not segment_id:
            failures.append({"code": "missing_segment_id", "row": row_number})
            continue
        if segment_id in seen:
            failures.append({"code": "duplicate_decision_segment", "segment_id": segment_id, "row": row_number})
            continue
        seen.add(segment_id)
        candidate = candidates_by_id.get(segment_id)
        if candidate is None:
            failures.append({"code": "decision_segment_missing_from_candidates", "segment_id": segment_id, "row": row_number})
            continue
        normalized = _normalize_decision(raw_decision)
        if normalized is None:
            failures.append({"code": "unsupported_decision", "segment_id": segment_id, "decision": raw_decision, "row": row_number})
            continue
        if _text(candidate.get("target_name")) != "paper":
            failures.append({"code": "non_paper_candidate_in_v7e_paper_review", "segment_id": segment_id, "row": row_number})
        if _text(candidate.get("proposal_role")) != "hard_paper_prompt_window":
            failures.append({"code": "unexpected_candidate_role", "segment_id": segment_id, "row": row_number})
        review_notes = _text(row.get("review_notes"))
        decision = {
            "segment_id": segment_id,
            "decision": normalized,
            "review_notes": review_notes,
            "row": row_number,
        }
        decisions.append(decision)
        if normalized == "approve":
            if not review_notes:
                approval_notes_missing.append(segment_id)
            audit = _evidence_audit_row(candidate=candidate, project_root=project_root)
            evidence_audit.append(audit)
            failures.extend(_evidence_failures(audit, segment_id=segment_id, row=row_number))

    approved = [decision for decision in decisions if decision["decision"] == "approve"]
    approve_count = len(approved)
    status = _status(
        failures=failures,
        decision_count=len(decisions),
        approval_notes_missing=approval_notes_missing,
        approve_count=approve_count,
        minimum_approved_count=int(config.minimum_approved_paper_seed_count),
    )
    summary: dict[str, Any] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "plan_root": _display_path(plan_root, base=project_root),
        "candidates_csv": _display_path(candidates_csv, base=project_root),
        "decisions_csv": _display_path(decisions_csv, base=project_root),
        "output_root": _display_path(output_root, base=project_root),
        "minimum_approved_paper_seed_count": int(config.minimum_approved_paper_seed_count),
        "candidate_count": len(candidate_rows),
        "decision_count": len(decisions),
        "approve_count": approve_count,
        "approved_paper_segment_ids": sorted(str(decision["segment_id"]) for decision in approved),
        "approval_decisions_missing_review_notes": sorted(approval_notes_missing),
        "approved_evidence_audit": evidence_audit,
        "approved_evidence_all_present": all(bool(row.get("evidence_files_present")) for row in evidence_audit) if evidence_audit else False,
        "approved_skeleton_npz_all_finite": all(bool(row.get("skeleton_npz_finite")) for row in evidence_audit) if evidence_audit else False,
        "failures": failures,
        "ready_for_seed_package": status == "ready_for_v7e_seed_package_inputs",
        "review_manifest_modified": False,
        "seed_package_created": False,
        "dataset_generated": False,
        "training_started": False,
        "validation_started": False,
        "heldout15_started": False,
        "promotion_eligible": False,
        "fallback_policy": "keep_v4_live_demo_fallback",
        "heldout_policy": "heldout */test paths are rejected from v7e paper seed review metadata",
        "next_action": _next_action(status),
        "config": _config_summary(project_root=project_root, config=config),
    }
    (output_root / "v7e_paper_seed_review_validation_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7e_paper_seed_review_validation_summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _status(
    *,
    failures: Sequence[Mapping[str, Any]],
    decision_count: int,
    approval_notes_missing: Sequence[str],
    approve_count: int,
    minimum_approved_count: int,
) -> str:
    if failures:
        return "invalid_decisions"
    if decision_count <= 0:
        return "no_review_decisions"
    if approval_notes_missing:
        return "approval_notes_missing"
    if approve_count < minimum_approved_count:
        return "insufficient_approved_paper_seeds"
    return "ready_for_v7e_seed_package_inputs"


def _next_action(status: str) -> str:
    if status == "ready_for_v7e_seed_package_inputs":
        return "build a v7e paper-expanded seed package and stage1-only dataset after preserving this validation summary"
    if status == "no_review_decisions":
        return "manually review v7e paper temporal evidence and fill approve decisions with notes"
    if status == "approval_notes_missing":
        return "add review_notes for every approved v7e paper seed before seed-package work"
    if status == "insufficient_approved_paper_seeds":
        return "approve enough reviewed v7e hard-paper prompt-window seeds before seed-package work"
    return "fix invalid decisions, heldout paths, evidence links, or skeleton NPZ files before retrying"


def _evidence_audit_row(*, candidate: Mapping[str, str], project_root: Path) -> dict[str, Any]:
    temporal_strip = _resolve_path(project_root, Path(_text(candidate.get("temporal_strip"))))
    preview_image = _resolve_path(project_root, Path(_text(candidate.get("preview_image"))))
    skeleton_npz = _resolve_path(project_root, Path(_text(candidate.get("skeleton_npz"))))
    temporal_exists = temporal_strip.exists()
    preview_exists = preview_image.exists()
    skeleton_exists = skeleton_npz.exists()
    skeleton_finite = False
    skeleton_failure = ""
    if skeleton_exists:
        skeleton_finite, skeleton_failure = _check_skeleton_npz(skeleton_npz)
    return {
        "segment_id": _text(candidate.get("segment_id")),
        "target_name": _text(candidate.get("target_name")),
        "proposal_role": _text(candidate.get("proposal_role")),
        "temporal_strip": _display_path(temporal_strip, base=project_root),
        "temporal_strip_exists": temporal_exists,
        "preview_image": _display_path(preview_image, base=project_root),
        "preview_image_exists": preview_exists,
        "skeleton_npz": _display_path(skeleton_npz, base=project_root),
        "skeleton_npz_exists": skeleton_exists,
        "skeleton_npz_finite": skeleton_finite,
        "skeleton_npz_failure": skeleton_failure,
        "evidence_files_present": temporal_exists and preview_exists and skeleton_exists,
    }


def _evidence_failures(audit: Mapping[str, Any], *, segment_id: str, row: int) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not audit.get("temporal_strip_exists"):
        failures.append({"code": "approved_temporal_strip_missing", "segment_id": segment_id, "row": row})
    if not audit.get("preview_image_exists"):
        failures.append({"code": "approved_preview_image_missing", "segment_id": segment_id, "row": row})
    if not audit.get("skeleton_npz_exists"):
        failures.append({"code": "approved_skeleton_npz_missing", "segment_id": segment_id, "row": row})
    elif not audit.get("skeleton_npz_finite"):
        failures.append(
            {
                "code": "approved_skeleton_npz_nonfinite",
                "segment_id": segment_id,
                "row": row,
                "reason": _text(audit.get("skeleton_npz_failure")),
            }
        )
    return failures


def _check_skeleton_npz(path: Path) -> tuple[bool, str]:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "canonical_landmarks" not in data:
                return False, "missing canonical_landmarks"
            landmarks = data["canonical_landmarks"]
            if landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):
                return False, f"invalid canonical_landmarks shape {landmarks.shape}"
            if not np.isfinite(landmarks).all():
                return False, "nonfinite canonical_landmarks"
    except Exception as exc:  # pragma: no cover - exact numpy errors vary by version
        return False, str(exc)
    return True, ""


def _normalize_decision(value: str) -> str | None:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"approve", "approved", "accept", "accepted"}:
        return "approve"
    if normalized in {"reject", "rejected", "deny", "denied"}:
        return "reject"
    if normalized in {"needs_review", "pending", "pending_manual_review"}:
        return "needs_review"
    return None


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def _config_summary(*, project_root: Path, config: V7EPaperSeedReviewValidatorConfig) -> dict[str, object]:
    summary = asdict(config)
    summary["project_root"] = "."
    for key, value in list(summary.items()):
        if isinstance(value, Path):
            summary[key] = _display_path(_resolve_path(project_root, value), base=project_root)
    return _json_ready(summary)  # type: ignore[return-value]


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


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# V7e Paper Seed Review Validation",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Approve count: `{summary.get('approve_count')}`",
        f"- Minimum approved paper seed count: `{summary.get('minimum_approved_paper_seed_count')}`",
        f"- Ready for seed package: `{summary.get('ready_for_seed_package')}`",
        f"- Approved paper segments: `{summary.get('approved_paper_segment_ids')}`",
        f"- Next action: `{summary.get('next_action')}`",
        "",
    ]
    return "\n".join(lines)


__all__ = ["V7EPaperSeedReviewValidatorConfig", "validate_v7e_paper_seed_review"]
