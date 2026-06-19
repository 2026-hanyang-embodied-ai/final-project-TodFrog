"""Materialize a filled v7d three-row selection sheet into a validator-ready decision CSV."""

from __future__ import annotations

import csv
import json
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from embodied_rps.v7d_review_decision_validator import (
    V7DReviewDecisionValidatorConfig,
    validate_v7d_review_decisions,
)

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_REVIEW_ROOT = Path("artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618")
DEFAULT_SELECTION_ROOT = Path("artifacts/real_skeleton_v7d_manual_approval_selection_20260618")
DEFAULT_SHORTLIST_ROOT = Path("artifacts/real_skeleton_v7d_manual_review_shortlist_20260618")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_selection_decision_materialization_20260618")
REQUIRED_ROLES: tuple[str, ...] = (
    "hard_paper_prompt_window",
    "rock_wait_prompt_window",
    "scissors_boundary_control",
)
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = (
    "temporal_strip",
    "preview_image",
    "skeleton_npz",
    "decision_template_csv",
    "source_path",
)


@dataclass(frozen=True)
class V7DSelectionDecisionMaterializerConfig:
    """Inputs for turning the compact selection sheet into a decision CSV."""

    project_root: Path = field(default_factory=Path.cwd)
    review_root: Path = DEFAULT_REVIEW_ROOT
    selection_root: Path = DEFAULT_SELECTION_ROOT
    shortlist_root: Path = DEFAULT_SHORTLIST_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


def write_v7d_selection_decision_materialization(
    config: V7DSelectionDecisionMaterializerConfig,
) -> dict[str, object]:
    """Write a derived decision CSV and validation summary without mutating review state."""

    project_root = config.project_root.resolve()
    review_root = _resolve_path(project_root, config.review_root)
    selection_root = _resolve_path(project_root, config.selection_root)
    shortlist_root = _resolve_path(project_root, config.shortlist_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selection_csv = selection_root / "approval_selection_template.csv"
    options_csv = selection_root / "approval_selection_options.csv"
    decision_template_csv = shortlist_root / "seed_required_decision_template.csv"
    for path in (selection_csv, options_csv, decision_template_csv):
        if not path.exists():
            raise FileNotFoundError(f"Missing v7d selection materialization input: {path}")

    selection_rows, selection_fieldnames = _read_csv_with_fieldnames(selection_csv)
    option_rows, _ = _read_csv_with_fieldnames(options_csv)
    template_rows, template_fieldnames = _read_csv_with_fieldnames(decision_template_csv)
    if not {"segment_id", "decision", "review_notes"}.issubset(set(template_fieldnames)):
        raise ValueError(f"{decision_template_csv} is missing decision materialization columns")

    option_ids_by_role: dict[str, set[str]] = {role: set() for role in REQUIRED_ROLES}
    option_by_segment: dict[str, dict[str, str]] = {}
    for row in option_rows:
        _reject_heldout_metadata(row, context=options_csv)
        role = str(row.get("proposal_role", "")).strip()
        segment_id = str(row.get("segment_id", "")).strip()
        if role in option_ids_by_role and segment_id:
            option_ids_by_role[role].add(segment_id)
            option_by_segment[segment_id] = dict(row)

    selected_by_role: dict[str, dict[str, str]] = {}
    failures: list[dict[str, object]] = []
    seen_selected_ids: set[str] = set()
    for row_number, row in enumerate(selection_rows, start=2):
        _reject_heldout_metadata(row, context=selection_csv)
        role = str(row.get("proposal_role", "")).strip()
        if role not in REQUIRED_ROLES:
            continue
        selected_id = str(row.get("selected_segment_id", "")).strip()
        decision = _normalize_decision(str(row.get("decision", "")).strip())
        review_notes = str(row.get("review_notes", "")).strip()
        available_ids = _split_available_ids(str(row.get("available_segment_ids", "")).strip())
        if not selected_id and not decision and not review_notes:
            continue
        if not selected_id or not decision:
            failures.append({"code": "incomplete_selection_row", "proposal_role": role, "row": row_number})
            continue
        if decision != "approve":
            failures.append(
                {"code": "unsupported_selection_decision", "proposal_role": role, "decision": decision, "row": row_number}
            )
            continue
        if not review_notes:
            failures.append({"code": "selection_approval_missing_review_notes", "proposal_role": role, "row": row_number})
            continue
        if selected_id not in available_ids:
            failures.append(
                {
                    "code": "selected_segment_not_in_available_ids",
                    "proposal_role": role,
                    "segment_id": selected_id,
                    "row": row_number,
                }
            )
            continue
        if selected_id not in option_ids_by_role.get(role, set()):
            failures.append(
                {
                    "code": "selected_segment_missing_from_selection_options",
                    "proposal_role": role,
                    "segment_id": selected_id,
                    "row": row_number,
                }
            )
            continue
        if selected_id in seen_selected_ids:
            failures.append({"code": "duplicate_selected_segment_id", "segment_id": selected_id, "row": row_number})
            continue
        if role in selected_by_role:
            failures.append({"code": "duplicate_selected_role", "proposal_role": role, "row": row_number})
            continue
        seen_selected_ids.add(selected_id)
        selected_by_role[role] = {
            "segment_id": selected_id,
            "decision": "approve",
            "review_notes": review_notes,
        }

    template_by_segment = {
        str(row.get("segment_id", "")).strip(): row
        for row in template_rows
        if str(row.get("segment_id", "")).strip()
    }
    for role, selected in selected_by_role.items():
        segment_id = selected["segment_id"]
        template_row = template_by_segment.get(segment_id)
        if template_row is None:
            failures.append(
                {
                    "code": "selected_segment_missing_from_decision_template",
                    "proposal_role": role,
                    "segment_id": segment_id,
                }
            )
        else:
            _reject_heldout_metadata(template_row, context=decision_template_csv)

    selected_evidence_audit = _selected_evidence_audit(
        project_root=project_root,
        selection_root=selection_root,
        selected_by_role=selected_by_role,
        option_by_segment=option_by_segment,
    )
    failures.extend(_selected_evidence_failures(selected_evidence_audit))

    materialized_rows = _materialized_rows(template_rows=template_rows, selected_by_role=selected_by_role)
    materialized_csv = output_root / "seed_required_decision_template_from_selection.csv"
    _write_csv(materialized_csv, template_fieldnames, materialized_rows)

    validator_output_root = output_root / "validator"
    validation_summary: dict[str, object] | None = None
    if not failures:
        validation_summary = validate_v7d_review_decisions(
            V7DReviewDecisionValidatorConfig(
                project_root=project_root,
                review_root=review_root,
                decisions_csv=materialized_csv,
                output_root=validator_output_root,
            )
        )

    selected_count = len(selected_by_role)
    missing_selected_roles = [role for role in REQUIRED_ROLES if role not in selected_by_role]
    selected_evidence_all_present = all(bool(row.get("evidence_files_present")) for row in selected_evidence_audit)
    selected_skeleton_npz_all_finite = all(bool(row.get("skeleton_npz_finite")) for row in selected_evidence_audit)
    validation_status = str(validation_summary.get("status", "")) if validation_summary else ""
    decisions_apply_safe = bool(validation_summary and validation_summary.get("decisions_apply_safe") is True and not failures)
    status = _status(
        failures=failures,
        selected_count=selected_count,
        missing_selected_roles=missing_selected_roles,
        validation_status=validation_status,
        decisions_apply_safe=decisions_apply_safe,
    )
    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "review_root": _display_path(review_root, base=project_root),
        "selection_root": _display_path(selection_root, base=project_root),
        "shortlist_root": _display_path(shortlist_root, base=project_root),
        "output_root": _display_path(output_root, base=project_root),
        "selection_csv": _display_path(selection_csv, base=project_root),
        "options_csv": _display_path(options_csv, base=project_root),
        "source_decision_template_csv": _display_path(decision_template_csv, base=project_root),
        "materialized_decisions_csv": _display_path(materialized_csv, base=project_root),
        "validation_output_root": _display_path(validator_output_root, base=project_root),
        "selected_count": selected_count,
        "selected_segment_ids_by_role": {
            role: selected_by_role[role]["segment_id"] for role in REQUIRED_ROLES if role in selected_by_role
        },
        "selected_evidence_audit": selected_evidence_audit,
        "selected_evidence_all_present": selected_evidence_all_present,
        "selected_skeleton_npz_all_finite": selected_skeleton_npz_all_finite,
        "missing_required_selected_roles": missing_selected_roles,
        "failures": failures,
        "validation_status": validation_status,
        "validation_summary": validation_summary,
        "decisions_apply_safe": decisions_apply_safe,
        "decision_template_modified": False,
        "review_manifest_modified": False,
        "seed_package_created": False,
        "dataset_generated": False,
        "training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test paths are rejected from selection and materialized decision metadata",
        "next_action": _next_action(status),
        "config": _config_summary(project_root=project_root, config=config),
    }
    (output_root / "selection_decision_materialization_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "selection_decision_materialization_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _materialized_rows(
    *,
    template_rows: Sequence[Mapping[str, object]],
    selected_by_role: Mapping[str, Mapping[str, str]],
) -> list[dict[str, object]]:
    selected_by_segment = {
        selected["segment_id"]: selected
        for selected in selected_by_role.values()
        if selected.get("segment_id")
    }
    rows: list[dict[str, object]] = []
    for row in template_rows:
        materialized = dict(row)
        segment_id = str(row.get("segment_id", "")).strip()
        materialized["decision"] = ""
        materialized["review_notes"] = ""
        if segment_id in selected_by_segment:
            materialized["decision"] = selected_by_segment[segment_id]["decision"]
            materialized["review_notes"] = selected_by_segment[segment_id]["review_notes"]
        rows.append(materialized)
    return rows


def _selected_evidence_audit(
    *,
    project_root: Path,
    selection_root: Path,
    selected_by_role: Mapping[str, Mapping[str, str]],
    option_by_segment: Mapping[str, Mapping[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role in REQUIRED_ROLES:
        selected = selected_by_role.get(role)
        if selected is None:
            continue
        segment_id = selected["segment_id"]
        option = option_by_segment.get(segment_id, {})
        temporal_path = _resolve_selection_link(selection_root, str(option.get("temporal_strip", "")).strip())
        preview_path = _resolve_selection_link(selection_root, str(option.get("preview_image", "")).strip())
        skeleton_path = _resolve_selection_link(selection_root, str(option.get("skeleton_npz", "")).strip())
        temporal_exists = temporal_path.exists()
        preview_exists = preview_path.exists()
        skeleton_exists = skeleton_path.exists()
        skeleton_finite = False
        skeleton_failure = ""
        if skeleton_exists:
            skeleton_finite, skeleton_failure = _check_skeleton_npz(skeleton_path)
        rows.append(
            {
                "proposal_role": role,
                "segment_id": segment_id,
                "target_name": str(option.get("target_name", "")).strip(),
                "temporal_strip": _display_path(temporal_path, base=project_root),
                "temporal_strip_exists": temporal_exists,
                "temporal_strip_sha256": _file_sha256(temporal_path) if temporal_exists else "",
                "preview_image": _display_path(preview_path, base=project_root),
                "preview_image_exists": preview_exists,
                "preview_image_sha256": _file_sha256(preview_path) if preview_exists else "",
                "skeleton_npz": _display_path(skeleton_path, base=project_root),
                "skeleton_npz_exists": skeleton_exists,
                "skeleton_npz_sha256": _file_sha256(skeleton_path) if skeleton_exists else "",
                "skeleton_npz_finite": skeleton_finite,
                "skeleton_npz_failure": skeleton_failure,
                "evidence_files_present": temporal_exists and preview_exists and skeleton_exists,
            }
        )
    return rows


def _selected_evidence_failures(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for row in rows:
        role = str(row.get("proposal_role", "")).strip()
        segment_id = str(row.get("segment_id", "")).strip()
        if not row.get("temporal_strip_exists"):
            failures.append({"code": "selected_temporal_strip_missing", "proposal_role": role, "segment_id": segment_id})
        if not row.get("preview_image_exists"):
            failures.append({"code": "selected_preview_image_missing", "proposal_role": role, "segment_id": segment_id})
        if not row.get("skeleton_npz_exists"):
            failures.append({"code": "selected_skeleton_npz_missing", "proposal_role": role, "segment_id": segment_id})
        elif not row.get("skeleton_npz_finite"):
            failures.append(
                {
                    "code": "selected_skeleton_npz_nonfinite",
                    "proposal_role": role,
                    "segment_id": segment_id,
                    "reason": str(row.get("skeleton_npz_failure", "")).strip(),
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


def _status(
    *,
    failures: Sequence[Mapping[str, object]],
    selected_count: int,
    missing_selected_roles: Sequence[str],
    validation_status: str,
    decisions_apply_safe: bool,
) -> str:
    if failures:
        return "invalid_selection"
    if selected_count <= 0:
        return "no_selection_decisions"
    if missing_selected_roles:
        return "incomplete_selection"
    if decisions_apply_safe and validation_status == "ready_for_apply":
        return "ready_for_review_decision_apply"
    return "selection_materialized_but_validation_blocked"


def _next_action(status: str) -> str:
    if status == "ready_for_review_decision_apply":
        return (
            "use the materialized decision CSV with run_v7d_post_approval_pipeline --review-decision-mode dry-run, "
            "then apply only after confirming the validator summary"
        )
    if status == "no_selection_decisions":
        return "fill approval_selection_template.csv with one approved segment_id and nonblank review_notes per required role"
    if status == "incomplete_selection":
        return "complete every required role in approval_selection_template.csv before materializing decisions"
    if status == "invalid_selection":
        return "fix invalid selected segment IDs, decisions, review notes, or heldout metadata before retrying"
    return "inspect the embedded validation summary before any apply step"


def _summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V7d Selection Decision Materialization",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Selected count: `{summary.get('selected_count')}`",
        f"- Missing selected roles: `{summary.get('missing_required_selected_roles')}`",
        f"- Validation status: `{summary.get('validation_status')}`",
        f"- Decisions apply safe: `{summary.get('decisions_apply_safe')}`",
        f"- Materialized decisions CSV: `{summary.get('materialized_decisions_csv')}`",
        "- This artifact does not edit `seed_required_decision_template.csv` or `segment_review_manifest.csv`.",
        "- Run `validate_v7d_review_decisions` or the v7d post-approval pipeline dry-run before any apply step.",
        "",
        "## Next Command",
        "",
        "```powershell",
        "$env:PYTHONPATH='src'",
        (
            "python -m embodied_rps.tools.run_v7d_post_approval_pipeline "
            "--decisions-csv artifacts/real_skeleton_v7d_selection_decision_materialization_20260618/"
            "seed_required_decision_template_from_selection.csv --review-decision-mode dry-run"
        ),
        "```",
        "",
    ]
    return "\n".join(lines)


def _read_csv_with_fieldnames(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _reject_heldout_metadata(row: Mapping[str, object], *, context: Path) -> None:
    for field_name in PATH_FIELDS_TO_AUDIT:
        value = str(row.get(field_name, "")).strip()
        if value and _is_heldout_test_path(value):
            raise ValueError(f"{context} contains held-out test path in {field_name}: {value}")


def _is_heldout_test_path(value: str) -> bool:
    normalized = value.replace("\\", "/").lower().rstrip("/")
    return "/test/" in normalized or normalized.endswith("/test")


def _split_available_ids(value: str) -> set[str]:
    return {part.strip() for part in value.split(";") if part.strip()}


def _normalize_decision(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _resolve_selection_link(selection_root: Path, value: str) -> Path:
    if not value:
        return selection_root / "__missing_evidence_link__"
    path = Path(value)
    return path if path.is_absolute() else selection_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(base.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _config_summary(*, project_root: Path, config: V7DSelectionDecisionMaterializerConfig) -> dict[str, object]:
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
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["V7DSelectionDecisionMaterializerConfig", "write_v7d_selection_decision_materialization"]
