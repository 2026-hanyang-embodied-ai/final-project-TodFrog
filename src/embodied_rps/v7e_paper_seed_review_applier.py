"""Guarded applier for v7e paper seed review decisions."""

from __future__ import annotations

import csv
import difflib
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from embodied_rps.v7e_paper_seed_review_validator import (
    DEFAULT_MINIMUM_APPROVED_PAPER_SEED_COUNT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_VALIDATOR_OUTPUT_ROOT,
    V7EPaperSeedReviewValidatorConfig,
    validate_v7e_paper_seed_review,
)


BRANCH_LABEL = "v7e_stage1_paper_transition_rescue"
APPLY_CONFIRMATION_PHRASE = "reviewed_v7e_paper_seed_patch"
DEFAULT_PLAN_ROOT = Path("artifacts/real_skeleton_v7e_stage1_paper_transition_rescue_plan_20260619")
DEFAULT_FILL_GUIDE_ROOT = Path("artifacts/real_skeleton_v7e_paper_seed_review_fill_guide_20260619")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7e_paper_seed_review_applier_20260619")
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = ("temporal_strip", "preview_image", "skeleton_npz", "source_path")
ApplyMode = Literal["dry-run", "apply"]


@dataclass(frozen=True)
class V7EPaperSeedReviewApplierConfig:
    """Inputs for guarded v7e paper decision application."""

    project_root: Path = field(default_factory=Path.cwd)
    plan_root: Path = DEFAULT_PLAN_ROOT
    fill_guide_root: Path = DEFAULT_FILL_GUIDE_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    validator_output_root: Path = DEFAULT_VALIDATOR_OUTPUT_ROOT
    minimum_approved_paper_seed_count: int = DEFAULT_MINIMUM_APPROVED_PAPER_SEED_COUNT
    mode: ApplyMode = "dry-run"
    apply_confirmation: str = ""


def write_v7e_paper_seed_review_applier(config: V7EPaperSeedReviewApplierConfig) -> dict[str, Any]:
    """Dry-run or explicitly apply v7e paper review decisions from the fill guide."""

    project_root = config.project_root.resolve()
    plan_root = _resolve_path(project_root, config.plan_root)
    fill_guide_root = _resolve_path(project_root, config.fill_guide_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    decision_template = plan_root / "v7e_paper_seed_review_decision_template.csv"
    suggestions_csv = fill_guide_root / "v7e_paper_seed_review_fill_suggestions.csv"
    if not decision_template.exists():
        raise FileNotFoundError(f"Missing v7e paper decision template CSV: {decision_template}")
    if not suggestions_csv.exists():
        raise FileNotFoundError(f"Missing v7e paper fill suggestions CSV: {suggestions_csv}")

    template_rows = _read_rows(decision_template)
    suggestion_rows = _read_rows(suggestions_csv)
    _reject_heldout_rows(template_rows, context=decision_template)
    _reject_heldout_rows(suggestion_rows, context=suggestions_csv)
    _reject_pipeline_input_suggestions(suggestion_rows, context=suggestions_csv)
    patched_rows, applied_segment_ids = _patched_rows(template_rows=template_rows, suggestion_rows=suggestion_rows)
    before_hash = _sha256(decision_template)
    original_csv = _rows_to_csv(template_rows)
    patched_csv = _rows_to_csv(patched_rows)
    patch_path = output_root / "v7e_paper_seed_review_decision_template.patch"
    patch_path.write_text(
        "".join(
            difflib.unified_diff(
                original_csv.splitlines(keepends=True),
                patched_csv.splitlines(keepends=True),
                fromfile=_display_path(decision_template, base=project_root),
                tofile=f"{_display_path(decision_template, base=project_root)}.after_manual_review",
            )
        ),
        encoding="utf-8",
    )

    confirmation_guard = _confirmation_guard(config.apply_confirmation)
    validator_summary: dict[str, Any] | None = None
    patch_applied = False
    if config.mode == "apply":
        if not confirmation_guard["confirmation_valid"]:
            summary = _summary(
                status="apply_confirmation_required",
                project_root=project_root,
                plan_root=plan_root,
                fill_guide_root=fill_guide_root,
                output_root=output_root,
                decision_template=decision_template,
                suggestions_csv=suggestions_csv,
                patch_path=patch_path,
                config=config,
                applied_segment_ids=applied_segment_ids,
                before_hash=before_hash,
                after_hash=_sha256(decision_template),
                patch_applied=False,
                confirmation_guard=confirmation_guard,
                validator_summary=None,
            )
            _write_summary(output_root, summary)
            return summary
        backup_path = output_root / "v7e_paper_seed_review_decision_template.before_apply.csv"
        backup_path.write_text(original_csv, encoding="utf-8")
        decision_template.write_text(patched_csv, encoding="utf-8")
        patch_applied = True
        validator_summary = validate_v7e_paper_seed_review(
            V7EPaperSeedReviewValidatorConfig(
                project_root=project_root,
                plan_root=config.plan_root,
                output_root=config.validator_output_root,
                minimum_approved_paper_seed_count=config.minimum_approved_paper_seed_count,
            )
        )
    after_hash = _sha256(decision_template)
    if config.mode == "dry-run":
        status = "dry_run_ready_for_v7e_paper_seed_apply"
    elif validator_summary and validator_summary.get("status") == "ready_for_v7e_seed_package_inputs":
        status = "applied_ready_for_v7e_seed_package_inputs"
    else:
        status = "apply_validation_failed"
    summary = _summary(
        status=status,
        project_root=project_root,
        plan_root=plan_root,
        fill_guide_root=fill_guide_root,
        output_root=output_root,
        decision_template=decision_template,
        suggestions_csv=suggestions_csv,
        patch_path=patch_path,
        config=config,
        applied_segment_ids=applied_segment_ids,
        before_hash=before_hash,
        after_hash=after_hash,
        patch_applied=patch_applied,
        confirmation_guard=confirmation_guard,
        validator_summary=validator_summary,
    )
    _write_summary(output_root, summary)
    return summary


def _summary(
    *,
    status: str,
    project_root: Path,
    plan_root: Path,
    fill_guide_root: Path,
    output_root: Path,
    decision_template: Path,
    suggestions_csv: Path,
    patch_path: Path,
    config: V7EPaperSeedReviewApplierConfig,
    applied_segment_ids: Sequence[str],
    before_hash: str,
    after_hash: str,
    patch_applied: bool,
    confirmation_guard: Mapping[str, Any],
    validator_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _json_ready(
        {
            "status": status,
            "branch_label": BRANCH_LABEL,
            "mode": config.mode,
            "plan_root": _display_path(plan_root, base=project_root),
            "fill_guide_root": _display_path(fill_guide_root, base=project_root),
            "output_root": _display_path(output_root, base=project_root),
            "decision_template_csv": _display_path(decision_template, base=project_root),
            "fill_suggestions_csv": _display_path(suggestions_csv, base=project_root),
            "patch_path": _display_path(patch_path, base=project_root),
            "suggested_approval_count": len(applied_segment_ids),
            "suggested_segment_ids": list(applied_segment_ids),
            "decision_template_sha256_before": before_hash,
            "decision_template_sha256_after": after_hash,
            "decision_template_modified": before_hash != after_hash,
            "patch_applied": patch_applied,
            "apply_confirmation_guard": dict(confirmation_guard),
            "validator_summary": validator_summary,
            "review_manifest_modified": False,
            "seed_package_created": False,
            "dataset_generated": False,
            "training_started": False,
            "validation_started": False,
            "heldout15_started": False,
            "promotion_eligible": False,
            "fallback_policy": "keep_v4_live_demo_fallback",
            "heldout_policy": "heldout */test paths are rejected from v7e paper review applier metadata",
            "next_action": _next_action(status),
            "config": _config_summary(project_root=project_root, config=config),
        }
    )


def _next_action(status: str) -> str:
    if status == "dry_run_ready_for_v7e_paper_seed_apply":
        return f"rerun with --mode apply --apply-confirmation {APPLY_CONFIRMATION_PHRASE} only after manual patch review"
    if status == "apply_confirmation_required":
        return f"apply requires --apply-confirmation {APPLY_CONFIRMATION_PHRASE}"
    if status == "applied_ready_for_v7e_seed_package_inputs":
        return "run the v7e seed-package builder, then stage1 dataset generation and smoke gates"
    return "inspect validator_summary and fix v7e paper decisions before seed-package work"


def _patched_rows(*, template_rows: Sequence[Mapping[str, str]], suggestion_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    suggestions_by_id = {_text(row.get("segment_id")): row for row in suggestion_rows if _text(row.get("segment_id"))}
    patched: list[dict[str, str]] = []
    applied: list[str] = []
    for row in template_rows:
        copied = dict(row)
        segment_id = _text(copied.get("segment_id"))
        suggestion = suggestions_by_id.get(segment_id)
        if suggestion is not None:
            copied["decision"] = _text(suggestion.get("suggested_decision"))
            copied["review_notes"] = _text(suggestion.get("suggested_review_notes"))
            applied.append(segment_id)
        patched.append(copied)
    return patched, applied


def _confirmation_guard(actual_confirmation: str) -> dict[str, Any]:
    actual = actual_confirmation.strip()
    return {
        "required_confirmation": APPLY_CONFIRMATION_PHRASE,
        "confirmation_present": bool(actual),
        "confirmation_valid": actual == APPLY_CONFIRMATION_PHRASE,
    }


def _reject_pipeline_input_suggestions(rows: Sequence[Mapping[str, str]], *, context: Path) -> None:
    for row in rows:
        if _text(row.get("pipeline_input")).lower() not in {"false", "0", "no"}:
            raise ValueError(f"{context} suggestions must not be a pipeline approval input")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _rows_to_csv(rows: Sequence[Mapping[str, str]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


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


def _config_summary(*, project_root: Path, config: V7EPaperSeedReviewApplierConfig) -> dict[str, object]:
    summary = asdict(config)
    summary["project_root"] = "."
    for key, value in list(summary.items()):
        if isinstance(value, Path):
            summary[key] = _display_path(_resolve_path(project_root, value), base=project_root)
    return _json_ready(summary)  # type: ignore[return-value]


def _write_summary(output_root: Path, summary: Mapping[str, Any]) -> None:
    (output_root / "v7e_paper_seed_review_applier_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7e_paper_seed_review_applier_summary.md").write_text(_markdown(summary), encoding="utf-8")


def _markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# V7e Paper Seed Review Applier",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Mode: `{summary.get('mode')}`",
            f"- Patch applied: `{summary.get('patch_applied')}`",
            f"- Decision template modified: `{summary.get('decision_template_modified')}`",
            f"- Suggested approval count: `{summary.get('suggested_approval_count')}`",
            f"- Training started: `{summary.get('training_started')}`",
            f"- Promotion eligible: `{summary.get('promotion_eligible')}`",
            f"- Next action: `{summary.get('next_action')}`",
            "",
        ]
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


__all__ = ["APPLY_CONFIRMATION_PHRASE", "V7EPaperSeedReviewApplierConfig", "write_v7e_paper_seed_review_applier"]
