"""Non-mutating approval patch plan for v7e paper seed review."""

from __future__ import annotations

import csv
import difflib
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


BRANCH_LABEL = "v7e_stage1_paper_transition_rescue"
DEFAULT_PLAN_ROOT = Path("artifacts/real_skeleton_v7e_stage1_paper_transition_rescue_plan_20260619")
DEFAULT_FILL_GUIDE_ROOT = Path("artifacts/real_skeleton_v7e_paper_seed_review_fill_guide_20260619")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7e_paper_seed_review_apply_plan_20260619")
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = ("temporal_strip", "preview_image", "skeleton_npz", "source_path")


@dataclass(frozen=True)
class V7EPaperSeedReviewApplyPlanConfig:
    """Inputs for writing a reviewable v7e decision-template patch plan."""

    project_root: Path = field(default_factory=Path.cwd)
    plan_root: Path = DEFAULT_PLAN_ROOT
    fill_guide_root: Path = DEFAULT_FILL_GUIDE_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


def write_v7e_paper_seed_review_apply_plan(config: V7EPaperSeedReviewApplyPlanConfig) -> dict[str, Any]:
    """Write a patch preview for manual approval without applying it."""

    project_root = config.project_root.resolve()
    plan_root = _resolve_path(project_root, config.plan_root)
    fill_guide_root = _resolve_path(project_root, config.fill_guide_root)
    output_root = _resolve_path(project_root, config.output_root)
    decision_template = plan_root / "v7e_paper_seed_review_decision_template.csv"
    suggestions_csv = fill_guide_root / "v7e_paper_seed_review_fill_suggestions.csv"
    if not decision_template.exists():
        raise FileNotFoundError(f"Missing v7e paper decision template CSV: {decision_template}")
    if not suggestions_csv.exists():
        raise FileNotFoundError(f"Missing v7e paper fill suggestions CSV: {suggestions_csv}")

    template_hash_before = _sha256(decision_template)
    template_rows = _read_rows(decision_template)
    suggestion_rows = _read_rows(suggestions_csv)
    _reject_heldout_rows(template_rows, context=decision_template)
    _reject_heldout_rows(suggestion_rows, context=suggestions_csv)
    _reject_pipeline_input_suggestions(suggestion_rows, context=suggestions_csv)

    output_root.mkdir(parents=True, exist_ok=True)
    patched_rows, applied_segment_ids = _patched_rows(template_rows=template_rows, suggestion_rows=suggestion_rows)
    original_csv = _rows_to_csv(template_rows)
    patched_csv = _rows_to_csv(patched_rows)
    patch_lines = list(
        difflib.unified_diff(
            original_csv.splitlines(keepends=True),
            patched_csv.splitlines(keepends=True),
            fromfile=_display_path(decision_template, base=project_root),
            tofile=f"{_display_path(decision_template, base=project_root)}.after_manual_review",
        )
    )
    patch_text = "".join(patch_lines)
    patch_path = output_root / "v7e_paper_seed_review_decision_template.patch"
    patch_path.write_text(patch_text, encoding="utf-8")
    (output_root / "v7e_paper_seed_review_apply_plan.md").write_text(
        _markdown(
            status="ready_for_manual_v7e_paper_seed_patch_review",
            patch_path=_display_path(patch_path, base=project_root),
            applied_segment_ids=applied_segment_ids,
        ),
        encoding="utf-8",
    )
    template_hash_after = _sha256(decision_template)
    summary: dict[str, Any] = {
        "status": "ready_for_manual_v7e_paper_seed_patch_review",
        "branch_label": BRANCH_LABEL,
        "plan_root": _display_path(plan_root, base=project_root),
        "fill_guide_root": _display_path(fill_guide_root, base=project_root),
        "output_root": _display_path(output_root, base=project_root),
        "decision_template_csv": _display_path(decision_template, base=project_root),
        "fill_suggestions_csv": _display_path(suggestions_csv, base=project_root),
        "patch_path": _display_path(patch_path, base=project_root),
        "candidate_template_row_count": len(template_rows),
        "suggested_approval_count": len(applied_segment_ids),
        "suggested_segment_ids": applied_segment_ids,
        "decision_template_sha256_before": template_hash_before,
        "decision_template_sha256_after": template_hash_after,
        "decision_template_modified": template_hash_before != template_hash_after,
        "patch_applied": False,
        "review_manifest_modified": False,
        "seed_package_created": False,
        "dataset_generated": False,
        "training_started": False,
        "validation_started": False,
        "heldout15_started": False,
        "promotion_eligible": False,
        "fallback_policy": "keep_v4_live_demo_fallback",
        "heldout_policy": "heldout */test paths are rejected from v7e paper review apply-plan metadata",
        "next_action": "review the patch and manually apply approved decision/review_notes rows to the real v7e paper decision template",
        "config": _config_summary(project_root=project_root, config=config),
    }
    (output_root / "v7e_paper_seed_review_apply_plan_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return _json_ready(summary)


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


def _config_summary(*, project_root: Path, config: V7EPaperSeedReviewApplyPlanConfig) -> dict[str, object]:
    summary = asdict(config)
    summary["project_root"] = "."
    for key, value in list(summary.items()):
        if isinstance(value, Path):
            summary[key] = _display_path(_resolve_path(project_root, value), base=project_root)
    return _json_ready(summary)  # type: ignore[return-value]


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


def _markdown(*, status: str, patch_path: str, applied_segment_ids: Sequence[str]) -> str:
    lines = [
        "# V7e Paper Seed Review Apply Plan",
        "",
        f"- Status: `{status}`",
        f"- Patch path: `{patch_path}`",
        "- Patch applied: `False`",
        "- This is not a pipeline approval input.",
        "",
        "## Suggested Segments",
        "",
    ]
    for segment_id in applied_segment_ids:
        lines.append(f"- `{segment_id}`")
    return "\n".join(lines) + "\n"


__all__ = ["V7EPaperSeedReviewApplyPlanConfig", "write_v7e_paper_seed_review_apply_plan"]
