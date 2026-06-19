"""Non-mutating fill guide for v7e paper seed review approvals."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


BRANCH_LABEL = "v7e_stage1_paper_transition_rescue"
DEFAULT_PLAN_ROOT = Path("artifacts/real_skeleton_v7e_stage1_paper_transition_rescue_plan_20260619")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7e_paper_seed_review_fill_guide_20260619")
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = ("temporal_strip", "preview_image", "skeleton_npz", "source_path")


@dataclass(frozen=True)
class V7EPaperSeedReviewFillGuideConfig:
    """Inputs for writing a reviewer-facing v7e paper approval guide."""

    project_root: Path = field(default_factory=Path.cwd)
    plan_root: Path = DEFAULT_PLAN_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


def write_v7e_paper_seed_review_fill_guide(config: V7EPaperSeedReviewFillGuideConfig) -> dict[str, Any]:
    """Write review suggestions without modifying the approval template or training state."""

    project_root = config.project_root.resolve()
    plan_root = _resolve_path(project_root, config.plan_root)
    output_root = _resolve_path(project_root, config.output_root)
    candidates_csv = plan_root / "v7e_paper_seed_review_candidates.csv"
    decision_template_csv = plan_root / "v7e_paper_seed_review_decision_template.csv"
    if not candidates_csv.exists():
        raise FileNotFoundError(f"Missing v7e paper candidates CSV: {candidates_csv}")
    if not decision_template_csv.exists():
        raise FileNotFoundError(f"Missing v7e paper decision template CSV: {decision_template_csv}")

    template_hash_before = _sha256(decision_template_csv)
    candidates = _read_rows(candidates_csv)
    decisions = _read_rows(decision_template_csv)
    _reject_heldout_rows(candidates, context=candidates_csv)
    _reject_heldout_rows(decisions, context=decision_template_csv)

    output_root.mkdir(parents=True, exist_ok=True)
    evidence_audit = [_candidate_audit(row, project_root=project_root) for row in candidates]
    suggestions = [_suggestion_row(row, audit=audit, destination=decision_template_csv, project_root=project_root) for row, audit in zip(candidates, evidence_audit)]

    suggestions_csv = output_root / "v7e_paper_seed_review_fill_suggestions.csv"
    _write_csv(suggestions_csv, suggestions)
    html_path = output_root / "v7e_paper_seed_review_fill_guide.html"
    html_path.write_text(
        _html_guide(
            suggestions=suggestions,
            evidence_audit=evidence_audit,
            output_root=output_root,
            project_root=project_root,
        ),
        encoding="utf-8",
    )
    template_hash_after = _sha256(decision_template_csv)
    failures = _audit_failures(evidence_audit)
    status = "ready_for_manual_v7e_paper_seed_approval" if not failures else "evidence_incomplete"
    summary: dict[str, Any] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "plan_root": _display_path(plan_root, base=project_root),
        "candidates_csv": _display_path(candidates_csv, base=project_root),
        "decision_template_csv": _display_path(decision_template_csv, base=project_root),
        "output_root": _display_path(output_root, base=project_root),
        "fill_guide_html": _display_path(html_path, base=project_root),
        "fill_suggestions_csv": _display_path(suggestions_csv, base=project_root),
        "candidate_count": len(candidates),
        "suggested_approval_count": len(suggestions),
        "suggested_segment_ids": [row["segment_id"] for row in suggestions],
        "evidence_audit": evidence_audit,
        "evidence_all_present": all(bool(row.get("evidence_files_present")) for row in evidence_audit) if evidence_audit else False,
        "skeleton_npz_all_finite": all(bool(row.get("skeleton_npz_finite")) for row in evidence_audit) if evidence_audit else False,
        "failures": failures,
        "decision_template_sha256_before": template_hash_before,
        "decision_template_sha256_after": template_hash_after,
        "decision_template_modified": template_hash_before != template_hash_after,
        "review_manifest_modified": False,
        "seed_package_created": False,
        "dataset_generated": False,
        "training_started": False,
        "validation_started": False,
        "heldout15_started": False,
        "promotion_eligible": False,
        "fallback_policy": "keep_v4_live_demo_fallback",
        "heldout_policy": "heldout */test paths are rejected from v7e paper review guide metadata",
        "next_action": "copy reviewed approve decisions and nonblank notes into the real v7e paper decision template, then rerun the validator",
        "config": _config_summary(project_root=project_root, config=config),
    }
    (output_root / "v7e_paper_seed_review_fill_guide_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7e_paper_seed_review_fill_guide_summary.md").write_text(_markdown(summary), encoding="utf-8")
    return _json_ready(summary)


def _candidate_audit(row: Mapping[str, str], *, project_root: Path) -> dict[str, Any]:
    temporal_strip = _resolve_path(project_root, Path(_text(row.get("temporal_strip"))))
    preview_image = _resolve_path(project_root, Path(_text(row.get("preview_image"))))
    skeleton_npz = _resolve_path(project_root, Path(_text(row.get("skeleton_npz"))))
    skeleton_finite = False
    skeleton_failure = ""
    if skeleton_npz.exists():
        skeleton_finite, skeleton_failure = _check_skeleton_npz(skeleton_npz)
    return {
        "segment_id": _text(row.get("segment_id")),
        "target_name": _text(row.get("target_name")),
        "proposal_role": _text(row.get("proposal_role")),
        "frame_count": _text(row.get("frame_count")),
        "detection_coverage": _text(row.get("detection_coverage")),
        "temporal_strip": _display_path(temporal_strip, base=project_root),
        "temporal_strip_exists": temporal_strip.exists(),
        "preview_image": _display_path(preview_image, base=project_root),
        "preview_image_exists": preview_image.exists(),
        "skeleton_npz": _display_path(skeleton_npz, base=project_root),
        "skeleton_npz_exists": skeleton_npz.exists(),
        "skeleton_npz_finite": skeleton_finite,
        "skeleton_npz_failure": skeleton_failure,
        "evidence_files_present": temporal_strip.exists() and preview_image.exists() and skeleton_npz.exists(),
    }


def _suggestion_row(
    row: Mapping[str, str],
    *,
    audit: Mapping[str, Any],
    destination: Path,
    project_root: Path,
) -> dict[str, str]:
    segment_id = _text(row.get("segment_id"))
    return {
        "review_priority": _text(row.get("review_priority")),
        "segment_id": segment_id,
        "target_name": _text(row.get("target_name")),
        "proposal_role": _text(row.get("proposal_role")),
        "suggested_decision": "approve",
        "suggested_review_notes": (
            "Approved after temporal review: bounded PROMPT SCISSORS paper resolution for "
            f"v7e stage1 paper-transition rescue ({segment_id})."
        ),
        "destination_decisions_csv": _display_path(destination, base=project_root),
        "copy_required": "true",
        "pipeline_input": "false",
        "temporal_strip": str(audit.get("temporal_strip", "")),
        "preview_image": str(audit.get("preview_image", "")),
        "skeleton_npz": str(audit.get("skeleton_npz", "")),
        "evidence_files_present": str(bool(audit.get("evidence_files_present"))).lower(),
        "skeleton_npz_finite": str(bool(audit.get("skeleton_npz_finite"))).lower(),
    }


def _html_guide(
    *,
    suggestions: Sequence[Mapping[str, str]],
    evidence_audit: Sequence[Mapping[str, Any]],
    output_root: Path,
    project_root: Path,
) -> str:
    rows: list[str] = []
    audits = {str(row.get("segment_id")): row for row in evidence_audit}
    for suggestion in suggestions:
        segment_id = suggestion["segment_id"]
        audit = audits.get(segment_id, {})
        strip_src = _html_src(audit.get("temporal_strip"), output_root=output_root, project_root=project_root)
        preview_src = _html_src(audit.get("preview_image"), output_root=output_root, project_root=project_root)
        rows.append(
            "\n".join(
                [
                    '<section class="candidate">',
                    f"<h2>{html.escape(segment_id)}</h2>",
                    f"<p><strong>Suggested decision:</strong> {html.escape(suggestion['suggested_decision'])}</p>",
                    f"<p><strong>Suggested review notes:</strong> {html.escape(suggestion['suggested_review_notes'])}</p>",
                    f'<img src="{html.escape(strip_src)}" alt="{html.escape(segment_id)} temporal strip">',
                    f'<img src="{html.escape(preview_src)}" alt="{html.escape(segment_id)} preview">',
                    "</section>",
                ]
            )
        )
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>V7e Paper Seed Review Fill Guide</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:24px;color:#1f2933;background:#f8fafc}",
            ".notice{padding:12px;border:1px solid #b45309;background:#fffbeb;margin-bottom:20px}",
            ".candidate{background:white;border:1px solid #d9e2ec;margin:16px 0;padding:16px;border-radius:6px}",
            "img{display:block;max-width:100%;margin:10px 0;border:1px solid #bcccdc}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>V7e Paper Seed Review Fill Guide</h1>",
            '<div class="notice">This file is not a pipeline approval input. Copy only human-reviewed decisions and notes into the real v7e paper decision template.</div>',
            *rows,
            "</body>",
            "</html>",
        ]
    )


def _html_src(value: object, *, output_root: Path, project_root: Path) -> str:
    relative = _text(value)
    if not relative:
        return ""
    source = _resolve_path(project_root, Path(relative))
    try:
        return source.resolve(strict=False).relative_to(output_root.resolve(strict=False)).as_posix()
    except ValueError:
        return os.path.relpath(source.resolve(strict=False), output_root.resolve(strict=False)).replace("\\", "/")


def _audit_failures(evidence_audit: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in evidence_audit:
        segment_id = str(row.get("segment_id"))
        if not row.get("temporal_strip_exists"):
            failures.append({"code": "temporal_strip_missing", "segment_id": segment_id})
        if not row.get("preview_image_exists"):
            failures.append({"code": "preview_image_missing", "segment_id": segment_id})
        if not row.get("skeleton_npz_exists"):
            failures.append({"code": "skeleton_npz_missing", "segment_id": segment_id})
        elif not row.get("skeleton_npz_finite"):
            failures.append({"code": "skeleton_npz_nonfinite", "segment_id": segment_id, "reason": _text(row.get("skeleton_npz_failure"))})
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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["segment_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _config_summary(*, project_root: Path, config: V7EPaperSeedReviewFillGuideConfig) -> dict[str, object]:
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


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# V7e Paper Seed Review Fill Guide",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Candidate count: `{summary.get('candidate_count')}`",
        f"- Suggested approval count: `{summary.get('suggested_approval_count')}`",
        f"- Evidence all present: `{summary.get('evidence_all_present')}`",
        f"- Skeleton NPZ all finite: `{summary.get('skeleton_npz_all_finite')}`",
        f"- Decision template modified: `{summary.get('decision_template_modified')}`",
        f"- Next action: `{summary.get('next_action')}`",
        "",
    ]
    return "\n".join(lines)


__all__ = ["V7EPaperSeedReviewFillGuideConfig", "write_v7e_paper_seed_review_fill_guide"]
