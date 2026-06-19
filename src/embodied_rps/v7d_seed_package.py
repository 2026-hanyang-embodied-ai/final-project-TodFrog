"""Build the v7d prompt-pose seed package after manual review approval."""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodied_rps.v7_rps_seed_package import build_v7_rps_seed_package
from embodied_rps.v7d_prompt_pose_collection import BRANCH_LABEL, DEFAULT_REVIEW_ROOT

DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_prompt_pose_seed_package_20260618")
DEFAULT_READINESS_ROOT = Path("artifacts/real_skeleton_v7d_seed_readiness_20260618")
REQUIRED_APPROVED_ROLES: tuple[str, ...] = (
    "hard_paper_prompt_window",
    "rock_wait_prompt_window",
    "scissors_boundary_control",
)
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = (
    "source_path",
    "source_overlay_video",
    "source_skeleton_npz",
    "source_frame_log",
    "skeleton_npz",
    "preview_image",
)
REQUIRED_REVIEW_FILES: tuple[str, ...] = ("proposed_segments.jsonl", "segment_review_manifest.csv")
OPTIONAL_REVIEW_FILES: tuple[str, ...] = (
    "auto_quality_pass_segments.csv",
    "segment_review_packet.md",
    "segment_review_worklist.csv",
    "segment_review_worklist.md",
    "segment_review_gallery.html",
    "segment_review_decision_template.csv",
    "segment_review_decision_template.md",
    "review_contact_sheet.png",
    "segment_proposal_summary.json",
    "segment_proposal_summary.md",
)
OPTIONAL_REVIEW_DIRS: tuple[str, ...] = ("segments", "previews")


@dataclass(frozen=True)
class V7DSeedPackageConfig:
    """Inputs for building the approved v7d seed package."""

    project_root: Path = field(default_factory=Path.cwd)
    review_root: Path = DEFAULT_REVIEW_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    sequence_length: int = 72
    overwrite: bool = False


@dataclass(frozen=True)
class V7DSeedReadinessConfig:
    """Inputs for checking whether v7d seed-package creation is allowed."""

    project_root: Path = field(default_factory=Path.cwd)
    review_root: Path = DEFAULT_REVIEW_ROOT
    output_root: Path = DEFAULT_READINESS_ROOT


def build_v7d_prompt_pose_seed_package(config: V7DSeedPackageConfig) -> dict[str, object]:
    """Copy approved review artifacts into the v7d seed root and build the seed NPZ."""

    project_root = config.project_root.resolve()
    review_root = _resolve_path(project_root, config.review_root)
    output_root = _resolve_path(project_root, config.output_root)
    if not review_root.exists():
        raise FileNotFoundError(f"Missing v7d prompt-pose review root: {review_root}")
    for filename in REQUIRED_REVIEW_FILES:
        if not (review_root / filename).exists():
            raise FileNotFoundError(f"Missing required v7d review file: {review_root / filename}")
    readiness = check_v7d_prompt_pose_seed_readiness(
        V7DSeedReadinessConfig(
            project_root=project_root,
            review_root=review_root,
            output_root=project_root / DEFAULT_READINESS_ROOT,
        )
    )
    if readiness["status"] != "ready_for_v7d_seed_package":
        missing = ", ".join(str(role) for role in readiness.get("missing_required_approved_roles", []))
        raise ValueError(f"v7d manual review gate is not cleared; missing approved roles: {missing}")
    if output_root.exists():
        if not config.overwrite:
            raise FileExistsError(f"v7d seed package root already exists: {output_root}")
        _safe_remove_output_root(project_root=project_root, output_root=output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for filename in (*REQUIRED_REVIEW_FILES, *OPTIONAL_REVIEW_FILES):
        source = review_root / filename
        if source.exists():
            shutil.copy2(source, output_root / filename)
    for dirname in OPTIONAL_REVIEW_DIRS:
        source_dir = review_root / dirname
        if source_dir.exists():
            shutil.copytree(source_dir, output_root / dirname)

    builder_summary = build_v7_rps_seed_package(output_root=output_root, sequence_length=config.sequence_length)
    summary: dict[str, object] = {
        "status": builder_summary.get("status"),
        "branch_label": BRANCH_LABEL,
        "review_root": _display_path(review_root, base=project_root),
        "output_root": _display_path(output_root, base=project_root),
        "sequence_length": config.sequence_length,
        "readiness_summary": _relativize_paths(readiness, project_root=project_root),
        "builder_summary": _relativize_paths(builder_summary, project_root=project_root),
        "training_started": False,
        "dataset_generated": False,
        "promotion_eligible": False,
        "review_gate": "manual_approval_required_before_seed_npz_contains_samples",
        "heldout_policy": "heldout */test MP4s remain validation-only and are rejected before seed packaging",
    }
    (output_root / "v7d_seed_package_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7d_seed_package_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def check_v7d_prompt_pose_seed_readiness(config: V7DSeedReadinessConfig) -> dict[str, object]:
    """Write and return the fail-closed v7d manual-review readiness status."""

    project_root = config.project_root.resolve()
    review_root = _resolve_path(project_root, config.review_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    proposed_path = review_root / "proposed_segments.jsonl"
    manifest_path = review_root / "segment_review_manifest.csv"
    if not proposed_path.exists():
        raise FileNotFoundError(f"Missing v7d proposed segments: {proposed_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing v7d review manifest: {manifest_path}")

    proposed_rows = _read_jsonl(proposed_path)
    manifest_rows = _read_csv(manifest_path)
    manifest_by_id = {str(row.get("segment_id", "")).strip(): row for row in manifest_rows}
    approved_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    quality_pass_counts = {role: 0 for role in REQUIRED_APPROVED_ROLES}
    approved_counts = {role: 0 for role in REQUIRED_APPROVED_ROLES}
    target_counts: dict[str, int] = {}
    approved_segment_ids_by_role: dict[str, list[str]] = {role: [] for role in REQUIRED_APPROVED_ROLES}

    for proposed in proposed_rows:
        segment_id = str(proposed.get("segment_id", "")).strip()
        role = str(proposed.get("proposal_role", "")).strip()
        if role in quality_pass_counts and proposed.get("quality_status") == "auto_quality_pass":
            quality_pass_counts[role] += 1
        _audit_heldout_paths(proposed, failures=failures, context=proposed_path, segment_id=segment_id)
        manifest = manifest_by_id.get(segment_id)
        if manifest is None:
            failures.append({"code": "proposed_segment_missing_from_review_manifest", "segment_id": segment_id})
            continue
        approved = _truthy(manifest.get("approved_for_training")) and str(manifest.get("review_status", "")).lower() == "approved"
        if not approved:
            continue
        if proposed.get("quality_status") != "auto_quality_pass":
            failures.append({"code": "approved_segment_failed_auto_quality", "segment_id": segment_id, "role": role})
            continue
        if role in approved_counts:
            approved_counts[role] += 1
            approved_segment_ids_by_role[role].append(segment_id)
        target = str(proposed.get("target_name", "")).strip()
        if target:
            target_counts[target] = target_counts.get(target, 0) + 1
        approved_rows.append(dict(proposed))

    missing_required = [role for role, count in approved_counts.items() if count <= 0]
    status = "ready_for_v7d_seed_package" if not missing_required and not failures else "blocked_manual_approval_required"
    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "review_root": _display_path(review_root, base=project_root),
        "output_root": _display_path(output_root, base=project_root),
        "required_approved_roles": list(REQUIRED_APPROVED_ROLES),
        "missing_required_approved_roles": missing_required,
        "approved_counts_by_role": approved_counts,
        "quality_pass_counts_by_role": quality_pass_counts,
        "approved_counts_by_target": target_counts,
        "approved_segment_count": len(approved_rows),
        "approved_segment_ids_by_role": approved_segment_ids_by_role,
        "failures": failures,
        "training_started": False,
        "dataset_generated": False,
        "seed_package_created": False,
        "heldout_policy": "heldout */test MP4s remain validation-only and are rejected from v7d seed readiness metadata",
        "next_actions": _readiness_next_actions(missing_required=missing_required, failures=failures),
    }
    (output_root / "v7d_seed_readiness_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7d_seed_readiness_summary.md").write_text(_readiness_markdown(summary), encoding="utf-8")
    return summary


def _safe_remove_output_root(*, project_root: Path, output_root: Path) -> None:
    resolved = output_root.resolve()
    resolved.relative_to(project_root.resolve())
    if "v7d_prompt_pose_seed_package" not in resolved.name:
        raise ValueError(f"Refusing to overwrite unexpected v7d seed package root: {output_root}")
    shutil.rmtree(resolved)


def _summary_markdown(summary: Mapping[str, object]) -> str:
    builder = summary.get("builder_summary", {})
    builder_status = builder.get("status") if isinstance(builder, Mapping) else None
    approved = builder.get("approved_segment_count") if isinstance(builder, Mapping) else None
    return "\n".join(
        [
            "# V7d Prompt-Pose Seed Package Summary",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Branch: `{summary.get('branch_label')}`",
            f"- Builder status: `{builder_status}`",
            f"- Approved segments: `{approved}`",
            f"- Review root: `{summary.get('review_root')}`",
            f"- Output root: `{summary.get('output_root')}`",
            "- Training started: `False`",
            "- Dataset generated: `False`",
            "- Heldout policy: validation-only `*/test` MP4s are rejected before seed packaging.",
            "",
        ]
    )


def _readiness_next_actions(*, missing_required: Sequence[str], failures: Sequence[Mapping[str, object]]) -> list[str]:
    if failures:
        return ["fix v7d review manifest/proposal failures before applying approvals or building seed package"]
    if missing_required:
        return [
            "fill explicit manual decisions for the missing required roles",
            "dry-run apply_v7_segment_review_decisions before applying approvals",
            "rerun v7d seed readiness after approvals are applied",
        ]
    return ["build the v7d prompt-pose seed package, then generate/remap v7d datasets"]


def _readiness_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V7d Seed Readiness Summary",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Review root: `{summary.get('review_root')}`",
        f"- Approved segment count: `{summary.get('approved_segment_count')}`",
        "- Training started: `False`",
        "- Dataset generated: `False`",
        "- Seed package created: `False`",
        "",
        "## Required Roles",
        "",
    ]
    approved_counts = summary.get("approved_counts_by_role", {})
    missing = set(str(role) for role in summary.get("missing_required_approved_roles", []))
    if isinstance(approved_counts, Mapping):
        for role, count in approved_counts.items():
            state = "missing" if str(role) in missing else "present"
            lines.append(f"- `{role}`: approved={count}, state=`{state}`")
    lines.extend(["", "## Next Actions", ""])
    actions = summary.get("next_actions", [])
    if isinstance(actions, Sequence):
        for action in actions:
            lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, Mapping):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(dict(value))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _audit_heldout_paths(
    row: Mapping[str, object],
    *,
    failures: list[dict[str, object]],
    context: Path,
    segment_id: str,
) -> None:
    for field_name in PATH_FIELDS_TO_AUDIT:
        value = str(row.get(field_name, "")).strip()
        if value and _is_heldout_test_path(value):
            failures.append(
                {
                    "code": "heldout_test_path_in_seed_candidate",
                    "segment_id": segment_id,
                    "field": field_name,
                    "context": context.as_posix(),
                }
            )


def _is_heldout_test_path(value: str) -> bool:
    normalized = value.replace("\\", "/").lower().rstrip("/")
    return "/test/" in normalized or normalized.endswith("/test")


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "approved"}


def _relativize_paths(value: Any, *, project_root: Path) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _relativize_paths(item, project_root=project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_relativize_paths(item, project_root=project_root) for item in value]
    if isinstance(value, tuple):
        return tuple(_relativize_paths(item, project_root=project_root) for item in value)
    if isinstance(value, str) and value:
        path = Path(value)
        if path.is_absolute():
            return _display_path(path, base=project_root)
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_READINESS_ROOT",
    "REQUIRED_APPROVED_ROLES",
    "V7DSeedPackageConfig",
    "V7DSeedReadinessConfig",
    "build_v7d_prompt_pose_seed_package",
    "check_v7d_prompt_pose_seed_readiness",
]
