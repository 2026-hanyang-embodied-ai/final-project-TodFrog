"""Build the v7e paper-expanded seed package after paper review approval."""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodied_rps.v7_rps_seed_package import build_v7_rps_seed_package
from embodied_rps.v7e_paper_seed_review_validator import (
    BRANCH_LABEL,
    DEFAULT_PLAN_ROOT,
)
from embodied_rps.v7e_seed_package_preflight import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_PREFLIGHT_ROOT,
    DEFAULT_V7D_SEED_PACKAGE_ROOT,
    DEFAULT_V7E_SEED_PACKAGE_ROOT,
    V7ESeedPackagePreflightConfig,
    write_v7e_seed_package_preflight,
)


DEFAULT_V7D_REVIEW_ROOT = Path("artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618")
DEFAULT_PAPER_REVIEW_VALIDATION_ROOT = Path("artifacts/real_skeleton_v7e_paper_seed_review_validation_20260619")
DEFAULT_SEQUENCE_LENGTH = 72

REVIEW_COPY_FILES: tuple[str, ...] = (
    "proposed_segments.jsonl",
    "segment_review_manifest.csv",
    "auto_quality_pass_segments.csv",
    "segment_review_packet.md",
    "review_contact_sheet.png",
)
REVIEW_COPY_DIRS: tuple[str, ...] = ("segments", "previews")


@dataclass(frozen=True)
class V7ESeedPackageBuilderConfig:
    """Inputs for building the approved v7e seed package."""

    project_root: Path = field(default_factory=Path.cwd)
    v7d_seed_package_root: Path = DEFAULT_V7D_SEED_PACKAGE_ROOT
    v7d_review_root: Path = DEFAULT_V7D_REVIEW_ROOT
    v7e_plan_root: Path = DEFAULT_PLAN_ROOT
    paper_review_validation_root: Path = DEFAULT_PAPER_REVIEW_VALIDATION_ROOT
    output_root: Path = DEFAULT_V7E_SEED_PACKAGE_ROOT
    preflight_root: Path = DEFAULT_PREFLIGHT_ROOT
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    minimum_approved_paper_seed_count: int = 5
    overwrite: bool = False


def build_v7e_stage1_paper_transition_rescue_seed_package(config: V7ESeedPackageBuilderConfig) -> dict[str, Any]:
    """Build v7e seed package from v7d approved base seeds plus approved paper additions."""

    project_root = config.project_root.resolve()
    output_root = _resolve_path(project_root, config.output_root)
    v7d_review_root = _resolve_path(project_root, config.v7d_review_root)
    v7e_plan_root = _resolve_path(project_root, config.v7e_plan_root)
    paper_review_validation_root = _resolve_path(project_root, config.paper_review_validation_root)
    preflight = write_v7e_seed_package_preflight(
        V7ESeedPackagePreflightConfig(
            project_root=project_root,
            v7d_seed_package_root=config.v7d_seed_package_root,
            paper_review_validation_root=config.paper_review_validation_root,
            output_root=config.preflight_root,
            minimum_approved_paper_seed_count=config.minimum_approved_paper_seed_count,
            planned_v7e_seed_package_root=config.output_root,
        )
    )
    if preflight["status"] != "ready_for_v7e_seed_package_build":
        raise ValueError(f"v7e seed-package preflight is not ready: {preflight['status']}")
    if not v7d_review_root.exists():
        raise FileNotFoundError(f"Missing v7d review root: {v7d_review_root}")
    if not v7e_plan_root.exists():
        raise FileNotFoundError(f"Missing v7e plan root: {v7e_plan_root}")
    if output_root.exists():
        if not config.overwrite:
            raise FileExistsError(f"v7e seed package root already exists: {output_root}")
        _safe_remove_output_root(project_root=project_root, output_root=output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_review_base(v7d_review_root=v7d_review_root, output_root=output_root)

    paper_summary = _read_json(paper_review_validation_root / "v7e_paper_seed_review_validation_summary.json")
    approved_ids = _string_list(paper_summary.get("approved_paper_segment_ids", []))
    candidate_rows = _read_csv(v7e_plan_root / "v7e_paper_seed_review_candidates.csv")
    candidates_by_id = {str(row.get("segment_id", "")).strip(): row for row in candidate_rows}
    source_records = _read_jsonl(v7d_review_root / "proposed_segments.jsonl")
    source_records_by_id = {str(row.get("segment_id", "")).strip(): row for row in source_records}

    added_records: list[dict[str, Any]] = []
    for segment_id in approved_ids:
        candidate = candidates_by_id.get(segment_id)
        if candidate is None:
            raise ValueError(f"approved v7e paper segment {segment_id} is missing from candidates CSV")
        _reject_heldout_paths(candidate, context=f"candidate:{segment_id}")
        source_record = source_records_by_id.get(segment_id)
        if source_record is None:
            source_record = _record_from_candidate(candidate)
        record = _materialize_added_paper_record(
            project_root=project_root,
            plan_root=v7e_plan_root,
            output_root=output_root,
            source_record=source_record,
            candidate=candidate,
        )
        added_records.append(record)

    _append_jsonl(output_root / "proposed_segments.jsonl", added_records)
    _append_review_manifest(output_root / "segment_review_manifest.csv", added_records)
    _append_auto_quality_pass(output_root / "auto_quality_pass_segments.csv", added_records)

    builder_summary = build_v7_rps_seed_package(output_root=output_root, sequence_length=config.sequence_length)
    _rewrite_child_json_summaries_relative(output_root=output_root, project_root=project_root)
    base_approved_count = _count_base_approved(v7d_review_root / "segment_review_manifest.csv")
    summary: dict[str, Any] = {
        "status": builder_summary.get("status"),
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "v7d_review_root": _display_path(v7d_review_root, base=project_root),
        "v7e_plan_root": _display_path(v7e_plan_root, base=project_root),
        "paper_review_validation_root": _display_path(paper_review_validation_root, base=project_root),
        "sequence_length": int(config.sequence_length),
        "approved_base_seed_count": base_approved_count,
        "added_paper_seed_count": len(added_records),
        "added_paper_segment_ids": approved_ids,
        "preflight_summary": _relativize_paths(preflight, project_root=project_root),
        "builder_summary": _relativize_paths(builder_summary, project_root=project_root),
        "stage1_training_scope": "retrain_rock_vs_transition_only",
        "stage2_policy": "reuse_v7d_paper_scissors_stage2_unless_new_evidence_requires_change",
        "seed_package_created": builder_summary.get("status") == "passed",
        "dataset_generated": False,
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "heldout15_started": False,
        "promotion_eligible": False,
        "fallback_policy": "keep_v4_live_demo_fallback",
        "heldout_policy": "heldout */test MP4s remain validation-only and are rejected before v7e seed packaging",
    }
    (output_root / "v7e_seed_package_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7e_seed_package_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def blocked_v7e_seed_package_summary(config: V7ESeedPackageBuilderConfig) -> dict[str, Any]:
    """Return the current v7e preflight status without mutating the seed package root."""

    project_root = config.project_root.resolve()
    preflight = write_v7e_seed_package_preflight(
        V7ESeedPackagePreflightConfig(
            project_root=project_root,
            v7d_seed_package_root=config.v7d_seed_package_root,
            paper_review_validation_root=config.paper_review_validation_root,
            output_root=config.preflight_root,
            minimum_approved_paper_seed_count=config.minimum_approved_paper_seed_count,
            planned_v7e_seed_package_root=config.output_root,
        )
    )
    return {
        "status": preflight.get("status"),
        "branch_label": BRANCH_LABEL,
        "preflight_summary": preflight,
        "seed_package_created": False,
        "dataset_generated": False,
        "training_started": False,
        "heldout15_started": False,
        "promotion_eligible": False,
        "next_action": preflight.get("next_action"),
    }


def _copy_review_base(*, v7d_review_root: Path, output_root: Path) -> None:
    for filename in REVIEW_COPY_FILES:
        source = v7d_review_root / filename
        if source.exists():
            shutil.copy2(source, output_root / filename)
    for dirname in REVIEW_COPY_DIRS:
        source_dir = v7d_review_root / dirname
        if source_dir.exists():
            shutil.copytree(source_dir, output_root / dirname)
    missing = [filename for filename in ("proposed_segments.jsonl", "segment_review_manifest.csv") if not (output_root / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required v7d review files after copy: {missing}")


def _materialize_added_paper_record(
    *,
    project_root: Path,
    plan_root: Path,
    output_root: Path,
    source_record: Mapping[str, Any],
    candidate: Mapping[str, str],
) -> dict[str, Any]:
    segment_id = str(candidate.get("segment_id", "")).strip()
    if not segment_id:
        raise ValueError("v7e paper candidate is missing segment_id")
    record = dict(source_record)
    record["segment_id"] = segment_id
    record["target_name"] = "paper"
    record["proposal_role"] = "hard_paper_prompt_window"
    record["quality_status"] = "auto_quality_pass"
    record["review_status"] = "approved"
    record["approved_for_training"] = True
    record["source_name"] = "v7e_reviewed_paper_prompt_window_rescue"
    record["training_policy"] = "v7e_manual_review_approved_paper_prompt_window_seed"
    record["v7e_stage1_paper_transition_rescue"] = True
    skeleton_source = _resolve_candidate_path(
        project_root=project_root,
        plan_root=plan_root,
        value=str(candidate.get("skeleton_npz", "")).strip() or str(record.get("skeleton_npz", "")),
    )
    preview_source = _resolve_candidate_path(
        project_root=project_root,
        plan_root=plan_root,
        value=str(candidate.get("preview_image", "")).strip() or str(record.get("preview_image", "")),
    )
    if not skeleton_source.exists():
        raise FileNotFoundError(f"Missing approved v7e paper skeleton NPZ: {skeleton_source}")
    if not preview_source.exists():
        raise FileNotFoundError(f"Missing approved v7e paper preview image: {preview_source}")
    _reject_heldout_paths(record, context=f"record:{segment_id}")
    segments_dir = output_root / "segments"
    previews_dir = output_root / "previews"
    segments_dir.mkdir(exist_ok=True)
    previews_dir.mkdir(exist_ok=True)
    skeleton_target = segments_dir / f"{segment_id}.npz"
    preview_target = previews_dir / f"{segment_id}.png"
    shutil.copy2(skeleton_source, skeleton_target)
    shutil.copy2(preview_source, preview_target)
    record["skeleton_npz"] = _display_path(skeleton_target, base=output_root)
    record["preview_image"] = _display_path(preview_target, base=output_root)
    return record


def _record_from_candidate(candidate: Mapping[str, str]) -> dict[str, Any]:
    segment_id = str(candidate.get("segment_id", "")).strip()
    return {
        "segment_id": segment_id,
        "target_name": "paper",
        "proposal_role": "hard_paper_prompt_window",
        "quality_status": "auto_quality_pass",
        "source_name": "v7e_reviewed_paper_prompt_window_rescue",
        "source_path": str(candidate.get("source_path", "")),
        "source_run_id": segment_id,
        "skeleton_npz": str(candidate.get("skeleton_npz", "")),
        "preview_image": str(candidate.get("preview_image", "")),
        "frame_count": int(float(str(candidate.get("frame_count", 0) or 0))),
        "detection_coverage": float(str(candidate.get("detection_coverage", 0.0) or 0.0)),
        "severe_landmark_jump_count": 0,
        "prompt_conditioned_sequence": True,
        "target_prompt": "scissors",
    }


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    existing_rows = _read_jsonl(path)
    row_index_by_id = {
        str(row.get("segment_id", "")).strip(): index
        for index, row in enumerate(existing_rows)
        if str(row.get("segment_id", "")).strip()
    }
    for row in rows:
        segment_id = str(row.get("segment_id", "")).strip()
        if segment_id in row_index_by_id:
            index = row_index_by_id[segment_id]
            existing_rows[index] = {**existing_rows[index], **dict(row)}
            continue
        row_index_by_id[segment_id] = len(existing_rows)
        existing_rows.append(dict(row))
    with path.open("w", encoding="utf-8") as handle:
        for row in existing_rows:
            handle.write(json.dumps(_json_ready(row), ensure_ascii=False, sort_keys=True) + "\n")


def _append_review_manifest(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = _read_csv(path)
    fieldnames = _csv_fieldnames(path, fallback=("segment_id", "approved_for_training", "review_status", "review_notes"))
    for field in ("segment_id", "approved_for_training", "review_status", "review_notes"):
        if field not in fieldnames:
            fieldnames.append(field)
    row_index_by_id = {
        str(row.get("segment_id", "")).strip(): index
        for index, row in enumerate(rows)
        if str(row.get("segment_id", "")).strip()
    }
    for record in records:
        segment_id = str(record.get("segment_id", "")).strip()
        approved_row = {
            "segment_id": segment_id,
            "approved_for_training": "true",
            "review_status": "approved",
            "review_notes": "v7e approved paper prompt-window rescue seed",
        }
        if segment_id in row_index_by_id:
            rows[row_index_by_id[segment_id]].update(approved_row)
            continue
        row_index_by_id[segment_id] = len(rows)
        rows.append(approved_row)
    _write_csv(path, rows, fieldnames=fieldnames)


def _append_auto_quality_pass(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = _read_csv(path) if path.exists() else []
    fieldnames = _csv_fieldnames(
        path,
        fallback=("segment_id", "target_name", "proposal_role", "preview_image", "skeleton_npz"),
    )
    for field in ("segment_id", "target_name", "proposal_role", "preview_image", "skeleton_npz"):
        if field not in fieldnames:
            fieldnames.append(field)
    row_index_by_id = {
        str(row.get("segment_id", "")).strip(): index
        for index, row in enumerate(rows)
        if str(row.get("segment_id", "")).strip()
    }
    for record in records:
        segment_id = str(record.get("segment_id", "")).strip()
        quality_row = {field: record.get(field, "") for field in fieldnames}
        if segment_id in row_index_by_id:
            rows[row_index_by_id[segment_id]].update(quality_row)
            continue
        row_index_by_id[segment_id] = len(rows)
        rows.append(quality_row)
    _write_csv(path, rows, fieldnames=fieldnames)


def _count_base_approved(path: Path) -> int:
    return sum(
        1
        for row in _read_csv(path)
        if _truthy(row.get("approved_for_training", "")) and str(row.get("review_status", "")).strip().lower() == "approved"
    )


def _safe_remove_output_root(*, project_root: Path, output_root: Path) -> None:
    resolved = output_root.resolve()
    resolved.relative_to(project_root.resolve())
    if "v7e_stage1_paper_transition_rescue_seed_package" not in resolved.name and resolved.name != "v7e_seed":
        raise ValueError(f"Refusing to overwrite unexpected v7e seed package root: {output_root}")
    shutil.rmtree(resolved)


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# V7e Seed Package Summary",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Branch: `{summary.get('branch_label')}`",
            f"- Approved base seeds: `{summary.get('approved_base_seed_count')}`",
            f"- Added paper seeds: `{summary.get('added_paper_seed_count')}`",
            "- Dataset generated: `False`",
            "- Training started: `False`",
            "- Heldout15 started: `False`",
            "- Promotion eligible: `False`",
            "- V4 remains the live/demo fallback.",
            "",
        ]
    )


def _rewrite_child_json_summaries_relative(*, output_root: Path, project_root: Path) -> None:
    for filename in ("review_readiness_summary.json", "seed_package_summary.json"):
        path = output_root / filename
        if not path.exists():
            continue
        summary = _read_json(path)
        path.write_text(
            json.dumps(_json_ready(_relativize_paths(summary, project_root=project_root)), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object in {path}")
    return dict(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _csv_fieldnames(path: Path, *, fallback: Sequence[str]) -> list[str]:
    if not path.exists():
        return list(fallback)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or fallback)


def _reject_heldout_paths(row: Mapping[str, Any], *, context: str) -> None:
    for key, value in row.items():
        if isinstance(value, str) and _is_heldout_test_path(value):
            raise ValueError(f"heldout */test path is not allowed in v7e seed package metadata: {context}.{key}={value}")


def _is_heldout_test_path(value: str) -> bool:
    normalized = value.replace("\\", "/").lower().rstrip("/")
    return "/test/" in normalized or normalized.endswith("/test")


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "approved"}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _resolve_candidate_path(*, project_root: Path, plan_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_path = project_root / path
    if project_path.exists():
        return project_path
    return plan_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _relativize_paths(value: Any, *, project_root: Path) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _relativize_paths(item, project_root=project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_relativize_paths(item, project_root=project_root) for item in value]
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
    "DEFAULT_PAPER_REVIEW_VALIDATION_ROOT",
    "DEFAULT_V7D_REVIEW_ROOT",
    "DEFAULT_SEQUENCE_LENGTH",
    "V7ESeedPackageBuilderConfig",
    "blocked_v7e_seed_package_summary",
    "build_v7e_stage1_paper_transition_rescue_seed_package",
]
