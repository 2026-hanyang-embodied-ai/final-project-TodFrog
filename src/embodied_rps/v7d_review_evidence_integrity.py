"""Integrity audit for v7d required-role temporal review evidence."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_PACKET_ROOT = Path("artifacts/real_skeleton_v7d_required_role_approval_packet_20260618")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_review_evidence_integrity_20260618")
REQUIRED_ROLES: tuple[str, ...] = (
    "hard_paper_prompt_window",
    "rock_wait_prompt_window",
    "scissors_boundary_control",
)
PACKET_CSV = "required_role_approval_packet.csv"
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = (
    "temporal_strip",
    "preview_image",
    "skeleton_npz",
    "decision_template_csv",
)
OUTPUT_FIELDS: tuple[str, ...] = (
    "proposal_role",
    "segment_id",
    "target_name",
    "temporal_strip",
    "temporal_strip_exists",
    "preview_image",
    "preview_image_exists",
    "skeleton_npz",
    "skeleton_npz_exists",
    "skeleton_npz_finite",
    "decision_template_csv",
    "decision_template_exists",
    "status",
    "failure_codes",
)


@dataclass(frozen=True)
class V7DReviewEvidenceIntegrityConfig:
    """Inputs for auditing v7d manual-review evidence file integrity."""

    project_root: Path = field(default_factory=Path.cwd)
    packet_root: Path = DEFAULT_PACKET_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


def write_v7d_review_evidence_integrity(config: V7DReviewEvidenceIntegrityConfig) -> dict[str, object]:
    """Write a non-mutating integrity audit over the v7d required-role review packet."""

    project_root = config.project_root.resolve()
    packet_root = _resolve_path(project_root, config.packet_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    packet_csv = packet_root / PACKET_CSV
    if not packet_csv.exists():
        raise FileNotFoundError(f"Missing v7d required-role approval packet CSV: {packet_csv}")

    packet_rows = _read_csv(packet_csv)
    role_rows = [row for row in packet_rows if str(row.get("proposal_role", "")).strip() in REQUIRED_ROLES]
    missing_required_roles = [
        role
        for role in REQUIRED_ROLES
        if role not in {str(row.get("proposal_role", "")).strip() for row in role_rows}
    ]
    audit_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for row in role_rows:
        _reject_heldout_metadata(row, context=packet_csv)
        audit_row, row_failures = _audit_packet_row(row=row, packet_root=packet_root, project_root=project_root)
        audit_rows.append(audit_row)
        failures.extend(row_failures)

    for role in missing_required_roles:
        failures.append({"code": "missing_required_role", "proposal_role": role})

    all_evidence_files_exist = all(
        bool(row["temporal_strip_exists"])
        and bool(row["preview_image_exists"])
        and bool(row["skeleton_npz_exists"])
        and bool(row["decision_template_exists"])
        for row in audit_rows
    )
    all_skeleton_npz_finite = all(bool(row["skeleton_npz_finite"]) for row in audit_rows)
    status = (
        "ready_for_manual_temporal_approval"
        if not failures and len(audit_rows) == len(REQUIRED_ROLES)
        else "evidence_integrity_failed"
    )

    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "packet_root": _display_path(packet_root, base=project_root),
        "output_root": _display_path(output_root, base=project_root),
        "packet_csv": _display_path(packet_csv, base=project_root),
        "integrity_csv": _display_path(output_root / "required_role_evidence_integrity.csv", base=project_root),
        "packet_row_count": len(role_rows),
        "verified_row_count": len([row for row in audit_rows if row.get("status") == "verified"]),
        "required_roles": list(REQUIRED_ROLES),
        "missing_required_roles": missing_required_roles,
        "all_evidence_files_exist": all_evidence_files_exist,
        "all_skeleton_npz_finite": all_skeleton_npz_finite,
        "failures": failures,
        "decisions_applied": False,
        "review_manifest_modified": False,
        "seed_package_created": False,
        "dataset_generated": False,
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test paths are rejected from v7d review evidence metadata",
        "next_action": (
            "inspect the verified temporal strips and previews, then fill the real approval selection sheet "
            "only after temporal review"
        ),
        "config": _config_summary(project_root=project_root, config=config),
    }
    _write_csv(output_root / "required_role_evidence_integrity.csv", OUTPUT_FIELDS, audit_rows)
    (output_root / "review_evidence_integrity_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "review_evidence_integrity_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _audit_packet_row(
    *,
    row: Mapping[str, object],
    packet_root: Path,
    project_root: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    role = str(row.get("proposal_role", "")).strip()
    segment_id = str(row.get("first_candidate_segment_id", "")).strip()
    target_name = str(row.get("target_name", "")).strip()
    failures: list[dict[str, object]] = []

    temporal_path = _resolve_packet_link(packet_root, str(row.get("temporal_strip", "")).strip())
    preview_path = _resolve_packet_link(packet_root, str(row.get("preview_image", "")).strip())
    skeleton_path = _resolve_packet_link(packet_root, str(row.get("skeleton_npz", "")).strip())
    decision_path = _resolve_packet_link(packet_root, str(row.get("decision_template_csv", "")).strip())

    temporal_exists = _nonempty_file(temporal_path)
    preview_exists = _nonempty_file(preview_path)
    skeleton_exists = _nonempty_file(skeleton_path)
    decision_exists = _nonempty_file(decision_path)
    skeleton_finite = False

    if not segment_id:
        failures.append({"code": "missing_segment_id", "proposal_role": role})
    if not temporal_exists:
        failures.append({"code": "missing_temporal_strip", "proposal_role": role, "segment_id": segment_id})
    if not preview_exists:
        failures.append({"code": "missing_preview_image", "proposal_role": role, "segment_id": segment_id})
    if not skeleton_exists:
        failures.append({"code": "missing_skeleton_npz", "proposal_role": role, "segment_id": segment_id})
    else:
        skeleton_finite, skeleton_failure = _check_skeleton_npz(skeleton_path)
        if skeleton_failure:
            failures.append({"code": skeleton_failure, "proposal_role": role, "segment_id": segment_id})
    if not decision_exists:
        failures.append({"code": "missing_decision_template", "proposal_role": role, "segment_id": segment_id})

    audit_row: dict[str, object] = {
        "proposal_role": role,
        "segment_id": segment_id,
        "target_name": target_name,
        "temporal_strip": _display_path(temporal_path, base=project_root),
        "temporal_strip_exists": temporal_exists,
        "preview_image": _display_path(preview_path, base=project_root),
        "preview_image_exists": preview_exists,
        "skeleton_npz": _display_path(skeleton_path, base=project_root),
        "skeleton_npz_exists": skeleton_exists,
        "skeleton_npz_finite": skeleton_finite,
        "decision_template_csv": _display_path(decision_path, base=project_root),
        "decision_template_exists": decision_exists,
        "status": "verified" if not failures else "failed",
        "failure_codes": ";".join(str(failure["code"]) for failure in failures),
    }
    return audit_row, failures


def _check_skeleton_npz(path: Path) -> tuple[bool, str | None]:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "canonical_landmarks" not in data.files:
                return False, "missing_canonical_landmarks"
            canonical = data["canonical_landmarks"]
            if canonical.ndim != 3 or canonical.shape[1:] != (21, 3) or canonical.shape[0] <= 0:
                return False, "invalid_canonical_landmarks_shape"
            if not bool(np.isfinite(canonical).all()):
                return False, "nonfinite_canonical_landmarks"
            if "detected" in data.files and len(data["detected"]) != canonical.shape[0]:
                return False, "detected_length_mismatch"
    except Exception as exc:  # pragma: no cover - summarized for artifact diagnostics
        return False, f"invalid_npz:{type(exc).__name__}"
    return True, None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def _resolve_packet_link(packet_root: Path, value: str) -> Path:
    if not value:
        return packet_root
    path = Path(value)
    return path if path.is_absolute() else (packet_root / path).resolve(strict=False)


def _nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _summary_markdown(summary: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# V7d Review Evidence Integrity",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Packet rows: `{summary.get('packet_row_count')}`",
            f"- Verified rows: `{summary.get('verified_row_count')}`",
            f"- Missing required roles: `{summary.get('missing_required_roles')}`",
            f"- All evidence files exist: `{summary.get('all_evidence_files_exist')}`",
            f"- All skeleton NPZ files finite: `{summary.get('all_skeleton_npz_finite')}`",
            "- This audit does not approve rows, build seeds, generate datasets, train, validate, or promote v7d.",
            f"- Next action: `{summary.get('next_action')}`",
            "",
        ]
    )


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(base.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _config_summary(*, project_root: Path, config: V7DReviewEvidenceIntegrityConfig) -> dict[str, object]:
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


__all__ = ["V7DReviewEvidenceIntegrityConfig", "write_v7d_review_evidence_integrity"]
