"""Dry-run assignment planner for v4 recording staging MP4s."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2, move

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER, natural_key


@dataclass(frozen=True)
class V4RecordingSlotAssignmentConfig:
    """Configuration for assigning staging MP4 files to v4 recording slots."""

    source_root: Path
    calibration_root: Path
    output_root: Path
    slot_manifest_path: Path | None = None
    execute_copy: bool = False
    archive_copied_sources: bool = True
    archive_root: Path | None = None


def plan_v4_recording_slot_assignment(config: V4RecordingSlotAssignmentConfig) -> dict[str, object]:
    """Plan, and optionally copy, staging MP4s into missing v4 recording slots."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = config.slot_manifest_path or (config.calibration_root / "recording_slot_manifest.json")
    if not manifest_path.exists():
        summary = _base_summary(config, manifest_path, status="missing_slot_manifest")
        summary["failures"] = [{"code": "missing_slot_manifest", "path": manifest_path.as_posix()}]
        _write_outputs(config.output_root, summary)
        return summary
    overlap_failure = _overlap_failure(config.source_root, config.calibration_root)
    if overlap_failure is not None:
        summary = _base_summary(config, manifest_path, status="invalid_roots")
        summary["failures"] = [overlap_failure]
        _write_outputs(config.output_root, summary)
        return summary

    slots = _load_slots(manifest_path)
    missing_slots = [slot for slot in slots if not Path(str(slot["target_path"])).exists()]
    source_records = _discover_staging_sources(config.source_root)
    sources_by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for source in source_records:
        sources_by_label[str(source["label"])].append(source)
    assignments = _build_assignments(missing_slots, sources_by_label)
    copied: list[dict[str, object]] = []
    copy_failures: list[dict[str, object]] = []
    archive_root = config.archive_root or config.source_root.with_name(f"{config.source_root.name}_copied")
    if config.execute_copy:
        copied, copy_failures = _execute_assignments(
            assignments,
            archive_sources=config.archive_copied_sources,
            archive_root=archive_root,
        )
    assigned_counts = Counter(str(record["label"]) for record in assignments)
    source_counts = Counter(str(record["label"]) for record in source_records)
    missing_counts = Counter(str(slot["label"]) for slot in missing_slots)
    remaining_after_assignment = {
        label: max(0, int(missing_counts.get(label, 0)) - int(assigned_counts.get(label, 0)))
        for label in REVIEW_LABEL_ORDER
    }
    status = _status(
        execute_copy=config.execute_copy,
        assignments=assignments,
        missing_slots=missing_slots,
        remaining_after_assignment=remaining_after_assignment,
        copy_failures=copy_failures,
    )
    summary = {
        **_base_summary(config, manifest_path, status=status),
        "slot_count": len(slots),
        "missing_slot_count": len(missing_slots),
        "source_count": len(source_records),
        "assignment_count": len(assignments),
        "copied_count": len(copied),
        "archived_count": sum(1 for record in copied if record.get("archived_source_path")),
        "archive_copied_sources": bool(config.archive_copied_sources),
        "archive_root": archive_root.as_posix(),
        "source_label_counts": {label: int(source_counts.get(label, 0)) for label in REVIEW_LABEL_ORDER},
        "missing_label_counts": {label: int(missing_counts.get(label, 0)) for label in REVIEW_LABEL_ORDER},
        "assigned_label_counts": {label: int(assigned_counts.get(label, 0)) for label in REVIEW_LABEL_ORDER},
        "remaining_after_assignment": remaining_after_assignment,
        "assignments": assignments,
        "copied": copied,
        "failures": copy_failures,
        "assignment_table": (config.output_root / "recording_slot_assignment_table.csv").as_posix(),
    }
    _write_outputs(config.output_root, summary)
    return summary


def _base_summary(config: V4RecordingSlotAssignmentConfig, manifest_path: Path, *, status: str) -> dict[str, object]:
    return {
        "status": status,
        "execute_copy": bool(config.execute_copy),
        "source_root": config.source_root.as_posix(),
        "calibration_root": config.calibration_root.as_posix(),
        "slot_manifest_path": manifest_path.as_posix(),
    }


def _load_slots(path: Path) -> list[dict[str, object]]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("recording slot manifest must be a JSON list")
    return [{str(key): value for key, value in dict(item).items()} for item in loaded]


def _discover_staging_sources(source_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not source_root.exists():
        return records
    for path in sorted(source_root.rglob("*.mp4"), key=natural_key):
        label = _infer_label(path, source_root)
        if label is None:
            continue
        records.append(
            {
                "label": label,
                "source_path": path.as_posix(),
                "filename": path.name,
                "size_bytes": int(path.stat().st_size),
            }
        )
    return records


def _infer_label(path: Path, source_root: Path) -> str | None:
    try:
        relative_parts = path.relative_to(source_root).parts
    except ValueError:
        relative_parts = path.parts
    for part in relative_parts[:-1]:
        if part in REVIEW_LABEL_ORDER:
            return part
    stem_lower = path.stem.lower()
    for label in REVIEW_LABEL_ORDER:
        if stem_lower.startswith(label):
            return label
    return None


def _build_assignments(
    missing_slots: list[dict[str, object]],
    sources_by_label: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    used_by_label: dict[str, int] = defaultdict(int)
    assignments: list[dict[str, object]] = []
    for slot in missing_slots:
        label = str(slot["label"])
        source_index = used_by_label[label]
        sources = sources_by_label.get(label, [])
        if source_index >= len(sources):
            continue
        used_by_label[label] += 1
        source = sources[source_index]
        assignments.append(
            {
                "slot_id": str(slot["slot_id"]),
                "label": label,
                "source_path": str(source["source_path"]),
                "target_path": str(slot["target_path"]),
                "target_filename": str(slot.get("filename", Path(str(slot["target_path"])).name)),
                "planned_action": "copy",
                "motion_focus": str(slot.get("motion_focus", "")),
                "viewpoint": str(slot.get("viewpoint", "")),
                "background": str(slot.get("background", "")),
            }
        )
    return assignments


def _execute_assignments(
    assignments: list[dict[str, object]],
    *,
    archive_sources: bool,
    archive_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    copied: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for assignment in assignments:
        source = Path(str(assignment["source_path"]))
        target = Path(str(assignment["target_path"]))
        if not source.exists():
            failures.append({"code": "source_missing", "slot_id": assignment["slot_id"], "source_path": source.as_posix()})
            continue
        if target.exists():
            failures.append({"code": "target_exists", "slot_id": assignment["slot_id"], "target_path": target.as_posix()})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, target)
        copied_record = {**assignment, "copied": True, "size_bytes": int(target.stat().st_size)}
        if archive_sources:
            archive_path = _unique_archive_path(archive_root / str(assignment["label"]) / source.name)
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            move(str(source), str(archive_path))
            copied_record["archived_source_path"] = archive_path.as_posix()
        copied.append(copied_record)
    return copied, failures


def _unique_archive_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{index:04d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate archive path for {path}")


def _status(
    *,
    execute_copy: bool,
    assignments: list[dict[str, object]],
    missing_slots: list[dict[str, object]],
    remaining_after_assignment: dict[str, int],
    copy_failures: list[dict[str, object]],
) -> str:
    if copy_failures:
        return "copy_failed"
    if not assignments and missing_slots:
        return "no_staging_sources"
    if any(value > 0 for value in remaining_after_assignment.values()):
        return "partial_assignment_ready" if not execute_copy else "partial_copy_complete"
    return "assignment_ready" if not execute_copy else "copy_complete"


def _overlap_failure(source_root: Path, calibration_root: Path) -> dict[str, object] | None:
    source = source_root.expanduser().resolve(strict=False)
    calibration = calibration_root.expanduser().resolve(strict=False)
    if _is_within(source, calibration) or _is_within(calibration, source):
        return {
            "code": "source_overlaps_calibration_root",
            "source_root": source_root.as_posix(),
            "calibration_root": calibration_root.as_posix(),
        }
    return None


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _write_outputs(output_root: Path, summary: dict[str, object]) -> None:
    (output_root / "recording_slot_assignment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_table(output_root / "recording_slot_assignment_table.csv", list(summary.get("assignments", [])))
    (output_root / "recording_slot_assignment_summary.md").write_text(_markdown(summary), encoding="utf-8")


def _write_table(path: Path, assignments: list[object]) -> None:
    fieldnames = [
        "slot_id",
        "label",
        "source_path",
        "target_path",
        "target_filename",
        "planned_action",
        "motion_focus",
        "viewpoint",
        "background",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for assignment in assignments:
            if isinstance(assignment, dict):
                writer.writerow({field: assignment.get(field) for field in fieldnames})


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# V4 Recording Slot Assignment",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Execute copy: `{summary.get('execute_copy')}`",
        f"- Source root: `{summary.get('source_root')}`",
        f"- Calibration root: `{summary.get('calibration_root')}`",
        f"- Source MP4s: `{summary.get('source_count', 0)}`",
        f"- Missing slots: `{summary.get('missing_slot_count', 0)}`",
        f"- Planned assignments: `{summary.get('assignment_count', 0)}`",
        f"- Copied files: `{summary.get('copied_count', 0)}`",
        f"- Archived staging sources: `{summary.get('archived_count', 0)}`",
        "",
        "## Label Counts",
        "",
        "| Label | Sources | Missing slots | Planned assignments | Remaining after assignment |",
        "|---|---:|---:|---:|---:|",
    ]
    sources = summary.get("source_label_counts")
    missing = summary.get("missing_label_counts")
    assigned = summary.get("assigned_label_counts")
    remaining = summary.get("remaining_after_assignment")
    if isinstance(sources, dict) and isinstance(missing, dict) and isinstance(assigned, dict) and isinstance(remaining, dict):
        for label in REVIEW_LABEL_ORDER:
            lines.append(
                f"| `{label}` | `{sources.get(label, 0)}` | `{missing.get(label, 0)}` | `{assigned.get(label, 0)}` | `{remaining.get(label, 0)}` |"
            )
    lines.extend(["", "## Next Step", "", _next_step(str(summary.get("status"))), ""])
    return "\n".join(lines)


def _next_step(status: str) -> str:
    if status == "assignment_ready":
        return "Review the assignment table, then rerun with `--execute-copy` if the mapping is acceptable."
    if status == "partial_assignment_ready":
        return "Add more staging MP4s for labels that still have remaining slots, or execute a partial copy intentionally."
    if status == "copy_complete":
        return "Run the v4 recording slot audit and MP4 preflight."
    if status == "partial_copy_complete":
        return "Add more staging MP4s, then rerun assignment for the remaining slots."
    if status == "no_staging_sources":
        return "Record MP4s into a separate staging folder such as `v4_recording_staging/<label>/`."
    return "Fix the reported failure before continuing."


__all__ = ["V4RecordingSlotAssignmentConfig", "plan_v4_recording_slot_assignment"]
