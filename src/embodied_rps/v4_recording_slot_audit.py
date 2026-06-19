"""Coverage audit for v4 calibration recording slots."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER, natural_key


@dataclass(frozen=True)
class V4RecordingSlotAuditConfig:
    """Configuration for auditing recorded MP4 coverage against the slot manifest."""

    calibration_root: Path
    output_root: Path
    slot_manifest_path: Path | None = None


def audit_v4_recording_slots(config: V4RecordingSlotAuditConfig) -> dict[str, object]:
    """Audit whether the v4 calibration MP4s fill the planned recording slots."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = config.slot_manifest_path or (config.calibration_root / "recording_slot_manifest.json")
    if not manifest_path.exists():
        summary = {
            "status": "missing_slot_manifest",
            "calibration_root": config.calibration_root.as_posix(),
            "slot_manifest_path": manifest_path.as_posix(),
            "slot_count": 0,
            "filled_slot_count": 0,
            "missing_slot_count": 0,
            "extra_mp4_count": 0,
            "failures": [{"code": "missing_slot_manifest", "path": manifest_path.as_posix()}],
            "records": [],
        }
        _write_outputs(config.output_root, summary)
        return summary

    slots = _load_slots(manifest_path)
    records, manifest_failures = _slot_records(config.calibration_root, slots)
    extra_records = _extra_mp4s(config.calibration_root, records)
    missing_records = [record for record in records if not bool(record["exists"])]
    filled_records = [record for record in records if bool(record["exists"])]
    failures: list[dict[str, object]] = list(manifest_failures)
    failures.extend(
        {"code": "missing_slot_file", "slot_id": record["slot_id"], "target_path": record["target_path"]}
        for record in missing_records
    )
    failures.extend({"code": "extra_mp4", **record} for record in extra_records)
    status = _status(manifest_failures=manifest_failures, missing_records=missing_records, extra_records=extra_records)
    planned_counts = Counter(str(record["label"]) for record in records)
    filled_counts = Counter(str(record["label"]) for record in filled_records)
    summary = {
        "status": status,
        "calibration_root": config.calibration_root.as_posix(),
        "slot_manifest_path": manifest_path.as_posix(),
        "slot_count": len(records),
        "filled_slot_count": len(filled_records),
        "missing_slot_count": len(missing_records),
        "extra_mp4_count": len(extra_records),
        "planned_label_counts": {label: int(planned_counts.get(label, 0)) for label in REVIEW_LABEL_ORDER},
        "filled_label_counts": {label: int(filled_counts.get(label, 0)) for label in REVIEW_LABEL_ORDER},
        "missing_by_label": {
            label: int(planned_counts.get(label, 0) - filled_counts.get(label, 0))
            for label in REVIEW_LABEL_ORDER
        },
        "failures": failures,
        "records": records,
        "extra_records": extra_records,
        "audit_table": (config.output_root / "recording_slot_audit_table.csv").as_posix(),
    }
    _write_outputs(config.output_root, summary)
    return summary


def _load_slots(path: Path) -> list[dict[str, object]]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("recording slot manifest must be a JSON list")
    return [{str(key): value for key, value in dict(item).items()} for item in loaded]


def _slot_records(calibration_root: Path, slots: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    seen_slot_ids: set[str] = set()
    seen_targets: set[str] = set()
    root = _resolved(calibration_root)
    for index, slot in enumerate(slots):
        slot_id = str(slot.get("slot_id", f"slot_{index:03d}"))
        label = str(slot.get("label", ""))
        target_path = Path(str(slot.get("target_path", "")))
        resolved_target = _resolved(target_path)
        target_key = _path_key(resolved_target)
        if slot_id in seen_slot_ids:
            failures.append({"code": "duplicate_slot_id", "slot_id": slot_id})
        seen_slot_ids.add(slot_id)
        if target_key in seen_targets:
            failures.append({"code": "duplicate_target_path", "target_path": target_path.as_posix()})
        seen_targets.add(target_key)
        if label not in REVIEW_LABEL_ORDER:
            failures.append({"code": "invalid_label", "slot_id": slot_id, "label": label})
        if not _is_within(resolved_target, root):
            failures.append({"code": "target_outside_calibration_root", "slot_id": slot_id, "target_path": target_path.as_posix()})
        exists = target_path.exists()
        size_bytes = int(target_path.stat().st_size) if exists else 0
        records.append(
            {
                "slot_id": slot_id,
                "label": label,
                "filename": str(slot.get("filename", target_path.name)),
                "target_path": target_path.as_posix(),
                "exists": exists,
                "size_bytes": size_bytes,
                "viewpoint": str(slot.get("viewpoint", "")),
                "background": str(slot.get("background", "")),
                "motion_focus": str(slot.get("motion_focus", "")),
            }
        )
    return records, failures


def _extra_mp4s(calibration_root: Path, records: list[dict[str, object]]) -> list[dict[str, object]]:
    planned = {_path_key(_resolved(Path(str(record["target_path"])))) for record in records}
    extra: list[dict[str, object]] = []
    if not calibration_root.exists():
        return extra
    for path in sorted(calibration_root.rglob("*.mp4"), key=natural_key):
        if _path_key(_resolved(path)) in planned:
            continue
        label = path.parent.name
        extra.append(
            {
                "label": label,
                "source_path": path.as_posix(),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return extra


def _status(
    *,
    manifest_failures: list[dict[str, object]],
    missing_records: list[dict[str, object]],
    extra_records: list[dict[str, object]],
) -> str:
    if manifest_failures or extra_records:
        return "slot_audit_failed"
    if missing_records:
        return "awaiting_recordings"
    return "ready_for_mp4_preflight"


def _write_outputs(output_root: Path, summary: dict[str, object]) -> None:
    (output_root / "recording_slot_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_table(output_root / "recording_slot_audit_table.csv", list(summary.get("records", [])))
    (output_root / "recording_slot_audit_summary.md").write_text(_markdown(summary), encoding="utf-8")


def _write_table(path: Path, records: list[object]) -> None:
    fieldnames = [
        "slot_id",
        "label",
        "filename",
        "target_path",
        "exists",
        "size_bytes",
        "viewpoint",
        "background",
        "motion_focus",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if isinstance(record, dict):
                writer.writerow({field: record.get(field) for field in fieldnames})


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# V4 Recording Slot Audit",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Calibration root: `{summary.get('calibration_root')}`",
        f"- Slot manifest: `{summary.get('slot_manifest_path')}`",
        f"- Filled slots: `{summary.get('filled_slot_count')}` / `{summary.get('slot_count')}`",
        f"- Missing slots: `{summary.get('missing_slot_count')}`",
        f"- Extra MP4s: `{summary.get('extra_mp4_count')}`",
        "",
        "## Label Counts",
        "",
        "| Label | Planned | Filled | Missing |",
        "|---|---:|---:|---:|",
    ]
    planned = summary.get("planned_label_counts")
    filled = summary.get("filled_label_counts")
    missing = summary.get("missing_by_label")
    if isinstance(planned, dict) and isinstance(filled, dict) and isinstance(missing, dict):
        for label in REVIEW_LABEL_ORDER:
            lines.append(f"| `{label}` | `{planned.get(label, 0)}` | `{filled.get(label, 0)}` | `{missing.get(label, 0)}` |")
    lines.extend(["", "## Blocking Issues", ""])
    failures = summary.get("failures")
    if isinstance(failures, list) and failures:
        for failure in failures[:25]:
            lines.append(f"- `{json.dumps(failure, ensure_ascii=False)}`")
        if len(failures) > 25:
            lines.append(f"- ... `{len(failures) - 25}` more failures omitted from markdown.")
    else:
        lines.append("- None.")
    lines.extend(["", "## Next Step", "", _next_step(str(summary.get("status"))), ""])
    return "\n".join(lines)


def _next_step(status: str) -> str:
    if status == "ready_for_mp4_preflight":
        return "Run the v4 MP4 preflight audit before MediaPipe skeleton review."
    if status == "awaiting_recordings":
        return "Record or add MP4 files for the missing manifest slots."
    if status == "missing_slot_manifest":
        return "Generate the v4 recording slot manifest before recording audit."
    return "Fix manifest integrity or remove extra MP4s before continuing."


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _path_key(path: Path) -> str:
    return str(path).casefold()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


__all__ = ["V4RecordingSlotAuditConfig", "audit_v4_recording_slots"]
