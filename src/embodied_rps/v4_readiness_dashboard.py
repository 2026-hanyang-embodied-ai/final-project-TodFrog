"""Human-readable dashboard for v4 calibration and training readiness."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER, natural_key
from embodied_rps.v4_recording_slot_audit import V4RecordingSlotAuditConfig, audit_v4_recording_slots


@dataclass(frozen=True)
class V4ReadinessDashboardConfig:
    """Configuration for the v4 readiness dashboard."""

    calibration_root: Path
    output_root: Path
    end_to_end_summary_path: Path
    expected_per_label: int = 20
    heldout_root: Path | None = None
    recording_slot_audit_output_root: Path | None = None


def build_v4_readiness_dashboard(config: V4ReadinessDashboardConfig) -> dict[str, object]:
    """Write a compact dashboard for the current v4 calibration gate."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    checklist = _load_json_if_exists(config.calibration_root / "recording_checklist.json")
    expected_per_label = int(checklist.get("expected_per_label", config.expected_per_label)) if checklist else config.expected_per_label
    video_records = _scan_videos(config.calibration_root)
    slot_audit_output_root = config.recording_slot_audit_output_root or (config.output_root / "recording_slot_audit")
    slot_audit = audit_v4_recording_slots(
        V4RecordingSlotAuditConfig(
            calibration_root=config.calibration_root,
            output_root=slot_audit_output_root,
        )
    )
    label_counts = Counter(str(record["label"]) for record in video_records)
    remaining = {
        label: max(0, int(expected_per_label) - int(label_counts.get(label, 0)))
        for label in REVIEW_LABEL_ORDER
    }
    end_to_end = _load_json_if_exists(config.end_to_end_summary_path)
    current_gate = str(end_to_end.get("current_gate", "unknown")) if end_to_end else "missing_end_to_end_summary"
    blocking_stage = end_to_end.get("blocking_stage") if end_to_end else "end_to_end_summary"
    next_action = end_to_end.get("next_action") if end_to_end else "run_v4_end_to_end"
    status = "ready_for_v4_end_to_end_rerun" if all(value == 0 for value in remaining.values()) else "recording_incomplete"
    if end_to_end and end_to_end.get("status") == "strict_gates_passed":
        status = "strict_gates_passed"

    dashboard = {
        "status": status,
        "calibration_root": config.calibration_root.as_posix(),
        "heldout_root": config.heldout_root.as_posix() if config.heldout_root is not None else None,
        "expected_per_label": int(expected_per_label),
        "expected_total": int(expected_per_label) * len(REVIEW_LABEL_ORDER),
        "video_count": len(video_records),
        "label_counts": {label: int(label_counts.get(label, 0)) for label in REVIEW_LABEL_ORDER},
        "remaining_counts": remaining,
        "current_gate": current_gate,
        "blocking_stage": blocking_stage,
        "next_action": next_action,
        "end_to_end_summary": config.end_to_end_summary_path.as_posix(),
        "records": video_records,
        "recording_slot_audit": _compact_slot_audit(slot_audit),
        "commands": _commands(config),
    }
    (config.output_root / "readiness_dashboard.json").write_text(
        json.dumps(dashboard, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (config.output_root / "readiness_dashboard.md").write_text(_dashboard_markdown(dashboard), encoding="utf-8")
    return dashboard


def _scan_videos(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not root.exists():
        return records
    for label in REVIEW_LABEL_ORDER:
        label_dir = root / label
        if not label_dir.is_dir():
            continue
        for path in sorted(label_dir.glob("*.mp4"), key=natural_key):
            stat = path.stat()
            records.append(
                {
                    "label": label,
                    "source_path": path.as_posix(),
                    "size_bytes": int(stat.st_size),
                    "last_write_time": stat.st_mtime,
                }
            )
    return records


def _commands(config: V4ReadinessDashboardConfig) -> dict[str, str]:
    return {
        "audit_recording_slots": (
            "python -m embodied_rps.tools.audit_v4_recording_slots "
            f"--calibration-root {config.calibration_root.as_posix()} "
            "--output-root artifacts\\real_skeleton_v4_recording_slot_audit_20260612"
        ),
        "run_end_to_end_safe": (
            "python -m embodied_rps.tools.run_v4_end_to_end "
            "--output-root artifacts\\real_skeleton_v4_end_to_end_20260612"
        ),
        "run_end_to_end_with_skeleton_review": (
            "python -m embodied_rps.tools.run_v4_end_to_end "
            "--output-root artifacts\\real_skeleton_v4_end_to_end_20260612 "
            "--execute-skeleton-review"
        ),
        "run_end_to_end_with_dataset_generation": (
            "python -m embodied_rps.tools.run_v4_end_to_end "
            "--output-root artifacts\\real_skeleton_v4_end_to_end_20260612 "
            "--execute-dataset-generation --overwrite-dataset"
        ),
        "open_calibration_root": config.calibration_root.as_posix(),
    }


def _dashboard_markdown(dashboard: dict[str, object]) -> str:
    lines = [
        "# V4 Readiness Dashboard",
        "",
        f"- Status: `{dashboard.get('status')}`",
        f"- Calibration root: `{dashboard.get('calibration_root')}`",
        f"- Held-out root: `{dashboard.get('heldout_root')}`",
        f"- Current gate: `{dashboard.get('current_gate')}`",
        f"- Blocking stage: `{dashboard.get('blocking_stage')}`",
        f"- Next action: `{dashboard.get('next_action')}`",
        "",
        "## Recording Counts",
        "",
        "| Label | Current | Required | Remaining |",
        "|---|---:|---:|---:|",
    ]
    label_counts = dashboard.get("label_counts")
    remaining_counts = dashboard.get("remaining_counts")
    expected = int(dashboard.get("expected_per_label", 0))
    if isinstance(label_counts, dict) and isinstance(remaining_counts, dict):
        for label in REVIEW_LABEL_ORDER:
            lines.append(
                f"| `{label}` | `{label_counts.get(label, 0)}` | `{expected}` | `{remaining_counts.get(label, expected)}` |"
            )
    slot_audit = dashboard.get("recording_slot_audit")
    if isinstance(slot_audit, dict):
        lines.extend(
            [
                "",
                "## Recording Slot Coverage",
                "",
                f"- Status: `{slot_audit.get('status')}`",
                f"- Filled slots: `{slot_audit.get('filled_slot_count')}` / `{slot_audit.get('slot_count')}`",
                f"- Missing slots: `{slot_audit.get('missing_slot_count')}`",
                f"- Extra MP4s: `{slot_audit.get('extra_mp4_count')}`",
                f"- Audit table: `{slot_audit.get('audit_table')}`",
                "",
            ]
        )
    lines.extend(["", "## Commands", ""])
    commands = dashboard.get("commands")
    if isinstance(commands, dict):
        for name, command in commands.items():
            lines.extend([f"### {name}", "", "```powershell", str(command), "```", ""])
    lines.extend(
        [
            "## Notes",
            "",
            "- The held-out test folder remains validation-only.",
            "- Run skeleton review only after MP4 preflight and intake pass.",
            "- Run dataset generation only after visual skeleton approval.",
            "- SCHUNK/Isaac remains blocked until both strict video gates pass.",
            "",
        ]
    )
    return "\n".join(lines)


def _compact_slot_audit(slot_audit: dict[str, object]) -> dict[str, object]:
    return {
        "status": slot_audit.get("status"),
        "slot_count": slot_audit.get("slot_count"),
        "filled_slot_count": slot_audit.get("filled_slot_count"),
        "missing_slot_count": slot_audit.get("missing_slot_count"),
        "extra_mp4_count": slot_audit.get("extra_mp4_count"),
        "planned_label_counts": slot_audit.get("planned_label_counts"),
        "filled_label_counts": slot_audit.get("filled_label_counts"),
        "missing_by_label": slot_audit.get("missing_by_label"),
        "audit_table": slot_audit.get("audit_table"),
    }


def _load_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {}
    return {str(key): value for key, value in loaded.items()}


__all__ = ["V4ReadinessDashboardConfig", "build_v4_readiness_dashboard"]
