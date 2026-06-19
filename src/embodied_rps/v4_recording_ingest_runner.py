"""Safe ingest runner for v4 recording staging clips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.v4_readiness_dashboard import V4ReadinessDashboardConfig, build_v4_readiness_dashboard
from embodied_rps.v4_recording_slot_assignment import (
    V4RecordingSlotAssignmentConfig,
    plan_v4_recording_slot_assignment,
)
from embodied_rps.v4_recording_slot_audit import V4RecordingSlotAuditConfig, audit_v4_recording_slots
from embodied_rps.v4_recording_staging_audit import V4RecordingStagingAuditConfig, audit_v4_recording_staging
from embodied_rps.v4_recording_staging_audit import VideoProbe as StagingVideoProbe


@dataclass(frozen=True)
class V4RecordingIngestConfig:
    """Configuration for one safe v4 recording ingest pass."""

    source_root: Path
    calibration_root: Path
    heldout_root: Path
    output_root: Path
    end_to_end_summary_path: Path
    expected_per_label: int = 20
    execute_copy: bool = False
    slot_manifest_path: Path | None = None
    staging_video_probe: StagingVideoProbe | None = None


def run_v4_recording_ingest(config: V4RecordingIngestConfig) -> dict[str, object]:
    """Run assignment, slot audit, and readiness dashboard for v4 recordings."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    staging_audit = audit_v4_recording_staging(
        V4RecordingStagingAuditConfig(
            staging_root=config.source_root,
            calibration_root=config.calibration_root,
            heldout_roots=(config.heldout_root,),
            expected_per_label=config.expected_per_label,
            output_root=config.output_root / "staging_audit",
        ),
        video_probe=config.staging_video_probe,
    )
    if _staging_audit_blocks_assignment(staging_audit):
        assignment = _skipped_assignment(config, staging_audit)
    else:
        assignment = plan_v4_recording_slot_assignment(
            V4RecordingSlotAssignmentConfig(
                source_root=config.source_root,
                calibration_root=config.calibration_root,
                slot_manifest_path=config.slot_manifest_path,
                output_root=config.output_root / "assignment",
                execute_copy=config.execute_copy,
            )
        )
    slot_audit = audit_v4_recording_slots(
        V4RecordingSlotAuditConfig(
            calibration_root=config.calibration_root,
            slot_manifest_path=config.slot_manifest_path,
            output_root=config.output_root / "slot_audit",
        )
    )
    dashboard = build_v4_readiness_dashboard(
        V4ReadinessDashboardConfig(
            calibration_root=config.calibration_root,
            heldout_root=config.heldout_root,
            expected_per_label=config.expected_per_label,
            output_root=config.output_root / "readiness_dashboard",
            end_to_end_summary_path=config.end_to_end_summary_path,
            recording_slot_audit_output_root=config.output_root / "slot_audit",
        )
    )
    status = _ingest_status(staging_audit, assignment, slot_audit, execute_copy=config.execute_copy)
    next_action = _next_action(staging_audit, assignment, slot_audit, execute_copy=config.execute_copy)
    summary = {
        "status": status,
        "execute_copy": config.execute_copy,
        "source_root": config.source_root.as_posix(),
        "calibration_root": config.calibration_root.as_posix(),
        "heldout_root": config.heldout_root.as_posix(),
        "staging_audit": _compact_staging_audit(staging_audit),
        "assignment": _compact_assignment(assignment),
        "slot_audit": _compact_slot_audit(slot_audit),
        "readiness_dashboard": _compact_dashboard(dashboard),
        "next_action": next_action,
        "gate_decision": _gate_decision(
            config=config,
            status=status,
            next_action=next_action,
            staging_audit=staging_audit,
            assignment=assignment,
            slot_audit=slot_audit,
        ),
        "outputs": {
            "staging_audit_summary": (config.output_root / "staging_audit" / "recording_staging_audit_summary.json").as_posix(),
            "assignment_summary": (config.output_root / "assignment" / "recording_slot_assignment_summary.json").as_posix(),
            "slot_audit_summary": (config.output_root / "slot_audit" / "recording_slot_audit_summary.json").as_posix(),
            "readiness_dashboard": (config.output_root / "readiness_dashboard" / "readiness_dashboard.json").as_posix(),
        },
    }
    _write_outputs(config.output_root, summary)
    return summary


def _ingest_status(
    staging_audit: dict[str, object],
    assignment: dict[str, object],
    slot_audit: dict[str, object],
    *,
    execute_copy: bool,
) -> str:
    staging_status = str(staging_audit.get("status"))
    assignment_status = str(assignment.get("status"))
    slot_status = str(slot_audit.get("status"))
    if staging_status in {"invalid_roots", "staging_needs_review"}:
        return "recording_ingest_blocked"
    if staging_status in {"missing_staging_root", "awaiting_staging_mp4s"}:
        return "awaiting_staging_sources"
    if assignment_status in {"missing_slot_manifest", "invalid_roots", "copy_failed"}:
        return "recording_ingest_blocked"
    if assignment_status == "no_staging_sources":
        return "awaiting_staging_sources"
    if not execute_copy and assignment_status in {"assignment_ready", "partial_assignment_ready"}:
        return "assignment_ready_for_review"
    if execute_copy and slot_status == "ready_for_mp4_preflight":
        return "ready_for_mp4_preflight"
    if execute_copy and assignment_status == "partial_copy_complete":
        return "awaiting_more_staging_sources"
    if slot_status == "awaiting_recordings":
        return "awaiting_recordings"
    return "recording_ingest_needs_review"


def _next_action(
    staging_audit: dict[str, object],
    assignment: dict[str, object],
    slot_audit: dict[str, object],
    *,
    execute_copy: bool,
) -> str:
    status = _ingest_status(staging_audit, assignment, slot_audit, execute_copy=execute_copy)
    if status == "assignment_ready_for_review":
        return "review_assignment_table_then_rerun_with_execute_copy"
    if status == "ready_for_mp4_preflight":
        return "run_v4_mp4_preflight_then_skeleton_review"
    if status == "awaiting_staging_sources":
        return "record_or_add_mp4s_to_v4_recording_staging"
    if status == "awaiting_more_staging_sources":
        return "add_more_staging_mp4s_for_remaining_slots"
    if status == "recording_ingest_blocked":
        return "fix_ingest_failure"
    return "inspect_ingest_outputs"


def _gate_decision(
    *,
    config: V4RecordingIngestConfig,
    status: str,
    next_action: str,
    staging_audit: dict[str, object],
    assignment: dict[str, object],
    slot_audit: dict[str, object],
) -> dict[str, object]:
    copy_execution_allowed = status == "assignment_ready_for_review" and _int(assignment.get("assignment_count")) > 0
    mp4_preflight_allowed = status == "ready_for_mp4_preflight" and str(slot_audit.get("status")) == "ready_for_mp4_preflight"
    return {
        "status": status,
        "next_action": next_action,
        "hard_stop": status == "recording_ingest_blocked",
        "hard_stop_reasons": _hard_stop_reasons(staging_audit, assignment),
        "copy_execution_allowed": copy_execution_allowed,
        "copy_execution_command": _execute_copy_command(config) if copy_execution_allowed else None,
        "mp4_preflight_allowed": mp4_preflight_allowed,
        "mp4_preflight_command": _mp4_preflight_command(config) if mp4_preflight_allowed else None,
        "skeleton_review_allowed": False,
        "skeleton_review_reason": "run_and_pass_mp4_preflight_first",
        "schunk_allowed": False,
        "schunk_reason": "strict_original20_and_heldout15_gates_not_passed",
    }


def _hard_stop_reasons(staging_audit: dict[str, object], assignment: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    staging_status = str(staging_audit.get("status"))
    if staging_status in {"invalid_roots", "staging_needs_review"}:
        reasons.append(f"staging_audit:{staging_status}")
    assignment_status = str(assignment.get("status"))
    if assignment_status in {"missing_slot_manifest", "invalid_roots", "copy_failed", "skipped_due_to_staging_audit"}:
        reasons.append(f"assignment:{assignment_status}")
    return reasons


def _execute_copy_command(config: V4RecordingIngestConfig) -> str:
    parts = [
        "python",
        "-m",
        "embodied_rps.tools.run_v4_recording_ingest",
        "--source-root",
        config.source_root.as_posix(),
        "--calibration-root",
        config.calibration_root.as_posix(),
        "--heldout-root",
        config.heldout_root.as_posix(),
        "--output-root",
        config.output_root.as_posix(),
        "--end-to-end-summary",
        config.end_to_end_summary_path.as_posix(),
        "--expected-per-label",
        str(config.expected_per_label),
        "--execute-copy",
    ]
    if config.slot_manifest_path is not None:
        parts.extend(["--slot-manifest", config.slot_manifest_path.as_posix()])
    return _join_command(parts)


def _mp4_preflight_command(config: V4RecordingIngestConfig) -> str:
    return _join_command(
        [
            "python",
            "-m",
            "embodied_rps.tools.audit_v4_calibration_mp4s",
            "--input-root",
            config.calibration_root.as_posix(),
            "--heldout-root",
            config.heldout_root.as_posix(),
            "--expected-min-per-label",
            str(config.expected_per_label),
            "--output-root",
            "artifacts/real_skeleton_v4_mp4_preflight_20260612",
        ]
    )


def _staging_audit_blocks_assignment(staging_audit: dict[str, object]) -> bool:
    return str(staging_audit.get("status")) in {"invalid_roots", "staging_needs_review"}


def _skipped_assignment(config: V4RecordingIngestConfig, staging_audit: dict[str, object]) -> dict[str, object]:
    output_root = config.output_root / "assignment"
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "skipped_due_to_staging_audit",
        "execute_copy": False,
        "requested_execute_copy": bool(config.execute_copy),
        "source_root": config.source_root.as_posix(),
        "calibration_root": config.calibration_root.as_posix(),
        "slot_manifest_path": (config.slot_manifest_path or (config.calibration_root / "recording_slot_manifest.json")).as_posix(),
        "skip_reason": str(staging_audit.get("status")),
        "source_count": 0,
        "missing_slot_count": None,
        "assignment_count": 0,
        "copied_count": 0,
        "remaining_after_assignment": None,
        "assignments": [],
        "copied": [],
        "failures": [
            {
                "code": "staging_audit_blocked_assignment",
                "staging_status": str(staging_audit.get("status")),
                "failure_count": staging_audit.get("failure_count"),
            }
        ],
        "assignment_table": (output_root / "recording_slot_assignment_table.csv").as_posix(),
    }
    _write_skipped_assignment_outputs(output_root, summary)
    return summary


def _compact_staging_audit(staging_audit: dict[str, object]) -> dict[str, object]:
    return {
        "status": staging_audit.get("status"),
        "mp4_count": staging_audit.get("mp4_count"),
        "valid_mp4_count": staging_audit.get("valid_mp4_count"),
        "hashed_mp4_count": staging_audit.get("hashed_mp4_count"),
        "heldout_hash_count": staging_audit.get("heldout_hash_count"),
        "failed_video_probe_count": staging_audit.get("failed_video_probe_count"),
        "label_counts": staging_audit.get("label_counts"),
        "remaining_counts": staging_audit.get("remaining_counts"),
        "failure_count": staging_audit.get("failure_count"),
        "warning_count": staging_audit.get("warning_count"),
        "audit_table": staging_audit.get("audit_table"),
    }


def _compact_assignment(assignment: dict[str, object]) -> dict[str, object]:
    return {
        "status": assignment.get("status"),
        "requested_execute_copy": assignment.get("requested_execute_copy", assignment.get("execute_copy")),
        "source_count": assignment.get("source_count"),
        "missing_slot_count": assignment.get("missing_slot_count"),
        "assignment_count": assignment.get("assignment_count"),
        "copied_count": assignment.get("copied_count"),
        "archived_count": assignment.get("archived_count"),
        "remaining_after_assignment": assignment.get("remaining_after_assignment"),
        "assignment_table": assignment.get("assignment_table"),
    }


def _compact_slot_audit(slot_audit: dict[str, object]) -> dict[str, object]:
    return {
        "status": slot_audit.get("status"),
        "slot_count": slot_audit.get("slot_count"),
        "filled_slot_count": slot_audit.get("filled_slot_count"),
        "missing_slot_count": slot_audit.get("missing_slot_count"),
        "extra_mp4_count": slot_audit.get("extra_mp4_count"),
        "missing_by_label": slot_audit.get("missing_by_label"),
        "audit_table": slot_audit.get("audit_table"),
    }


def _compact_dashboard(dashboard: dict[str, object]) -> dict[str, object]:
    return {
        "status": dashboard.get("status"),
        "video_count": dashboard.get("video_count"),
        "label_counts": dashboard.get("label_counts"),
        "remaining_counts": dashboard.get("remaining_counts"),
        "recording_slot_audit": dashboard.get("recording_slot_audit"),
    }


def _write_outputs(output_root: Path, summary: dict[str, object]) -> None:
    (output_root / "recording_ingest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "recording_ingest_summary.md").write_text(_markdown(summary), encoding="utf-8")


def _write_skipped_assignment_outputs(output_root: Path, summary: dict[str, object]) -> None:
    (output_root / "recording_slot_assignment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "recording_slot_assignment_table.csv").write_text(
        "slot_id,label,source_path,target_path,target_filename,planned_action,motion_focus,viewpoint,background\n",
        encoding="utf-8",
    )
    lines = [
        "# V4 Recording Slot Assignment",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Requested execute copy: `{summary.get('requested_execute_copy')}`",
        f"- Copied files: `{summary.get('copied_count')}`",
        f"- Skip reason: `{summary.get('skip_reason')}`",
        "",
        "## Next Step",
        "",
        "Fix the staging audit failures before reviewing assignments or copying files.",
        "",
    ]
    (output_root / "recording_slot_assignment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown(summary: dict[str, object]) -> str:
    staging_audit = summary.get("staging_audit")
    assignment = summary.get("assignment")
    slot_audit = summary.get("slot_audit")
    dashboard = summary.get("readiness_dashboard")
    gate_decision = summary.get("gate_decision")
    lines = [
        "# V4 Recording Ingest Summary",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Execute copy: `{summary.get('execute_copy')}`",
        f"- Source root: `{summary.get('source_root')}`",
        f"- Calibration root: `{summary.get('calibration_root')}`",
        f"- Next action: `{summary.get('next_action')}`",
        "",
    ]
    if isinstance(gate_decision, dict):
        lines.extend(
            [
                "## Gate Decision",
                "",
                f"- Hard stop: `{gate_decision.get('hard_stop')}`",
                f"- Copy execution allowed: `{gate_decision.get('copy_execution_allowed')}`",
                f"- MP4 preflight allowed: `{gate_decision.get('mp4_preflight_allowed')}`",
                f"- Skeleton review allowed: `{gate_decision.get('skeleton_review_allowed')}`",
                f"- SCHUNK allowed: `{gate_decision.get('schunk_allowed')}`",
            ]
        )
        copy_command = gate_decision.get("copy_execution_command")
        if isinstance(copy_command, str) and copy_command:
            lines.extend(["", "### Execute Copy Command", "", "```powershell", copy_command, "```"])
        preflight_command = gate_decision.get("mp4_preflight_command")
        if isinstance(preflight_command, str) and preflight_command:
            lines.extend(["", "### MP4 Preflight Command", "", "```powershell", preflight_command, "```"])
        lines.append("")
    if isinstance(staging_audit, dict):
        lines.extend(
            [
                "## Staging Audit",
                "",
                f"- Status: `{staging_audit.get('status')}`",
                f"- MP4s: `{staging_audit.get('mp4_count')}`",
                f"- Valid MP4s: `{staging_audit.get('valid_mp4_count')}`",
                f"- Hashed MP4s: `{staging_audit.get('hashed_mp4_count')}`",
                f"- Held-out hashes: `{staging_audit.get('heldout_hash_count')}`",
                f"- Failed video probes: `{staging_audit.get('failed_video_probe_count')}`",
                f"- Failure count: `{staging_audit.get('failure_count')}`",
                f"- Warning count: `{staging_audit.get('warning_count')}`",
                f"- Audit table: `{staging_audit.get('audit_table')}`",
                "",
            ]
        )
    if isinstance(assignment, dict):
        lines.extend(
            [
                "## Assignment",
                "",
                f"- Status: `{assignment.get('status')}`",
                f"- Source MP4s: `{assignment.get('source_count')}`",
                f"- Planned assignments: `{assignment.get('assignment_count')}`",
                f"- Copied files: `{assignment.get('copied_count')}`",
                f"- Assignment table: `{assignment.get('assignment_table')}`",
                "",
            ]
        )
    if isinstance(slot_audit, dict):
        lines.extend(
            [
                "## Slot Audit",
                "",
                f"- Status: `{slot_audit.get('status')}`",
                f"- Filled slots: `{slot_audit.get('filled_slot_count')}` / `{slot_audit.get('slot_count')}`",
                f"- Missing slots: `{slot_audit.get('missing_slot_count')}`",
                f"- Extra MP4s: `{slot_audit.get('extra_mp4_count')}`",
                "",
            ]
        )
    if isinstance(dashboard, dict):
        lines.extend(
            [
                "## Readiness Dashboard",
                "",
                f"- Status: `{dashboard.get('status')}`",
                f"- Video count: `{dashboard.get('video_count')}`",
                f"- Label counts: `{json.dumps(dashboard.get('label_counts'), ensure_ascii=False)}`",
                "",
            ]
        )
    return "\n".join(lines)


def _join_command(parts: list[str]) -> str:
    return " ".join(_quote_part(str(part)) for part in parts)


def _quote_part(part: str) -> str:
    if not part or any(char.isspace() for char in part) or any(ord(char) > 127 for char in part) or "\\" in part or ":" in part:
        escaped = part.replace("'", "''")
        return f"'{escaped}'"
    return part


def _int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


__all__ = ["V4RecordingIngestConfig", "run_v4_recording_ingest"]
