"""Post-recording check for v4 staging clips."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER
from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_ingest_runner import V4RecordingIngestConfig, run_v4_recording_ingest
from embodied_rps.v4_recording_staging_audit import VideoProbe as StagingVideoProbe


@dataclass(frozen=True)
class V4RecordingPostcheckConfig:
    """Configuration for the recording postcheck."""

    staging_root: Path = DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging"
    calibration_root: Path = DEFAULT_LOCAL_DATA_ROOT / "v4_calibration"
    heldout_root: Path = DEFAULT_LOCAL_DATA_ROOT / "test"
    output_root: Path = Path("artifacts/real_skeleton_v4_recording_postcheck_20260612")
    slot_manifest_path: Path | None = DEFAULT_LOCAL_DATA_ROOT / "v4_calibration" / "recording_slot_manifest.json"
    end_to_end_summary_path: Path = Path("artifacts/real_skeleton_v4_end_to_end_20260612/end_to_end_summary.json")
    expected_per_label: int = 20
    expected_new_per_label: int = 1
    session_output_root: Path = Path("artifacts/real_skeleton_v4_recording_session_20260612")
    pre_roll_s: float = 1.5
    duration_s: float = 3.0
    fps: float = 30.0


def run_v4_recording_postcheck(
    config: V4RecordingPostcheckConfig,
    *,
    staging_video_probe: StagingVideoProbe | None = None,
) -> dict[str, object]:
    """Summarize whether newly staged clips are ready for assignment review."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    ingest = run_v4_recording_ingest(
        V4RecordingIngestConfig(
            source_root=config.staging_root,
            calibration_root=config.calibration_root,
            heldout_root=config.heldout_root,
            output_root=config.output_root / "ingest_dry_run",
            end_to_end_summary_path=config.end_to_end_summary_path,
            expected_per_label=config.expected_per_label,
            execute_copy=False,
            slot_manifest_path=config.slot_manifest_path,
            staging_video_probe=staging_video_probe,
        )
    )
    staging_audit = ingest.get("staging_audit") if isinstance(ingest.get("staging_audit"), Mapping) else {}
    assignment = ingest.get("assignment") if isinstance(ingest.get("assignment"), Mapping) else {}
    label_counts = {
        label: int((staging_audit.get("label_counts") or {}).get(label, 0)) if isinstance(staging_audit.get("label_counts"), Mapping) else 0
        for label in REVIEW_LABEL_ORDER
    }
    missing_for_first_batch = {
        label: max(0, int(config.expected_new_per_label) - int(label_counts.get(label, 0)))
        for label in REVIEW_LABEL_ORDER
    }
    status = _postcheck_status(ingest, missing_for_first_batch)
    summary = {
        "status": status,
        "staging_root": config.staging_root.as_posix(),
        "calibration_root": config.calibration_root.as_posix(),
        "heldout_root": config.heldout_root.as_posix(),
        "expected_new_per_label": int(config.expected_new_per_label),
        "staging_label_counts": label_counts,
        "missing_for_first_batch": missing_for_first_batch,
        "ingest_status": ingest.get("status"),
        "staging_audit_status": staging_audit.get("status"),
        "valid_mp4_count": staging_audit.get("valid_mp4_count"),
        "failed_video_probe_count": staging_audit.get("failed_video_probe_count"),
        "assignment_status": assignment.get("status"),
        "assignment_count": assignment.get("assignment_count"),
        "copy_execution_allowed": (ingest.get("gate_decision") or {}).get("copy_execution_allowed")
        if isinstance(ingest.get("gate_decision"), Mapping)
        else False,
        "copy_execution_command": (ingest.get("gate_decision") or {}).get("copy_execution_command")
        if isinstance(ingest.get("gate_decision"), Mapping)
        else None,
        "recording_command": _recording_command(config),
        "next_action": _next_action(status),
        "ingest_summary": (config.output_root / "ingest_dry_run" / "recording_ingest_summary.json").as_posix(),
        "postcheck_summary": (config.output_root / "recording_postcheck_summary.json").as_posix(),
    }
    _write_outputs(config.output_root, summary)
    return summary


def _postcheck_status(ingest: Mapping[str, object], missing_for_first_batch: Mapping[str, int]) -> str:
    if str(ingest.get("status")) == "recording_ingest_blocked":
        return "postcheck_blocked"
    if any(int(value) > 0 for value in missing_for_first_batch.values()):
        return "awaiting_recorded_clips"
    if str(ingest.get("status")) == "assignment_ready_for_review":
        return "ready_to_review_assignment"
    return "postcheck_needs_review"


def _next_action(status: str) -> str:
    if status == "awaiting_recorded_clips":
        return "record_missing_staging_clips"
    if status == "ready_to_review_assignment":
        return "review_assignment_table_then_execute_copy"
    if status == "postcheck_blocked":
        return "fix_staging_or_video_probe_failure"
    return "inspect_postcheck_outputs"


def _recording_command(config: V4RecordingPostcheckConfig) -> str:
    parts = [
        "python",
        "-m",
        "embodied_rps.tools.record_v4_staging_session",
        "--staging-root",
        config.staging_root.as_posix(),
        "--output-root",
        config.session_output_root.as_posix(),
        "--count-per-label",
        str(config.expected_new_per_label),
        "--pre-roll-s",
        str(config.pre_roll_s),
        "--duration-s",
        str(config.duration_s),
        "--fps",
        str(config.fps),
    ]
    if config.slot_manifest_path is not None:
        parts.extend(["--slot-manifest", config.slot_manifest_path.as_posix()])
    parts.append("--execute")
    return " ".join(_quote(part) for part in parts)


def _write_outputs(output_root: Path, summary: Mapping[str, object]) -> None:
    (output_root / "recording_postcheck_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "recording_postcheck_summary.md").write_text(_markdown(summary), encoding="utf-8")


def _markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V4 Recording Postcheck",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Next action: `{summary.get('next_action')}`",
        f"- Ingest status: `{summary.get('ingest_status')}`",
        f"- Staging audit status: `{summary.get('staging_audit_status')}`",
        f"- Assignment status: `{summary.get('assignment_status')}`",
        f"- Valid MP4 count: `{summary.get('valid_mp4_count')}`",
        f"- Failed video probe count: `{summary.get('failed_video_probe_count')}`",
        "",
        "## Staging Counts",
        "",
        "| Label | Staged | Missing For First Batch |",
        "|---|---:|---:|",
    ]
    label_counts = summary.get("staging_label_counts")
    missing = summary.get("missing_for_first_batch")
    if isinstance(label_counts, Mapping) and isinstance(missing, Mapping):
        for label in REVIEW_LABEL_ORDER:
            lines.append(f"| `{label}` | `{label_counts.get(label, 0)}` | `{missing.get(label, 0)}` |")
    if summary.get("copy_execution_allowed") and summary.get("copy_execution_command"):
        lines.extend(["", "## Execute Copy Command", "", "```powershell", str(summary.get("copy_execution_command")), "```"])
    else:
        lines.extend(["", "## Recording Command", "", "```powershell", str(summary.get("recording_command")), "```"])
    lines.extend(
        [
            "",
            "## Linked Outputs",
            "",
            f"- Ingest summary: `{summary.get('ingest_summary')}`",
            f"- Postcheck summary: `{summary.get('postcheck_summary')}`",
            "",
        ]
    )
    return "\n".join(lines)


def _quote(part: str) -> str:
    if not part or any(char.isspace() for char in part) or any(ord(char) > 127 for char in part) or "\\" in part or ":" in part:
        return "'" + part.replace("'", "''") + "'"
    return part


__all__ = ["V4RecordingPostcheckConfig", "run_v4_recording_postcheck"]
