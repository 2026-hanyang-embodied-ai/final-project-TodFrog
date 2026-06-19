"""Opt-in monitor for refreshing v4 recording ingest status while clips arrive."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER, natural_key
from embodied_rps.v4_recording_ingest_runner import V4RecordingIngestConfig, run_v4_recording_ingest
from embodied_rps.v4_recording_staging_audit import VideoProbe as StagingVideoProbe

SleepFn = Callable[[float], None]


@dataclass(frozen=True)
class V4RecordingIngestMonitorConfig:
    """Configuration for a bounded v4 recording ingest monitor."""

    source_root: Path
    calibration_root: Path
    heldout_root: Path
    output_root: Path
    ingest_output_root: Path
    end_to_end_summary_path: Path
    expected_per_label: int = 20
    iterations: int = 1
    poll_interval_s: float = 5.0
    slot_manifest_path: Path | None = None


def monitor_v4_recording_ingest(
    config: V4RecordingIngestMonitorConfig,
    *,
    sleep_fn: SleepFn = time.sleep,
    staging_video_probe: StagingVideoProbe | None = None,
) -> dict[str, object]:
    """Refresh ingest status on the first pass and whenever MP4 snapshots change."""

    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.poll_interval_s < 0:
        raise ValueError("poll_interval_s must be non-negative")
    config.output_root.mkdir(parents=True, exist_ok=True)
    previous_snapshot_id: str | None = None
    refresh_count = 0
    last_ingest: dict[str, object] | None = None
    last_snapshot: dict[str, object] | None = None
    events: list[dict[str, object]] = []
    for index in range(config.iterations):
        snapshot = _snapshot(config.source_root, config.calibration_root)
        last_snapshot = snapshot
        changed = previous_snapshot_id != snapshot["snapshot_id"]
        trigger = "initial" if previous_snapshot_id is None else ("changed" if changed else "no_change")
        ingest_summary: dict[str, object] | None = None
        if changed:
            ingest_summary = run_v4_recording_ingest(
                V4RecordingIngestConfig(
                    source_root=config.source_root,
                    calibration_root=config.calibration_root,
                    heldout_root=config.heldout_root,
                    output_root=config.ingest_output_root,
                    end_to_end_summary_path=config.end_to_end_summary_path,
                    expected_per_label=config.expected_per_label,
                    execute_copy=False,
                    slot_manifest_path=config.slot_manifest_path,
                    staging_video_probe=staging_video_probe,
                )
            )
            refresh_count += 1
            last_ingest = ingest_summary
        events.append(_event(index=index, trigger=trigger, snapshot=snapshot, ingest_summary=ingest_summary))
        previous_snapshot_id = str(snapshot["snapshot_id"])
        if index < config.iterations - 1:
            sleep_fn(float(config.poll_interval_s))
    summary = {
        "status": str(last_ingest.get("status")) if isinstance(last_ingest, dict) else "monitor_no_refresh",
        "next_action": last_ingest.get("next_action") if isinstance(last_ingest, dict) else None,
        "iterations": int(config.iterations),
        "poll_interval_s": float(config.poll_interval_s),
        "refresh_count": int(refresh_count),
        "source_root": config.source_root.as_posix(),
        "calibration_root": config.calibration_root.as_posix(),
        "heldout_root": config.heldout_root.as_posix(),
        "ingest_output_root": config.ingest_output_root.as_posix(),
        "last_snapshot": last_snapshot or {},
        "last_gate_decision": last_ingest.get("gate_decision") if isinstance(last_ingest, dict) else {},
        "events": events,
        "monitor_summary": (config.output_root / "recording_ingest_monitor_summary.json").as_posix(),
    }
    _write_outputs(config.output_root, summary)
    return summary


def _event(
    *,
    index: int,
    trigger: str,
    snapshot: dict[str, object],
    ingest_summary: dict[str, object] | None,
) -> dict[str, object]:
    event = {
        "iteration": int(index),
        "trigger": trigger,
        "snapshot_id": snapshot["snapshot_id"],
        "staging_mp4_count": snapshot["staging"]["mp4_count"] if isinstance(snapshot.get("staging"), dict) else None,
        "calibration_mp4_count": snapshot["calibration"]["mp4_count"] if isinstance(snapshot.get("calibration"), dict) else None,
        "refreshed_ingest": ingest_summary is not None,
    }
    if isinstance(ingest_summary, dict):
        gate_decision = ingest_summary.get("gate_decision")
        event.update(
            {
                "ingest_status": ingest_summary.get("status"),
                "next_action": ingest_summary.get("next_action"),
                "copy_execution_allowed": gate_decision.get("copy_execution_allowed") if isinstance(gate_decision, dict) else None,
                "mp4_preflight_allowed": gate_decision.get("mp4_preflight_allowed") if isinstance(gate_decision, dict) else None,
            }
        )
    return event


def _snapshot(staging_root: Path, calibration_root: Path) -> dict[str, object]:
    staging = _root_snapshot(staging_root)
    calibration = _root_snapshot(calibration_root)
    digest_payload = {
        "staging": staging["records"],
        "calibration": calibration["records"],
    }
    digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        "snapshot_id": digest,
        "staging": staging,
        "calibration": calibration,
    }


def _root_snapshot(root: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    if root.exists():
        for path in sorted(root.rglob("*.mp4"), key=natural_key):
            if not path.is_file():
                continue
            stat = path.stat()
            records.append(
                {
                    "relative_path": _relative(path, root),
                    "label": _label(path, root),
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    label_counts = Counter(str(record["label"]) for record in records if record.get("label"))
    return {
        "root": root.as_posix(),
        "exists": root.exists(),
        "mp4_count": len(records),
        "total_size_bytes": sum(int(record["size_bytes"]) for record in records),
        "label_counts": {label: int(label_counts.get(label, 0)) for label in REVIEW_LABEL_ORDER},
        "records": records,
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _label(path: Path, root: Path) -> str:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    if len(parts) >= 2 and parts[0] in REVIEW_LABEL_ORDER:
        return parts[0]
    return ""


def _write_outputs(output_root: Path, summary: dict[str, object]) -> None:
    (output_root / "recording_ingest_monitor_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "recording_ingest_monitor_summary.md").write_text(_markdown(summary), encoding="utf-8")


def _markdown(summary: dict[str, object]) -> str:
    gate = summary.get("last_gate_decision") if isinstance(summary.get("last_gate_decision"), dict) else {}
    snapshot = summary.get("last_snapshot") if isinstance(summary.get("last_snapshot"), dict) else {}
    staging = snapshot.get("staging") if isinstance(snapshot.get("staging"), dict) else {}
    calibration = snapshot.get("calibration") if isinstance(snapshot.get("calibration"), dict) else {}
    lines = [
        "# V4 Recording Ingest Monitor",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Next action: `{summary.get('next_action')}`",
        f"- Iterations: `{summary.get('iterations')}`",
        f"- Refresh count: `{summary.get('refresh_count')}`",
        f"- Staging MP4s: `{staging.get('mp4_count', 0)}`",
        f"- Calibration MP4s: `{calibration.get('mp4_count', 0)}`",
        f"- Copy execution allowed: `{gate.get('copy_execution_allowed')}`",
        f"- MP4 preflight allowed: `{gate.get('mp4_preflight_allowed')}`",
        "",
        "## Events",
        "",
        "| Iteration | Trigger | Refreshed | Staging MP4s | Calibration MP4s | Status | Next action |",
        "|---:|---|---|---:|---:|---|---|",
    ]
    events = summary.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            lines.append(
                "| "
                f"`{event.get('iteration')}` | `{event.get('trigger')}` | `{event.get('refreshed_ingest')}` | "
                f"`{event.get('staging_mp4_count')}` | `{event.get('calibration_mp4_count')}` | "
                f"`{event.get('ingest_status', '')}` | `{event.get('next_action', '')}` |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This monitor is opt-in and bounded by `iterations`; it does not install hooks or background services.",
            "- The monitor refreshes ingest in dry-run mode only; it never passes `--execute-copy`.",
            "- Use the generated ingest summary's `gate_decision.copy_execution_command` manually after reviewing assignments.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["V4RecordingIngestMonitorConfig", "monitor_v4_recording_ingest"]
