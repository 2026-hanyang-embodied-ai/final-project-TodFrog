"""Batch session planner/runner for v4 non-held-out staging recordings."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER, natural_key
from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_ingest_monitor import V4RecordingIngestMonitorConfig, monitor_v4_recording_ingest
from embodied_rps.v4_staging_recorder import (
    DEFAULT_V4_RECORDING_CAPTURE_ROOT,
    RecorderCallable,
    V4StagingRecorderConfig,
    record_v4_staging_clip,
    validate_staging_recording_label,
)

SleepFn = Callable[[float], None]


@dataclass(frozen=True)
class V4RecordingSessionConfig:
    """Configuration for a bounded v4 staging recording session."""

    staging_root: Path
    output_root: Path = Path("artifacts/real_skeleton_v4_recording_session_20260612")
    labels: tuple[str, ...] = REVIEW_LABEL_ORDER
    count_per_label: int = 1
    camera_index: int = 0
    pre_roll_s: float = 0.0
    duration_s: float = 3.0
    fps: float = 30.0
    width: int | None = None
    height: int | None = None
    prefix: str = "v4"
    codec: str = "mp4v"
    execute: bool = False
    inter_clip_pause_s: float = 0.0
    refresh_ingest: bool = False
    calibration_root: Path = DEFAULT_LOCAL_DATA_ROOT / "v4_calibration"
    heldout_root: Path = DEFAULT_LOCAL_DATA_ROOT / "test"
    ingest_output_root: Path = Path("artifacts/real_skeleton_v4_recording_ingest_20260612")
    end_to_end_summary_path: Path = Path("artifacts/real_skeleton_v4_end_to_end_20260612/end_to_end_summary.json")
    expected_per_label: int = 20
    slot_manifest_path: Path | None = DEFAULT_LOCAL_DATA_ROOT / "v4_calibration" / "recording_slot_manifest.json"


def plan_v4_recording_session(config: V4RecordingSessionConfig) -> list[dict[str, object]]:
    """Return deterministic recording plan rows for the session."""

    labels = _validated_labels(config.labels)
    if config.count_per_label <= 0:
        raise ValueError("count_per_label must be positive")
    if config.pre_roll_s < 0 or not math.isfinite(config.pre_roll_s):
        raise ValueError("pre_roll_s must be a finite non-negative value")
    if config.inter_clip_pause_s < 0:
        raise ValueError("inter_clip_pause_s must be non-negative")
    slot_prompts = _load_slot_prompts(config.slot_manifest_path)
    staged_counts = _staged_counts(config.staging_root)
    rows: list[dict[str, object]] = []
    session_index = 0
    for label in labels:
        label_dir = config.staging_root / label
        start_index = _next_index(label_dir, label=label, prefix=config.prefix)
        for offset in range(config.count_per_label):
            filename = f"{config.prefix}_{label}_{start_index + offset:06d}.mp4"
            output_path = label_dir / filename
            rows.append(
                {
                    "session_index": session_index,
                    "label": label,
                    "filename": filename,
                    "output_path": output_path.as_posix(),
                    "status": "planned",
                    "command": _record_clip_command(config=config, label=label, filename=filename),
                    **_slot_prompt_fields(
                        slot_prompts,
                        label=label,
                        prompt_index=int(staged_counts.get(label, 0)) + offset,
                    ),
                }
            )
            session_index += 1
    return rows


def run_v4_recording_session(
    config: V4RecordingSessionConfig,
    *,
    recorder: RecorderCallable | None = None,
    sleep_fn: SleepFn = time.sleep,
) -> dict[str, object]:
    """Plan or execute a bounded v4 recording session."""

    plan_rows = plan_v4_recording_session(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    clip_results: list[dict[str, object]] = []
    if config.execute:
        for index, row in enumerate(plan_rows):
            clip_summary = record_v4_staging_clip(
                V4StagingRecorderConfig(
                    staging_root=config.staging_root,
                    label=str(row["label"]),
                    output_root=DEFAULT_V4_RECORDING_CAPTURE_ROOT,
                    camera_index=config.camera_index,
                    pre_roll_s=config.pre_roll_s,
                    duration_s=config.duration_s,
                    fps=config.fps,
                    width=config.width,
                    height=config.height,
                    filename=str(row["filename"]),
                    prefix=config.prefix,
                    codec=config.codec,
                    dry_run=False,
                ),
                recorder=recorder,
            )
            row["status"] = "recorded"
            row["recording_summary"] = clip_summary.get("metadata_path", clip_summary.get("output_path"))
            clip_results.append(clip_summary)
            if index < len(plan_rows) - 1 and config.inter_clip_pause_s > 0:
                sleep_fn(float(config.inter_clip_pause_s))
    ingest_summary: dict[str, object] | None = None
    if config.execute and config.refresh_ingest:
        ingest_summary = monitor_v4_recording_ingest(
            V4RecordingIngestMonitorConfig(
                source_root=config.staging_root,
                calibration_root=config.calibration_root,
                heldout_root=config.heldout_root,
                output_root=config.output_root / "ingest_monitor",
                ingest_output_root=config.ingest_output_root,
                end_to_end_summary_path=config.end_to_end_summary_path,
                expected_per_label=config.expected_per_label,
                iterations=1,
                poll_interval_s=0.0,
            )
        )
    status = "recorded" if config.execute else "planned"
    summary = {
        "status": status,
        "execute": bool(config.execute),
        "staging_root": config.staging_root.as_posix(),
        "output_root": config.output_root.as_posix(),
        "labels": list(_validated_labels(config.labels)),
        "count_per_label": int(config.count_per_label),
        "planned_count": len(plan_rows),
        "recorded_count": len(clip_results),
        "duration_s": float(config.duration_s),
        "pre_roll_s": float(config.pre_roll_s),
        "fps": float(config.fps),
        "refresh_ingest": bool(config.refresh_ingest),
        "ingest_status": ingest_summary.get("status") if isinstance(ingest_summary, dict) else None,
        "ingest_next_action": ingest_summary.get("next_action") if isinstance(ingest_summary, dict) else None,
        "slot_manifest_path": config.slot_manifest_path.as_posix() if config.slot_manifest_path is not None else None,
        "slot_prompts_attached": sum(1 for row in plan_rows if row.get("slot_id")),
        "plan": plan_rows,
        "session_summary": (config.output_root / "recording_session_summary.json").as_posix(),
    }
    _write_outputs(config.output_root, summary)
    return summary


def _validated_labels(labels: Sequence[str]) -> tuple[str, ...]:
    if not labels:
        raise ValueError("at least one label is required")
    return tuple(validate_staging_recording_label(label) for label in labels)


def _next_index(label_dir: Path, *, label: str, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}_{re.escape(label)}_(\d{{6}})\.mp4$", re.IGNORECASE)
    max_index = 0
    if label_dir.exists():
        for path in label_dir.glob("*.mp4"):
            match = pattern.match(path.name)
            if match is not None:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _load_slot_prompts(path: Path | None) -> dict[str, list[dict[str, object]]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"slot manifest must contain a JSON list: {path}")
    prompts: dict[str, list[dict[str, object]]] = {label: [] for label in REVIEW_LABEL_ORDER}
    for item in data:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if isinstance(label, str) and label in prompts:
            item_dict = dict(item)
            target_path = Path(str(item_dict.get("target_path", "")))
            if not target_path.exists():
                prompts[label].append(item_dict)
    return prompts


def _staged_counts(staging_root: Path) -> dict[str, int]:
    counts = {label: 0 for label in REVIEW_LABEL_ORDER}
    if not staging_root.exists():
        return counts
    for path in sorted(staging_root.rglob("*.mp4"), key=natural_key):
        if not path.is_file():
            continue
        label = _infer_staging_label(path, staging_root)
        if label is not None:
            counts[label] += 1
    return counts


def _infer_staging_label(path: Path, staging_root: Path) -> str | None:
    try:
        parts = path.relative_to(staging_root).parts
    except ValueError:
        parts = path.parts
    for part in parts[:-1]:
        if part in REVIEW_LABEL_ORDER:
            return part
    stem = path.stem.lower()
    for label in REVIEW_LABEL_ORDER:
        if stem.startswith(label):
            return label
    return None


def _slot_prompt_fields(
    slot_prompts: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    label: str,
    prompt_index: int,
) -> dict[str, object]:
    slots = slot_prompts.get(label, ())
    slot = slots[prompt_index] if 0 <= prompt_index < len(slots) else None
    if slot is None:
        return {
            "slot_id": None,
            "target_filename": None,
            "motion_focus": None,
            "recording_prompt": _fallback_prompt(label),
        }
    prompt = (
        f"{label}: {slot.get('motion_focus')}; view={slot.get('viewpoint')}; "
        f"background={slot.get('background')}; distance={slot.get('distance')}; "
        f"handedness={slot.get('handedness_target')}; speed={slot.get('speed_focus')}; "
        f"stability={slot.get('stability_focus')}; {slot.get('review_focus')}"
    )
    return {
        "slot_id": slot.get("slot_id"),
        "target_filename": slot.get("filename"),
        "target_path": slot.get("target_path"),
        "viewpoint": slot.get("viewpoint"),
        "background": slot.get("background"),
        "distance": slot.get("distance"),
        "handedness_target": slot.get("handedness_target"),
        "speed_focus": slot.get("speed_focus"),
        "stability_focus": slot.get("stability_focus"),
        "motion_focus": slot.get("motion_focus"),
        "review_focus": slot.get("review_focus"),
        "recording_prompt": prompt,
    }


def _fallback_prompt(label: str) -> str:
    if label == "rock":
        return "rock: hold a stable fist; avoid paper/scissors transition."
    if label == "paper":
        return "paper: start fist-like, then open clearly to paper."
    return "scissors: start fist-like, then finish with clear scissors."


def _record_clip_command(*, config: V4RecordingSessionConfig, label: str, filename: str) -> str:
    parts = [
        "python",
        "-m",
        "embodied_rps.tools.record_v4_staging_clip",
        "--label",
        label,
        "--staging-root",
        config.staging_root.as_posix(),
        "--output-root",
        DEFAULT_V4_RECORDING_CAPTURE_ROOT.as_posix(),
        "--pre-roll-s",
        str(config.pre_roll_s),
        "--duration-s",
        str(config.duration_s),
        "--fps",
        str(config.fps),
        "--filename",
        filename,
    ]
    return " ".join(_quote(part) for part in parts)


def _write_outputs(output_root: Path, summary: Mapping[str, object]) -> None:
    (output_root / "recording_session_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "recording_session_summary.md").write_text(_markdown(summary), encoding="utf-8")


def _markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V4 Recording Session",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Execute: `{summary.get('execute')}`",
        f"- Planned clips: `{summary.get('planned_count')}`",
        f"- Recorded clips: `{summary.get('recorded_count')}`",
        f"- Staging root: `{summary.get('staging_root')}`",
        f"- Ingest status: `{summary.get('ingest_status')}`",
        f"- Slot prompts attached: `{summary.get('slot_prompts_attached')}`",
        "",
        "## Clip Plan",
        "",
        "| Index | Label | Filename | Slot | Motion focus | Prompt | Status |",
        "|---:|---|---|---|---|---|---|",
    ]
    plan = summary.get("plan")
    if isinstance(plan, list):
        for row in plan:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"| `{row.get('session_index')}` | `{row.get('label')}` | `{row.get('filename')}` | "
                f"`{row.get('slot_id')}` | `{row.get('motion_focus')}` | {row.get('recording_prompt')} | `{row.get('status')}` |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The default mode only plans filenames and commands; actual camera recording requires `--execute`.",
            "- These clips are intended for non-held-out v4 calibration only.",
            "- The held-out `test` folder must remain validation-only.",
            "",
        ]
    )
    return "\n".join(lines)


def _quote(part: str) -> str:
    if not part or any(char.isspace() for char in part) or any(ord(char) > 127 for char in part) or "\\" in part or ":" in part:
        return "'" + part.replace("'", "''") + "'"
    return part


__all__ = [
    "V4RecordingSessionConfig",
    "plan_v4_recording_session",
    "run_v4_recording_session",
]
