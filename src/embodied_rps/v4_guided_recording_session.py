"""Guided v4 staging recording session."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER
from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_session import V4RecordingSessionConfig, plan_v4_recording_session
from embodied_rps.v4_staging_recorder import (
    DEFAULT_V4_RECORDING_CAPTURE_ROOT,
    RecorderCallable,
    V4StagingRecorderConfig,
    record_v4_staging_clip,
)

InputFn = Callable[[str], str]
SleepFn = Callable[[float], None]


@dataclass(frozen=True)
class V4GuidedRecordingSessionConfig:
    """Configuration for prompt-by-prompt v4 staging recording."""

    staging_root: Path = DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging"
    output_root: Path = Path("artifacts/real_skeleton_v4_guided_recording_session_20260612")
    labels: tuple[str, ...] = REVIEW_LABEL_ORDER
    count_per_label: int = 1
    camera_index: int = 0
    pre_roll_s: float = 1.5
    duration_s: float = 3.0
    fps: float = 30.0
    width: int | None = None
    height: int | None = None
    prefix: str = "v4"
    codec: str = "mp4v"
    execute: bool = False
    assume_yes: bool = False
    inter_clip_pause_s: float = 0.0
    slot_manifest_path: Path | None = DEFAULT_LOCAL_DATA_ROOT / "v4_calibration" / "recording_slot_manifest.json"


def run_v4_guided_recording_session(
    config: V4GuidedRecordingSessionConfig,
    *,
    recorder: RecorderCallable | None = None,
    input_fn: InputFn = input,
    sleep_fn: SleepFn = time.sleep,
) -> dict[str, object]:
    """Plan or execute a guided recording session one cue at a time."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    plan_rows = plan_v4_recording_session(
        V4RecordingSessionConfig(
            staging_root=config.staging_root,
            labels=config.labels,
            count_per_label=config.count_per_label,
            camera_index=config.camera_index,
            pre_roll_s=config.pre_roll_s,
            duration_s=config.duration_s,
            fps=config.fps,
            width=config.width,
            height=config.height,
            prefix=config.prefix,
            codec=config.codec,
            execute=False,
            inter_clip_pause_s=config.inter_clip_pause_s,
            slot_manifest_path=config.slot_manifest_path,
        )
    )
    prompts: list[dict[str, object]] = [_cue_row(index, row) for index, row in enumerate(plan_rows, start=1)]
    recorded_results: list[dict[str, object]] = []
    aborted_at: int | None = None
    if config.execute:
        for index, row in enumerate(plan_rows, start=1):
            if not config.assume_yes:
                answer = input_fn(_confirmation_prompt(index, len(plan_rows), row))
                if str(answer).strip().lower() in {"q", "quit", "abort", "stop", "n", "no"}:
                    row["status"] = "skipped_operator_abort"
                    aborted_at = index
                    break
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
            recorded_results.append(clip_summary)
            if index < len(plan_rows) and config.inter_clip_pause_s > 0:
                sleep_fn(float(config.inter_clip_pause_s))
    status = _status(execute=config.execute, planned_count=len(plan_rows), recorded_count=len(recorded_results), aborted_at=aborted_at)
    summary = {
        "status": status,
        "execute": bool(config.execute),
        "assume_yes": bool(config.assume_yes),
        "staging_root": config.staging_root.as_posix(),
        "output_root": config.output_root.as_posix(),
        "labels": list(config.labels),
        "count_per_label": int(config.count_per_label),
        "planned_count": len(plan_rows),
        "recorded_count": len(recorded_results),
        "aborted_at": aborted_at,
        "duration_s": float(config.duration_s),
        "pre_roll_s": float(config.pre_roll_s),
        "fps": float(config.fps),
        "slot_manifest_path": config.slot_manifest_path.as_posix() if config.slot_manifest_path is not None else None,
        "slot_prompts_attached": sum(1 for row in plan_rows if row.get("slot_id")),
        "cues": prompts,
        "plan": plan_rows,
        "guide_cue_sheet": (config.output_root / "guided_recording_cue_sheet.md").as_posix(),
        "guided_recording_summary": (config.output_root / "guided_recording_summary.json").as_posix(),
    }
    _write_outputs(config.output_root, summary)
    return summary


def _status(*, execute: bool, planned_count: int, recorded_count: int, aborted_at: int | None) -> str:
    if not execute:
        return "planned"
    if aborted_at is not None and recorded_count == 0:
        return "aborted_before_recording"
    if aborted_at is not None:
        return "partial_recorded_operator_abort"
    if recorded_count == planned_count:
        return "recorded"
    return "partial_recorded"


def _cue_row(index: int, row: Mapping[str, object]) -> dict[str, object]:
    return {
        "order": index,
        "label": row.get("label"),
        "filename": row.get("filename"),
        "slot_id": row.get("slot_id"),
        "motion_focus": row.get("motion_focus"),
        "viewpoint": row.get("viewpoint"),
        "background": row.get("background"),
        "distance": row.get("distance"),
        "handedness_target": row.get("handedness_target"),
        "speed_focus": row.get("speed_focus"),
        "stability_focus": row.get("stability_focus"),
        "review_focus": row.get("review_focus"),
        "recording_prompt": row.get("recording_prompt"),
        "output_path": row.get("output_path"),
        "target_path": row.get("target_path"),
    }


def _confirmation_prompt(index: int, total: int, row: Mapping[str, object]) -> str:
    return (
        f"\n[{index}/{total}] {row.get('filename')} | {row.get('recording_prompt')}\n"
        f"Output: {row.get('output_path')}\n"
        "Press Enter to record this clip, or type q to stop: "
    )


def _write_outputs(output_root: Path, summary: Mapping[str, object]) -> None:
    (output_root / "guided_recording_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "guided_recording_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    (output_root / "guided_recording_cue_sheet.json").write_text(
        json.dumps(summary.get("cues", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "guided_recording_cue_sheet.md").write_text(_cue_markdown(summary), encoding="utf-8")


def _summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V4 Guided Recording Session",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Execute: `{summary.get('execute')}`",
        f"- Planned clips: `{summary.get('planned_count')}`",
        f"- Recorded clips: `{summary.get('recorded_count')}`",
        f"- Aborted at: `{summary.get('aborted_at')}`",
        f"- Cue sheet: `{summary.get('guide_cue_sheet')}`",
        "",
    ]
    lines.extend(_cue_table_lines(summary))
    return "\n".join(lines)


def _cue_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V4 Guided Recording Cue Sheet",
        "",
        "Run the guided command with `--execute` when ready to record. Each clip waits for Enter before capture.",
        "",
        f"- Pre-roll seconds: `{summary.get('pre_roll_s')}`",
        f"- Clip duration seconds: `{summary.get('duration_s')}`",
        f"- FPS: `{summary.get('fps')}`",
        "",
    ]
    lines.extend(_cue_table_lines(summary))
    return "\n".join(lines)


def _cue_table_lines(summary: Mapping[str, object]) -> list[str]:
    lines = [
        "## Cues",
        "",
        "| Order | Label | Slot | Filename | Focus | View | Speed | Stability |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    cues = summary.get("cues")
    if isinstance(cues, list):
        for cue in cues:
            if not isinstance(cue, Mapping):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(cue.get("order", "")),
                        f"`{cue.get('label', '')}`",
                        f"`{cue.get('slot_id', '')}`",
                        f"`{cue.get('filename', '')}`",
                        str(cue.get("motion_focus", "")),
                        str(cue.get("viewpoint", "")),
                        str(cue.get("speed_focus", "")),
                        str(cue.get("stability_focus", "")),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Full Prompts", ""])
    if isinstance(cues, list):
        for cue in cues:
            if not isinstance(cue, Mapping):
                continue
            lines.extend(
                [
                    f"### {cue.get('order')}. {cue.get('filename')}",
                    "",
                    f"- Prompt: {cue.get('recording_prompt')}",
                    f"- Output path: `{cue.get('output_path')}`",
                    f"- Calibration target: `{cue.get('target_path')}`",
                    "",
                ]
            )
    return lines


__all__ = ["InputFn", "V4GuidedRecordingSessionConfig", "run_v4_guided_recording_session"]
