"""One-command safe flow for v4 guided recording."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER
from embodied_rps.v4_guided_recording_session import (
    InputFn,
    SleepFn,
    V4GuidedRecordingSessionConfig,
    run_v4_guided_recording_session,
)
from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_postcheck import V4RecordingPostcheckConfig, run_v4_recording_postcheck
from embodied_rps.v4_recording_preflight import OpenCvProbe, V4RecordingPreflightConfig, preflight_v4_recording_session
from embodied_rps.v4_recording_staging_audit import VideoProbe as StagingVideoProbe
from embodied_rps.v4_staging_recorder import RecorderCallable


@dataclass(frozen=True)
class V4GuidedRecordingFlowConfig:
    """Configuration for the safe guided recording flow."""

    staging_root: Path = DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging"
    calibration_root: Path = DEFAULT_LOCAL_DATA_ROOT / "v4_calibration"
    heldout_root: Path = DEFAULT_LOCAL_DATA_ROOT / "test"
    output_root: Path = Path("artifacts/real_skeleton_v4_guided_recording_flow_20260612")
    labels: tuple[str, ...] = REVIEW_LABEL_ORDER
    count_per_label: int = 1
    camera_index: int = 0
    check_camera: bool = False
    pre_roll_s: float = 1.5
    duration_s: float = 3.0
    fps: float = 30.0
    width: int | None = None
    height: int | None = None
    execute: bool = False
    assume_yes: bool = False
    inter_clip_pause_s: float = 0.0
    slot_manifest_path: Path | None = DEFAULT_LOCAL_DATA_ROOT / "v4_calibration" / "recording_slot_manifest.json"
    end_to_end_summary_path: Path = Path("artifacts/real_skeleton_v4_end_to_end_20260612/end_to_end_summary.json")
    expected_per_label: int = 20


def run_v4_guided_recording_flow(
    config: V4GuidedRecordingFlowConfig,
    *,
    opencv_probe: OpenCvProbe | None = None,
    recorder: RecorderCallable | None = None,
    input_fn: InputFn = input,
    sleep_fn: SleepFn | None = None,
    staging_video_probe: StagingVideoProbe | None = None,
) -> dict[str, object]:
    """Run preflight, guided recording, and postcheck in order."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    preflight = preflight_v4_recording_session(
        V4RecordingPreflightConfig(
            staging_root=config.staging_root,
            calibration_root=config.calibration_root,
            heldout_root=config.heldout_root,
            output_root=config.output_root / "preflight",
            slot_manifest_path=config.slot_manifest_path,
            labels=config.labels,
            count_per_label=config.count_per_label,
            camera_index=config.camera_index,
            check_camera=config.check_camera,
            pre_roll_s=config.pre_roll_s,
            duration_s=config.duration_s,
            fps=config.fps,
            expected_per_label=config.expected_per_label,
        ),
        opencv_probe=opencv_probe,
    )
    guided: dict[str, object] | None = None
    postcheck: dict[str, object] | None = None
    if _can_continue_after_preflight(preflight, execute=config.execute):
        guided_kwargs: dict[str, object] = {"recorder": recorder, "input_fn": input_fn}
        if sleep_fn is not None:
            guided_kwargs["sleep_fn"] = sleep_fn
        guided = run_v4_guided_recording_session(
            V4GuidedRecordingSessionConfig(
                staging_root=config.staging_root,
                output_root=config.output_root / "guided_session",
                labels=config.labels,
                count_per_label=config.count_per_label,
                camera_index=config.camera_index,
                pre_roll_s=config.pre_roll_s,
                duration_s=config.duration_s,
                fps=config.fps,
                width=config.width,
                height=config.height,
                execute=config.execute,
                assume_yes=config.assume_yes,
                inter_clip_pause_s=config.inter_clip_pause_s,
                slot_manifest_path=config.slot_manifest_path,
            ),
            **guided_kwargs,
        )
        postcheck = run_v4_recording_postcheck(
            V4RecordingPostcheckConfig(
                staging_root=config.staging_root,
                calibration_root=config.calibration_root,
                heldout_root=config.heldout_root,
                output_root=config.output_root / "postcheck",
                slot_manifest_path=config.slot_manifest_path,
                end_to_end_summary_path=config.end_to_end_summary_path,
                expected_per_label=config.expected_per_label,
                expected_new_per_label=config.count_per_label,
                pre_roll_s=config.pre_roll_s,
                duration_s=config.duration_s,
                fps=config.fps,
            ),
            staging_video_probe=staging_video_probe,
        )
    status = _flow_status(preflight=preflight, guided=guided, postcheck=postcheck, execute=config.execute)
    summary = {
        "status": status,
        "execute": bool(config.execute),
        "check_camera": bool(config.check_camera),
        "staging_root": config.staging_root.as_posix(),
        "calibration_root": config.calibration_root.as_posix(),
        "heldout_root": config.heldout_root.as_posix(),
        "preflight_status": preflight.get("status"),
        "guided_status": guided.get("status") if isinstance(guided, Mapping) else None,
        "postcheck_status": postcheck.get("status") if isinstance(postcheck, Mapping) else None,
        "planned_count": guided.get("planned_count") if isinstance(guided, Mapping) else (preflight.get("session") or {}).get("planned_count")
        if isinstance(preflight.get("session"), Mapping)
        else None,
        "recorded_count": guided.get("recorded_count") if isinstance(guided, Mapping) else 0,
        "slot_prompts_attached": guided.get("slot_prompts_attached") if isinstance(guided, Mapping) else (preflight.get("session") or {}).get("slot_prompts_attached")
        if isinstance(preflight.get("session"), Mapping)
        else None,
        "copy_execution_allowed": postcheck.get("copy_execution_allowed") if isinstance(postcheck, Mapping) else False,
        "next_action": _next_action(status, execute=config.execute),
        "preflight_summary": (config.output_root / "preflight" / "recording_preflight_summary.json").as_posix(),
        "guided_summary": (config.output_root / "guided_session" / "guided_recording_summary.json").as_posix() if guided is not None else None,
        "postcheck_summary": (config.output_root / "postcheck" / "recording_postcheck_summary.json").as_posix() if postcheck is not None else None,
        "flow_summary": (config.output_root / "guided_recording_flow_summary.json").as_posix(),
    }
    _write_outputs(config.output_root, summary)
    return summary


def _can_continue_after_preflight(preflight: Mapping[str, object], *, execute: bool) -> bool:
    status = str(preflight.get("status"))
    if execute:
        return status == "ready_for_recording"
    return status in {"ready_for_recording", "ready_without_camera_check"}


def _flow_status(
    *,
    preflight: Mapping[str, object],
    guided: Mapping[str, object] | None,
    postcheck: Mapping[str, object] | None,
    execute: bool,
) -> str:
    if not _can_continue_after_preflight(preflight, execute=execute):
        return "blocked_at_preflight"
    if guided is None:
        return "blocked_before_guided_session"
    guided_status = str(guided.get("status"))
    if guided_status in {"aborted_before_recording", "partial_recorded_operator_abort", "partial_recorded"}:
        return guided_status
    if not execute:
        return "planned"
    if postcheck is None:
        return "recorded_without_postcheck"
    postcheck_status = str(postcheck.get("status"))
    if postcheck_status == "ready_to_review_assignment":
        return "ready_to_review_assignment"
    if postcheck_status == "awaiting_recorded_clips":
        return "recorded_but_missing_first_batch"
    if postcheck_status == "postcheck_blocked":
        return "postcheck_blocked"
    return "postcheck_needs_review"


def _next_action(status: str, *, execute: bool) -> str:
    if status == "planned":
        return "rerun_with_execute_when_operator_ready"
    if status == "blocked_at_preflight":
        return "fix_preflight_failure"
    if status == "ready_to_review_assignment":
        return "review_assignment_table_then_execute_copy"
    if status == "recorded_but_missing_first_batch":
        return "record_missing_staging_clips"
    if status in {"aborted_before_recording", "partial_recorded_operator_abort", "partial_recorded"}:
        return "rerun_guided_flow_for_remaining_clips"
    if status == "postcheck_blocked":
        return "fix_staging_or_video_probe_failure"
    if not execute:
        return "rerun_with_execute_when_operator_ready"
    return "inspect_guided_recording_flow_outputs"


def _write_outputs(output_root: Path, summary: Mapping[str, object]) -> None:
    (output_root / "guided_recording_flow_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "guided_recording_flow_summary.md").write_text(_markdown(summary), encoding="utf-8")


def _markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V4 Guided Recording Flow",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Execute: `{summary.get('execute')}`",
        f"- Check camera: `{summary.get('check_camera')}`",
        f"- Next action: `{summary.get('next_action')}`",
        f"- Preflight status: `{summary.get('preflight_status')}`",
        f"- Guided status: `{summary.get('guided_status')}`",
        f"- Postcheck status: `{summary.get('postcheck_status')}`",
        f"- Planned clips: `{summary.get('planned_count')}`",
        f"- Recorded clips: `{summary.get('recorded_count')}`",
        f"- Copy execution allowed: `{summary.get('copy_execution_allowed')}`",
        "",
        "## Linked Outputs",
        "",
        f"- Preflight summary: `{summary.get('preflight_summary')}`",
        f"- Guided summary: `{summary.get('guided_summary')}`",
        f"- Postcheck summary: `{summary.get('postcheck_summary')}`",
        f"- Flow summary: `{summary.get('flow_summary')}`",
        "",
    ]
    return "\n".join(lines)


__all__ = ["V4GuidedRecordingFlowConfig", "run_v4_guided_recording_flow"]
