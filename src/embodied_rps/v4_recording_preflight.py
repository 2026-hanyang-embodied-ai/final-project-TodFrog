"""Preflight checks before executing a v4 recording session."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER
from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_session import V4RecordingSessionConfig, run_v4_recording_session

OpenCvProbe = Callable[[int, bool], Mapping[str, object]]


@dataclass(frozen=True)
class V4RecordingPreflightConfig:
    """Configuration for recording-session preflight checks."""

    staging_root: Path
    calibration_root: Path
    heldout_root: Path
    output_root: Path = Path("artifacts/real_skeleton_v4_recording_preflight_20260612")
    slot_manifest_path: Path | None = DEFAULT_LOCAL_DATA_ROOT / "v4_calibration" / "recording_slot_manifest.json"
    labels: tuple[str, ...] = REVIEW_LABEL_ORDER
    count_per_label: int = 1
    camera_index: int = 0
    check_camera: bool = False
    pre_roll_s: float = 1.5
    duration_s: float = 3.0
    fps: float = 30.0
    expected_per_label: int = 20


def preflight_v4_recording_session(
    config: V4RecordingPreflightConfig,
    *,
    opencv_probe: OpenCvProbe | None = None,
) -> dict[str, object]:
    """Run metadata and optional camera checks before recording."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    root_checks = _root_checks(config)
    failures.extend(root_checks["failures"])
    warnings.extend(root_checks["warnings"])
    manifest_summary = _manifest_summary(config.slot_manifest_path)
    failures.extend(manifest_summary["failures"])

    camera_summary = dict((opencv_probe or _probe_opencv)(int(config.camera_index), bool(config.check_camera)))
    if not bool(camera_summary.get("opencv_available")):
        failures.append({"code": "opencv_unavailable", "detail": camera_summary.get("detail", "")})
    if config.check_camera and not bool(camera_summary.get("camera_opened")):
        failures.append({"code": "camera_not_opened", "camera_index": int(config.camera_index), "detail": camera_summary.get("detail", "")})
    if config.check_camera and bool(camera_summary.get("camera_opened")) and not bool(camera_summary.get("frame_read")):
        failures.append({"code": "camera_frame_not_read", "camera_index": int(config.camera_index), "detail": camera_summary.get("detail", "")})
    if not config.check_camera:
        warnings.append({"code": "camera_not_checked", "detail": "Run again with --check-camera before executing a live recording session."})

    session_summary = _session_plan_summary(config)
    if int(session_summary.get("planned_count", 0)) <= 0:
        failures.append({"code": "empty_session_plan"})
    if int(session_summary.get("slot_prompts_attached", 0)) < int(session_summary.get("planned_count", 0)):
        warnings.append(
            {
                "code": "missing_slot_prompts",
                "attached": int(session_summary.get("slot_prompts_attached", 0)),
                "planned": int(session_summary.get("planned_count", 0)),
            }
        )
    existing_outputs = [
        row["output_path"]
        for row in session_summary.get("plan", [])
        if isinstance(row, dict) and Path(str(row.get("output_path", ""))).exists()
    ]
    if existing_outputs:
        failures.append({"code": "planned_output_exists", "paths": existing_outputs})

    status = "ready_for_recording" if not failures else "preflight_failed"
    if status == "ready_for_recording" and not config.check_camera:
        status = "ready_without_camera_check"
    summary = {
        "status": status,
        "staging_root": config.staging_root.as_posix(),
        "calibration_root": config.calibration_root.as_posix(),
        "heldout_root": config.heldout_root.as_posix(),
        "slot_manifest_path": config.slot_manifest_path.as_posix() if config.slot_manifest_path is not None else None,
        "camera_index": int(config.camera_index),
        "check_camera": bool(config.check_camera),
        "opencv": camera_summary,
        "root_checks": root_checks["checks"],
        "manifest": manifest_summary["summary"],
        "session": session_summary,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "next_command": _next_command(config),
        "preflight_summary": (config.output_root / "recording_preflight_summary.json").as_posix(),
        "recording_cue_sheet": (config.output_root / "recording_cue_sheet.md").as_posix(),
        "recording_cue_sheet_json": (config.output_root / "recording_cue_sheet.json").as_posix(),
    }
    _write_outputs(config.output_root, summary)
    return summary


def _session_plan_summary(config: V4RecordingPreflightConfig) -> dict[str, object]:
    return run_v4_recording_session(
        V4RecordingSessionConfig(
            staging_root=config.staging_root,
            output_root=config.output_root / "session_plan",
            labels=config.labels,
            count_per_label=config.count_per_label,
            camera_index=config.camera_index,
            pre_roll_s=config.pre_roll_s,
            duration_s=config.duration_s,
            fps=config.fps,
            execute=False,
            calibration_root=config.calibration_root,
            heldout_root=config.heldout_root,
            expected_per_label=config.expected_per_label,
            slot_manifest_path=config.slot_manifest_path,
        )
    )


def _root_checks(config: V4RecordingPreflightConfig) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    for name, path in (
        ("staging_root", config.staging_root),
        ("calibration_root", config.calibration_root),
        ("heldout_root", config.heldout_root),
    ):
        exists = path.exists()
        checks.append({"name": name, "path": path.as_posix(), "exists": exists})
        if not exists and name != "staging_root":
            failures.append({"code": "missing_root", "name": name, "path": path.as_posix()})
        if not exists and name == "staging_root":
            warnings.append({"code": "missing_staging_root", "path": path.as_posix()})
    if _overlaps(config.staging_root, config.calibration_root):
        failures.append({"code": "staging_overlaps_calibration_root"})
    if _overlaps(config.staging_root, config.heldout_root):
        failures.append({"code": "staging_overlaps_heldout_root"})
    return {"checks": checks, "failures": failures, "warnings": warnings}


def _manifest_summary(path: Path | None) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    if path is None:
        return {"summary": {"exists": False, "slot_count": 0, "label_counts": {}}, "failures": [{"code": "missing_slot_manifest_path"}]}
    if not path.exists():
        return {"summary": {"path": path.as_posix(), "exists": False, "slot_count": 0, "label_counts": {}}, "failures": [{"code": "missing_slot_manifest", "path": path.as_posix()}]}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        return {"summary": {"path": path.as_posix(), "exists": True, "slot_count": 0, "label_counts": {}}, "failures": [{"code": "invalid_slot_manifest_payload", "path": path.as_posix()}]}
    label_counts = {
        label: sum(1 for row in loaded if isinstance(row, dict) and row.get("label") == label)
        for label in REVIEW_LABEL_ORDER
    }
    if any(label_counts[label] == 0 for label in REVIEW_LABEL_ORDER):
        failures.append({"code": "slot_manifest_missing_label", "label_counts": label_counts})
    return {
        "summary": {"path": path.as_posix(), "exists": True, "slot_count": len(loaded), "label_counts": label_counts},
        "failures": failures,
    }


def _probe_opencv(camera_index: int, check_camera: bool) -> dict[str, object]:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"opencv_available": False, "detail": f"{type(exc).__name__}: {exc}"}
    result: dict[str, object] = {
        "opencv_available": True,
        "opencv_version": getattr(cv2, "__version__", ""),
        "camera_checked": bool(check_camera),
        "camera_opened": None,
        "frame_read": None,
    }
    if not check_camera:
        return result
    capture = cv2.VideoCapture(int(camera_index))
    try:
        opened = bool(capture.isOpened())
        frame_read = False
        width = 0
        height = 0
        if opened:
            ok, frame = capture.read()
            frame_read = bool(ok and frame is not None)
            if frame_read:
                height, width = frame.shape[:2]
        result.update({"camera_opened": opened, "frame_read": frame_read, "frame_width": int(width), "frame_height": int(height)})
    finally:
        capture.release()
    return result


def _next_command(config: V4RecordingPreflightConfig) -> str:
    parts = [
        "python",
        "-m",
        "embodied_rps.tools.record_v4_staging_session",
        "--staging-root",
        config.staging_root.as_posix(),
        "--output-root",
        "artifacts/real_skeleton_v4_recording_session_20260612",
        "--count-per-label",
        str(config.count_per_label),
        "--pre-roll-s",
        str(config.pre_roll_s),
        "--duration-s",
        str(config.duration_s),
        "--fps",
        str(config.fps),
        "--slot-manifest",
        config.slot_manifest_path.as_posix() if config.slot_manifest_path is not None else "",
        "--execute",
    ]
    return " ".join(_quote(part) for part in parts if part)


def _write_outputs(output_root: Path, summary: Mapping[str, object]) -> None:
    (output_root / "recording_preflight_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "recording_preflight_summary.md").write_text(_markdown(summary), encoding="utf-8")
    cue_payload = _cue_sheet_payload(summary)
    (output_root / "recording_cue_sheet.json").write_text(
        json.dumps(cue_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "recording_cue_sheet.md").write_text(_cue_sheet_markdown(summary, cue_payload), encoding="utf-8")


def _markdown(summary: Mapping[str, object]) -> str:
    session = summary.get("session") if isinstance(summary.get("session"), Mapping) else {}
    lines = [
        "# V4 Recording Preflight",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Check camera: `{summary.get('check_camera')}`",
        f"- Failure count: `{summary.get('failure_count')}`",
        f"- Warning count: `{summary.get('warning_count')}`",
        f"- Planned clips: `{session.get('planned_count')}`",
        f"- Slot prompts attached: `{session.get('slot_prompts_attached')}`",
        "",
        "## Failures",
        "",
    ]
    failures = summary.get("failures")
    if isinstance(failures, list) and failures:
        for failure in failures:
            lines.append(f"- `{json.dumps(failure, ensure_ascii=False)}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    warnings = summary.get("warnings")
    if isinstance(warnings, list) and warnings:
        for warning in warnings:
            lines.append(f"- `{json.dumps(warning, ensure_ascii=False)}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Next Command", "", "```powershell", str(summary.get("next_command", "")), "```", ""])
    return "\n".join(lines)


def _cue_sheet_payload(summary: Mapping[str, object]) -> dict[str, object]:
    session = summary.get("session") if isinstance(summary.get("session"), Mapping) else {}
    rows: list[dict[str, object]] = []
    plan = session.get("plan") if isinstance(session, Mapping) else None
    if isinstance(plan, list):
        for index, row in enumerate(plan, start=1):
            if not isinstance(row, Mapping):
                continue
            rows.append(
                {
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
                    "command": row.get("command"),
                }
            )
    return {
        "status": summary.get("status"),
        "camera_checked": summary.get("check_camera"),
        "camera_index": summary.get("camera_index"),
        "preflight_summary": summary.get("preflight_summary"),
        "next_command": summary.get("next_command"),
        "planned_count": session.get("planned_count") if isinstance(session, Mapping) else None,
        "pre_roll_s": session.get("pre_roll_s") if isinstance(session, Mapping) else None,
        "duration_s": session.get("duration_s") if isinstance(session, Mapping) else None,
        "fps": session.get("fps") if isinstance(session, Mapping) else None,
        "cue_count": len(rows),
        "cues": rows,
    }


def _cue_sheet_markdown(summary: Mapping[str, object], cue_payload: Mapping[str, object]) -> str:
    lines = [
        "# V4 Recording Cue Sheet",
        "",
        f"- Preflight status: `{summary.get('status')}`",
        f"- Camera checked: `{summary.get('check_camera')}`",
        f"- Camera index: `{summary.get('camera_index')}`",
        f"- Planned clips: `{cue_payload.get('cue_count')}`",
        f"- Pre-roll seconds: `{cue_payload.get('pre_roll_s')}`",
        f"- Clip duration seconds: `{cue_payload.get('duration_s')}`",
        "",
        "## Operator Steps",
        "",
        "1. Confirm the next cue before starting the recording command.",
        "2. During pre-roll, prepare the hand pose and framing.",
        "3. Perform only the prompted motion for the clip.",
        "4. Keep the final gesture clear until capture ends.",
        "5. After recording, run ingest in dry-run mode before copying to calibration slots.",
        "",
        "## Live Recording Command",
        "",
        "```powershell",
        str(summary.get("next_command", "")),
        "```",
        "",
        "## Cues",
        "",
        "| Order | Label | Slot | Filename | Focus | View | Speed | Stability | Review Target |",
        "|---:|---|---|---|---|---|---|---|---|",
    ]
    cues = cue_payload.get("cues")
    if isinstance(cues, list):
        for row in cues:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("order", "")),
                        f"`{row.get('label', '')}`",
                        f"`{row.get('slot_id', '')}`",
                        f"`{row.get('filename', '')}`",
                        str(row.get("motion_focus", "")),
                        str(row.get("viewpoint", "")),
                        str(row.get("speed_focus", "")),
                        str(row.get("stability_focus", "")),
                        str(row.get("review_focus", "")),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Full Prompts", ""])
    if isinstance(cues, list):
        for row in cues:
            if not isinstance(row, Mapping):
                continue
            lines.extend(
                [
                    f"### {row.get('order')}. {row.get('filename')}",
                    "",
                    f"- Prompt: {row.get('recording_prompt')}",
                    f"- Output path: `{row.get('output_path')}`",
                    f"- Calibration target: `{row.get('target_path')}`",
                    "",
                ]
            )
    return "\n".join(lines)


def _overlaps(first: Path, second: Path) -> bool:
    first_resolved = first.expanduser().resolve(strict=False)
    second_resolved = second.expanduser().resolve(strict=False)
    return _is_within(first_resolved, second_resolved) or _is_within(second_resolved, first_resolved)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _quote(part: str) -> str:
    if not part or any(char.isspace() for char in part) or any(ord(char) > 127 for char in part) or "\\" in part or ":" in part:
        return "'" + part.replace("'", "''") + "'"
    return part


__all__ = ["OpenCvProbe", "V4RecordingPreflightConfig", "preflight_v4_recording_session"]
