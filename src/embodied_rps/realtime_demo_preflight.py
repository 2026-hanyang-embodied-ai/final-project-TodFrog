"""Preflight checks for the prompt-gated realtime RPS skeleton demo."""

from __future__ import annotations

import json
import inspect
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodied_rps.realtime_demo_launcher import load_realtime_demo_config

CameraProbe = Callable[[int], Mapping[str, object]]
HandVisibilityProbe = Callable[..., Mapping[str, object]]


@dataclass(frozen=True)
class RealtimeDemoPreflightConfig:
    """Configuration for local realtime demo readiness checks."""

    project_root: Path = Path(".")
    config_path: Path = Path("configs/realtime_two_stage_selector_demo_conditional_scissors_rescue.yaml")
    python_executable: Path = field(default_factory=lambda: Path(sys.executable))
    output_root: Path = Path("artifacts/realtime_demo_preflight_20260616")
    camera_index: int = 0
    check_camera: bool = False
    check_hand_visibility: bool = False
    hand_visibility_max_frames: int = 60
    hand_visibility_min_detection_rate: float = 0.80
    require_response_prompt: str | None = "scissors"
    require_reset_on_prompt_change: bool = True


def run_realtime_demo_preflight(
    config: RealtimeDemoPreflightConfig,
    *,
    camera_probe: CameraProbe | None = None,
    hand_visibility_probe: HandVisibilityProbe | None = None,
) -> dict[str, object]:
    """Run file, policy, and optional camera checks for the realtime demo."""

    project_root = config.project_root.resolve()
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    config_path = _resolve_path(project_root, config.config_path)
    checks["config_exists"] = config_path.exists()
    demo_config = None
    config_error: str | None = None
    if checks["config_exists"]:
        try:
            demo_config = load_realtime_demo_config(config_path)
            checks["config_loads"] = True
        except Exception as exc:  # pragma: no cover - exact parser exception is not important.
            checks["config_loads"] = False
            config_error = str(exc)
            failures.append("config_load_failed")
    else:
        checks["config_loads"] = False
        failures.append("config_missing")

    python_executable = _resolve_path(project_root, config.python_executable)
    checks["python_executable_exists"] = python_executable.exists()
    if not checks["python_executable_exists"]:
        failures.append("python_executable_missing")

    profile_summaries: list[dict[str, object]] = []
    if demo_config is not None:
        _check_demo_policy(config, demo_config, checks, failures)
        profile_summaries = _check_profiles(project_root, demo_config.profiles, checks, failures)
    else:
        checks["response_prompt_ok"] = False
        checks["reset_on_prompt_change_ok"] = False
        checks["profile_paths_exist"] = False
        checks["model_state_paths_exist"] = False

    camera_summary = _check_camera(
        camera_index=int(config.camera_index),
        check_camera=bool(config.check_camera),
        camera_probe=camera_probe,
        checks=checks,
        failures=failures,
        warnings=warnings,
    )
    hand_visibility_summary = _check_hand_visibility(
        camera_index=int(config.camera_index),
        check_hand_visibility=bool(config.check_hand_visibility),
        max_frames=int(config.hand_visibility_max_frames),
        min_detection_rate=float(config.hand_visibility_min_detection_rate),
        output_root=output_root,
        hand_visibility_probe=hand_visibility_probe,
        checks=checks,
        failures=failures,
        warnings=warnings,
    )

    checks["output_root_writable"] = True
    if failures:
        status = "blocked"
        ok = False
    elif config.check_camera:
        status = "ready_for_live_demo"
        ok = True
    else:
        status = "ready_without_camera_check"
        ok = True

    summary: dict[str, object] = {
        "status": status,
        "ok": ok,
        "project_root": project_root.as_posix(),
        "config_path": config_path.as_posix(),
        "python_executable": python_executable.as_posix(),
        "output_root": output_root.as_posix(),
        "camera_index": int(config.camera_index),
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
        "config_error": config_error,
        "profiles": profile_summaries,
        "camera": camera_summary,
        "hand_visibility": hand_visibility_summary,
        "required_policy": {
            "response_prompt": config.require_response_prompt,
            "reset_on_prompt_change": bool(config.require_reset_on_prompt_change),
            "hand_visibility_min_detection_rate": float(config.hand_visibility_min_detection_rate),
        },
    }
    (output_root / "preflight_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (output_root / "preflight_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _check_demo_policy(
    config: RealtimeDemoPreflightConfig,
    demo_config: Any,
    checks: dict[str, bool],
    failures: list[str],
) -> None:
    if config.require_response_prompt is None:
        response_prompt_ok = demo_config.response_prompt is None
    else:
        response_prompt_ok = demo_config.response_prompt == config.require_response_prompt
    checks["response_prompt_ok"] = bool(response_prompt_ok)
    if not response_prompt_ok:
        failures.append("response_prompt_mismatch")

    reset_ok = (not config.require_reset_on_prompt_change) or bool(demo_config.reset_on_prompt_change)
    checks["reset_on_prompt_change_ok"] = bool(reset_ok)
    if not reset_ok:
        failures.append("reset_on_prompt_change_disabled")


def _check_profiles(
    project_root: Path,
    profiles: tuple[Path, ...],
    checks: dict[str, bool],
    failures: list[str],
) -> list[dict[str, object]]:
    profile_summaries: list[dict[str, object]] = []
    all_profiles_exist = True
    all_model_states_exist = True
    for profile_path_value in profiles:
        profile_path = _resolve_path(project_root, profile_path_value)
        profile_exists = profile_path.exists()
        all_profiles_exist = all_profiles_exist and profile_exists
        profile_summary: dict[str, object] = {
            "path": profile_path.as_posix(),
            "exists": profile_exists,
            "profile_name": None,
            "model_state_path": None,
            "model_state_exists": False,
        }
        if profile_exists:
            try:
                profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                profile_summary["error"] = str(exc)
                all_model_states_exist = False
            else:
                profile_summary["profile_name"] = profile_data.get("profile_name")
                state_value = profile_data.get("model_state_path")
                if isinstance(state_value, str) and state_value.strip():
                    model_state_path = _resolve_path(project_root, Path(state_value))
                    model_state_exists = model_state_path.exists()
                    profile_summary["model_state_path"] = model_state_path.as_posix()
                    profile_summary["model_state_exists"] = model_state_exists
                    all_model_states_exist = all_model_states_exist and model_state_exists
                else:
                    profile_summary["error"] = "missing_model_state_path"
                    all_model_states_exist = False
        else:
            all_model_states_exist = False
        profile_summaries.append(profile_summary)

    checks["profile_paths_exist"] = all_profiles_exist
    checks["model_state_paths_exist"] = all_model_states_exist
    if not all_profiles_exist:
        failures.append("profile_missing")
    if not all_model_states_exist:
        failures.append("model_state_missing")
    return profile_summaries


def _check_camera(
    *,
    camera_index: int,
    check_camera: bool,
    camera_probe: CameraProbe | None,
    checks: dict[str, bool],
    failures: list[str],
    warnings: list[str],
) -> dict[str, object]:
    if not check_camera:
        checks["camera_checked"] = False
        checks["camera_opened"] = False
        warnings.append("camera_not_checked")
        return {"checked": False, "opened": False, "frame_read": False}

    probe = camera_probe or _opencv_camera_probe
    try:
        camera_summary = dict(probe(camera_index))
    except Exception as exc:  # pragma: no cover - depends on local camera stack.
        checks["camera_checked"] = True
        checks["camera_opened"] = False
        failures.append("camera_probe_failed")
        return {"checked": True, "opened": False, "frame_read": False, "error": str(exc)}

    camera_summary["checked"] = True
    opened = bool(camera_summary.get("opened"))
    frame_read = bool(camera_summary.get("frame_read"))
    checks["camera_checked"] = True
    checks["camera_opened"] = opened and frame_read
    if not checks["camera_opened"]:
        failures.append("camera_unavailable")
    return camera_summary


def _opencv_camera_probe(camera_index: int) -> dict[str, object]:
    import cv2  # type: ignore[import-untyped]

    capture = cv2.VideoCapture(int(camera_index))
    try:
        opened = bool(capture.isOpened())
        frame_read = False
        frame_width: int | None = None
        frame_height: int | None = None
        if opened:
            frame_read, frame = capture.read()
            if frame_read and frame is not None:
                frame_height, frame_width = frame.shape[:2]
        return {
            "opened": opened,
            "frame_read": bool(frame_read),
            "frame_width": frame_width,
            "frame_height": frame_height,
        }
    finally:
        capture.release()


def _check_hand_visibility(
    *,
    camera_index: int,
    check_hand_visibility: bool,
    max_frames: int,
    min_detection_rate: float,
    output_root: Path,
    hand_visibility_probe: HandVisibilityProbe | None,
    checks: dict[str, bool],
    failures: list[str],
    warnings: list[str],
) -> dict[str, object]:
    if not check_hand_visibility:
        checks["hand_visibility_checked"] = False
        checks["hand_visibility_ok"] = False
        warnings.append("hand_visibility_not_checked")
        return {"checked": False, "frame_count": 0, "detected_frames": 0, "detection_rate": None}

    probe = hand_visibility_probe or _mediapipe_hand_visibility_probe
    try:
        summary = dict(_run_hand_visibility_probe(probe, camera_index, max_frames, output_root))
    except Exception as exc:  # pragma: no cover - depends on local camera and mediapipe stack.
        checks["hand_visibility_checked"] = True
        checks["hand_visibility_ok"] = False
        failures.append("hand_visibility_probe_failed")
        return {"checked": True, "frame_count": 0, "detected_frames": 0, "detection_rate": None, "error": str(exc)}

    detection_rate = _optional_float(summary.get("detection_rate"))
    if detection_rate is None:
        frame_count = int(summary.get("frame_count", 0) or 0)
        detected_frames = int(summary.get("detected_frames", 0) or 0)
        detection_rate = detected_frames / frame_count if frame_count > 0 else 0.0
        summary["detection_rate"] = detection_rate
    checked = bool(summary.get("checked", True))
    ok = checked and detection_rate >= min_detection_rate
    summary["checked"] = checked
    summary["min_detection_rate"] = min_detection_rate
    checks["hand_visibility_checked"] = checked
    checks["hand_visibility_ok"] = ok
    if not ok:
        failures.append("hand_visibility_low")
    return summary


def _run_hand_visibility_probe(
    probe: HandVisibilityProbe,
    camera_index: int,
    max_frames: int,
    output_root: Path,
) -> Mapping[str, object]:
    try:
        parameters = inspect.signature(probe).parameters
    except (TypeError, ValueError):
        return probe(camera_index, max_frames)
    if len(parameters) >= 3:
        return probe(camera_index, max_frames, output_root)
    return probe(camera_index, max_frames)


def _mediapipe_hand_visibility_probe(camera_index: int, max_frames: int, output_root: Path) -> dict[str, object]:
    import cv2  # type: ignore[import-untyped]
    import mediapipe as mp  # type: ignore[import-untyped]

    capture = cv2.VideoCapture(int(camera_index))
    diagnostic_root = output_root / "hand_visibility"
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    try:
        opened = bool(capture.isOpened())
        frame_count = 0
        detected_frames = 0
        frame_width: int | None = None
        frame_height: int | None = None
        first_frame_path: str | None = None
        first_detected_path: str | None = None
        first_missing_path: str | None = None
        if not opened:
            return {
                "checked": True,
                "opened": False,
                "frame_count": 0,
                "detected_frames": 0,
                "detection_rate": 0.0,
                "frame_width": None,
                "frame_height": None,
                "diagnostic_image_paths": [],
            }
        with mp.solutions.hands.Hands(  # type: ignore[attr-defined]
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            for _ in range(max(1, int(max_frames))):
                read_ok, frame = capture.read()
                if not read_ok or frame is None:
                    continue
                frame_count += 1
                frame_height, frame_width = frame.shape[:2]
                if first_frame_path is None:
                    first_frame_path = (diagnostic_root / "first_frame.png").as_posix()
                    cv2.imwrite(first_frame_path, frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                if result.multi_hand_landmarks:
                    detected_frames += 1
                    if first_detected_path is None:
                        annotated = frame.copy()
                        for hand_landmarks in result.multi_hand_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(  # type: ignore[attr-defined]
                                annotated,
                                hand_landmarks,
                                mp.solutions.hands.HAND_CONNECTIONS,  # type: ignore[attr-defined]
                            )
                        first_detected_path = (diagnostic_root / "first_detected.png").as_posix()
                        cv2.imwrite(first_detected_path, annotated)
                elif first_missing_path is None:
                    missing_frame = frame.copy()
                    cv2.putText(
                        missing_frame,
                        "NO HAND DETECTED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                    )
                    first_missing_path = (diagnostic_root / "first_missing.png").as_posix()
                    cv2.imwrite(first_missing_path, missing_frame)
        diagnostic_paths = [
            path
            for path in (first_frame_path, first_detected_path, first_missing_path)
            if isinstance(path, str) and path
        ]
        return {
            "checked": True,
            "opened": opened,
            "frame_count": frame_count,
            "detected_frames": detected_frames,
            "detection_rate": detected_frames / frame_count if frame_count > 0 else 0.0,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "diagnostic_image_paths": diagnostic_paths,
        }
    finally:
        capture.release()


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _resolve_path(project_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else project_root / value


def _summary_markdown(summary: Mapping[str, object]) -> str:
    failures = summary.get("failures", [])
    warnings = summary.get("warnings", [])
    hand_visibility = summary.get("hand_visibility", {})
    lines = [
        "# Realtime Demo Preflight",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Config: `{summary.get('config_path')}`",
        f"- Python: `{summary.get('python_executable')}`",
        f"- Camera index: `{summary.get('camera_index')}`",
        "",
        "## Failures",
        "",
    ]
    if isinstance(failures, list) and failures:
        lines.extend(f"- `{item}`" for item in failures)
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    if isinstance(warnings, list) and warnings:
        lines.extend(f"- `{item}`" for item in warnings)
    else:
        lines.append("- None")
    lines.extend(["", "## Hand Visibility", ""])
    if isinstance(hand_visibility, Mapping):
        for key in ("checked", "frame_count", "detected_frames", "detection_rate", "min_detection_rate"):
            if key in hand_visibility:
                lines.append(f"- `{key}`: `{hand_visibility[key]}`")
        diagnostic_paths = hand_visibility.get("diagnostic_image_paths")
        if isinstance(diagnostic_paths, list) and diagnostic_paths:
            lines.extend(["", "### Diagnostic Images", ""])
            lines.extend(f"- `{path}`" for path in diagnostic_paths)
        else:
            lines.extend(["", "### Diagnostic Images", "", "- None"])
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoPreflightConfig", "run_realtime_demo_preflight"]
