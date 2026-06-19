"""Record the three final live robot-counterattack submission takes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-untyped]
import yaml

from embodied_rps.final_submission_live_counterattack_recording import (
    LiveCounterattackTakeSpec,
    LiveTakeInput,
    build_live_recording_artifacts_from_take_inputs,
    default_take_specs,
    validate_archived_schunk_style_assets,
)
from embodied_rps.realtime_demo_launcher import load_realtime_demo_config
from embodied_rps.tools.run_current_best_realtime_demo import main as run_current_best_realtime_demo

DEFAULT_CONFIG = Path("configs/realtime_two_stage_selector_demo_conditional_scissors_rescue_rock_guard.yaml")
DEFAULT_POSE_CONFIG = Path("configs/kinematic_rps.yaml")
DEFAULT_STYLE_ROOT = Path("artifacts/schunk_joint_target_skeleton_passed")
DEFAULT_OUTPUT_ROOT = Path("artifacts/final_submission_live_counterattack_recording_20260619")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the final live counterattack recording workflow."""

    parser = argparse.ArgumentParser(description="Record final live RPS robot counterattack takes.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Frozen v4 realtime demo config.")
    parser.add_argument("--pose-config", type=Path, default=DEFAULT_POSE_CONFIG, help="Kinematic actuator feasibility config.")
    parser.add_argument("--style-asset-root", type=Path, default=DEFAULT_STYLE_ROOT, help="Archived SCHUNK yaw45/pitch20 image root.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Final live-recording artifact root.")
    parser.add_argument("--camera", type=int, default=0, help="Live camera index.")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum frames per live take.")
    parser.add_argument("--preflight-frames", type=int, default=45, help="Frames to sample for camera hand-detection preflight.")
    parser.add_argument("--min-detection-rate", type=float, default=0.80, help="Required preflight hand-detection rate.")
    parser.add_argument("--skip-camera-preflight", action="store_true", help="Skip hand-detection preflight, but still validate config/assets.")
    parser.add_argument("--dry-run", action="store_true", help="Write take configs and delegated argv without recording.")
    parser.add_argument(
        "--take-inputs-json",
        type=Path,
        default=None,
        help="Build final artifacts from existing raw take inputs instead of recording live.",
    )
    parser.add_argument(
        "--display-window",
        dest="display_window",
        action="store_true",
        default=None,
        help="Show the OpenCV live predictor window. Defaults to predictor behavior.",
    )
    parser.add_argument(
        "--no-display-window",
        dest="display_window",
        action="store_false",
        help="Disable the OpenCV live predictor window.",
    )
    args = parser.parse_args(argv)

    project_root = Path.cwd()
    args.output_root.mkdir(parents=True, exist_ok=True)
    preflight = _run_preflight(
        config_path=args.config,
        style_asset_root=args.style_asset_root,
        output_root=args.output_root,
        project_root=project_root,
        camera=args.camera,
        preflight_frames=args.preflight_frames,
        min_detection_rate=args.min_detection_rate,
        skip_camera_preflight=args.skip_camera_preflight or args.take_inputs_json is not None or args.dry_run,
    )
    if preflight["status"] != "passed":
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return 2

    specs = default_take_specs()
    if args.take_inputs_json is not None:
        take_inputs = _load_take_inputs(args.take_inputs_json, project_root=project_root)
    else:
        take_inputs = {}
        for spec in specs:
            take_input = _record_take(
                spec=spec,
                base_config_path=args.config,
                output_root=args.output_root,
                camera=args.camera,
                max_frames=args.max_frames,
                display_window=args.display_window,
                project_root=project_root,
                dry_run=args.dry_run,
            )
            if take_input is not None:
                take_inputs[spec.take_id] = take_input
        if args.dry_run:
            payload = {
                "status": "passed",
                "mode": "dry-run",
                "preflight": preflight,
                "take_configs": [_relative_path(args.output_root / spec.take_id / "take_prompt_policy.yaml", project_root=project_root) for spec in specs],
            }
            _write_json(args.output_root / "dry_run_summary.json", payload)
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

    manifest = build_live_recording_artifacts_from_take_inputs(
        take_inputs=take_inputs,
        style_asset_root=args.style_asset_root,
        pose_config_path=args.pose_config,
        output_root=args.output_root,
        project_root=project_root,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if manifest["status"] == "passed" else 3


def _run_preflight(
    *,
    config_path: Path,
    style_asset_root: Path,
    output_root: Path,
    project_root: Path,
    camera: int,
    preflight_frames: int,
    min_detection_rate: float,
    skip_camera_preflight: bool,
) -> dict[str, object]:
    checks: dict[str, object] = {}
    status = "passed"
    failure_reason: str | None = None
    try:
        config = load_realtime_demo_config(config_path)
        profile_rows: list[dict[str, object]] = []
        for profile in config.profiles:
            normalized = profile.as_posix()
            exists = (project_root / profile).exists()
            is_v4 = "v4" in normalized.lower() and "v7" not in normalized.lower()
            profile_rows.append({"path": normalized, "exists": exists, "v4_live_demo_profile": is_v4})
            if not exists or not is_v4:
                raise ValueError(f"Final live recording requires existing v4 profile, got {normalized}")
        checks["v4_profiles"] = profile_rows
        checks["config_path"] = _relative_path(config_path, project_root=project_root)
        style_assets = validate_archived_schunk_style_assets(style_asset_root)
        checks["archived_schunk_style_assets"] = {
            gesture: _relative_path(path, project_root=project_root) for gesture, path in style_assets.items()
        }
        if skip_camera_preflight:
            checks["camera_preflight"] = {"skipped": True}
        else:
            checks["camera_preflight"] = _camera_hand_detection_preflight(
                camera=camera,
                frame_count=preflight_frames,
                min_detection_rate=min_detection_rate,
            )
            if float(checks["camera_preflight"]["detection_rate"]) < min_detection_rate:
                raise ValueError(
                    f"Camera hand-detection preflight below threshold: {checks['camera_preflight']['detection_rate']}"
                )
        output_root.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        status = "failed"
        failure_reason = str(exc)
    payload = {
        "status": status,
        "claim_scope": "preflight for final live recording; no retraining and no final packaging",
        "failure_reason": failure_reason,
        "checks": checks,
    }
    _write_json(output_root / "preflight_summary.json", payload)
    return payload


def _camera_hand_detection_preflight(*, camera: int, frame_count: int, min_detection_rate: float) -> dict[str, object]:
    try:
        import mediapipe as mp  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("mediapipe is required for live hand-detection preflight") from exc

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera}")
    sampled = 0
    detected = 0
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    try:
        while sampled < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            sampled += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            if bool(result.multi_hand_landmarks):
                detected += 1
    finally:
        hands.close()
        capture.release()
    detection_rate = float(detected / sampled) if sampled else 0.0
    return {
        "skipped": False,
        "camera": camera,
        "sampled_frames": sampled,
        "detected_frames": detected,
        "detection_rate": detection_rate,
        "min_detection_rate": min_detection_rate,
    }


def _record_take(
    *,
    spec: LiveCounterattackTakeSpec,
    base_config_path: Path,
    output_root: Path,
    camera: int,
    max_frames: int,
    display_window: bool | None,
    project_root: Path,
    dry_run: bool,
) -> LiveTakeInput | None:
    take_dir = output_root / spec.take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    take_config = take_dir / "take_prompt_policy.yaml"
    overlay_video = take_dir / "raw_camera_overlay.mp4"
    frame_log = take_dir / "raw_camera_frames.jsonl"
    skeleton_npz = take_dir / "raw_camera_skeletons.npz"
    _write_take_prompt_config(base_config_path=base_config_path, output_path=take_config, spec=spec, display_window=display_window)
    delegated_argv = [
        "--config",
        _path_arg(take_config, project_root=project_root),
        "--camera",
        str(camera),
        "--output",
        _path_arg(overlay_video, project_root=project_root),
        "--frame-log-jsonl",
        _path_arg(frame_log, project_root=project_root),
        "--skeleton-npz",
        _path_arg(skeleton_npz, project_root=project_root),
        "--max-frames",
        str(max_frames),
        "--expected-actual-gesture",
        spec.human_target,
    ]
    if dry_run:
        delegated_argv.append("--dry-run")
    _write_json(
        take_dir / "recording_command.json",
        {
            "take_id": spec.take_id,
            "delegated_module": "embodied_rps.tools.run_current_best_realtime_demo",
            "argv": delegated_argv,
        },
    )
    result = run_current_best_realtime_demo(delegated_argv)
    if result != 0:
        _write_json(
            take_dir / "capture_failure.json",
            {
                "status": "failed",
                "take_id": spec.take_id,
                "return_code": result,
            },
        )
        return None
    if dry_run:
        return None
    if not overlay_video.exists() or not frame_log.exists():
        _write_json(
            take_dir / "capture_failure.json",
            {
                "status": "failed",
                "take_id": spec.take_id,
                "reason": "Live predictor did not produce required raw overlay/frame-log artifacts",
            },
        )
        return None
    return LiveTakeInput(overlay_video=overlay_video, frame_log=frame_log, skeleton_npz=skeleton_npz if skeleton_npz.exists() else None)


def _write_take_prompt_config(
    *,
    base_config_path: Path,
    output_path: Path,
    spec: LiveCounterattackTakeSpec,
    display_window: bool | None,
) -> None:
    with base_config_path.open("r", encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ValueError("Base realtime config must be a mapping")
    payload = dict(loaded)
    payload["prompt_sequence"] = spec.prompt_sequence
    payload["response_prompt"] = spec.response_prompt
    payload["hold_response_prompt_until_decision"] = True
    payload["response_hold_max_frames"] = max(int(payload.get("response_hold_max_frames", 0)), 180)
    payload["stop_after_confirmed_response_decision"] = True
    payload["post_decision_hold_frames"] = max(int(payload.get("post_decision_hold_frames", 0)), 90)
    payload["reset_on_prompt_cycle"] = True
    payload["reset_on_prompt_change"] = True
    if display_window is not None:
        payload["display_window"] = bool(display_window)
    payload["final_live_take"] = {
        "take_id": spec.take_id,
        "human_target": spec.human_target,
        "robot_counter": spec.robot_counter,
        "accept_wait_as_rock": spec.accept_wait_as_rock,
        "source_config": base_config_path.as_posix(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _load_take_inputs(path: Path, *, project_root: Path) -> dict[str, LiveTakeInput]:
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("--take-inputs-json must contain a JSON object")
    take_inputs: dict[str, LiveTakeInput] = {}
    for take_id, raw_value in loaded.items():
        if not isinstance(raw_value, Mapping):
            raise ValueError(f"Take input for {take_id!r} must be an object")
        overlay = _required_project_path(raw_value, "overlay_video", project_root=project_root)
        frame_log = _required_project_path(raw_value, "frame_log", project_root=project_root)
        skeleton_value = raw_value.get("skeleton_npz")
        skeleton_npz = _project_path(str(skeleton_value), project_root=project_root) if skeleton_value else None
        take_inputs[str(take_id)] = LiveTakeInput(overlay_video=overlay, frame_log=frame_log, skeleton_npz=skeleton_npz)
    return take_inputs


def _required_project_path(mapping: Mapping[str, Any], key: str, *, project_root: Path) -> Path:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Take input must include {key!r}")
    return _project_path(value, project_root=project_root)


def _project_path(path_text: str, *, project_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _relative_path(path: Path | None, *, project_root: Path) -> str | None:
    if path is None:
        return None
    raw = Path(path)
    if not raw.is_absolute():
        return raw.as_posix()
    try:
        return raw.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return raw.as_posix()


def _path_arg(path: Path, *, project_root: Path) -> str:
    relative = _relative_path(path, project_root=project_root)
    return relative if relative is not None else path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
