"""Guarded live-retake readiness audit for the realtime RPS demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_rps.realtime_demo_launcher import build_realtime_demo_argv, load_realtime_demo_config


DEFAULT_STALE_ARTIFACT_PATHS = (
    Path("artifacts/realtime_demo_rehearsal_20260616/live_camera_overlay.mp4"),
    Path("artifacts/realtime_demo_rehearsal_20260616/live_camera_frames.jsonl"),
    Path("artifacts/realtime_demo_rehearsal_20260616/postcapture"),
    Path("artifacts/realtime_schunk_live_demo_composite_20260616"),
    Path("artifacts/realtime_demo_overlay_contract_20260616"),
)

REQUIRED_GUARD_ARGS = (
    "--rock-hold-guard-min-history-frames",
    "--rock-hold-guard-max-latest-finger-extension",
    "--rock-hold-guard-max-extension-delta",
)

REQUIRED_VERIFIER_ARGS = (
    "--gesture-verifier-min-history-frames",
    "--gesture-verifier-rock-max-ring-pinky-extension",
    "--gesture-verifier-rock-max-index-middle-extension",
    "--gesture-verifier-rock-max-index-middle-minus-ring-pinky",
    "--gesture-verifier-rock-max-extension-delta",
    "--gesture-verifier-scissors-min-index-middle-extension",
    "--gesture-verifier-scissors-min-index-middle-delta",
    "--gesture-verifier-scissors-min-index-middle-minus-ring-pinky",
    "--gesture-verifier-scissors-max-ring-pinky-extension",
    "--gesture-verifier-paper-min-ring-pinky-extension",
    "--gesture-verifier-paper-min-ring-pinky-delta",
    "--gesture-verifier-paper-max-index-middle-minus-ring-pinky",
)


@dataclass(frozen=True)
class RealtimeDemoGuardedRetakeReadinessConfig:
    """Input and output paths for guarded live-retake readiness auditing."""

    output_root: Path = Path("artifacts/realtime_demo_guarded_retake_readiness_20260616")
    readiness_summary: Path = Path("artifacts/realtime_demo_readiness_20260616/readiness_summary.json")
    live_status_snapshot: Path = Path("artifacts/realtime_demo_live_status_snapshot_20260616/live_status_snapshot.json")
    prelaunch_audit: Path = Path("artifacts/realtime_demo_prelaunch_audit_20260616/prelaunch_audit.json")
    operator_command_audit: Path = Path(
        "artifacts/realtime_demo_operator_command_audit_20260616/operator_command_audit.json"
    )
    live_artifact_cleanup: Path = Path(
        "artifacts/realtime_demo_live_artifact_cleanup_20260616/live_artifact_cleanup.json"
    )
    launch_summary: Path = Path("artifacts/realtime_demo_launch_20260616/demo_launch_summary.json")
    stale_artifact_paths: tuple[Path, ...] = DEFAULT_STALE_ARTIFACT_PATHS


def build_realtime_demo_guarded_retake_readiness(
    config: RealtimeDemoGuardedRetakeReadinessConfig,
) -> dict[str, object]:
    """Write an audit proving the next live retake is using the rock-hold guard."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    output_json = config.output_root / "guarded_retake_readiness.json"
    output_md = config.output_root / "guarded_retake_readiness.md"
    readiness = _read_json_if_exists(config.readiness_summary) or {}
    live_status = _read_json_if_exists(config.live_status_snapshot) or {}
    prelaunch = _read_json_if_exists(config.prelaunch_audit) or {}
    operator_audit = _read_json_if_exists(config.operator_command_audit) or {}
    cleanup = _read_json_if_exists(config.live_artifact_cleanup) or {}
    launch_summary = _read_json_if_exists(config.launch_summary) or {}
    audit = _guarded_retake_audit(
        config=config,
        readiness=readiness,
        live_status=live_status,
        prelaunch=prelaunch,
        operator_audit=operator_audit,
        cleanup=cleanup,
        launch_summary=launch_summary,
        output_json=output_json,
        output_md=output_md,
    )
    output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    output_md.write_text(_guarded_retake_markdown(audit), encoding="utf-8")
    return audit


def _guarded_retake_audit(
    *,
    config: RealtimeDemoGuardedRetakeReadinessConfig,
    readiness: dict[str, Any],
    live_status: dict[str, Any],
    prelaunch: dict[str, Any],
    operator_audit: dict[str, Any],
    cleanup: dict[str, Any],
    launch_summary: dict[str, Any],
    output_json: Path,
    output_md: Path,
) -> dict[str, object]:
    scripts = _dict_value(launch_summary, "scripts")
    delegated_argv, argv_error = _delegated_live_argv(launch_summary)
    delegated_guard_args = _arg_values(delegated_argv, REQUIRED_GUARD_ARGS)
    delegated_verifier_args = _arg_values(delegated_argv, REQUIRED_VERIFIER_ARGS)
    checks: list[dict[str, object]] = []
    _add_check(checks, "readiness_summary_loaded", bool(readiness), "readiness summary JSON is missing or empty")
    _add_check(
        checks,
        "readiness_waiting_for_live_capture",
        readiness.get("status") == "ready_for_live_capture"
        and "live_capture_missing" in _list_value(readiness, "remaining_actions"),
        "readiness summary must be ready_for_live_capture with live_capture_missing as the only remaining live step",
    )
    _add_check(
        checks,
        "live_status_awaiting_capture",
        live_status.get("snapshot_status") == "awaiting_live_capture",
        "live status snapshot must be awaiting_live_capture",
    )
    _add_check(
        checks,
        "prelaunch_ready",
        prelaunch.get("ready_for_operator_live_attempt") is True,
        "prelaunch audit is not ready for an operator live attempt",
    )
    _add_check(
        checks,
        "operator_command_audit_passed",
        operator_audit.get("audit_status") == "passed",
        "operator command audit has not passed",
    )
    _add_check(
        checks,
        "stale_artifact_cleanup_cleared",
        cleanup.get("cleanup_status") == "cleared",
        "stale live artifact cleanup has not reported cleared",
    )
    _add_check(checks, "launch_summary_loaded", bool(launch_summary), "launch summary JSON is missing or empty")
    config_path = _path_value(launch_summary.get("config_path"))
    _add_check(
        checks,
        "guard_config_path_exists",
        config_path is not None and config_path.exists(),
        f"guard config path is missing or does not exist: {launch_summary.get('config_path')}",
    )
    _add_check(
        checks,
        "guard_config_path_is_rock_guard_candidate",
        config_path is not None and "rock_guard" in config_path.as_posix(),
        f"launch config path must identify the rock guard candidate: {launch_summary.get('config_path')}",
    )
    live_script = _path_value(scripts.get("live_camera_demo"))
    _add_check(
        checks,
        "live_camera_script_exists",
        live_script is not None and live_script.exists(),
        f"live camera script is missing: {scripts.get('live_camera_demo')}",
    )
    if live_script is not None and live_script.exists() and config_path is not None:
        live_script_content = live_script.read_text(encoding="utf-8-sig")
        _add_check(
            checks,
            "live_camera_script_uses_guard_config",
            config_path.as_posix() in live_script_content or str(config_path) in live_script_content,
            "live camera script does not pass the guarded config path",
        )
    if scripts.get("run_live_demo_operator_confirmed_strict") is not None:
        strict_script = _path_value(scripts.get("run_live_demo_operator_confirmed_strict"))
        _add_check(
            checks,
            "strict_wrapper_exists",
            strict_script is not None and strict_script.exists(),
            f"strict wrapper is missing: {scripts.get('run_live_demo_operator_confirmed_strict')}",
        )
    _add_check(checks, "delegated_argv_builds", argv_error is None, argv_error or "ok")
    _add_guard_arg_checks(checks, delegated_guard_args)
    _add_verifier_arg_checks(checks, delegated_verifier_args)
    for stale_path in config.stale_artifact_paths:
        _add_check(
            checks,
            f"stale_artifact_absent_{_safe_check_id(stale_path)}",
            not stale_path.exists(),
            f"stale live artifact still exists: {stale_path.as_posix()}",
        )
    blocking_issues = [str(check["detail"]) for check in checks if check.get("passed") is not True]
    ready = not blocking_issues
    status = "ready_for_guarded_rock_retake" if ready else "guarded_retake_blocked"
    return {
        "guarded_retake_status": status,
        "ready_for_guarded_retake": ready,
        "recommended_expected_actual_gesture": "rock",
        "recommended_operator_command": (
            "powershell -ExecutionPolicy Bypass -File "
            "artifacts\\realtime_demo_launch_20260616\\24_run_live_demo_operator_confirmed_strict.ps1"
        ),
        "delegated_guard_args": delegated_guard_args,
        "delegated_verifier_args": delegated_verifier_args,
        "blocking_issues": blocking_issues,
        "checks": checks,
        "inputs": {
            "readiness_summary": config.readiness_summary.as_posix(),
            "live_status_snapshot": config.live_status_snapshot.as_posix(),
            "prelaunch_audit": config.prelaunch_audit.as_posix(),
            "operator_command_audit": config.operator_command_audit.as_posix(),
            "live_artifact_cleanup": config.live_artifact_cleanup.as_posix(),
            "launch_summary": config.launch_summary.as_posix(),
            "stale_artifact_paths": [path.as_posix() for path in config.stale_artifact_paths],
        },
        "outputs": {
            "guarded_retake_readiness_json": output_json.as_posix(),
            "guarded_retake_readiness_md": output_md.as_posix(),
        },
        "claim_scope": (
            "non-camera audit for guarded live-retake readiness; verifies launch wiring, guard argv, "
            "status artifacts, and stale artifact cleanup, but does not run camera capture"
        ),
    }


def _delegated_live_argv(launch_summary: dict[str, Any]) -> tuple[list[str], str | None]:
    config_path = _path_value(launch_summary.get("config_path"))
    if config_path is None:
        return [], "launch summary does not contain a valid config_path"
    try:
        demo_config = load_realtime_demo_config(config_path)
        camera = int(launch_summary.get("camera_index", 0))
        max_frames = int(launch_summary.get("camera_max_frames", 900))
        live_output = _path_value(launch_summary.get("live_output"))
        live_frame_log = _path_value(launch_summary.get("live_frame_log"))
        return (
            build_realtime_demo_argv(
                demo_config,
                video=None,
                camera=camera,
                output=live_output,
                frame_log_jsonl=live_frame_log,
                max_frames=max_frames,
                expected_actual_gesture="rock",
            ),
            None,
        )
    except Exception as exc:  # pragma: no cover - defensive error capture for audit artifacts.
        return [], f"failed to build delegated live argv: {exc}"


def _add_guard_arg_checks(checks: list[dict[str, object]], delegated_guard_args: dict[str, str]) -> None:
    min_history = delegated_guard_args.get("--rock-hold-guard-min-history-frames")
    try:
        min_history_enabled = min_history is not None and int(min_history) > 0
    except ValueError:
        min_history_enabled = False
    _add_check(
        checks,
        "rock_hold_guard_enabled_in_delegated_argv",
        min_history_enabled,
        "rock-hold guard is not enabled in delegated argv",
    )
    for arg_name in REQUIRED_GUARD_ARGS:
        _add_check(
            checks,
            f"delegated_arg_present_{arg_name.strip('-').replace('-', '_')}",
            arg_name in delegated_guard_args,
            f"required delegated guard argument is missing: {arg_name}",
        )


def _add_verifier_arg_checks(checks: list[dict[str, object]], delegated_verifier_args: dict[str, str]) -> None:
    min_history = delegated_verifier_args.get("--gesture-verifier-min-history-frames")
    try:
        min_history_enabled = min_history is not None and int(min_history) > 0
    except ValueError:
        min_history_enabled = False
    _add_check(
        checks,
        "gesture_verifier_enabled_in_delegated_argv",
        min_history_enabled,
        "gesture verifier is not enabled in delegated argv",
    )
    for arg_name in REQUIRED_VERIFIER_ARGS:
        _add_check(
            checks,
            f"delegated_arg_present_{arg_name.strip('-').replace('-', '_')}",
            arg_name in delegated_verifier_args,
            f"required delegated gesture verifier argument is missing: {arg_name}",
        )


def _arg_values(argv: list[str], names: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in names:
        if name in argv:
            index = argv.index(name)
            if index + 1 < len(argv):
                values[name] = argv[index + 1]
    return values


def _add_check(checks: list[dict[str, object]], check_id: str, passed: bool, detail: str) -> None:
    checks.append({"id": check_id, "passed": bool(passed), "detail": "ok" if passed else detail})


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _list_value(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _path_value(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def _safe_check_id(path: Path) -> str:
    return "".join(char if char.isalnum() else "_" for char in path.as_posix()).strip("_")


def _guarded_retake_markdown(audit: dict[str, object]) -> str:
    lines = [
        "# Guarded Retake Readiness",
        "",
        f"- Status: `{audit.get('guarded_retake_status')}`",
        f"- Ready for guarded retake: `{audit.get('ready_for_guarded_retake')}`",
        f"- Recommended expected actual gesture: `{audit.get('recommended_expected_actual_gesture')}`",
        f"- Recommended command: `{audit.get('recommended_operator_command')}`",
        "",
        "## Delegated Guard Args",
        "",
    ]
    guard_args = audit.get("delegated_guard_args", {})
    if isinstance(guard_args, dict) and guard_args:
        lines.extend(f"- `{key}`: `{value}`" for key, value in guard_args.items())
    else:
        lines.append("- None")
    lines.extend(["", "## Blocking Issues", ""])
    blocking = audit.get("blocking_issues", [])
    if isinstance(blocking, list) and blocking:
        lines.extend(f"- {item}" for item in blocking)
    else:
        lines.append("- None")
    lines.extend(["", "## Checks", ""])
    checks = audit.get("checks", [])
    if isinstance(checks, list):
        for check in checks:
            if isinstance(check, dict):
                lines.append(f"- `{check.get('id')}`: `{check.get('passed')}` - {check.get('detail')}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "RealtimeDemoGuardedRetakeReadinessConfig",
    "build_realtime_demo_guarded_retake_readiness",
]
