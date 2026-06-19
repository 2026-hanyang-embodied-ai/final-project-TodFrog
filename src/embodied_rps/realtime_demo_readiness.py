"""Readiness summary for the few-shot realtime RPS demo pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeDemoReadinessConfig:
    """Paths used to summarize current realtime demo readiness."""

    output_root: Path
    original20_validation_summary: Path = Path(
        "artifacts/real_mp4_prediction_validation_v4_late_geometry_paper_override_fullrescue_original20_20260616/"
        "validation_summary.json"
    )
    heldout15_validation_summary: Path = Path(
        "artifacts/real_mp4_prediction_validation_v4_late_geometry_paper_override_fullrescue_new15_20260616/"
        "validation_summary.json"
    )
    launch_summary: Path = Path("artifacts/realtime_demo_launch_20260616/demo_launch_summary.json")
    preflight_summary: Path = Path("artifacts/realtime_demo_rehearsal_20260616/preflight/preflight_summary.json")
    dry_run_postcapture_summary: Path = Path(
        "artifacts/realtime_demo_rehearsal_20260616/dry_run_postcapture/postcapture_summary.json"
    )
    dry_run_composite_manifest: Path = Path(
        "artifacts/realtime_schunk_demo_composite_response_frame_20260616/"
        "realtime_schunk_demo_composite_manifest.json"
    )
    prelaunch_audit_summary: Path = Path("artifacts/realtime_demo_prelaunch_audit_20260616/prelaunch_audit.json")
    wrapper_contract_probe_summary: Path = Path(
        "artifacts/realtime_demo_wrapper_contract_probe_20260616/wrapper_contract_probe.json"
    )
    operator_command_audit_summary: Path = Path(
        "artifacts/realtime_demo_operator_command_audit_20260616/operator_command_audit.json"
    )
    live_overlay_video: Path = Path("artifacts/realtime_demo_rehearsal_20260616/live_camera_overlay.mp4")
    live_postcapture_summary: Path = Path("artifacts/realtime_demo_rehearsal_20260616/postcapture/postcapture_summary.json")
    live_composite_manifest: Path = Path(
        "artifacts/realtime_schunk_live_demo_composite_20260616/realtime_schunk_demo_composite_manifest.json"
    )


def summarize_realtime_demo_readiness(config: RealtimeDemoReadinessConfig) -> dict[str, object]:
    """Summarize offline gates, dry-run readiness, and live-demo artifact state."""

    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "readiness_summary.json"
    summary_md = output_root / "readiness_summary.md"

    original20 = _read_json_if_exists(config.original20_validation_summary)
    heldout15 = _read_json_if_exists(config.heldout15_validation_summary)
    launch = _read_json_if_exists(config.launch_summary)
    preflight = _read_json_if_exists(config.preflight_summary)
    dry_postcapture = _read_json_if_exists(config.dry_run_postcapture_summary)
    dry_composite = _read_json_if_exists(config.dry_run_composite_manifest)
    prelaunch = _read_json_if_exists(config.prelaunch_audit_summary)
    wrapper_contract = _read_json_if_exists(config.wrapper_contract_probe_summary)
    operator_command_audit = _read_json_if_exists(config.operator_command_audit_summary)
    live_postcapture = _read_json_if_exists(config.live_postcapture_summary)
    live_composite = _read_json_if_exists(config.live_composite_manifest)

    checks = {
        "original20_gate_passed": _json_bool(original20, "passed"),
        "heldout15_gate_passed": _json_bool(heldout15, "passed"),
        "launch_summary_exists": launch is not None,
        "one_command_pipeline_available": _pipeline_available(launch),
        "preflight_ready": _preflight_ready(preflight),
        "dry_run_postcapture_ready": _status_ok(dry_postcapture, accepted_statuses={"ready_for_composite"}),
        "dry_run_frame_log_ready": _frame_log_ready(dry_postcapture),
        "dry_run_demo_success_gate_passed": _demo_success_gate_passed(dry_postcapture),
        "dry_run_composite_ready": _status_ok(dry_composite, accepted_statuses={"passed"}),
        "prelaunch_audit_ready": _prelaunch_audit_ready(prelaunch),
        "wrapper_contract_probe_passed": _wrapper_contract_probe_passed(wrapper_contract),
        "operator_command_audit_passed": _operator_command_audit_passed(operator_command_audit),
        "live_overlay_exists": config.live_overlay_video.exists(),
        "live_postcapture_ready": _status_ok(live_postcapture, accepted_statuses={"ready_for_composite"}),
        "live_frame_log_ready": _frame_log_ready(live_postcapture),
        "live_demo_success_gate_passed": _demo_success_gate_passed(live_postcapture),
        "live_composite_ready": _status_ok(live_composite, accepted_statuses={"passed"}),
    }
    remaining_actions = _remaining_actions(checks)
    status = _status_for_checks(checks, remaining_actions)
    next_action = _next_action(status, remaining_actions)
    ok = status in {"ready_for_live_capture", "live_demo_artifact_ready"}
    summary: dict[str, object] = {
        "status": status,
        "ok": ok,
        "checks": checks,
        "remaining_actions": remaining_actions,
        "next_action": next_action,
        "paths": {
            "original20_validation_summary": config.original20_validation_summary.as_posix(),
            "heldout15_validation_summary": config.heldout15_validation_summary.as_posix(),
            "launch_summary": config.launch_summary.as_posix(),
            "preflight_summary": config.preflight_summary.as_posix(),
            "dry_run_postcapture_summary": config.dry_run_postcapture_summary.as_posix(),
            "dry_run_composite_manifest": config.dry_run_composite_manifest.as_posix(),
            "prelaunch_audit_summary": config.prelaunch_audit_summary.as_posix(),
            "wrapper_contract_probe_summary": config.wrapper_contract_probe_summary.as_posix(),
            "operator_command_audit_summary": config.operator_command_audit_summary.as_posix(),
            "live_overlay_video": config.live_overlay_video.as_posix(),
            "live_postcapture_summary": config.live_postcapture_summary.as_posix(),
            "live_composite_manifest": config.live_composite_manifest.as_posix(),
        },
        "metrics": {
            "original20_passed_clip_count": _json_value(original20, "passed_clip_count"),
            "original20_clip_count": _json_value(original20, "clip_count"),
            "heldout15_passed_clip_count": _json_value(heldout15, "passed_clip_count"),
            "heldout15_clip_count": _json_value(heldout15, "clip_count"),
            "launch_script_count": _json_value(launch, "script_count"),
            "dry_run_frame_log_records": _frame_log_record_count(dry_postcapture),
            "dry_run_demo_success_binary_latency_s": _demo_success_binary_latency(dry_postcapture),
            "dry_run_composite_frame_count": _json_value(dry_composite, "frame_count"),
            "prelaunch_status": _json_value(prelaunch, "prelaunch_status"),
            "wrapper_contract_status": _json_value(wrapper_contract, "contract_status"),
            "wrapper_contract_passed_count": _json_value(wrapper_contract, "passed_count"),
            "wrapper_contract_scenario_count": _json_value(wrapper_contract, "scenario_count"),
            "operator_command_audit_status": _json_value(operator_command_audit, "audit_status"),
            "operator_command_audit_passed_count": _json_value(operator_command_audit, "passed_count"),
            "operator_command_audit_check_count": _json_value(operator_command_audit, "check_count"),
            "live_frame_log_records": _frame_log_record_count(live_postcapture),
            "live_demo_success_binary_latency_s": _demo_success_binary_latency(live_postcapture),
            "live_composite_frame_count": _json_value(live_composite, "frame_count"),
        },
        "outputs": {
            "summary_json": summary_json.as_posix(),
            "summary_md": summary_md.as_posix(),
        },
        "claim_scope": "readiness summary over existing artifacts; does not run model inference or camera capture",
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _json_bool(payload: dict[str, Any] | None, key: str) -> bool:
    return bool(payload is not None and payload.get(key) is True)


def _json_value(payload: dict[str, Any] | None, key: str) -> object:
    if payload is None:
        return None
    return payload.get(key)


def _status_ok(payload: dict[str, Any] | None, *, accepted_statuses: set[str]) -> bool:
    if payload is None:
        return False
    status = payload.get("status")
    if isinstance(status, str):
        return status in accepted_statuses
    return bool(payload.get("ok") is True)


def _preflight_ready(payload: dict[str, Any] | None) -> bool:
    if _status_ok(payload, accepted_statuses={"ready_for_live_demo"}):
        return True
    if payload is None:
        return False
    failures = payload.get("failures")
    if not isinstance(failures, list):
        return False
    if {str(item) for item in failures} != {"hand_visibility_low"}:
        return False
    camera = payload.get("camera")
    if not isinstance(camera, dict):
        return False
    return camera.get("opened") is True and camera.get("frame_read") is True


def _frame_log_ready(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    frame_log = payload.get("frame_log")
    if not isinstance(frame_log, dict):
        return False
    checks = frame_log.get("checks")
    if not isinstance(checks, dict):
        return False
    return checks.get("frame_log_ready") is True


def _frame_log_record_count(payload: dict[str, Any] | None) -> object:
    if payload is None:
        return None
    frame_log = payload.get("frame_log")
    if not isinstance(frame_log, dict):
        return None
    return frame_log.get("record_count")


def _demo_success_gate_passed(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    gate = payload.get("demo_success_gate")
    if not isinstance(gate, dict):
        return False
    return gate.get("passed") is True


def _demo_success_binary_latency(payload: dict[str, Any] | None) -> object:
    if payload is None:
        return None
    gate = payload.get("demo_success_gate")
    if not isinstance(gate, dict):
        return None
    return gate.get("binary_decision_latency_s")


def _prelaunch_audit_ready(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    return (
        payload.get("prelaunch_status") == "ready_for_operator_live_attempt"
        and payload.get("ready_for_operator_live_attempt") is True
        and payload.get("blocking_issues") in ([], None)
    )


def _wrapper_contract_probe_passed(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    return (
        payload.get("contract_status") == "passed"
        and payload.get("scenario_count") == payload.get("passed_count")
        and int(payload.get("scenario_count", 0) or 0) > 0
    )


def _operator_command_audit_passed(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    return (
        payload.get("audit_status") == "passed"
        and payload.get("check_count") == payload.get("passed_count")
        and int(payload.get("check_count", 0) or 0) > 0
    )


def _pipeline_available(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    scripts = payload.get("scripts")
    if not isinstance(scripts, dict):
        return False
    return "run_live_demo_pipeline" in scripts and int(payload.get("script_count", 0)) >= 8


def _remaining_actions(checks: dict[str, bool]) -> list[str]:
    actions: list[str] = []
    required_before_live = {
        "original20_gate_passed": "original20_gate_not_passed",
        "heldout15_gate_passed": "heldout15_gate_not_passed",
        "one_command_pipeline_available": "one_command_pipeline_missing",
        "preflight_ready": "preflight_not_ready",
        "dry_run_postcapture_ready": "dry_run_postcapture_not_ready",
        "dry_run_frame_log_ready": "dry_run_frame_log_not_ready",
        "dry_run_demo_success_gate_passed": "dry_run_demo_success_gate_not_passed",
        "dry_run_composite_ready": "dry_run_composite_not_ready",
        "wrapper_contract_probe_passed": "wrapper_contract_probe_not_passed",
        "operator_command_audit_passed": "operator_command_audit_not_passed",
    }
    for key, action in required_before_live.items():
        if not checks[key]:
            actions.append(action)
    if actions:
        return actions
    if not checks["live_overlay_exists"]:
        actions.append("live_capture_missing")
    elif checks["live_composite_ready"]:
        return actions
    elif not checks["live_postcapture_ready"]:
        actions.append("live_postcapture_missing_or_not_ready")
    elif not checks["live_composite_ready"]:
        actions.append("live_composite_missing_or_not_ready")
    return actions


def _status_for_checks(checks: dict[str, bool], remaining_actions: list[str]) -> str:
    if not remaining_actions:
        return "live_demo_artifact_ready"
    if remaining_actions == ["live_capture_missing"]:
        return "ready_for_live_capture"
    if checks["live_overlay_exists"] and not checks["live_composite_ready"]:
        return "live_capture_needs_postprocessing"
    return "blocked"


def _next_action(status: str, remaining_actions: list[str]) -> str:
    if status == "live_demo_artifact_ready":
        return "inspect live composite output and use it as final demo-video source material"
    if status == "ready_for_live_capture":
        return "run 24_run_live_demo_operator_confirmed_strict.ps1 with the user performing during PROMPT SCISSORS"
    if status == "live_capture_needs_postprocessing":
        return "run 06_verify_live_capture.ps1 and then 05_create_live_schunk_composite.ps1"
    if remaining_actions:
        return f"resolve {remaining_actions[0]}"
    return "inspect readiness summary"


def _summary_markdown(summary: dict[str, object]) -> str:
    checks = summary.get("checks", {})
    metrics = summary.get("metrics", {})
    remaining = summary.get("remaining_actions", [])
    lines = [
        "# Realtime Demo Readiness",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- OK: `{summary.get('ok')}`",
        f"- Next action: {summary.get('next_action')}",
        "",
        "## Checks",
        "",
    ]
    if isinstance(checks, dict):
        for key in sorted(checks):
            lines.append(f"- `{key}`: `{checks[key]}`")
    lines.extend(["", "## Metrics", ""])
    if isinstance(metrics, dict):
        for key in sorted(metrics):
            lines.append(f"- `{key}`: `{metrics[key]}`")
    lines.extend(["", "## Remaining Actions", ""])
    if isinstance(remaining, list) and remaining:
        lines.extend(f"- `{item}`" for item in remaining)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoReadinessConfig", "summarize_realtime_demo_readiness"]
