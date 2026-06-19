"""Submission-oriented evidence bundle for the realtime RPS demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeDemoEvidenceBundleConfig:
    """Artifact paths used to summarize final demo evidence readiness."""

    output_root: Path
    readiness_summary: Path = Path("artifacts/realtime_demo_readiness_20260616/readiness_summary.json")
    triage_summary: Path = Path("artifacts/realtime_demo_triage_20260616/triage_summary.json")
    dry_run_postcapture_summary: Path = Path(
        "artifacts/realtime_demo_rehearsal_20260616/dry_run_postcapture/postcapture_summary.json"
    )
    dry_run_composite_manifest: Path = Path(
        "artifacts/realtime_schunk_demo_composite_response_frame_20260616/"
        "realtime_schunk_demo_composite_manifest.json"
    )
    dry_run_overlay_contract_summary: Path = Path(
        "artifacts/realtime_demo_overlay_contract_dryrun_20260616/overlay_contract_summary.json"
    )
    live_overlay_video: Path = Path("artifacts/realtime_demo_rehearsal_20260616/live_camera_overlay.mp4")
    live_frame_log: Path = Path("artifacts/realtime_demo_rehearsal_20260616/live_camera_frames.jsonl")
    live_postcapture_summary: Path = Path("artifacts/realtime_demo_rehearsal_20260616/postcapture/postcapture_summary.json")
    live_composite_manifest: Path = Path(
        "artifacts/realtime_schunk_live_demo_composite_20260616/realtime_schunk_demo_composite_manifest.json"
    )
    live_overlay_contract_summary: Path = Path("artifacts/realtime_demo_overlay_contract_20260616/overlay_contract_summary.json")
    live_artifact_cleanup_summary: Path = Path(
        "artifacts/realtime_demo_live_artifact_cleanup_20260616/live_artifact_cleanup.json"
    )


def build_realtime_demo_evidence_bundle(config: RealtimeDemoEvidenceBundleConfig) -> dict[str, object]:
    """Write a submission-facing summary of current realtime demo evidence."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    summary_json = config.output_root / "demo_evidence_bundle.json"
    summary_md = config.output_root / "demo_evidence_bundle.md"

    readiness = _read_json_if_exists(config.readiness_summary)
    triage = _read_json_if_exists(config.triage_summary)
    dry_postcapture = _read_json_if_exists(config.dry_run_postcapture_summary)
    dry_composite = _read_json_if_exists(config.dry_run_composite_manifest)
    dry_overlay_contract = _read_json_if_exists(config.dry_run_overlay_contract_summary)
    live_postcapture = _read_json_if_exists(config.live_postcapture_summary)
    live_composite = _read_json_if_exists(config.live_composite_manifest)
    live_overlay_contract = _read_json_if_exists(config.live_overlay_contract_summary)
    live_artifact_freshness = _live_artifact_freshness(config, live_postcapture=live_postcapture)

    evidence = {
        "offline_gates": _offline_gates(readiness),
        "dry_run": _run_evidence(dry_postcapture, dry_composite),
        "live": _run_evidence(live_postcapture, live_composite),
        "live_artifact_freshness": live_artifact_freshness,
        "overlay_contracts": {
            "dry_run": _overlay_contract_evidence(dry_overlay_contract),
            "live": _overlay_contract_evidence(live_overlay_contract),
        },
        "triage": {
            "status": triage.get("status") if triage else None,
            "failure_category": triage.get("failure_category") if triage else None,
            "recommended_next_action": triage.get("recommended_next_action") if triage else None,
        },
    }
    missing = _missing_required_evidence(
        readiness=readiness,
        live_postcapture=live_postcapture,
        live_postcapture_summary=config.live_postcapture_summary,
        live_composite=live_composite,
        live_overlay_contract=live_overlay_contract,
        live_artifact_freshness=live_artifact_freshness,
    )
    ready_for_submission = not missing
    dry_run_ready = _dry_run_ready(
        readiness=readiness,
        dry_postcapture=dry_postcapture,
        dry_composite=dry_composite,
        dry_overlay_contract=dry_overlay_contract,
    )
    status = "ready_for_submission_demo" if ready_for_submission else (
        "awaiting_live_capture" if dry_run_ready and _missing_live_capture_evidence(missing) else "incomplete_demo_evidence"
    )
    level = "live_demo_ready" if ready_for_submission else (
        "dry_run_rehearsal_ready" if dry_run_ready else "insufficient_demo_evidence"
    )
    summary: dict[str, object] = {
        "status": status,
        "ready_for_submission_demo": ready_for_submission,
        "demo_evidence_level": level,
        "missing_required_evidence": missing,
        "evidence": evidence,
        "paths": {
            "readiness_summary": config.readiness_summary.as_posix(),
            "triage_summary": config.triage_summary.as_posix(),
            "dry_run_postcapture_summary": config.dry_run_postcapture_summary.as_posix(),
            "dry_run_composite_manifest": config.dry_run_composite_manifest.as_posix(),
            "dry_run_overlay_contract_summary": config.dry_run_overlay_contract_summary.as_posix(),
            "live_overlay_video": config.live_overlay_video.as_posix(),
            "live_frame_log": config.live_frame_log.as_posix(),
            "live_postcapture_summary": config.live_postcapture_summary.as_posix(),
            "live_composite_manifest": config.live_composite_manifest.as_posix(),
            "live_overlay_contract_summary": config.live_overlay_contract_summary.as_posix(),
            "live_artifact_cleanup_summary": config.live_artifact_cleanup_summary.as_posix(),
        },
        "outputs": {
            "summary_json": summary_json.as_posix(),
            "summary_md": summary_md.as_posix(),
        },
        "claim_scope": "submission evidence summary over existing artifacts; does not run camera capture, inference, or video rendering",
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _offline_gates(readiness: dict[str, Any] | None) -> dict[str, object]:
    checks = _dict_value(readiness, "checks")
    metrics = _dict_value(readiness, "metrics")
    return {
        "original20_passed": checks.get("original20_gate_passed") is True,
        "heldout15_passed": checks.get("heldout15_gate_passed") is True,
        "original20": _count_string(metrics.get("original20_passed_clip_count"), metrics.get("original20_clip_count")),
        "heldout15": _count_string(metrics.get("heldout15_passed_clip_count"), metrics.get("heldout15_clip_count")),
    }


def _run_evidence(postcapture: dict[str, Any] | None, composite: dict[str, Any] | None) -> dict[str, object]:
    gate = _dict_value(postcapture, "demo_success_gate")
    frame_log = _dict_value(postcapture, "frame_log")
    response_decision_frame = _dict_value(postcapture, "response_decision_frame")
    response_prompt_diagnostic_frame = _dict_value(postcapture, "response_prompt_diagnostic_frame")
    outputs = _dict_value(composite, "outputs")
    return {
        "postcapture_status": postcapture.get("status") if postcapture else None,
        "overlay_video": postcapture.get("overlay_video") if postcapture else None,
        "success_gate_passed": gate.get("passed") is True,
        "binary_decision_latency_s": _optional_float(gate.get("binary_decision_latency_s")),
        "detection_rate": _optional_float(frame_log.get("detection_rate")),
        "frame_log_records": frame_log.get("record_count"),
        "response_decision_frame_png": response_decision_frame.get("path"),
        "response_decision_frame_decision": response_decision_frame.get("decision_state"),
        "response_decision_frame_robot_action": response_decision_frame.get("robot_action"),
        "response_decision_frame_confidence": _optional_float(response_decision_frame.get("confidence")),
        "response_prompt_diagnostic_frame_png": response_prompt_diagnostic_frame.get("path"),
        "response_prompt_diagnostic_frame_reason": response_prompt_diagnostic_frame.get("reason"),
        "composite_status": composite.get("status") if composite else None,
        "composite_frame_count": composite.get("frame_count") if composite else None,
        "composite_mp4": outputs.get("mp4"),
    }


def _overlay_contract_evidence(payload: dict[str, Any] | None) -> dict[str, object]:
    metrics = _dict_value(payload, "metrics")
    failures = payload.get("failures") if payload else None
    return {
        "status": payload.get("status") if payload else None,
        "contract_passed": _overlay_contract_passed(payload),
        "overlay_video": payload.get("overlay_video") if payload else None,
        "frame_log_jsonl": payload.get("frame_log_jsonl") if payload else None,
        "failures": failures if isinstance(failures, list) else [],
        "video_frame_count": metrics.get("video_frame_count"),
        "frame_log_records": metrics.get("frame_log_records"),
        "detection_rate": _optional_float(metrics.get("detection_rate")),
        "binary_decision_latency_s": _optional_float(metrics.get("binary_decision_latency_s")),
    }


def _missing_required_evidence(
    *,
    readiness: dict[str, Any] | None,
    live_postcapture: dict[str, Any] | None,
    live_postcapture_summary: Path,
    live_composite: dict[str, Any] | None,
    live_overlay_contract: dict[str, Any] | None,
    live_artifact_freshness: dict[str, object],
) -> list[str]:
    missing: list[str] = []
    checks = _dict_value(readiness, "checks")
    if checks.get("original20_gate_passed") is not True:
        missing.append("original20_gate")
    if checks.get("heldout15_gate_passed") is not True:
        missing.append("heldout15_gate")
    if not _audited_live_launch_control_passed(checks):
        missing.append("audited_live_launch_control")
    if checks.get("live_overlay_exists") is not True:
        missing.append("live_overlay")
    if checks.get("live_demo_success_gate_passed") is not True or _dict_value(live_postcapture, "demo_success_gate").get("passed") is not True:
        missing.append("live_postcapture_success_gate")
    if not _response_decision_frame_ready(live_postcapture, live_postcapture_summary):
        missing.append("live_response_decision_frame")
    if checks.get("live_composite_ready") is not True or not _composite_passed(live_composite):
        missing.append("live_composite")
    if not _overlay_contract_passed(live_overlay_contract):
        missing.append("live_overlay_contract")
    if _live_evidence_appears_complete(
        checks=checks,
        live_postcapture=live_postcapture,
        live_composite=live_composite,
        live_overlay_contract=live_overlay_contract,
    ) and live_artifact_freshness.get("fresh_after_cleanup") is not True:
        missing.append("live_artifacts_fresh_after_cleanup")
    return missing


def _audited_live_launch_control_passed(checks: dict[str, Any]) -> bool:
    return (
        checks.get("one_command_pipeline_available") is True
        and checks.get("prelaunch_audit_ready") is True
        and checks.get("wrapper_contract_probe_passed") is True
        and checks.get("operator_command_audit_passed") is True
    )


def _missing_live_capture_evidence(missing: list[str]) -> bool:
    live_items = {
        "live_overlay",
        "live_postcapture_success_gate",
        "live_composite",
        "live_overlay_contract",
    }
    return any(item in live_items for item in missing)


def _dry_run_ready(
    *,
    readiness: dict[str, Any] | None,
    dry_postcapture: dict[str, Any] | None,
    dry_composite: dict[str, Any] | None,
    dry_overlay_contract: dict[str, Any] | None,
) -> bool:
    checks = _dict_value(readiness, "checks")
    return (
        checks.get("original20_gate_passed") is True
        and checks.get("heldout15_gate_passed") is True
        and (
            checks.get("dry_run_demo_success_gate_passed") is True
            or _dict_value(dry_postcapture, "demo_success_gate").get("passed") is True
        )
        and (checks.get("dry_run_composite_ready") is True or _composite_passed(dry_composite))
        and _overlay_contract_passed(dry_overlay_contract)
    )


def _composite_passed(payload: dict[str, Any] | None) -> bool:
    return bool(payload is not None and payload.get("status") == "passed")


def _overlay_contract_passed(payload: dict[str, Any] | None) -> bool:
    return bool(payload is not None and payload.get("contract_passed") is True)


def _live_evidence_appears_complete(
    *,
    checks: dict[str, Any],
    live_postcapture: dict[str, Any] | None,
    live_composite: dict[str, Any] | None,
    live_overlay_contract: dict[str, Any] | None,
) -> bool:
    return (
        checks.get("live_overlay_exists") is True
        and checks.get("live_demo_success_gate_passed") is True
        and _dict_value(live_postcapture, "demo_success_gate").get("passed") is True
        and checks.get("live_composite_ready") is True
        and _composite_passed(live_composite)
        and _overlay_contract_passed(live_overlay_contract)
    )


def _response_decision_frame_ready(live_postcapture: dict[str, Any] | None, live_postcapture_summary: Path) -> bool:
    return _resolve_response_decision_frame_path(live_postcapture, live_postcapture_summary) is not None


def _resolve_response_decision_frame_path(
    live_postcapture: dict[str, Any] | None, live_postcapture_summary: Path
) -> Path | None:
    response_decision_frame = _dict_value(live_postcapture, "response_decision_frame")
    path_value = response_decision_frame.get("path")
    if not isinstance(path_value, str) or not path_value:
        return None
    frame_path = Path(path_value)
    if frame_path.exists() and frame_path.is_file():
        return frame_path
    if frame_path.is_absolute():
        return None
    sibling_frame_path = live_postcapture_summary.parent / frame_path
    if sibling_frame_path.exists() and sibling_frame_path.is_file():
        return sibling_frame_path
    return None


def _live_artifact_freshness(
    config: RealtimeDemoEvidenceBundleConfig, *, live_postcapture: dict[str, Any] | None
) -> dict[str, object]:
    cleanup = config.live_artifact_cleanup_summary
    response_decision_frame = _resolve_response_decision_frame_path(live_postcapture, config.live_postcapture_summary)
    live_paths = _unique_paths(
        [
        config.live_overlay_video,
        config.live_frame_log,
        config.live_postcapture_summary,
        config.live_composite_manifest,
        config.live_overlay_contract_summary,
        ]
        + ([response_decision_frame] if response_decision_frame is not None else [])
    )
    if not cleanup.exists():
        return {
            "cleanup_summary": cleanup.as_posix(),
            "cleanup_status": None,
            "cleanup_mtime": None,
            "response_decision_frame_path": response_decision_frame.as_posix() if response_decision_frame else None,
            "fresh_after_cleanup": False,
            "stale_paths": [path.as_posix() for path in live_paths if path.exists()],
            "missing_paths": [path.as_posix() for path in live_paths if not path.exists()],
        }
    cleanup_payload = _read_json_if_exists(cleanup) or {}
    cleanup_mtime = cleanup.stat().st_mtime
    stale_paths = [path.as_posix() for path in live_paths if path.exists() and path.stat().st_mtime <= cleanup_mtime]
    missing_paths = [path.as_posix() for path in live_paths if not path.exists()]
    return {
        "cleanup_summary": cleanup.as_posix(),
        "cleanup_status": cleanup_payload.get("cleanup_status"),
        "cleanup_mtime": cleanup_mtime,
        "response_decision_frame_path": response_decision_frame.as_posix() if response_decision_frame else None,
        "fresh_after_cleanup": cleanup_payload.get("cleanup_status") == "cleared" and not stale_paths and not missing_paths,
        "stale_paths": stale_paths,
        "missing_paths": missing_paths,
    }


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = path.resolve(strict=False).as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _dict_value(payload: dict[str, Any] | None, key: str) -> dict[str, Any]:
    value = payload.get(key) if payload else None
    return value if isinstance(value, dict) else {}


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _count_string(passed: object, total: object) -> str | None:
    if passed is None or total is None:
        return None
    return f"{passed}/{total}"


def _summary_markdown(summary: dict[str, object]) -> str:
    evidence = summary.get("evidence", {})
    missing = summary.get("missing_required_evidence", [])
    lines = [
        "# Realtime Demo Evidence Bundle",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Ready for submission demo: `{summary.get('ready_for_submission_demo')}`",
        f"- Demo evidence level: `{summary.get('demo_evidence_level')}`",
        "",
        "## Missing Required Evidence",
        "",
    ]
    if isinstance(missing, list) and missing:
        lines.extend(f"- `{item}`" for item in missing)
    else:
        lines.append("- None")
    lines.extend(["", "## Evidence", ""])
    if isinstance(evidence, dict):
        lines.append("```json")
        lines.append(json.dumps(evidence, indent=2))
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoEvidenceBundleConfig", "build_realtime_demo_evidence_bundle"]
