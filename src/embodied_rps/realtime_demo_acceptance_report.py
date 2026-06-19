"""Final acceptance report for realtime RPS demo evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LIVE_PIPELINE_COMMAND = (
    "powershell -ExecutionPolicy Bypass -File "
    "artifacts\\realtime_demo_launch_20260616\\24_run_live_demo_operator_confirmed_strict.ps1"
)


@dataclass(frozen=True)
class RealtimeDemoAcceptanceReportConfig:
    """Artifact paths used to build the final live-demo acceptance report."""

    output_root: Path
    evidence_bundle: Path = Path("artifacts/realtime_demo_evidence_bundle_20260616/demo_evidence_bundle.json")
    review_packet_manifest: Path = Path("artifacts/realtime_demo_review_packet_20260616/review_packet_manifest.json")
    dry_run_overlay_contract_summary: Path = Path(
        "artifacts/realtime_demo_overlay_contract_dryrun_20260616/overlay_contract_summary.json"
    )
    live_overlay_contract_summary: Path = Path("artifacts/realtime_demo_overlay_contract_20260616/overlay_contract_summary.json")


def build_realtime_demo_acceptance_report(config: RealtimeDemoAcceptanceReportConfig) -> dict[str, object]:
    """Summarize whether the current artifacts can support final demo claims."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    output_json = config.output_root / "acceptance_report.json"
    output_md = config.output_root / "acceptance_report.md"

    evidence_bundle = _read_json_if_exists(config.evidence_bundle) or {}
    review_packet = _read_json_if_exists(config.review_packet_manifest)
    dry_contract = _read_json_if_exists(config.dry_run_overlay_contract_summary)
    live_contract = _read_json_if_exists(config.live_overlay_contract_summary)

    requirement_status = _requirement_status(
        evidence_bundle=evidence_bundle,
        review_packet=review_packet,
        dry_contract=dry_contract,
        live_contract=live_contract,
    )
    blocking = _blocking_requirements(requirement_status)
    ready_for_youtube = (
        evidence_bundle.get("ready_for_submission_demo") is True
        and not blocking
        and requirement_status["live_overlay_contract"]["status"] == "passed"
    )
    status = _overall_status(ready_for_youtube=ready_for_youtube, blocking=blocking, requirement_status=requirement_status)
    summary: dict[str, object] = {
        "status": status,
        "ready_for_youtube_demo": ready_for_youtube,
        "requirement_status": requirement_status,
        "blocking_requirements": blocking,
        "next_operator_command": _next_operator_command(status=status, blocking=blocking),
        "evidence_level": evidence_bundle.get("demo_evidence_level"),
        "source_status": {
            "evidence_bundle": evidence_bundle.get("status"),
            "review_packet": review_packet.get("status") if review_packet else None,
            "dry_overlay_contract": dry_contract.get("status") if dry_contract else None,
            "live_overlay_contract": live_contract.get("status") if live_contract else None,
        },
        "paths": {
            "evidence_bundle": config.evidence_bundle.as_posix(),
            "review_packet_manifest": config.review_packet_manifest.as_posix(),
            "dry_run_overlay_contract_summary": config.dry_run_overlay_contract_summary.as_posix(),
            "live_overlay_contract_summary": config.live_overlay_contract_summary.as_posix(),
        },
        "outputs": {
            "summary_json": output_json.as_posix(),
            "summary_md": output_md.as_posix(),
        },
        "claim_scope": "acceptance report over existing evidence artifacts; does not run camera capture, model inference, or rendering",
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _requirement_status(
    *,
    evidence_bundle: dict[str, Any],
    review_packet: dict[str, Any] | None,
    dry_contract: dict[str, Any] | None,
    live_contract: dict[str, Any] | None,
) -> dict[str, dict[str, object]]:
    evidence = _dict_value(evidence_bundle, "evidence")
    offline = _dict_value(evidence, "offline_gates")
    dry_run = _dict_value(evidence, "dry_run")
    live = _dict_value(evidence, "live")
    return {
        "submission_evidence_bundle": _evidence_bundle_status(evidence_bundle),
        "offline_original20_gate": _status_item(
            passed=offline.get("original20_passed") is True,
            detail=f"original20={offline.get('original20')}",
            when_missing="failed",
        ),
        "offline_heldout15_gate": _status_item(
            passed=offline.get("heldout15_passed") is True,
            detail=f"heldout15={offline.get('heldout15')}",
            when_missing="failed",
        ),
        "dry_run_success_gate": _status_item(
            passed=dry_run.get("success_gate_passed") is True,
            detail=f"latency_s={dry_run.get('binary_decision_latency_s')}",
            when_missing="failed",
        ),
        "dry_run_composite": _status_item(
            passed=dry_run.get("composite_status") == "passed",
            detail=f"frames={dry_run.get('composite_frame_count')}",
            when_missing="failed",
        ),
        "dry_run_overlay_contract": _contract_status(dry_contract, missing_status="failed"),
        "live_camera_capture": _status_item(
            passed=isinstance(live.get("overlay_video"), str) and bool(live.get("overlay_video")),
            detail=f"overlay={live.get('overlay_video')}",
            when_missing="awaiting",
        ),
        "live_success_gate": _status_item(
            passed=live.get("success_gate_passed") is True,
            detail=f"latency_s={live.get('binary_decision_latency_s')}",
            when_missing="awaiting",
        ),
        "live_composite": _status_item(
            passed=live.get("composite_status") == "passed",
            detail=f"mp4={live.get('composite_mp4')}",
            when_missing="awaiting",
        ),
        "live_overlay_contract": _contract_status(live_contract, missing_status="awaiting"),
        "review_packet": _status_item(
            passed=review_packet is not None,
            detail=f"status={review_packet.get('status') if review_packet else None}",
            when_missing="awaiting",
        ),
    }


def _status_item(*, passed: bool, detail: str, when_missing: str) -> dict[str, object]:
    return {
        "status": "passed" if passed else when_missing,
        "passed": passed,
        "detail": detail,
    }


def _contract_status(payload: dict[str, Any] | None, *, missing_status: str) -> dict[str, object]:
    checks = _dict_value(payload, "checks")
    metrics = _dict_value(payload, "metrics")
    failures = payload.get("failures") if payload else None
    passed = payload is not None and payload.get("contract_passed") is True
    if passed:
        status = "passed"
    elif payload is None:
        status = missing_status
    else:
        status = "failed"
    return {
        "status": status,
        "passed": passed,
        "detail": {
            "failures": failures if isinstance(failures, list) else [],
            "binary_decision_latency_s": metrics.get("binary_decision_latency_s"),
            "detection_rate": metrics.get("detection_rate"),
            "prompt_cycle_present": checks.get("prompt_cycle_present"),
            "probabilities_present": checks.get("probabilities_present"),
            "confidence_margin_present": checks.get("confidence_margin_present"),
            "transition_mass_present": checks.get("transition_mass_present"),
            "robot_action_present": checks.get("robot_action_present"),
        },
    }


def _evidence_bundle_status(payload: dict[str, Any]) -> dict[str, object]:
    missing = payload.get("missing_required_evidence")
    missing_list = missing if isinstance(missing, list) else []
    passed = payload.get("ready_for_submission_demo") is True and not missing_list
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "detail": {
            "status": payload.get("status"),
            "ready_for_submission_demo": payload.get("ready_for_submission_demo"),
            "missing_required_evidence": missing_list,
        },
    }


def _blocking_requirements(requirement_status: dict[str, dict[str, object]]) -> list[str]:
    required = [
        "submission_evidence_bundle",
        "offline_original20_gate",
        "offline_heldout15_gate",
        "dry_run_success_gate",
        "dry_run_composite",
        "dry_run_overlay_contract",
        "live_camera_capture",
        "live_success_gate",
        "live_composite",
        "live_overlay_contract",
        "review_packet",
    ]
    return [name for name in required if requirement_status[name].get("status") != "passed"]


def _overall_status(
    *,
    ready_for_youtube: bool,
    blocking: list[str],
    requirement_status: dict[str, dict[str, object]],
) -> str:
    if ready_for_youtube:
        return "ready_for_youtube_demo"
    dry_ready = all(
        requirement_status[name].get("status") == "passed"
        for name in (
            "offline_original20_gate",
            "offline_heldout15_gate",
            "dry_run_success_gate",
            "dry_run_composite",
            "dry_run_overlay_contract",
        )
    )
    if dry_ready and "live_camera_capture" in blocking:
        return "awaiting_live_capture"
    if dry_ready:
        return "live_demo_needs_repair"
    return "incomplete_demo_evidence"


def _next_operator_command(*, status: str, blocking: list[str]) -> str:
    if status == "ready_for_youtube_demo":
        return "inspect live composite and prepare final video upload"
    if "live_camera_capture" in blocking:
        return LIVE_PIPELINE_COMMAND
    if status == "live_demo_needs_repair":
        return "inspect acceptance_report.md and rerun the failed live post-processing stage"
    return "refresh readiness and evidence bundle artifacts before live rehearsal"


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _dict_value(payload: dict[str, Any] | None, key: str) -> dict[str, Any]:
    value = payload.get(key) if payload else None
    return value if isinstance(value, dict) else {}


def _summary_markdown(summary: dict[str, object]) -> str:
    statuses = summary.get("requirement_status", {})
    blocking = summary.get("blocking_requirements", [])
    lines = [
        "# Realtime Demo Acceptance Report",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Ready for YouTube demo: `{summary.get('ready_for_youtube_demo')}`",
        f"- Evidence level: `{summary.get('evidence_level')}`",
        f"- Next operator command: `{summary.get('next_operator_command')}`",
        "",
        "## Requirements",
        "",
        "| Requirement | Status | Detail |",
        "|---|---|---|",
    ]
    if isinstance(statuses, dict):
        for name, value in statuses.items():
            detail = value.get("detail") if isinstance(value, dict) else None
            status = value.get("status") if isinstance(value, dict) else "unknown"
            lines.append(f"| `{name}` | `{status}` | `{json.dumps(detail, ensure_ascii=False)}` |")
    lines.extend(["", "## Blocking Requirements", ""])
    if isinstance(blocking, list) and blocking:
        lines.extend(f"- `{item}`" for item in blocking)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "LIVE_PIPELINE_COMMAND",
    "RealtimeDemoAcceptanceReportConfig",
    "build_realtime_demo_acceptance_report",
]
