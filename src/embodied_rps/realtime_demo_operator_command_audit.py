"""Audit operator-facing realtime demo commands for strict-wrapper drift."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STRICT_WRAPPER_SCRIPT = "24_run_live_demo_operator_confirmed_strict.ps1"
RAW_PIPELINE_SCRIPT = "07_run_live_demo_pipeline.ps1"


@dataclass(frozen=True)
class RealtimeDemoOperatorCommandAuditConfig:
    """Artifact paths used to audit operator-facing live-demo commands."""

    output_root: Path = Path("artifacts/realtime_demo_operator_command_audit_20260616")
    triage_summary: Path = Path("artifacts/realtime_demo_triage_20260616/triage_summary.json")
    review_packet_manifest: Path = Path("artifacts/realtime_demo_review_packet_20260616/review_packet_manifest.json")
    acceptance_report: Path = Path("artifacts/realtime_demo_acceptance_report_20260616/acceptance_report.json")
    operator_outcome: Path = Path("artifacts/realtime_demo_operator_outcome_20260616/operator_outcome.json")
    live_run_checklist: Path = Path("artifacts/realtime_demo_live_run_checklist_20260616/live_run_checklist.json")
    final_run_card: Path = Path("artifacts/realtime_demo_final_run_card_20260616/final_run_card.json")
    learning_queue: Path = Path("artifacts/realtime_demo_learning_queue_20260616/learning_queue.json")
    goal_progress_audit: Path = Path("artifacts/realtime_demo_goal_progress_audit_20260616/goal_progress_audit.json")
    live_status_snapshot: Path = Path("artifacts/realtime_demo_live_status_snapshot_20260616/live_status_snapshot.json")
    operator_handoff_card: Path = Path("artifacts/realtime_demo_operator_handoff_card_20260616/operator_handoff_card.json")


def audit_realtime_demo_operator_commands(config: RealtimeDemoOperatorCommandAuditConfig) -> dict[str, object]:
    """Write a drift audit for user-facing live-demo command fields."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    output_json = config.output_root / "operator_command_audit.json"
    output_md = config.output_root / "operator_command_audit.md"
    operator_outcome_payload = _read_json_if_exists(config.operator_outcome)
    checks = [
        _check_command(
            check_id="triage_summary.recommended_next_action",
            path=config.triage_summary,
            command=_nested_value(_read_json_if_exists(config.triage_summary), ("recommended_next_action",)),
        ),
        _check_command(
            check_id="review_packet_manifest.operator_commands.live_pipeline",
            path=config.review_packet_manifest,
            command=_nested_value(
                _read_json_if_exists(config.review_packet_manifest), ("operator_commands", "live_pipeline")
            ),
        ),
        _check_command(
            check_id="acceptance_report.next_operator_command",
            path=config.acceptance_report,
            command=_nested_value(_read_json_if_exists(config.acceptance_report), ("next_operator_command",)),
        ),
        _check_command(
            check_id="operator_outcome.primary_command",
            path=config.operator_outcome,
            command=_nested_value(operator_outcome_payload, ("primary_command",)),
            payload=operator_outcome_payload,
        ),
        _check_command(
            check_id="live_run_checklist.live_pipeline_command",
            path=config.live_run_checklist,
            command=_nested_value(_read_json_if_exists(config.live_run_checklist), ("live_pipeline_command",)),
        ),
        _check_command(
            check_id="final_run_card.primary_command",
            path=config.final_run_card,
            command=_nested_value(_read_json_if_exists(config.final_run_card), ("primary_command",)),
        ),
        _check_command(
            check_id="learning_queue.source_evidence.primary_command",
            path=config.learning_queue,
            command=_nested_value(_read_json_if_exists(config.learning_queue), ("source_evidence", "primary_command")),
        ),
        _check_command(
            check_id="goal_progress_audit.next_action",
            path=config.goal_progress_audit,
            command=_nested_value(_read_json_if_exists(config.goal_progress_audit), ("next_action",)),
        ),
        _check_command(
            check_id="live_status_snapshot.recommended_command",
            path=config.live_status_snapshot,
            command=_nested_value(_read_json_if_exists(config.live_status_snapshot), ("recommended_command",)),
        ),
        _check_command(
            check_id="operator_handoff_card.recommended_command",
            path=config.operator_handoff_card,
            command=_nested_value(_read_json_if_exists(config.operator_handoff_card), ("recommended_command",)),
        ),
    ]
    failures = [str(check["id"]) for check in checks if check["passed"] is not True]
    summary: dict[str, object] = {
        "audit_status": "passed" if not failures else "failed",
        "check_count": len(checks),
        "passed_count": sum(1 for check in checks if check["passed"] is True),
        "failures": failures,
        "strict_wrapper_script": STRICT_WRAPPER_SCRIPT,
        "raw_pipeline_script": RAW_PIPELINE_SCRIPT,
        "checks": checks,
        "outputs": {
            "summary_json": output_json.as_posix(),
            "summary_md": output_md.as_posix(),
        },
        "claim_scope": (
            "operator-command drift audit over existing JSON artifacts; does not run camera capture, "
            "model inference, rendering, training, upload, or repository edits"
        ),
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(_audit_markdown(summary), encoding="utf-8")
    return summary


def _check_command(
    *,
    check_id: str,
    path: Path,
    command: object,
    payload: dict[str, Any] | None = None,
) -> dict[str, object]:
    command_text = command if isinstance(command, str) else None
    completion_state_without_command = command_text is None and _is_completion_state_without_command(payload)
    missing_without_command = command_text is None and not completion_state_without_command
    contains_strict = command_text is not None and STRICT_WRAPPER_SCRIPT in command_text
    contains_raw = command_text is not None and RAW_PIPELINE_SCRIPT in command_text
    if completion_state_without_command:
        command_kind = "completion_state_no_command"
    elif missing_without_command:
        command_kind = "missing_no_operator_command"
    else:
        command_kind = _command_kind(command_text)
    passed = (
        completion_state_without_command
        or missing_without_command
        or (
            command_text is not None
            and not contains_raw
            and (command_kind == "guidance_text" or contains_strict)
        )
    )
    if completion_state_without_command or missing_without_command:
        failure_reason = None
    elif contains_raw:
        failure_reason = "raw_pipeline_command_exposed"
    elif command_kind == "operator_command" and not contains_strict:
        failure_reason = "strict_wrapper_missing"
    else:
        failure_reason = None
    return {
        "id": check_id,
        "path": path.as_posix(),
        "command": command_text,
        "command_kind": command_kind,
        "passed": passed,
        "contains_strict_wrapper": contains_strict,
        "contains_raw_pipeline": contains_raw,
        "failure_reason": failure_reason,
    }


def _command_kind(command_text: str | None) -> str:
    if command_text is None:
        return "missing"
    lowered = command_text.lower()
    command_markers = (
        "powershell",
        ".ps1",
        "python ",
        "python.exe",
        " -m ",
        "embodied_rps.tools",
    )
    if any(marker in lowered for marker in command_markers):
        return "operator_command"
    return "guidance_text"


def _is_completion_state_without_command(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("operator_state") == "ready_for_final_video"
        and payload.get("primary_action_type") == "inspect_and_package_demo"
        and payload.get("primary_command") is None
    )


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _nested_value(payload: dict[str, Any], keys: tuple[str, ...]) -> object:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _audit_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Realtime Demo Operator Command Audit",
        "",
        f"- Audit status: `{summary.get('audit_status')}`",
        f"- Check count: `{summary.get('check_count')}`",
        f"- Passed count: `{summary.get('passed_count')}`",
        f"- Strict wrapper script: `{summary.get('strict_wrapper_script')}`",
        f"- Raw pipeline script: `{summary.get('raw_pipeline_script')}`",
        "",
        "## Failures",
        "",
    ]
    failures = summary.get("failures", [])
    if isinstance(failures, list) and failures:
        lines.extend(f"- `{failure}`" for failure in failures)
    else:
        lines.append("- None")
    lines.extend(["", "## Checks", "", "| Check | Passed | Failure | Command |", "|---|---:|---|---|"])
    checks = summary.get("checks", [])
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            command = str(check.get("command")).replace("|", "\\|")
            lines.append(
                f"| `{check.get('id')}` | `{check.get('passed')}` | "
                f"`{check.get('failure_reason')}` | `{command}` |"
            )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "RAW_PIPELINE_SCRIPT",
    "STRICT_WRAPPER_SCRIPT",
    "RealtimeDemoOperatorCommandAuditConfig",
    "audit_realtime_demo_operator_commands",
]
