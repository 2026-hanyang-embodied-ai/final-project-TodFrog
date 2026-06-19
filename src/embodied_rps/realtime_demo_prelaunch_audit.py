"""Prelaunch audit for the realtime RPS demo operator command."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeDemoPrelaunchAuditConfig:
    """Input and output paths for prelaunch audit generation."""

    output_root: Path = Path("artifacts/realtime_demo_prelaunch_audit_20260616")
    live_status_snapshot: Path = Path("artifacts/realtime_demo_live_status_snapshot_20260616/live_status_snapshot.json")
    operator_handoff_card: Path = Path("artifacts/realtime_demo_operator_handoff_card_20260616/operator_handoff_card.json")
    launch_summary: Path = Path("artifacts/realtime_demo_launch_20260616/demo_launch_summary.json")


REQUIRED_SCRIPT_KEYS = (
    "run_live_demo_operator_confirmed_strict",
    "check_prelaunch_audit",
    "check_operator_command_audit",
    "check_guarded_retake_readiness",
    "run_live_demo_operator_confirmed",
    "check_goal_progress_strict",
    "build_live_status_snapshot",
    "build_operator_handoff_card",
    "clear_stale_live_artifacts",
)


def build_realtime_demo_prelaunch_audit(config: RealtimeDemoPrelaunchAuditConfig) -> dict[str, object]:
    """Write a strict prelaunch audit for the recommended live command."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    output_json = config.output_root / "prelaunch_audit.json"
    output_md = config.output_root / "prelaunch_audit.md"
    snapshot = _read_json_if_exists(config.live_status_snapshot) or {}
    handoff = _read_json_if_exists(config.operator_handoff_card) or {}
    launch_summary = _read_json_if_exists(config.launch_summary) or {}
    audit = _prelaunch_audit(
        snapshot=snapshot,
        handoff=handoff,
        launch_summary=launch_summary,
        config=config,
        output_json=output_json,
        output_md=output_md,
    )
    output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    output_md.write_text(_prelaunch_markdown(audit), encoding="utf-8")
    return audit


def _prelaunch_audit(
    *,
    snapshot: dict[str, Any],
    handoff: dict[str, Any],
    launch_summary: dict[str, Any],
    config: RealtimeDemoPrelaunchAuditConfig,
    output_json: Path,
    output_md: Path,
) -> dict[str, object]:
    scripts = _dict_value(launch_summary, "scripts")
    recommended_command = _recommended_command(snapshot, handoff, scripts)
    checks = _build_checks(snapshot=snapshot, handoff=handoff, launch_summary=launch_summary, scripts=scripts)
    blocking_issues = [str(check["detail"]) for check in checks if check.get("passed") is not True]
    completion_candidate = snapshot.get("completion_candidate") is True or handoff.get("completion_candidate") is True
    ready = not blocking_issues and not completion_candidate
    if completion_candidate and not blocking_issues:
        status = "completion_candidate_already_available"
    elif ready:
        status = "ready_for_operator_live_attempt"
    else:
        status = "prelaunch_blocked"
    return {
        "prelaunch_status": status,
        "ready_for_operator_live_attempt": ready,
        "completion_candidate": completion_candidate,
        "recommended_command": recommended_command,
        "blocking_issues": blocking_issues,
        "warnings": [],
        "checks": checks,
        "inputs": {
            "live_status_snapshot": config.live_status_snapshot.as_posix(),
            "operator_handoff_card": config.operator_handoff_card.as_posix(),
            "launch_summary": config.launch_summary.as_posix(),
        },
        "outputs": {
            "prelaunch_audit_json": output_json.as_posix(),
            "prelaunch_audit_md": output_md.as_posix(),
        },
        "claim_scope": (
            "prelaunch audit over existing status and launch artifacts; does not run camera capture, "
            "inference, rendering, training, upload, or repository edits"
        ),
    }


def _build_checks(
    *,
    snapshot: dict[str, Any],
    handoff: dict[str, Any],
    launch_summary: dict[str, Any],
    scripts: dict[str, Any],
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    _add_check(
        checks,
        "live_status_snapshot_loaded",
        bool(snapshot),
        "live status snapshot JSON is missing or empty",
    )
    _add_check(
        checks,
        "operator_handoff_card_loaded",
        bool(handoff),
        "operator handoff card JSON is missing or empty",
    )
    _add_check(
        checks,
        "launch_summary_loaded",
        bool(launch_summary),
        "launch summary JSON is missing or empty",
    )
    completion_candidate = snapshot.get("completion_candidate") is True or handoff.get("completion_candidate") is True
    if not completion_candidate:
        _add_check(
            checks,
            "status_is_awaiting_live_capture",
            snapshot.get("snapshot_status") == "awaiting_live_capture"
            and handoff.get("handoff_status") == "awaiting_live_capture",
            "snapshot and handoff status must both be awaiting_live_capture before a live attempt",
        )
    script_count = launch_summary.get("script_count")
    _add_check(
        checks,
        "script_count_matches_summary",
        isinstance(script_count, int) and script_count == len(scripts),
        "launch summary script_count does not match the scripts map length",
    )
    for key in REQUIRED_SCRIPT_KEYS:
        value = scripts.get(key)
        _add_check(
            checks,
            f"script_key_present_{key}",
            isinstance(value, str) and bool(value),
            f"required launch script key missing: {key}",
        )
        if isinstance(value, str) and value:
            _add_check(
                checks,
                f"script_file_exists_{key}",
                Path(value).exists(),
                f"required launch script file is missing: {key} -> {value}",
            )
    _add_path_check(checks, launch_summary, "python_executable", "python executable")
    _add_path_check(checks, launch_summary, "config_path", "realtime demo config")
    _add_path_check(checks, launch_summary, "sample_video", "dry-run sample video")
    _add_path_check(checks, launch_summary, "response_preview_image", "SCHUNK response preview image")
    _add_check(
        checks,
        "recommended_command_targets_strict_wrapper",
        "24_run_live_demo_operator_confirmed_strict.ps1" in _recommended_command(snapshot, handoff, scripts),
        "recommended command does not target 24_run_live_demo_operator_confirmed_strict.ps1",
    )
    strict_script = scripts.get("run_live_demo_operator_confirmed_strict")
    if isinstance(strict_script, str) and strict_script and Path(strict_script).exists():
        content = Path(strict_script).read_text(encoding="utf-8-sig")
        _add_check(
            checks,
            "strict_wrapper_refreshes_status_before_prelaunch_and_capture",
            _contains_in_order(
                content,
                [
                    "30_clear_stale_live_artifacts.ps1",
                    "08_triage_live_capture.ps1",
                    "13_build_operator_outcome_report.ps1",
                    "18_build_live_run_checklist.ps1",
                    "19_build_final_run_card.ps1",
                    "20_build_learning_queue.ps1",
                    "21_build_goal_progress_audit.ps1",
                    "25_build_live_status_snapshot.ps1",
                    "26_build_operator_handoff_card.ps1",
                    "27_check_prelaunch_audit.ps1",
                    "29_check_operator_command_audit.ps1",
                    "32_check_guarded_retake_readiness.ps1",
                    "22_run_live_demo_operator_confirmed.ps1",
                    "23_check_goal_progress_strict.ps1",
                    "25_build_live_status_snapshot.ps1",
                    "26_build_operator_handoff_card.ps1",
                ],
            ),
            "strict wrapper does not refresh stale pre-capture state before prelaunch and capture",
        )
        _add_check(
            checks,
            "strict_wrapper_prints_post_run_handoff_summary",
            "operator_handoff_card.md" in content
            and _contains_in_order(
                content,
                [
                    "26_build_operator_handoff_card.ps1",
                    "--- Operator handoff summary ---",
                    "Get-Content $handoffMarkdownPath",
                ],
            ),
            "strict wrapper does not print the refreshed operator handoff summary after live execution",
        )
    return checks


def _add_path_check(checks: list[dict[str, object]], launch_summary: dict[str, Any], key: str, label: str) -> None:
    value = launch_summary.get(key)
    _add_check(
        checks,
        f"{key}_exists",
        isinstance(value, str) and bool(value) and Path(value).exists(),
        f"{label} path is missing or does not exist: {value}",
    )


def _add_check(checks: list[dict[str, object]], check_id: str, passed: bool, detail: str) -> None:
    checks.append({"id": check_id, "passed": bool(passed), "detail": "ok" if passed else detail})


def _recommended_command(snapshot: dict[str, Any], handoff: dict[str, Any], scripts: dict[str, Any]) -> str:
    command = snapshot.get("recommended_command") or handoff.get("recommended_command")
    if command:
        return str(command)
    strict_script = scripts.get("run_live_demo_operator_confirmed_strict")
    if strict_script:
        return f"powershell -ExecutionPolicy Bypass -File {strict_script}"
    return ""


def _contains_in_order(content: str, needles: list[str]) -> bool:
    offset = 0
    for needle in needles:
        found = content.find(needle, offset)
        if found < 0:
            return False
        offset = found + len(needle)
    return True


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _prelaunch_markdown(audit: dict[str, object]) -> str:
    lines = [
        "# Realtime Demo Prelaunch Audit",
        "",
        f"- Prelaunch status: `{audit.get('prelaunch_status')}`",
        f"- Ready for operator live attempt: `{audit.get('ready_for_operator_live_attempt')}`",
        f"- Completion candidate: `{audit.get('completion_candidate')}`",
        f"- Recommended command: `{audit.get('recommended_command')}`",
        "",
        "## Blocking Issues",
        "",
    ]
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


__all__ = ["RealtimeDemoPrelaunchAuditConfig", "build_realtime_demo_prelaunch_audit"]
