"""Goal-level progress audit for the few-shot realtime RPS demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeDemoGoalProgressAuditConfig:
    """Input and output paths for objective-level progress auditing."""

    output_root: Path = Path("artifacts/realtime_demo_goal_progress_audit_20260616")
    readiness_summary: Path = Path("artifacts/realtime_demo_readiness_20260616/readiness_summary.json")
    evidence_bundle: Path = Path("artifacts/realtime_demo_evidence_bundle_20260616/demo_evidence_bundle.json")
    final_run_card: Path = Path("artifacts/realtime_demo_final_run_card_20260616/final_run_card.json")
    learning_queue: Path = Path("artifacts/realtime_demo_learning_queue_20260616/learning_queue.json")
    final_candidate: Path = Path("artifacts/realtime_demo_final_candidate_20260616/final_demo_candidate.json")
    archive_index: Path = Path("artifacts/realtime_demo_run_archive_20260616/run_archive_index.json")


def build_realtime_demo_goal_progress_audit(config: RealtimeDemoGoalProgressAuditConfig) -> dict[str, object]:
    """Write a requirement-by-requirement audit of the active realtime demo goal."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    output_json = config.output_root / "goal_progress_audit.json"
    output_md = config.output_root / "goal_progress_audit.md"
    readiness = _read_json_if_exists(config.readiness_summary) or {}
    evidence_bundle = _read_json_if_exists(config.evidence_bundle) or {}
    final_card = _read_json_if_exists(config.final_run_card) or {}
    learning_queue = _read_json_if_exists(config.learning_queue) or {}
    final_candidate = _read_json_if_exists(config.final_candidate) or {}
    archive_index = _read_json_if_exists(config.archive_index) or {}

    requirements = _requirements(
        readiness=readiness,
        evidence_bundle=evidence_bundle,
        final_card=final_card,
        learning_queue=learning_queue,
        final_candidate=final_candidate,
        archive_index=archive_index,
    )
    missing = [name for name, item in requirements.items() if item.get("status") != "passed"]
    goal_complete = not missing
    summary: dict[str, object] = {
        "goal_status": "complete_evidence_ready" if goal_complete else _incomplete_status(final_card, learning_queue),
        "goal_complete": goal_complete,
        "missing_requirements": missing,
        "requirements": requirements,
        "next_action": _next_action(goal_complete=goal_complete, final_card=final_card, learning_queue=learning_queue),
        "next_action_absolute": _next_action_absolute(goal_complete=goal_complete, final_card=final_card),
        "inputs": {
            "readiness_summary": config.readiness_summary.as_posix(),
            "evidence_bundle": config.evidence_bundle.as_posix(),
            "final_run_card": config.final_run_card.as_posix(),
            "learning_queue": config.learning_queue.as_posix(),
            "final_candidate": config.final_candidate.as_posix(),
            "archive_index": config.archive_index.as_posix(),
        },
        "outputs": {
            "audit_json": output_json.as_posix(),
            "audit_md": output_md.as_posix(),
        },
        "claim_scope": "objective-level audit over existing artifacts; does not run capture, inference, rendering, training, upload, or repository edits",
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(_audit_markdown(summary), encoding="utf-8")
    return summary


def _requirements(
    *,
    readiness: dict[str, Any],
    evidence_bundle: dict[str, Any],
    final_card: dict[str, Any],
    learning_queue: dict[str, Any],
    final_candidate: dict[str, Any],
    archive_index: dict[str, Any],
) -> dict[str, dict[str, object]]:
    checks = _dict_value(readiness, "checks")
    evidence = _dict_value(evidence_bundle, "evidence")
    offline = _dict_value(evidence, "offline_gates")
    dry_run = _dict_value(evidence, "dry_run")
    live = _dict_value(evidence, "live")
    overlay_contracts = _dict_value(evidence, "overlay_contracts")
    dry_contract = _dict_value(overlay_contracts, "dry_run")
    live_contract = _dict_value(overlay_contracts, "live")
    approved_archive_candidate = _approved_archive_candidate(archive_index=archive_index, final_candidate=final_candidate)
    return {
        "few_shot_predictor": _item(
            passed=offline.get("original20_passed") is True and offline.get("heldout15_passed") is True,
            evidence="offline gates passed with the current skeleton predictor",
        ),
        "offline_mp4_gates": _item(
            passed=checks.get("original20_gate_passed") is True and checks.get("heldout15_gate_passed") is True,
            evidence=f"original20={offline.get('original20')}, heldout15={offline.get('heldout15')}",
        ),
        "audited_live_launch_control": _item(
            passed=(
                checks.get("one_command_pipeline_available") is True
                and checks.get("prelaunch_audit_ready") is True
                and checks.get("wrapper_contract_probe_passed") is True
                and checks.get("operator_command_audit_passed") is True
            ),
            evidence=(
                f"pipeline={checks.get('one_command_pipeline_available')}, "
                f"prelaunch={checks.get('prelaunch_audit_ready')}, "
                f"wrapper_contract={checks.get('wrapper_contract_probe_passed')}, "
                f"operator_command_audit={checks.get('operator_command_audit_passed')}"
            ),
        ),
        "dry_run_demo": _item(
            passed=dry_run.get("success_gate_passed") is True and dry_run.get("composite_status") == "passed",
            evidence=f"dry_run_success={dry_run.get('success_gate_passed')}, dry_run_composite={dry_run.get('composite_status')}",
        ),
        "prompt_timing_and_overlay_contract": _item(
            passed=dry_contract.get("contract_passed") is True,
            evidence=f"dry_run_overlay_contract={dry_contract.get('contract_passed')}, live_overlay_contract={live_contract.get('contract_passed')}",
        ),
        "submission_evidence_bundle": _item(
            passed=_submission_evidence_ready(evidence_bundle),
            evidence=_submission_evidence_detail(evidence_bundle),
        ),
        "live_camera_demo": _item(
            passed=checks.get("live_overlay_exists") is True and live.get("success_gate_passed") is True,
            evidence=f"live_overlay_exists={checks.get('live_overlay_exists')}, live_success={live.get('success_gate_passed')}",
        ),
        "early_prediction": _item(
            passed=_live_latency_ready(live),
            evidence=f"live_binary_decision_latency_s={live.get('binary_decision_latency_s')}",
        ),
        "live_response_decision_frame": _item(
            passed=_live_response_decision_frame_ready(live),
            evidence=(
                f"path={live.get('response_decision_frame_png')}, "
                f"decision={live.get('response_decision_frame_decision')}, "
                f"robot_action={live.get('response_decision_frame_robot_action')}, "
                f"confidence={live.get('response_decision_frame_confidence')}"
            ),
        ),
        "confidence_display": _item(
            passed=live_contract.get("contract_passed") is True,
            evidence="live overlay contract must prove probability/confidence fields are present",
        ),
        "robot_response_action": _item(
            passed=checks.get("live_composite_ready") is True and live.get("composite_status") == "passed",
            evidence=f"live_composite_ready={checks.get('live_composite_ready')}, live_composite_status={live.get('composite_status')}",
        ),
        "operator_ground_truth_validation": _item(
            passed=_operator_ground_truth_ready(approved_archive_candidate),
            evidence=_operator_ground_truth_evidence(approved_archive_candidate),
        ),
        "manual_visual_approval": _item(
            passed=_manual_visual_approval_ready(approved_archive_candidate),
            evidence=f"manual_review_status={approved_archive_candidate.get('manual_review_status') if approved_archive_candidate else None}",
        ),
        "final_demo_candidate": _item(
            passed=_final_candidate_ready(final_candidate=final_candidate, archive_candidate=approved_archive_candidate),
            evidence=_final_candidate_evidence(final_candidate=final_candidate, archive_candidate=approved_archive_candidate),
        ),
        "submission_handoff": _item(
            passed=final_card.get("ready_for_submission_linking") is True
            and learning_queue.get("queue_status") == "no_research_needed",
            evidence=f"final_card={final_card.get('card_status')}, learning_queue={learning_queue.get('queue_status')}",
        ),
    }


def _item(*, passed: bool, evidence: str) -> dict[str, object]:
    return {"status": "passed" if passed else "missing", "evidence": evidence}


def _live_latency_ready(live: dict[str, Any]) -> bool:
    latency = live.get("binary_decision_latency_s")
    return isinstance(latency, (int, float)) and not isinstance(latency, bool) and float(latency) <= 0.50


def _live_response_decision_frame_ready(live: dict[str, Any]) -> bool:
    path = live.get("response_decision_frame_png")
    decision = live.get("response_decision_frame_decision")
    robot_action = live.get("response_decision_frame_robot_action")
    confidence = live.get("response_decision_frame_confidence")
    return (
        isinstance(path, str)
        and bool(path)
        and isinstance(decision, str)
        and bool(decision)
        and isinstance(robot_action, str)
        and bool(robot_action)
        and isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
    )


def _approved_archive_candidate(
    *,
    archive_index: dict[str, Any],
    final_candidate: dict[str, Any],
) -> dict[str, Any]:
    candidate = _dict_value(archive_index, "latest_final_video_candidate")
    if candidate:
        return candidate
    selected_run_id = final_candidate.get("selected_run_id")
    runs = archive_index.get("runs")
    if not isinstance(selected_run_id, str) or not isinstance(runs, list):
        return {}
    for raw_run in runs:
        if isinstance(raw_run, dict) and raw_run.get("run_id") == selected_run_id:
            return raw_run
    return {}


def _operator_ground_truth_ready(candidate: dict[str, Any]) -> bool:
    return (
        candidate.get("expected_actual_gesture") in {"rock", "paper", "scissors"}
        and candidate.get("ground_truth_passed") is True
        and candidate.get("ground_truth_match") is True
        and candidate.get("robot_action_match") is True
    )


def _operator_ground_truth_evidence(candidate: dict[str, Any]) -> str:
    if not candidate:
        return "no selected archived candidate with operator-ground-truth metadata"
    return (
        f"run_id={candidate.get('run_id')}, "
        f"expected_actual_gesture={candidate.get('expected_actual_gesture')}, "
        f"ground_truth_passed={candidate.get('ground_truth_passed')}, "
        f"ground_truth_match={candidate.get('ground_truth_match')}, "
        f"robot_action_match={candidate.get('robot_action_match')}"
    )


def _manual_visual_approval_ready(candidate: dict[str, Any]) -> bool:
    return bool(candidate) and candidate.get("manual_review_status") == "approved"


def _final_candidate_ready(*, final_candidate: dict[str, Any], archive_candidate: dict[str, Any]) -> bool:
    return (
        final_candidate.get("candidate_status") == "selected_final_demo_candidate"
        and final_candidate.get("ready_for_video_packaging") is True
        and isinstance(final_candidate.get("selected_run_id"), str)
        and bool(final_candidate.get("selected_run_id"))
        and isinstance(final_candidate.get("primary_video"), str)
        and bool(final_candidate.get("primary_video"))
        and isinstance(final_candidate.get("live_response_decision_frame"), str)
        and bool(final_candidate.get("live_response_decision_frame"))
        and _operator_ground_truth_ready(archive_candidate)
        and _manual_visual_approval_ready(archive_candidate)
    )


def _final_candidate_evidence(*, final_candidate: dict[str, Any], archive_candidate: dict[str, Any]) -> str:
    return (
        f"candidate_status={final_candidate.get('candidate_status')}, "
        f"ready_for_video_packaging={final_candidate.get('ready_for_video_packaging')}, "
        f"selected_run_id={final_candidate.get('selected_run_id')}, "
        f"primary_video={final_candidate.get('primary_video')}, "
        f"live_response_decision_frame={final_candidate.get('live_response_decision_frame')}, "
        f"archive_run_id={archive_candidate.get('run_id') if archive_candidate else None}"
    )


def _submission_evidence_ready(evidence_bundle: dict[str, Any]) -> bool:
    missing = evidence_bundle.get("missing_required_evidence")
    missing_list = missing if isinstance(missing, list) else []
    return evidence_bundle.get("ready_for_submission_demo") is True and not missing_list


def _submission_evidence_detail(evidence_bundle: dict[str, Any]) -> str:
    missing = evidence_bundle.get("missing_required_evidence")
    missing_list = missing if isinstance(missing, list) else []
    return (
        f"ready_for_submission_demo={evidence_bundle.get('ready_for_submission_demo')}, "
        f"missing_required_evidence={missing_list}"
    )


def _incomplete_status(final_card: dict[str, Any], learning_queue: dict[str, Any]) -> str:
    if final_card.get("card_status") == "ready_to_record_live_demo" or learning_queue.get("queue_status") == "waiting_for_live_run":
        return "incomplete_awaiting_live_capture"
    if (
        final_card.get("card_status") == "postprocess_or_repair_capture"
        or learning_queue.get("queue_status") == "postprocess_repair_required"
    ):
        return "incomplete_postprocess_repair_needed"
    if learning_queue.get("queue_status") == "simulation_first_research_queued":
        return "incomplete_research_iteration_needed"
    return "incomplete_missing_evidence"


def _next_action(*, goal_complete: bool, final_card: dict[str, Any], learning_queue: dict[str, Any]) -> str:
    if goal_complete:
        return "Proceed to manual final-video review and submission linking."
    command = final_card.get("primary_command")
    if command:
        return str(command)
    actions = learning_queue.get("next_actions")
    if isinstance(actions, list) and actions:
        return str(actions[0])
    return "Inspect final run card and learning queue."


def _next_action_absolute(*, goal_complete: bool, final_card: dict[str, Any]) -> str | None:
    if goal_complete:
        return None
    command = final_card.get("primary_command_absolute")
    return str(command) if command else None


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _audit_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Realtime Demo Goal Progress Audit",
        "",
        f"- Goal status: `{summary.get('goal_status')}`",
        f"- Goal complete: `{summary.get('goal_complete')}`",
        f"- Next action: {summary.get('next_action')}",
        f"- Next action absolute: {summary.get('next_action_absolute')}",
        "",
        "## Missing Requirements",
        "",
    ]
    missing = summary.get("missing_requirements", [])
    if isinstance(missing, list) and missing:
        lines.extend(f"- `{item}`" for item in missing)
    else:
        lines.append("- None")
    lines.extend(["", "## Requirements", "", "| Requirement | Status | Evidence |", "|---|---|---|"])
    requirements = summary.get("requirements", {})
    if isinstance(requirements, dict):
        for name, item in requirements.items():
            if not isinstance(item, dict):
                continue
            lines.append(f"| `{name}` | `{item.get('status')}` | {item.get('evidence')} |")
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoGoalProgressAuditConfig", "build_realtime_demo_goal_progress_audit"]
