"""Post-run learning queue for the realtime RPS demo workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeDemoLearningQueueConfig:
    """Input and output paths for post-run learning queue generation."""

    output_root: Path = Path("artifacts/realtime_demo_learning_queue_20260616")
    final_run_card: Path = Path("artifacts/realtime_demo_final_run_card_20260616/final_run_card.json")
    triage_summary: Path = Path("artifacts/realtime_demo_triage_20260616/triage_summary.json")
    operator_outcome: Path = Path("artifacts/realtime_demo_operator_outcome_20260616/operator_outcome.json")


def build_realtime_demo_learning_queue(config: RealtimeDemoLearningQueueConfig) -> dict[str, object]:
    """Write the next learning/capture/setup branch after a live demo attempt."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    output_json = config.output_root / "learning_queue.json"
    output_md = config.output_root / "learning_queue.md"
    final_card = _read_json_if_exists(config.final_run_card) or {}
    triage = _read_json_if_exists(config.triage_summary) or {}
    operator = _read_json_if_exists(config.operator_outcome) or {}

    summary = _queue_summary(
        final_card=final_card,
        triage=triage,
        operator=operator,
        config=config,
        output_json=output_json,
        output_md=output_md,
    )
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(_queue_markdown(summary), encoding="utf-8")
    return summary


def _queue_summary(
    *,
    final_card: dict[str, Any],
    triage: dict[str, Any],
    operator: dict[str, Any],
    config: RealtimeDemoLearningQueueConfig,
    output_json: Path,
    output_md: Path,
) -> dict[str, object]:
    card_status = str(final_card.get("card_status") or "")
    operator_state = str(operator.get("operator_state") or "")
    triage_status = str(triage.get("status") or "")
    failure_category = _failure_category(final_card=final_card, triage=triage, operator=operator)
    queue_status = _queue_status(card_status=card_status, operator_state=operator_state, triage_status=triage_status)
    recommended_track = _recommended_track(queue_status)
    model_change_allowed = queue_status == "simulation_first_research_queued"
    simulation_targets = _simulation_targets(failure_category) if model_change_allowed else []
    triage_evidence = triage.get("evidence") if isinstance(triage.get("evidence"), dict) else {}
    return {
        "queue_status": queue_status,
        "recommended_track": recommended_track,
        "model_change_allowed": model_change_allowed,
        "failure_category": failure_category,
        "simulation_targets": simulation_targets,
        "next_actions": _next_actions(queue_status=queue_status),
        "source_evidence": {
            "card_status": final_card.get("card_status"),
            "triage_status": triage.get("status"),
            "operator_state": operator.get("operator_state"),
            "triage_failure_category": triage.get("failure_category"),
            "operator_failure_category": operator.get("failure_category"),
            "final_card_failure_category": final_card.get("failure_category"),
            "primary_command": final_card.get("primary_command") or operator.get("primary_command"),
            "primary_action": final_card.get("primary_action") or operator.get("primary_action"),
            "response_prompt_diagnostic_frame": triage_evidence.get("response_prompt_diagnostic_frame"),
            "response_prompt_diagnostic_frame_reason": triage_evidence.get(
                "response_prompt_diagnostic_frame_reason"
            ),
            "expected_actual_gesture": triage_evidence.get("expected_actual_gesture"),
            "first_binary_decision": triage_evidence.get("first_binary_decision"),
            "first_binary_robot_action": triage_evidence.get("first_binary_robot_action"),
            "live_rock_retake_gate_status": triage_evidence.get("live_rock_retake_gate_status"),
            "live_rock_retake_binary_decision_frame_count": triage_evidence.get(
                "live_rock_retake_binary_decision_frame_count"
            ),
            "live_rock_retake_confirmed_binary_decision_count": triage_evidence.get(
                "live_rock_retake_confirmed_binary_decision_count"
            ),
            "live_rock_retake_first_binary_decision": triage_evidence.get(
                "live_rock_retake_first_binary_decision"
            ),
            "live_rock_retake_first_binary_robot_action": triage_evidence.get(
                "live_rock_retake_first_binary_robot_action"
            ),
        },
        "inputs": {
            "final_run_card": config.final_run_card.as_posix(),
            "triage_summary": config.triage_summary.as_posix(),
            "operator_outcome": config.operator_outcome.as_posix(),
        },
        "outputs": {
            "queue_json": output_json.as_posix(),
            "queue_md": output_md.as_posix(),
        },
        "claim_scope": "post-run learning/capture/setup branch over existing artifacts; does not generate data, train models, run capture, or edit reports",
    }


def _queue_status(*, card_status: str, operator_state: str, triage_status: str = "") -> str:
    if triage_status == "needs_research_iteration":
        return "simulation_first_research_queued"
    if card_status == "ready_for_final_submission_review":
        return "no_research_needed"
    if card_status == "simulation_first_research_needed" or operator_state == "research_iteration_needed":
        return "simulation_first_research_queued"
    if card_status == "retake_live_capture" or operator_state == "retake_capture":
        return "capture_retake_required"
    if card_status == "ready_to_record_live_demo" or operator_state == "ready_to_record":
        return "waiting_for_live_run"
    if card_status == "postprocess_or_repair_capture" or operator_state == "postprocess_or_repair_capture":
        return "postprocess_repair_required"
    if card_status == "setup_fix_needed" or operator_state == "setup_fix_needed":
        return "setup_fix_required"
    return "manual_review_required"


def _recommended_track(queue_status: str) -> str:
    return {
        "no_research_needed": "final_packaging",
        "simulation_first_research_queued": "simulated_skeleton_augmentation",
        "capture_retake_required": "capture_setup",
        "postprocess_repair_required": "postprocess_repair",
        "setup_fix_required": "local_setup",
        "waiting_for_live_run": "live_capture",
        "manual_review_required": "manual_review",
    }.get(queue_status, "manual_review")


def _next_actions(*, queue_status: str) -> list[str]:
    return {
        "no_research_needed": [
            "manual_video_review",
            "youtube_demo_video_upload",
            "readme_and_report_link_update",
        ],
        "simulation_first_research_queued": [
            "inspect_live_frame_log",
            "generate_targeted_simulated_skeletons",
            "retrain_and_validate_strict_gates",
        ],
        "capture_retake_required": [
            "adjust_camera_framing_or_lighting",
            "retake_live_capture",
        ],
        "postprocess_repair_required": [
            "rerun_live_verification",
            "rerun_schunk_composite",
            "refresh_acceptance_and_goal_audit",
        ],
        "setup_fix_required": [
            "fix_local_setup",
            "rerun_preflight",
        ],
        "waiting_for_live_run": [
            "run_live_pipeline",
            "archive_live_run",
            "refresh_final_run_card",
        ],
        "manual_review_required": [
            "inspect_operator_outcome_and_triage",
        ],
    }.get(queue_status, ["inspect_operator_outcome_and_triage"])


def _simulation_targets(failure_category: str | None) -> list[str]:
    category = failure_category or "unknown"
    targets_by_category = {
        "wrong_class": [
            "class_boundary_hard_examples",
            "viewpoint_robustness",
            "gesture_pair_confusers",
        ],
        "late_decision": [
            "early_motion_prefixes",
            "slow_transition_prefixes",
            "decision_latency_ablation",
        ],
        "unstable_prediction": [
            "temporal_jitter_robustness",
            "rolling_confirmation_ablation",
        ],
        "rock_false_trigger": [
            "rock_wait_negatives",
            "transition_mass_threshold_ablation",
        ],
        "missing_binary_decision": [
            "response_prompt_prefix_hard_examples",
            "transition_mass_threshold_ablation",
            "early_motion_prefixes",
            "prompt_timing_robustness",
        ],
        "robot_action_mismatch": [
            "response_policy_manifest_audit",
        ],
    }
    return targets_by_category.get(category, ["inspect_failure_frame_log", "targeted_simulated_skeletons"])


def _failure_category(
    *,
    final_card: dict[str, Any],
    triage: dict[str, Any],
    operator: dict[str, Any],
) -> str | None:
    if triage.get("status") == "needs_research_iteration" and triage.get("failure_category"):
        return str(triage.get("failure_category"))
    for payload in (final_card, triage, operator):
        value = payload.get("failure_category")
        if value:
            return str(value)
    return None


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _queue_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Realtime Demo Learning Queue",
        "",
        f"- Queue status: `{summary.get('queue_status')}`",
        f"- Recommended track: `{summary.get('recommended_track')}`",
        f"- Model change allowed: `{summary.get('model_change_allowed')}`",
        f"- Failure category: `{summary.get('failure_category')}`",
        "",
        "## Simulation Targets",
        "",
    ]
    targets = summary.get("simulation_targets", [])
    if isinstance(targets, list) and targets:
        lines.extend(f"- `{target}`" for target in targets)
    else:
        lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    actions = summary.get("next_actions", [])
    if isinstance(actions, list):
        lines.extend(f"- `{action}`" for action in actions)
    lines.extend(["", "## Source Evidence", ""])
    evidence = summary.get("source_evidence", {})
    if isinstance(evidence, dict):
        for key in sorted(evidence):
            lines.append(f"- `{key}`: `{evidence[key]}`")
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoLearningQueueConfig", "build_realtime_demo_learning_queue"]
