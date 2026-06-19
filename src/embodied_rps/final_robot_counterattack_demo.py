"""Final screen-rendered robot counterattack demo artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import cv2  # type: ignore[import-untyped]
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-untyped]

from embodied_rps.config import KinematicConfig, load_kinematic_config
from embodied_rps.domain import FeasibilityResult, Gesture, HandPose
from embodied_rps.feasibility import check_actuator_feasibility
from embodied_rps.policy import CounterMovePolicy, OpponentGesture
from embodied_rps.realtime_demo_launcher import RealtimeDemoConfig, load_realtime_demo_config
from embodied_rps.schunk import SchunkPoseConfig, generate_pose_skeleton, load_schunk_pose_config, render_schunk_pose_previews

RobotGesture = Literal["rock", "paper", "scissors"]
FINAL_DEMO_CONFIG = Path("configs/realtime_two_stage_selector_demo_conditional_scissors_rescue_rock_guard.yaml")
LEGACY_DEFAULT_CONFIG = Path("configs/realtime_two_stage_selector_demo_conditional_scissors_rescue.yaml")
V7E_DIAGNOSTIC_SOURCE = Path("sources/v7e-stage1-paper-transition-rescue-2026-06-19.md")


@dataclass(frozen=True)
class ResponseWindowDecision:
    """First confirmed non-wait prediction inside the response prompt window."""

    predicted_gesture: OpponentGesture
    counter_move: OpponentGesture
    decision_frame: int
    decision_time_s: float
    response_window_start_time_s: float
    decision_latency_s: float
    confidence: float
    confidence_margin: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class MotionFeasibility:
    """Final-demo actuator result from the required rock start pose."""

    feasible: bool
    remaining_time_s: float
    joint_limits_ok: bool
    joint_limit_violations: tuple[str, ...]
    actuator_result: FeasibilityResult


@dataclass(frozen=True)
class KinematicMotionConfig:
    """Kinematic fallback motion policy used for final-demo feasibility."""

    config: KinematicConfig
    joint_min: float = 0.0
    joint_max: float = 1.0

    def check_from_rock(self, target_gesture: RobotGesture, *, decision_latency_s: float) -> MotionFeasibility:
        """Check whether the robot can move from rock to the target before deadline."""

        current_pose = self.config.gestures["rock"]
        target_pose = self.config.gestures[target_gesture]
        remaining_time_s = self.config.deadline_s - decision_latency_s
        actuator_result = check_actuator_feasibility(
            current_pose=current_pose,
            target_pose=target_pose,
            limits=self.config.actuator_limits(),
            remaining_time_s=remaining_time_s,
        )
        violations = _joint_limit_violations(target_pose, joint_min=self.joint_min, joint_max=self.joint_max)
        joint_limits_ok = len(violations) == 0
        return MotionFeasibility(
            feasible=actuator_result.feasible and joint_limits_ok,
            remaining_time_s=remaining_time_s,
            joint_limits_ok=joint_limits_ok,
            joint_limit_violations=tuple(violations),
            actuator_result=actuator_result,
        )


def load_kinematic_motion_config(path: Path) -> KinematicMotionConfig:
    """Load the final-demo kinematic fallback motion config."""

    return KinematicMotionConfig(load_kinematic_config(path))


def final_robot_pose_for_decision(
    decision_state: str | None,
    *,
    confirmed: bool,
    in_response_window: bool,
) -> RobotGesture:
    """Return the robot pose used by the final demo for one frame decision."""

    if not confirmed or not in_response_window:
        return "rock"
    if decision_state not in ("rock", "paper", "scissors"):
        return "rock"
    return cast(RobotGesture, CounterMovePolicy().counter(cast(OpponentGesture, decision_state)))


def extract_response_window_decision(frame_log: Path, *, response_prompt: str = "scissors") -> ResponseWindowDecision:
    """Extract the first confirmed RPS prediction inside the response prompt window."""

    records = _load_jsonl_records(frame_log)
    response_window_start_time_s: float | None = None
    for record in records:
        if _in_response_window(record, response_prompt=response_prompt):
            response_window_start_time_s = _float_record(record, "time_s")
            break
    if response_window_start_time_s is None:
        raise ValueError(f"No response window records found for response_prompt={response_prompt!r}")

    policy = CounterMovePolicy()
    for record in records:
        if not _in_response_window(record, response_prompt=response_prompt):
            continue
        if not bool(record.get("confirmed_decision", False)):
            continue
        decision_state = record.get("decision_state")
        if decision_state not in ("rock", "paper", "scissors"):
            continue
        predicted = cast(OpponentGesture, decision_state)
        decision_time_s = _float_record(record, "time_s")
        return ResponseWindowDecision(
            predicted_gesture=predicted,
            counter_move=policy.counter(predicted),
            decision_frame=int(_float_record(record, "frame_index")),
            decision_time_s=decision_time_s,
            response_window_start_time_s=response_window_start_time_s,
            decision_latency_s=decision_time_s - response_window_start_time_s,
            confidence=_optional_float(record.get("confidence"), default=0.0),
            confidence_margin=_optional_float(record.get("margin"), default=0.0),
            probabilities={
                "rock": _optional_float(record.get("p_rock"), default=0.0),
                "paper": _optional_float(record.get("p_paper"), default=0.0),
                "scissors": _optional_float(record.get("p_scissors"), default=0.0),
            },
        )
    raise ValueError(f"No confirmed RPS decision found in response_prompt={response_prompt!r}")


def freeze_final_demo_policy(
    *,
    config_path: Path,
    output_root: Path,
    project_root: Path,
) -> dict[str, object]:
    """Write the final v4 live/demo policy freeze artifact."""

    config = load_realtime_demo_config(config_path)
    profile_weights = _parse_profile_weights(config.profile_weights, len(config.profiles))
    profile_rows: list[dict[str, object]] = []
    for index, profile_path in enumerate(config.profiles):
        normalized_profile = _relative_path(profile_path, project_root=project_root)
        if "v4" not in normalized_profile.lower():
            raise ValueError(f"Final live/demo policy must use v4 profiles only, got {normalized_profile}")
        if "v7" in normalized_profile.lower():
            raise ValueError(f"Final live/demo policy must not promote v7 profiles, got {normalized_profile}")
        profile_json = _read_json_if_exists(project_root / profile_path)
        model_state_path = profile_json.get("model_state_path") if isinstance(profile_json, Mapping) else None
        profile_rows.append(
            {
                "index": index,
                "path": normalized_profile,
                "weight": profile_weights[index],
                "exists": (project_root / profile_path).exists(),
                "model_state_path": _relative_path(Path(str(model_state_path)), project_root=project_root)
                if isinstance(model_state_path, str)
                else None,
                "selected_accuracy": profile_json.get("selected_accuracy") if isinstance(profile_json, Mapping) else None,
                "macro_f1": profile_json.get("macro_f1") if isinstance(profile_json, Mapping) else None,
                "model_type": profile_json.get("model_type") if isinstance(profile_json, Mapping) else None,
            }
        )

    payload: dict[str, object] = {
        "status": "passed",
        "scope": "final screen-rendered robot counterattack demo live/demo policy freeze",
        "live_demo_model_family": "v4",
        "preserved_diagnostic_model_family": "v7e",
        "v7e_policy": "report diagnostics only; not promoted for live/demo",
        "v7f_policy": "not started; retraining remains blocked unless explicitly requested",
        "config_path": _relative_path(config_path, project_root=project_root),
        "legacy_launcher_default_preserved": _relative_path(LEGACY_DEFAULT_CONFIG, project_root=project_root),
        "profile_weights": profile_weights,
        "profiles": profile_rows,
        "gates": {
            "confidence_threshold": config.confidence_threshold,
            "margin_threshold": config.margin_threshold,
            "confirmation_count": config.confirmation_count,
            "transition_mass_threshold": config.transition_mass_threshold,
            "binary_transition_mass_threshold": config.binary_transition_mass_threshold,
            "min_binary_decision_progress": config.min_binary_decision_progress,
            "prompt_cycle": config.prompt_cycle,
            "prompt_sequence": config.prompt_sequence,
            "response_prompt": config.response_prompt,
            "rock_hold_guard_enabled": config.rock_hold_guard_min_history_frames > 0,
            "gesture_verifier_enabled": config.gesture_verifier_min_history_frames > 0,
            "hold_response_prompt_until_decision": config.hold_response_prompt_until_decision,
            "stop_after_confirmed_response_decision": config.stop_after_confirmed_response_decision,
        },
        "protected_inputs": {
            "heldout_test_mp4s": "validation-only",
            "proposal_pdf": "not edited",
            "presentation_slides_pdf": "not edited",
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "v4_live_demo_policy_freeze.json"
    summary_md = output_root / "v4_live_demo_policy_freeze.md"
    _write_json(summary_json, payload)
    _write_freeze_md(summary_md, payload)
    payload["outputs"] = {
        "policy_freeze_json": _relative_path(summary_json, project_root=project_root),
        "policy_freeze_md": _relative_path(summary_md, project_root=project_root),
    }
    _write_json(summary_json, payload)
    return payload


def validate_no_heldout_test_inputs(paths: Iterable[Path]) -> None:
    """Reject heldout */test MP4 paths for demo-generation inputs."""

    for path in paths:
        parts = {part.lower() for part in path.parts}
        if path.suffix.lower() == ".mp4" and "test" in parts:
            raise ValueError(f"Heldout */test MP4s remain validation-only and cannot be demo inputs: {path}")


def write_render_audit_artifacts(
    *,
    render_artifact_root: Path,
    output_root: Path,
    project_root: Path,
) -> dict[str, object]:
    """Record the archived Isaac/SCHUNK render evidence used by this pass."""

    postcondition_path = render_artifact_root / "render_postcondition.json"
    diagnostics_path = render_artifact_root / "render_diagnostics.json"
    postcondition = _read_required_json(postcondition_path)
    diagnostics = _read_required_json(diagnostics_path)
    status = "passed" if postcondition.get("status") == "passed" and diagnostics.get("render_validation_status") == "passed" else "failed"
    payload: dict[str, object] = {
        "status": status,
        "claim_scope": "archived Isaac/SCHUNK render evidence plus local kinematic SCHUNK proxy for final overlay",
        "fresh_remote_render_run": False,
        "archived_postcondition_path": _relative_path(postcondition_path, project_root=project_root),
        "archived_diagnostics_path": _relative_path(diagnostics_path, project_root=project_root),
        "postcondition_status": postcondition.get("status"),
        "articulation_status": postcondition.get("articulation_status"),
        "render_validation_status": postcondition.get("render_validation_status"),
        "dof_count": postcondition.get("dof_count"),
        "required_visual_evidence": postcondition.get("required_visual_evidence"),
        "renderer_backend_for_dynamic_overlay": "kinematic_schunk_proxy",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "robot_hand_render_audit.json"
    summary_md = output_root / "robot_hand_render_audit.md"
    payload["outputs"] = {
        "render_audit_json": _relative_path(summary_json, project_root=project_root),
        "render_audit_md": _relative_path(summary_md, project_root=project_root),
    }
    _write_json(summary_json, payload)
    _write_render_audit_md(summary_md, payload)
    return payload


def build_final_counterattack_artifacts(
    *,
    overlay_video: Path | None,
    frame_log: Path,
    config_path: Path,
    pose_config_path: Path,
    schunk_pose_config_path: Path,
    output_root: Path,
    project_root: Path,
    run_label: str = "replay",
) -> dict[str, object]:
    """Build final replay/demo logs, previews, optional overlay, and report tables."""

    candidate_inputs = [frame_log]
    if overlay_video is not None:
        candidate_inputs.append(overlay_video)
    validate_no_heldout_test_inputs(candidate_inputs)

    config = load_realtime_demo_config(config_path)
    response_prompt = config.response_prompt or "scissors"
    decision = extract_response_window_decision(frame_log, response_prompt=response_prompt)
    motion = load_kinematic_motion_config(pose_config_path)
    target = cast(RobotGesture, decision.counter_move)
    feasibility = motion.check_from_rock(target, decision_latency_s=decision.decision_latency_s)

    output_root.mkdir(parents=True, exist_ok=True)
    preview_dir = output_root / "motion_previews"
    preview_metadata = output_root / "motion_preview_metadata.jsonl"
    schunk_pose_config = load_schunk_pose_config(schunk_pose_config_path)
    render_schunk_pose_previews(
        pose_config=schunk_pose_config,
        out_dir=preview_dir,
        metadata_path=preview_metadata,
        yaw_degrees=(0.0,),
        pitch_degrees=(20.0,),
        gestures=("rock", "paper", "scissors"),
        image_width=720,
        image_height=520,
    )

    robot_motion_log = output_root / "robot_motion_log.jsonl"
    feasibility_summary_json = output_root / "feasibility_summary.json"
    episode_summary_csv = output_root / "episode_summary.csv"
    report_metrics_csv = output_root / "report_metrics.csv"
    final_overlay_video: Path | None = None
    poster_path: Path | None = None
    if overlay_video is not None:
        final_overlay_video = output_root / "final_robot_counterattack_overlay.mp4"
        poster_path = output_root / "final_robot_counterattack_poster.png"
        _write_final_overlay_video(
            overlay_video=overlay_video,
            output_video=final_overlay_video,
            poster_path=poster_path,
            decision=decision,
            feasibility=feasibility,
            response_delay_s=motion.config.response_delay_s,
            schunk_pose_config=schunk_pose_config,
        )

    episode_result = _episode_result(decision.predicted_gesture, decision.counter_move, feasibility.feasible)
    motion_record = _motion_record(
        decision=decision,
        feasibility=feasibility,
        episode_result=episode_result,
        run_label=run_label,
        frame_log=frame_log,
        overlay_video=overlay_video,
        config_path=config_path,
        pose_config_path=pose_config_path,
        project_root=project_root,
    )
    _write_jsonl(robot_motion_log, [motion_record])
    _write_episode_csv(episode_summary_csv, [motion_record])
    _write_report_metrics_csv(report_metrics_csv, decision=decision, feasibility=feasibility, episode_result=episode_result)

    summary: dict[str, object] = {
        "status": "passed" if feasibility.feasible else "infeasible",
        "run_label": run_label,
        "claim_scope": "screen-rendered replay/demo artifact; not a retraining result and not final packaging",
        "frame_log": _relative_path(frame_log, project_root=project_root),
        "overlay_video": _relative_path(overlay_video, project_root=project_root) if overlay_video else None,
        "config_path": _relative_path(config_path, project_root=project_root),
        "robot_start_pose": "rock",
        "predicted_gesture": decision.predicted_gesture,
        "selected_counter_move": decision.counter_move,
        "decision_frame": decision.decision_frame,
        "decision_time_s": decision.decision_time_s,
        "decision_latency_s": decision.decision_latency_s,
        "remaining_time_s": feasibility.remaining_time_s,
        "feasible": feasibility.feasible,
        "episode_result": episode_result,
        "outputs": {
            "robot_motion_log_jsonl": _relative_path(robot_motion_log, project_root=project_root),
            "feasibility_summary_json": _relative_path(feasibility_summary_json, project_root=project_root),
            "episode_summary_csv": _relative_path(episode_summary_csv, project_root=project_root),
            "report_metrics_csv": _relative_path(report_metrics_csv, project_root=project_root),
            "motion_preview_dir": _relative_path(preview_dir, project_root=project_root),
            "motion_preview_metadata_jsonl": _relative_path(preview_metadata, project_root=project_root),
            "final_overlay_video": _relative_path(final_overlay_video, project_root=project_root) if final_overlay_video else None,
            "poster_png": _relative_path(poster_path, project_root=project_root) if poster_path else None,
        },
    }
    feasibility_payload = {
        **summary,
        "actuator": {
            "deadline_s": motion.config.deadline_s,
            "response_delay_s": motion.config.response_delay_s,
            "required_time_s": feasibility.actuator_result.required_time_s,
            "completion_time_s": feasibility.actuator_result.completion_time_s,
            "limiting_joint": feasibility.actuator_result.limiting_joint,
            "failure_reason": feasibility.actuator_result.failure_reason,
            "joint_limits_ok": feasibility.joint_limits_ok,
            "joint_limit_violations": list(feasibility.joint_limit_violations),
        },
    }
    _write_json(feasibility_summary_json, feasibility_payload)
    return summary


def write_final_report_assets(
    *,
    output_root: Path,
    project_root: Path,
    replay_summary: Mapping[str, object],
    freeze_summary: Mapping[str, object],
    render_audit: Mapping[str, object],
    diagnostic_replay_summary_path: Path | None = None,
) -> dict[str, object]:
    """Write final report-ready metrics and figure inventory without packaging."""

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_root / "final_submission_metrics_table.csv"
    figure_inventory_csv = output_root / "final_submission_figure_inventory.csv"
    summary_json = output_root / "final_submission_report_assets_summary.json"
    rows = [
        {
            "section": "v4 fallback",
            "metric": "live_demo_policy_family",
            "value": freeze_summary.get("live_demo_model_family"),
            "notes": "frozen live/demo policy",
        },
        {
            "section": "v4 fallback",
            "metric": "profile_weights",
            "value": json.dumps(freeze_summary.get("profile_weights")),
            "notes": "fewshot_aug_tcn,rebalanced_tcn,final_gate_micro",
        },
        {
            "section": "v7e diagnostics",
            "metric": "original20_strict_result",
            "value": "17/20",
            "notes": "diagnostic only; not promoted",
        },
        {
            "section": "v7e diagnostics",
            "metric": "known_failures",
            "value": "scissors->paper x2; late paper x1",
            "notes": "from v7e rescue source page",
        },
        {
            "section": "decision timing",
            "metric": "decision_latency_s",
            "value": replay_summary.get("decision_latency_s"),
            "notes": "response-window replay",
        },
        {
            "section": "actuator feasibility",
            "metric": "remaining_time_s",
            "value": replay_summary.get("remaining_time_s"),
            "notes": "deadline minus decision latency",
        },
        {
            "section": "actuator feasibility",
            "metric": "episode_result",
            "value": replay_summary.get("episode_result"),
            "notes": "screen-rendered replay result",
        },
    ]
    _write_dict_csv(metrics_csv, rows)
    figure_rows = [
        {
            "artifact": "policy_freeze",
            "path": _output_path(freeze_summary, "policy_freeze_json"),
            "claim_scope": "v4 live/demo policy freeze",
        },
        {
            "artifact": "render_audit",
            "path": _output_path(render_audit, "render_audit_json"),
            "claim_scope": "archived Isaac/SCHUNK evidence",
        },
        {
            "artifact": "final_overlay_video",
            "path": _output_path(replay_summary, "final_overlay_video"),
            "claim_scope": "screen-rendered replay demo",
        },
        {
            "artifact": "motion_previews",
            "path": _output_path(replay_summary, "motion_preview_dir"),
            "claim_scope": "local kinematic SCHUNK proxy previews",
        },
        {
            "artifact": "v7e_diagnostic_source",
            "path": _relative_path(V7E_DIAGNOSTIC_SOURCE, project_root=project_root),
            "claim_scope": "diagnostic report source",
        },
    ]
    if diagnostic_replay_summary_path is not None:
        figure_rows.append(
            {
                "artifact": "secondary_failed_replay",
                "path": _relative_path(diagnostic_replay_summary_path, project_root=project_root),
                "claim_scope": "failed/blocked archived replay diagnostic",
            }
        )
    _write_dict_csv(figure_inventory_csv, figure_rows)
    summary: dict[str, object] = {
        "status": "passed",
        "claim_scope": "report-ready tables and figure inventory only; final packaging not started",
        "v4_policy": "live/demo fallback",
        "v7e_policy": "diagnostics only",
        "protected_pdf_policy": "proposal.pdf and presentation-slides.pdf not edited",
        "outputs": {
            "metrics_csv": _relative_path(metrics_csv, project_root=project_root),
            "figure_inventory_csv": _relative_path(figure_inventory_csv, project_root=project_root),
            "summary_json": _relative_path(summary_json, project_root=project_root),
        },
    }
    _write_json(summary_json, summary)
    return summary


def _motion_record(
    *,
    decision: ResponseWindowDecision,
    feasibility: MotionFeasibility,
    episode_result: str,
    run_label: str,
    frame_log: Path,
    overlay_video: Path | None,
    config_path: Path,
    pose_config_path: Path,
    project_root: Path,
) -> dict[str, object]:
    return {
        "run_label": run_label,
        "frame_log": _relative_path(frame_log, project_root=project_root),
        "overlay_video": _relative_path(overlay_video, project_root=project_root) if overlay_video else None,
        "config_path": _relative_path(config_path, project_root=project_root),
        "pose_config_path": _relative_path(pose_config_path, project_root=project_root),
        "robot_start_pose": "rock",
        "predicted_gesture": decision.predicted_gesture,
        "selected_counter_move": decision.counter_move,
        "decision_frame": decision.decision_frame,
        "decision_time_s": decision.decision_time_s,
        "response_window_start_time_s": decision.response_window_start_time_s,
        "decision_latency_s": decision.decision_latency_s,
        "confidence": decision.confidence,
        "confidence_margin": decision.confidence_margin,
        "remaining_time_s": feasibility.remaining_time_s,
        "feasible": feasibility.feasible,
        "required_time_s": feasibility.actuator_result.required_time_s,
        "completion_time_s": feasibility.actuator_result.completion_time_s,
        "limiting_joint": feasibility.actuator_result.limiting_joint,
        "failure_reason": feasibility.actuator_result.failure_reason,
        "joint_limits_ok": feasibility.joint_limits_ok,
        "joint_limit_violations": list(feasibility.joint_limit_violations),
        "episode_result": episode_result,
    }


def _write_final_overlay_video(
    *,
    overlay_video: Path,
    output_video: Path,
    poster_path: Path,
    decision: ResponseWindowDecision,
    feasibility: MotionFeasibility,
    response_delay_s: float,
    schunk_pose_config: SchunkPoseConfig,
) -> None:
    if not overlay_video.exists():
        raise FileNotFoundError(f"overlay_video does not exist: {overlay_video}")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(overlay_video))
    if not capture.isOpened():
        raise ValueError(f"could not open overlay video: {overlay_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width, height = 1920, 1080
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not open output video writer: {output_video}")

    font = ImageFont.load_default()
    target_pose = cast(Gesture, decision.counter_move)
    first_frame: Image.Image | None = None
    frame_index = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            source_time_s = frame_index / fps
            progress = _motion_progress(
                source_time_s,
                decision_time_s=decision.decision_time_s,
                response_delay_s=response_delay_s,
                motion_time_s=_motion_only_time(feasibility.actuator_result, response_delay_s=response_delay_s),
            )
            robot_image = _render_schunk_motion_frame(
                schunk_pose_config=schunk_pose_config,
                start_pose="rock",
                target_pose=target_pose,
                progress=progress,
                size=(720, 520),
            )
            overlay_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            composite = _compose_final_frame(
                overlay_rgb=overlay_rgb,
                robot_image=robot_image,
                decision=decision,
                feasibility=feasibility,
                source_time_s=source_time_s,
                output_size=(width, height),
                font=font,
            )
            if first_frame is None:
                first_frame = composite.copy()
            writer.write(cv2.cvtColor(np.asarray(composite), cv2.COLOR_RGB2BGR))
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    if frame_index == 0 or first_frame is None:
        raise ValueError("overlay video did not contain frames")
    first_frame.save(poster_path)


def _motion_only_time(result: FeasibilityResult, *, response_delay_s: float) -> float:
    return max(0.0, result.required_time_s - response_delay_s)


def _motion_progress(
    source_time_s: float,
    *,
    decision_time_s: float,
    response_delay_s: float,
    motion_time_s: float,
) -> float:
    if source_time_s <= decision_time_s + response_delay_s:
        return 0.0
    if motion_time_s <= 1e-9:
        return 1.0
    return max(0.0, min(1.0, (source_time_s - decision_time_s - response_delay_s) / motion_time_s))


def _render_schunk_motion_frame(
    *,
    schunk_pose_config: SchunkPoseConfig,
    start_pose: Gesture,
    target_pose: Gesture,
    progress: float,
    size: tuple[int, int],
) -> Image.Image:
    start_state = schunk_pose_config.gestures[start_pose]
    target_state = schunk_pose_config.gestures[target_pose]
    joint_state = {
        name: float(start_state[name]) + (float(target_state[name]) - float(start_state[name])) * progress
        for name in schunk_pose_config.joint_names
    }
    skeleton = generate_pose_skeleton(joint_state)
    return _draw_schunk_skeleton(skeleton, size=size, title=f"robot {start_pose}->{target_pose}  {progress:.0%}")


def _draw_schunk_skeleton(
    skeleton: Mapping[str, tuple[float, float, float]],
    *,
    size: tuple[int, int],
    title: str,
) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, (248, 250, 252))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, width, 42), fill=(226, 232, 240))
    draw.text((16, 15), title, fill=(15, 23, 42), font=font)
    projected = _project_skeleton_points(skeleton, width=width, height=height)
    chains = (
        ("thumb_base", "thumb_mid", "thumb_tip"),
        ("index_base", "index_mid", "index_tip"),
        ("middle_base", "middle_mid", "middle_tip"),
        ("ring_base", "ring_mid", "ring_tip"),
        ("pinky_base", "pinky_mid", "pinky_tip"),
    )
    colors = [(37, 99, 235), (22, 163, 74), (220, 38, 38), (147, 51, 234), (234, 88, 12)]
    for chain, color in zip(chains, colors):
        for start, end in zip(chain, chain[1:]):
            draw.line((projected[start], projected[end]), fill=color, width=10)
        for link in chain:
            x, y = projected[link]
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(15, 23, 42))
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 255, 255))
    x, y = projected["palm"]
    draw.ellipse((x - 30, y - 30, x + 30, y + 30), fill=(203, 213, 225), outline=(100, 116, 139), width=2)
    return image


def _project_skeleton_points(
    points: Mapping[str, tuple[float, float, float]],
    *,
    width: int,
    height: int,
) -> dict[str, tuple[int, int]]:
    xs = [point[0] for point in points.values()]
    ys = [point[1] for point in points.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    scale = min(width * 0.68, height * 0.68) / span
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    return {
        name: (
            int(round(width / 2.0 + (point[0] - cx) * scale)),
            int(round(height / 2.0 - (point[1] - cy) * scale + 36.0)),
        )
        for name, point in points.items()
    }


def _compose_final_frame(
    *,
    overlay_rgb: Image.Image,
    robot_image: Image.Image,
    decision: ResponseWindowDecision,
    feasibility: MotionFeasibility,
    source_time_s: float,
    output_size: tuple[int, int],
    font: ImageFont.ImageFont,
) -> Image.Image:
    width, height = output_size
    canvas = Image.new("RGB", (width, height), (243, 245, 248))
    draw = ImageDraw.Draw(canvas)
    header_h = 72
    draw.rectangle((0, 0, width, header_h), fill=(24, 30, 38))
    title = "Actuator-constrained RPS robot counterattack"
    draw.text((28, 18), title, fill=(248, 250, 252), font=font)
    status = "FEASIBLE" if feasibility.feasible else "INFEASIBLE"
    draw.text((width - 360, 18), f"{status}  remaining={feasibility.remaining_time_s:.3f}s", fill=(219, 234, 254), font=font)
    margin = 24
    gutter = 18
    content_top = header_h + margin
    content_h = height - content_top - margin - 70
    left_w = int((width - margin * 2 - gutter) * 0.62)
    right_w = width - margin * 2 - gutter - left_w
    left_box = (margin, content_top, margin + left_w, content_top + content_h)
    right_box = (margin + left_w + gutter, content_top, width - margin, content_top + content_h)
    _paste_fit(canvas, overlay_rgb, left_box, background=(226, 232, 240))
    _paste_fit(canvas, robot_image, right_box, background=(235, 238, 242))
    draw.rectangle(left_box, outline=(148, 163, 184), width=2)
    draw.rectangle(right_box, outline=(148, 163, 184), width=2)
    footer_top = height - margin - 58
    draw.rectangle((margin, footer_top, width - margin, height - margin), fill=(255, 255, 255), outline=(203, 213, 225))
    footer = (
        f"prompt=scissors | prediction={decision.predicted_gesture} "
        f"conf={decision.confidence:.2f} | counter={decision.counter_move} | "
        f"t={source_time_s:.2f}s | latency={decision.decision_latency_s:.3f}s | "
        f"deadline check={feasibility.actuator_result.failure_reason or 'passed'}"
    )
    draw.text((margin + 16, footer_top + 22), footer, fill=(15, 23, 42), font=font)
    return canvas


def _paste_fit(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    background: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    tile = Image.new("RGB", (width, height), background)
    copied = image.copy()
    copied.thumbnail((width - 24, height - 48), _resampling_lanczos())
    tile.paste(copied, ((width - copied.width) // 2, (height - copied.height) // 2 + 18))
    canvas.paste(tile, (left, top))


def _resampling_lanczos() -> int:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return int(resampling.LANCZOS)
    return int(Image.LANCZOS)


def _episode_result(predicted: OpponentGesture, counter_move: OpponentGesture, feasible: bool) -> str:
    if not feasible:
        return "infeasible_counter"
    expected = CounterMovePolicy().counter(predicted)
    return "actuator_feasible_win" if counter_move == expected else "counter_policy_mismatch"


def _in_response_window(record: Mapping[str, object], *, response_prompt: str) -> bool:
    if bool(record.get("response_window", False)) or bool(record.get("response_window_latched", False)):
        return True
    return record.get("active_prompt") == response_prompt or record.get("raw_prompt") == response_prompt


def _joint_limit_violations(pose: HandPose, *, joint_min: float, joint_max: float) -> list[str]:
    return [name for name, value in pose.joints.items() if value < joint_min or value > joint_max]


def _parse_profile_weights(raw_weights: str | None, profile_count: int) -> list[float]:
    if raw_weights is None:
        return [1.0 / profile_count for _ in range(profile_count)]
    parsed = [float(item.strip()) for item in raw_weights.split(",") if item.strip()]
    if len(parsed) != profile_count:
        raise ValueError("profile_weights must match number of profiles")
    return parsed


def _load_jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        loaded: object = json.loads(line)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"{path} contains a non-object JSONL row")
        records.append({str(key): value for key, value in loaded.items()})
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_episode_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "run_label",
        "robot_start_pose",
        "predicted_gesture",
        "selected_counter_move",
        "decision_frame",
        "decision_time_s",
        "decision_latency_s",
        "remaining_time_s",
        "feasible",
        "required_time_s",
        "limiting_joint",
        "failure_reason",
        "episode_result",
    ]
    _write_dict_csv(path, [{key: record.get(key) for key in fieldnames} for record in records], fieldnames=fieldnames)


def _write_report_metrics_csv(
    path: Path,
    *,
    decision: ResponseWindowDecision,
    feasibility: MotionFeasibility,
    episode_result: str,
) -> None:
    rows = [
        {"metric": "decision_latency_s", "value": decision.decision_latency_s, "notes": "response prompt start to confirmed RPS decision"},
        {"metric": "remaining_time_s", "value": feasibility.remaining_time_s, "notes": "actuator deadline minus decision latency"},
        {"metric": "required_time_s", "value": feasibility.actuator_result.required_time_s, "notes": "response delay plus max joint travel time"},
        {"metric": "feasible", "value": feasibility.feasible, "notes": "deadline and joint-limit result"},
        {"metric": "episode_result", "value": episode_result, "notes": "final screen-rendered replay outcome"},
    ]
    _write_dict_csv(path, rows)


def _write_dict_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(str(key))
        fieldnames = keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_freeze_md(path: Path, payload: Mapping[str, object]) -> None:
    gates = cast(Mapping[str, object], payload["gates"])
    lines = [
        "# Final Demo v4 Policy Freeze",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Live/demo model family: `{payload.get('live_demo_model_family')}`",
        f"- Diagnostic family preserved: `{payload.get('preserved_diagnostic_model_family')}`",
        f"- Config: `{payload.get('config_path')}`",
        f"- Legacy default preserved: `{payload.get('legacy_launcher_default_preserved')}`",
        f"- Weights: `{payload.get('profile_weights')}`",
        f"- Confidence threshold: `{gates.get('confidence_threshold')}`",
        f"- Margin threshold: `{gates.get('margin_threshold')}`",
        f"- Confirmation count: `{gates.get('confirmation_count')}`",
        f"- Response prompt: `{gates.get('response_prompt')}`",
        "",
        "## Scope",
        "",
        "v4 is the only live/demo predictor. v7e remains report diagnostics only, and v7f retraining is not started.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_render_audit_md(path: Path, payload: Mapping[str, object]) -> None:
    lines = [
        "# Robot-Hand Render Audit",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Fresh remote render run: `{payload.get('fresh_remote_render_run')}`",
        f"- Archived postcondition: `{payload.get('archived_postcondition_path')}`",
        f"- Archived diagnostics: `{payload.get('archived_diagnostics_path')}`",
        f"- Articulation status: `{payload.get('articulation_status')}`",
        f"- Render validation status: `{payload.get('render_validation_status')}`",
        f"- DOF count: `{payload.get('dof_count')}`",
        f"- Dynamic overlay backend: `{payload.get('renderer_backend_for_dynamic_overlay')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json_if_exists(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, Mapping) else {}


def _read_required_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON artifact does not exist: {path}")
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _float_record(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"frame-log record field {key} must be numeric")
    return float(value)


def _optional_float(value: object, *, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


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


def _output_path(summary: Mapping[str, object], key: str) -> object:
    outputs = summary.get("outputs")
    if isinstance(outputs, Mapping):
        return outputs.get(key)
    return None


__all__ = [
    "KinematicMotionConfig",
    "MotionFeasibility",
    "ResponseWindowDecision",
    "build_final_counterattack_artifacts",
    "extract_response_window_decision",
    "final_robot_pose_for_decision",
    "freeze_final_demo_policy",
    "load_kinematic_motion_config",
    "validate_no_heldout_test_inputs",
    "write_final_report_assets",
    "write_render_audit_artifacts",
]
