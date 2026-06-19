"""Final submission live counterattack recording artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import cv2  # type: ignore[import-untyped]
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-untyped]

from embodied_rps.final_robot_counterattack_demo import (
    KinematicMotionConfig,
    MotionFeasibility,
    ResponseWindowDecision,
    load_kinematic_motion_config,
    validate_no_heldout_test_inputs,
)
from embodied_rps.policy import CounterMovePolicy, OpponentGesture
from embodied_rps.real_skeleton_video_eval import WAIT_COUNTER_PAPER_STATE

RobotGesture = Literal["rock", "paper", "scissors"]
STYLE_GESTURES: tuple[RobotGesture, ...] = ("rock", "paper", "scissors")
PREFERRED_FINAL_TAKE_ORDER: tuple[str, ...] = (
    "take_02_human_paper_robot_scissors",
    "take_01_human_rock_robot_paper",
    "take_03_human_scissors_robot_rock",
)


@dataclass(frozen=True)
class LiveCounterattackTakeSpec:
    """One final-submission live take contract."""

    take_id: str
    human_target: OpponentGesture
    robot_counter: OpponentGesture
    response_prompt: OpponentGesture
    prompt_sequence: str
    accept_wait_as_rock: bool = False
    selection_priority: int = 99


@dataclass(frozen=True)
class LiveTakeInput:
    """Raw capture artifacts that can be styled into one take."""

    overlay_video: Path
    frame_log: Path
    skeleton_npz: Path | None = None


def default_take_specs() -> tuple[LiveCounterattackTakeSpec, ...]:
    """Return the three required final live counterattack takes."""

    return (
        LiveCounterattackTakeSpec(
            take_id="take_01_human_rock_robot_paper",
            human_target="rock",
            robot_counter="paper",
            response_prompt="scissors",
            prompt_sequence="rock,paper,scissors",
            accept_wait_as_rock=True,
            selection_priority=2,
        ),
        LiveCounterattackTakeSpec(
            take_id="take_02_human_paper_robot_scissors",
            human_target="paper",
            robot_counter="scissors",
            response_prompt="scissors",
            prompt_sequence="rock,paper,scissors",
            selection_priority=1,
        ),
        LiveCounterattackTakeSpec(
            take_id="take_03_human_scissors_robot_rock",
            human_target="scissors",
            robot_counter="rock",
            response_prompt="scissors",
            prompt_sequence="rock,paper,scissors",
            selection_priority=3,
        ),
    )


def validate_archived_schunk_style_assets(style_asset_root: Path) -> dict[RobotGesture, Path]:
    """Validate archived yaw45/pitch20 SCHUNK keyframes for the styled overlay."""

    assets: dict[RobotGesture, Path] = {}
    for gesture in STYLE_GESTURES:
        path = style_asset_root / f"{gesture}_view_yaw45_pitch20.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing archived SCHUNK style image: {path}")
        with Image.open(path) as image:
            if image.width <= 0 or image.height <= 0:
                raise ValueError(f"Archived SCHUNK style image is empty: {path}")
        assets[gesture] = path
    return assets


def extract_live_take_decision(frame_log: Path, spec: LiveCounterattackTakeSpec) -> ResponseWindowDecision:
    """Extract the first confirmed take-specific decision in the response window."""

    records = _load_jsonl_records(frame_log)
    response_window_start_time_s: float | None = None
    for record in records:
        if _in_response_window(record, response_prompt=spec.response_prompt):
            response_window_start_time_s = _float_record(record, "time_s")
            break
    if response_window_start_time_s is None:
        raise ValueError(f"No confirmed response-window decision found for take {spec.take_id}")

    policy = CounterMovePolicy()
    for record in records:
        if not _in_response_window(record, response_prompt=spec.response_prompt):
            continue
        record_time_s = _float_record(record, "time_s")
        if not bool(record.get("confirmed_decision", False)):
            continue
        predicted = _normalize_take_decision(record.get("decision_state"), spec)
        if predicted is None:
            continue
        if spec.accept_wait_as_rock and record.get("decision_state") == WAIT_COUNTER_PAPER_STATE and record_time_s <= response_window_start_time_s:
            continue
        if predicted != spec.human_target:
            continue
        counter_move = policy.counter(predicted)
        if counter_move != spec.robot_counter:
            raise ValueError(
                f"Take {spec.take_id} expected robot counter {spec.robot_counter!r}, got {counter_move!r}"
            )
        return ResponseWindowDecision(
            predicted_gesture=predicted,
            counter_move=counter_move,
            decision_frame=int(_float_record(record, "frame_index")),
            decision_time_s=record_time_s,
            response_window_start_time_s=response_window_start_time_s,
            decision_latency_s=record_time_s - response_window_start_time_s,
            confidence=_optional_float(record.get("confidence"), default=0.0),
            confidence_margin=_optional_float(record.get("margin"), default=0.0),
            probabilities={
                "rock": _optional_float(record.get("p_rock"), default=0.0),
                "paper": _optional_float(record.get("p_paper"), default=0.0),
                "scissors": _optional_float(record.get("p_scissors"), default=0.0),
            },
        )
    raise ValueError(f"No confirmed response-window decision found for take {spec.take_id}")


def build_live_take_artifacts(
    *,
    spec: LiveCounterattackTakeSpec,
    overlay_video: Path,
    frame_log: Path,
    style_asset_root: Path,
    pose_config_path: Path,
    output_root: Path,
    project_root: Path,
    skeleton_npz: Path | None = None,
) -> dict[str, object]:
    """Build styled SCHUNK live-take video, logs, and summary artifacts."""

    validate_no_heldout_test_inputs([overlay_video, frame_log])
    style_assets = validate_archived_schunk_style_assets(style_asset_root)
    output_root.mkdir(parents=True, exist_ok=True)

    decision: ResponseWindowDecision | None = None
    feasibility: MotionFeasibility | None = None
    consistency_summary: dict[str, object] | None = None
    episode_result = "no_confirmed_response_window_decision"
    failure_reason: str | None = None
    status = "failed"
    try:
        decision = extract_live_take_decision(frame_log, spec)
        consistency_summary = summarize_take_decision_consistency(frame_log, spec, decision)
        if not bool(consistency_summary["passed"]):
            raise ValueError(str(consistency_summary["failure_reason"]))
        motion = load_kinematic_motion_config(pose_config_path)
        feasibility = motion.check_from_rock(cast(RobotGesture, decision.counter_move), decision_latency_s=decision.decision_latency_s)
        episode_result = _episode_result(decision.predicted_gesture, decision.counter_move, feasibility.feasible)
        status = "passed" if feasibility.feasible else "infeasible"
    except Exception as exc:
        failure_reason = str(exc)
        motion = load_kinematic_motion_config(pose_config_path)

    raw_capture_summary = output_root / "raw_capture_summary.json"
    feasibility_summary_json = output_root / "feasibility_summary.json"
    robot_motion_log = output_root / "robot_motion_log.jsonl"
    take_summary_json = output_root / "take_summary.json"
    take_summary_md = output_root / "take_summary.md"
    styled_video = output_root / "styled_counterattack.mp4"
    poster_png = output_root / "poster_frame.png"
    episode_summary_csv = output_root / "episode_summary.csv"

    if decision is not None and feasibility is not None:
        _write_styled_counterattack_video(
            overlay_video=overlay_video,
            output_video=styled_video,
            poster_path=poster_png,
            style_assets=style_assets,
            decision=decision,
            feasibility=feasibility,
            response_delay_s=motion.config.response_delay_s,
            spec=spec,
        )
        motion_record = _motion_record(
            spec=spec,
            decision=decision,
            feasibility=feasibility,
            episode_result=episode_result,
            overlay_video=overlay_video,
            frame_log=frame_log,
            skeleton_npz=skeleton_npz,
            pose_config_path=pose_config_path,
            style_asset_root=style_asset_root,
            project_root=project_root,
        )
        _write_jsonl(robot_motion_log, [motion_record])
        _write_episode_csv(episode_summary_csv, [motion_record])
        feasibility_payload: dict[str, object] = {
            **motion_record,
            "status": status,
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
    else:
        _unlink_if_exists(styled_video)
        _unlink_if_exists(poster_png)
        _write_jsonl(robot_motion_log, [])
        _write_episode_csv(episode_summary_csv, [])
        feasibility_payload = {
            "status": status,
            "take_id": spec.take_id,
            "human_target": spec.human_target,
            "robot_counter": spec.robot_counter,
            "failure_reason": failure_reason,
        }

    raw_payload = {
        "take_id": spec.take_id,
        "overlay_video": _relative_path(overlay_video, project_root=project_root),
        "frame_log": _relative_path(frame_log, project_root=project_root),
        "skeleton_npz": _relative_path(skeleton_npz, project_root=project_root),
        "source_scope": "new live capture for final submission candidate selection",
    }
    _write_json(raw_capture_summary, raw_payload)
    _write_json(feasibility_summary_json, feasibility_payload)

    summary: dict[str, object] = {
        "status": status,
        "take_id": spec.take_id,
        "human_target": spec.human_target,
        "robot_counter": spec.robot_counter,
        "response_prompt": spec.response_prompt,
        "prompt_sequence": spec.prompt_sequence,
        "accept_wait_as_rock": spec.accept_wait_as_rock,
        "selection_priority": spec.selection_priority,
        "claim_scope": "new live counterattack take; replay rehearsal artifacts are validation-only",
        "robot_start_pose": "rock",
        "archived_schunk_style": True,
        "style_asset_root": _relative_path(style_asset_root, project_root=project_root),
        "overlay_video": _relative_path(overlay_video, project_root=project_root),
        "frame_log": _relative_path(frame_log, project_root=project_root),
        "skeleton_npz": _relative_path(skeleton_npz, project_root=project_root),
        "failure_reason": failure_reason,
        "decision_consistency": consistency_summary,
        "predicted_gesture": decision.predicted_gesture if decision else None,
        "selected_counter_move": decision.counter_move if decision else None,
        "decision_frame": decision.decision_frame if decision else None,
        "decision_time_s": decision.decision_time_s if decision else None,
        "decision_latency_s": decision.decision_latency_s if decision else None,
        "remaining_time_s": feasibility.remaining_time_s if feasibility else None,
        "feasible": feasibility.feasible if feasibility else False,
        "episode_result": episode_result,
        "outputs": {
            "raw_capture_summary_json": _relative_path(raw_capture_summary, project_root=project_root),
            "robot_motion_log_jsonl": _relative_path(robot_motion_log, project_root=project_root),
            "feasibility_summary_json": _relative_path(feasibility_summary_json, project_root=project_root),
            "episode_summary_csv": _relative_path(episode_summary_csv, project_root=project_root),
            "styled_counterattack_mp4": _relative_path(styled_video, project_root=project_root)
            if decision is not None and feasibility is not None and styled_video.exists()
            else None,
            "poster_png": _relative_path(poster_png, project_root=project_root)
            if decision is not None and feasibility is not None and poster_png.exists()
            else None,
            "take_summary_json": _relative_path(take_summary_json, project_root=project_root),
            "take_summary_md": _relative_path(take_summary_md, project_root=project_root),
        },
    }
    _write_json(take_summary_json, summary)
    _write_take_summary_md(take_summary_md, summary)
    return summary


def summarize_take_decision_consistency(
    frame_log: Path,
    spec: LiveCounterattackTakeSpec,
    decision: ResponseWindowDecision,
    *,
    min_target_confirmed_count: int = 1,
    min_target_ratio: float = 0.70,
) -> dict[str, object]:
    """Check that the selected live decision is not contradicted by later confirmed states."""

    records = _load_jsonl_records(frame_log)
    target_count = 0
    conflict_count = 0
    wait_count = 0
    checked_count = 0
    for record in records:
        if not _in_response_window(record, response_prompt=spec.response_prompt):
            continue
        if _float_record(record, "time_s") < decision.decision_time_s:
            continue
        if not bool(record.get("confirmed_decision", False)):
            continue
        raw_state = record.get("decision_state")
        normalized = _normalize_take_decision(raw_state, spec)
        if normalized is None:
            continue
        checked_count += 1
        if raw_state == WAIT_COUNTER_PAPER_STATE:
            wait_count += 1
        if normalized == spec.human_target:
            target_count += 1
        else:
            conflict_count += 1
    denominator = target_count + conflict_count
    target_ratio = float(target_count / denominator) if denominator else 0.0
    passed = target_count >= min_target_confirmed_count and target_ratio >= min_target_ratio and conflict_count == 0
    failure_reason = None
    if not passed:
        failure_reason = (
            f"Confirmed decision consistency failed for {spec.take_id}: "
            f"target_count={target_count}, conflict_count={conflict_count}, target_ratio={target_ratio:.3f}"
        )
    return {
        "passed": passed,
        "checked_confirmed_count": checked_count,
        "target_confirmed_count": target_count,
        "conflict_confirmed_count": conflict_count,
        "wait_confirmed_count": wait_count,
        "target_ratio": target_ratio,
        "min_target_confirmed_count": min_target_confirmed_count,
        "min_target_ratio": min_target_ratio,
        "failure_reason": failure_reason,
    }


def build_live_recording_artifacts_from_take_inputs(
    *,
    take_inputs: Mapping[str, LiveTakeInput],
    style_asset_root: Path,
    pose_config_path: Path,
    output_root: Path,
    project_root: Path,
) -> dict[str, object]:
    """Build all three take directories and selected-final directory from raw captures."""

    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for spec in default_take_specs():
        take_input = take_inputs.get(spec.take_id)
        if take_input is None:
            summary = _missing_take_summary(spec, output_root=output_root / spec.take_id, project_root=project_root)
        else:
            summary = build_live_take_artifacts(
                spec=spec,
                overlay_video=take_input.overlay_video,
                frame_log=take_input.frame_log,
                skeleton_npz=take_input.skeleton_npz,
                style_asset_root=style_asset_root,
                pose_config_path=pose_config_path,
                output_root=output_root / spec.take_id,
                project_root=project_root,
            )
        summaries.append(summary)

    selected = select_final_submission_take(summaries)
    selected_dir = output_root / "selected_final_submission_take"
    selected_payload = _write_selected_final_take(selected, selected_dir=selected_dir, project_root=project_root)
    manifest = {
        "status": "passed" if selected is not None else "failed",
        "claim_scope": "three new live counterattack takes plus one selected final submission candidate; final packaging not started",
        "selection_policy": "paper->scissors first, rock->paper second, scissors->rock only if it is the only passed take",
        "replay_rehearsal_policy": "validation-only; not selected as final submission video",
        "takes": summaries,
        "selected_final_submission_take": selected_payload,
        "outputs": {
            "selected_final_dir": _relative_path(selected_dir, project_root=project_root),
            "manifest_json": _relative_path(output_root / "live_recording_manifest.json", project_root=project_root),
        },
    }
    _write_json(output_root / "live_recording_manifest.json", manifest)
    return manifest


def select_final_submission_take(summaries: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    """Select the final submission candidate by the requested priority policy."""

    passed_by_take_id = {
        str(summary.get("take_id")): summary
        for summary in summaries
        if summary.get("status") == "passed" and _has_required_selected_outputs(summary)
    }
    for take_id in PREFERRED_FINAL_TAKE_ORDER:
        if take_id in passed_by_take_id:
            return passed_by_take_id[take_id]
    return None


def _write_selected_final_take(
    selected: Mapping[str, object] | None,
    *,
    selected_dir: Path,
    project_root: Path,
) -> dict[str, object] | None:
    selected_dir.mkdir(parents=True, exist_ok=True)
    if selected is None:
        payload = {
            "status": "failed",
            "reason": "No take met the final-submission pass criteria",
        }
        _write_json(selected_dir / "selected_take_summary.json", payload)
        return payload

    outputs = selected.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Selected take summary does not contain outputs")
    styled_value = outputs.get("styled_counterattack_mp4")
    if not isinstance(styled_value, str):
        raise ValueError("Selected take summary does not contain a styled MP4")
    styled_source = _project_path(styled_value, project_root=project_root)
    selected_mp4 = selected_dir / "final_submission_live_counterattack.mp4"
    shutil.copy2(styled_source, selected_mp4)

    copied_assets: dict[str, str | None] = {
        "final_submission_mp4": _relative_path(selected_mp4, project_root=project_root),
    }
    for output_key, target_name in (
        ("poster_png", "poster_frame.png"),
        ("take_summary_json", "take_summary.json"),
        ("feasibility_summary_json", "feasibility_summary.json"),
    ):
        value = outputs.get(output_key)
        if isinstance(value, str):
            source = _project_path(value, project_root=project_root)
            target = selected_dir / target_name
            shutil.copy2(source, target)
            copied_assets[output_key] = _relative_path(target, project_root=project_root)
        else:
            copied_assets[output_key] = None

    payload = {
        "status": "passed",
        "selected_take_id": selected.get("take_id"),
        "human_target": selected.get("human_target"),
        "robot_counter": selected.get("robot_counter"),
        "episode_result": selected.get("episode_result"),
        "selection_reason": "highest-priority passed live take under the final recording policy",
        "outputs": copied_assets,
    }
    _write_json(selected_dir / "selected_take_summary.json", payload)
    return payload


def _missing_take_summary(
    spec: LiveCounterattackTakeSpec,
    *,
    output_root: Path,
    project_root: Path,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "missing",
        "take_id": spec.take_id,
        "human_target": spec.human_target,
        "robot_counter": spec.robot_counter,
        "response_prompt": spec.response_prompt,
        "failure_reason": "Raw live capture inputs were not provided",
        "outputs": {
            "take_summary_json": _relative_path(output_root / "take_summary.json", project_root=project_root),
        },
    }
    _write_json(output_root / "take_summary.json", summary)
    return summary


def _write_styled_counterattack_video(
    *,
    overlay_video: Path,
    output_video: Path,
    poster_path: Path,
    style_assets: Mapping[RobotGesture, Path],
    decision: ResponseWindowDecision,
    feasibility: MotionFeasibility,
    response_delay_s: float,
    spec: LiveCounterattackTakeSpec,
) -> None:
    if not overlay_video.exists():
        raise FileNotFoundError(f"overlay_video does not exist: {overlay_video}")
    capture = cv2.VideoCapture(str(overlay_video))
    if not capture.isOpened():
        raise ValueError(f"could not open overlay video: {overlay_video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width, height = 1920, 1080
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not open output video writer: {output_video}")

    start_image = _load_style_image(style_assets["rock"])
    target_image = _load_style_image(style_assets[cast(RobotGesture, decision.counter_move)])
    font = ImageFont.load_default()
    poster_frame: Image.Image | None = None
    frame_index = 0
    motion_time_s = _motion_only_time(feasibility, response_delay_s=response_delay_s)
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
                motion_time_s=motion_time_s,
            )
            robot_image = _blend_robot_style_frame(
                start_image=start_image,
                target_image=target_image,
                progress=progress,
                spec=spec,
            )
            overlay_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            composite = _compose_live_frame(
                overlay_rgb=overlay_rgb,
                robot_image=robot_image,
                decision=decision,
                feasibility=feasibility,
                source_time_s=source_time_s,
                spec=spec,
                output_size=(width, height),
                font=font,
            )
            if progress >= 1.0:
                poster_frame = composite.copy()
            elif poster_frame is None:
                poster_frame = composite.copy()
            writer.write(cv2.cvtColor(np.asarray(composite), cv2.COLOR_RGB2BGR))
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    if frame_index == 0 or poster_frame is None:
        raise ValueError("overlay video did not contain frames")
    poster_frame.save(poster_path)


def _load_style_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _blend_robot_style_frame(
    *,
    start_image: Image.Image,
    target_image: Image.Image,
    progress: float,
    spec: LiveCounterattackTakeSpec,
) -> Image.Image:
    if spec.robot_counter == "rock":
        blended = start_image.copy()
    else:
        blended = Image.blend(start_image, target_image.resize(start_image.size, _resampling_lanczos()), progress)
    draw = ImageDraw.Draw(blended)
    font = ImageFont.load_default()
    label = f"robot rock->{spec.robot_counter}  {progress:.0%}"
    draw.rectangle((0, 0, blended.width, 36), fill=(18, 24, 32))
    draw.text((14, 12), label, fill=(244, 247, 251), font=font)
    return blended


def _compose_live_frame(
    *,
    overlay_rgb: Image.Image,
    robot_image: Image.Image,
    decision: ResponseWindowDecision,
    feasibility: MotionFeasibility,
    source_time_s: float,
    spec: LiveCounterattackTakeSpec,
    output_size: tuple[int, int],
    font: ImageFont.ImageFont,
) -> Image.Image:
    width, height = output_size
    canvas = Image.new("RGB", (width, height), (238, 241, 245))
    draw = ImageDraw.Draw(canvas)
    header_h = 74
    draw.rectangle((0, 0, width, header_h), fill=(21, 28, 38))
    draw.text((28, 18), "Final live RPS robot counterattack recording", fill=(247, 249, 252), font=font)
    status = "FEASIBLE" if feasibility.feasible else "INFEASIBLE"
    draw.text((width - 390, 18), f"{status}  remaining={feasibility.remaining_time_s:.3f}s", fill=(219, 234, 254), font=font)

    margin = 24
    gutter = 20
    footer_h = 86
    content_top = header_h + margin
    content_h = height - content_top - margin - footer_h
    left_w = int((width - margin * 2 - gutter) * 0.60)
    right_w = width - margin * 2 - gutter - left_w
    left_box = (margin, content_top, margin + left_w, content_top + content_h)
    right_box = (margin + left_w + gutter, content_top, width - margin, content_top + content_h)
    _paste_fit(canvas, overlay_rgb, left_box, background=(218, 225, 233))
    _paste_fit(canvas, robot_image, right_box, background=(230, 234, 240))
    draw.rectangle(left_box, outline=(122, 138, 158), width=2)
    draw.rectangle(right_box, outline=(122, 138, 158), width=2)
    draw.text((left_box[0] + 14, left_box[1] + 12), f"human target: {spec.human_target}", fill=(15, 23, 42), font=font)
    draw.text((right_box[0] + 14, right_box[1] + 12), "archived SCHUNK render style", fill=(15, 23, 42), font=font)

    footer_top = height - margin - footer_h
    draw.rectangle((margin, footer_top, width - margin, height - margin), fill=(255, 255, 255), outline=(190, 202, 216))
    row1 = (
        f"prompt={spec.response_prompt} | prediction={decision.predicted_gesture} | "
        f"confidence={decision.confidence:.2f} | counter={decision.counter_move} | "
        f"time={source_time_s:.2f}s"
    )
    row2 = (
        f"latency={decision.decision_latency_s:.3f}s | deadline remaining={feasibility.remaining_time_s:.3f}s | "
        f"required={feasibility.actuator_result.required_time_s:.3f}s | "
        f"result={_episode_result(decision.predicted_gesture, decision.counter_move, feasibility.feasible)}"
    )
    draw.text((margin + 16, footer_top + 22), row1, fill=(15, 23, 42), font=font)
    draw.text((margin + 16, footer_top + 50), row2, fill=(51, 65, 85), font=font)
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
    copied.thumbnail((width - 26, height - 54), _resampling_lanczos())
    tile.paste(copied, ((width - copied.width) // 2, (height - copied.height) // 2 + 18))
    canvas.paste(tile, (left, top))


def _motion_only_time(feasibility: MotionFeasibility, *, response_delay_s: float) -> float:
    return max(0.0, feasibility.actuator_result.required_time_s - response_delay_s)


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


def _motion_record(
    *,
    spec: LiveCounterattackTakeSpec,
    decision: ResponseWindowDecision,
    feasibility: MotionFeasibility,
    episode_result: str,
    overlay_video: Path,
    frame_log: Path,
    skeleton_npz: Path | None,
    pose_config_path: Path,
    style_asset_root: Path,
    project_root: Path,
) -> dict[str, object]:
    return {
        "take_id": spec.take_id,
        "human_target": spec.human_target,
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
        "overlay_video": _relative_path(overlay_video, project_root=project_root),
        "frame_log": _relative_path(frame_log, project_root=project_root),
        "skeleton_npz": _relative_path(skeleton_npz, project_root=project_root),
        "pose_config_path": _relative_path(pose_config_path, project_root=project_root),
        "style_asset_root": _relative_path(style_asset_root, project_root=project_root),
    }


def _episode_result(predicted: OpponentGesture, counter_move: OpponentGesture, feasible: bool) -> str:
    if not feasible:
        return "infeasible_counter"
    expected = CounterMovePolicy().counter(predicted)
    return "actuator_feasible_win" if counter_move == expected else "counter_policy_mismatch"


def _normalize_take_decision(value: object, spec: LiveCounterattackTakeSpec) -> OpponentGesture | None:
    if value in ("rock", "paper", "scissors"):
        return cast(OpponentGesture, value)
    if spec.accept_wait_as_rock and value == WAIT_COUNTER_PAPER_STATE:
        return "rock"
    return None


def _in_response_window(record: Mapping[str, object], *, response_prompt: str) -> bool:
    if bool(record.get("response_window", False)) or bool(record.get("response_window_latched", False)):
        return True
    return record.get("active_prompt") == response_prompt or record.get("raw_prompt") == response_prompt


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


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def _write_episode_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "take_id",
        "human_target",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def _write_take_summary_md(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        f"# {summary.get('take_id')}",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Human target: `{summary.get('human_target')}`",
        f"- Robot counter: `{summary.get('robot_counter')}`",
        f"- Prediction: `{summary.get('predicted_gesture')}`",
        f"- Decision latency: `{summary.get('decision_latency_s')}`",
        f"- Episode result: `{summary.get('episode_result')}`",
        f"- Failure reason: `{summary.get('failure_reason')}`",
        "",
        "This is a new live recording take. Archived replay MP4s remain validation artifacts only.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _float_record(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"frame-log record field {key} must be numeric")
    return float(value)


def _optional_float(value: object, *, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _has_required_selected_outputs(summary: Mapping[str, object]) -> bool:
    outputs = summary.get("outputs")
    if not isinstance(outputs, Mapping):
        return True
    value = outputs.get("styled_counterattack_mp4")
    if value is None:
        return True
    return isinstance(value, str) and value != ""


def _nonempty_output(summary: Mapping[str, object], key: str) -> bool:
    outputs = summary.get("outputs")
    if not isinstance(outputs, Mapping):
        return False
    value = outputs.get(key)
    return isinstance(value, str) and value != ""


def _project_path(path_text: str, *, project_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


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


def _resampling_lanczos() -> int:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return int(resampling.LANCZOS)
    return int(Image.LANCZOS)


__all__ = [
    "LiveCounterattackTakeSpec",
    "LiveTakeInput",
    "build_live_recording_artifacts_from_take_inputs",
    "build_live_take_artifacts",
    "default_take_specs",
    "extract_live_take_decision",
    "select_final_submission_take",
    "summarize_take_decision_consistency",
    "validate_archived_schunk_style_assets",
]
