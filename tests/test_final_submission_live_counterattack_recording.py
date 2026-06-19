from __future__ import annotations

import json
from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np
import pytest
from PIL import Image

from embodied_rps.final_submission_live_counterattack_recording import (
    LiveTakeInput,
    build_live_recording_artifacts_from_take_inputs,
    build_live_take_artifacts,
    default_take_specs,
    extract_live_take_decision,
    select_final_submission_take,
    summarize_take_decision_consistency,
    validate_archived_schunk_style_assets,
)


def _write_frame_log(path: Path, *, prompt: str, decision_state: str, confirmed: bool = True) -> None:
    rows = [
        {
            "frame_index": 1,
            "time_s": 0.0,
            "active_prompt": prompt,
            "response_prompt": prompt,
            "decision_state": "wait_counter_paper",
            "confirmed_decision": True,
            "confidence": 0.80,
            "margin": 0.30,
        },
        {
            "frame_index": 20,
            "time_s": 0.033333333,
            "active_prompt": prompt,
            "response_prompt": prompt,
            "decision_state": decision_state,
            "confirmed_decision": confirmed,
            "confidence": 0.95,
            "margin": 0.70,
            "p_rock": 0.95 if decision_state in {"rock", "wait_counter_paper"} else 0.02,
            "p_paper": 0.95 if decision_state == "paper" else 0.02,
            "p_scissors": 0.95 if decision_state == "scissors" else 0.02,
        },
    ]
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _write_style_assets(root: Path) -> None:
    colors = {
        "rock": (80, 90, 110),
        "paper": (180, 210, 240),
        "scissors": (210, 180, 120),
    }
    root.mkdir(parents=True, exist_ok=True)
    for gesture, color in colors.items():
        image = Image.new("RGB", (160, 120), color)
        image.save(root / f"{gesture}_view_yaw45_pitch20.png")


def _write_overlay_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened()
    for index in range(8):
        frame = np.full((120, 160, 3), 30 + index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_default_take_specs_cover_three_counterattacks() -> None:
    specs = default_take_specs()

    assert [spec.take_id for spec in specs] == [
        "take_01_human_rock_robot_paper",
        "take_02_human_paper_robot_scissors",
        "take_03_human_scissors_robot_rock",
    ]
    assert [(spec.human_target, spec.robot_counter, spec.response_prompt) for spec in specs] == [
        ("rock", "paper", "scissors"),
        ("paper", "scissors", "scissors"),
        ("scissors", "rock", "scissors"),
    ]
    assert [spec.prompt_sequence for spec in specs] == [
        "rock,paper,scissors",
        "rock,paper,scissors",
        "rock,paper,scissors",
    ]
    assert specs[0].accept_wait_as_rock is True
    assert specs[1].accept_wait_as_rock is False
    assert specs[2].accept_wait_as_rock is False


def test_extract_live_take_decision_accepts_wait_state_only_for_rock_take(tmp_path: Path) -> None:
    frame_log = tmp_path / "rock_frames.jsonl"
    _write_frame_log(frame_log, prompt="scissors", decision_state="wait_counter_paper")
    rock_spec = default_take_specs()[0]
    paper_spec = default_take_specs()[1]

    decision = extract_live_take_decision(frame_log, rock_spec)

    assert decision.predicted_gesture == "rock"
    assert decision.counter_move == "paper"
    assert decision.decision_latency_s == pytest.approx(0.033333333)

    with pytest.raises(ValueError, match="No confirmed"):
        extract_live_take_decision(frame_log, paper_spec)


def test_rock_take_consistency_rejects_later_confirmed_conflicts(tmp_path: Path) -> None:
    frame_log = tmp_path / "rock_conflict_frames.jsonl"
    rows = [
        {
            "frame_index": 1,
            "time_s": 0.0,
            "active_prompt": "scissors",
            "decision_state": "wait_counter_paper",
            "confirmed_decision": True,
        },
        {
            "frame_index": 2,
            "time_s": 0.033333333,
            "active_prompt": "scissors",
            "decision_state": "wait_counter_paper",
            "confirmed_decision": True,
        },
        {
            "frame_index": 3,
            "time_s": 0.066666667,
            "active_prompt": "scissors",
            "decision_state": "scissors",
            "confirmed_decision": True,
        },
    ]
    frame_log.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    spec = default_take_specs()[0]

    decision = extract_live_take_decision(frame_log, spec)
    consistency = summarize_take_decision_consistency(frame_log, spec, decision)

    assert consistency["passed"] is False
    assert consistency["conflict_confirmed_count"] == 1


def test_select_final_submission_take_prefers_visible_paper_counter() -> None:
    summaries = [
        {"take_id": "take_01_human_rock_robot_paper", "status": "passed"},
        {"take_id": "take_02_human_paper_robot_scissors", "status": "passed"},
        {"take_id": "take_03_human_scissors_robot_rock", "status": "passed"},
    ]

    selected = select_final_submission_take(summaries)

    assert selected is not None
    assert selected["take_id"] == "take_02_human_paper_robot_scissors"


def test_validate_archived_schunk_style_assets_requires_three_yaw45_images(tmp_path: Path) -> None:
    _write_style_assets(tmp_path)

    assets = validate_archived_schunk_style_assets(tmp_path)

    assert set(assets) == {"rock", "paper", "scissors"}
    assert assets["paper"].name == "paper_view_yaw45_pitch20.png"


def test_build_live_take_artifacts_writes_styled_video_and_summary(tmp_path: Path) -> None:
    frame_log = tmp_path / "paper_frames.jsonl"
    overlay = tmp_path / "paper_overlay.mp4"
    style_root = tmp_path / "style"
    _write_frame_log(frame_log, prompt="scissors", decision_state="paper")
    _write_overlay_video(overlay)
    _write_style_assets(style_root)

    summary = build_live_take_artifacts(
        spec=default_take_specs()[1],
        overlay_video=overlay,
        frame_log=frame_log,
        style_asset_root=style_root,
        pose_config_path=Path("configs/kinematic_rps.yaml"),
        output_root=tmp_path / "take_02",
        project_root=Path.cwd(),
    )

    assert summary["status"] == "passed"
    assert summary["human_target"] == "paper"
    assert summary["robot_counter"] == "scissors"
    outputs = summary["outputs"]
    assert Path(outputs["styled_counterattack_mp4"]).exists()
    assert Path(outputs["poster_png"]).exists()
    assert Path(outputs["take_summary_json"]).exists()
    assert Path(outputs["feasibility_summary_json"]).exists()


def test_build_live_recording_artifacts_writes_three_takes_and_selected_dir(tmp_path: Path) -> None:
    style_root = tmp_path / "style"
    _write_style_assets(style_root)
    take_inputs: dict[str, LiveTakeInput] = {}
    for spec in default_take_specs():
        take_dir = tmp_path / "raw" / spec.take_id
        take_dir.mkdir(parents=True, exist_ok=True)
        frame_log = take_dir / "frames.jsonl"
        overlay = take_dir / "overlay.mp4"
        _write_frame_log(frame_log, prompt=spec.response_prompt, decision_state=spec.human_target)
        _write_overlay_video(overlay)
        take_inputs[spec.take_id] = LiveTakeInput(overlay_video=overlay, frame_log=frame_log)

    manifest = build_live_recording_artifacts_from_take_inputs(
        take_inputs=take_inputs,
        style_asset_root=style_root,
        pose_config_path=Path("configs/kinematic_rps.yaml"),
        output_root=tmp_path / "recording",
        project_root=Path.cwd(),
    )

    assert manifest["status"] == "passed"
    assert manifest["selected_final_submission_take"]["selected_take_id"] == "take_02_human_paper_robot_scissors"
    for spec in default_take_specs():
        assert (tmp_path / "recording" / spec.take_id / "take_summary.json").exists()
        assert (tmp_path / "recording" / spec.take_id / "styled_counterattack.mp4").exists()
    selected = tmp_path / "recording" / "selected_final_submission_take"
    assert (selected / "final_submission_live_counterattack.mp4").exists()
    assert (selected / "selected_take_summary.json").exists()
