from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodied_rps.final_robot_counterattack_demo import (
    build_final_counterattack_artifacts,
    extract_response_window_decision,
    final_robot_pose_for_decision,
    freeze_final_demo_policy,
    load_kinematic_motion_config,
    validate_no_heldout_test_inputs,
)


def _write_fixture_frame_log(path: Path) -> None:
    rows = [
        {
            "frame_index": 1,
            "time_s": 0.0,
            "active_prompt": "rock",
            "response_prompt": "scissors",
            "decision_state": "wait_counter_paper",
            "confirmed_decision": True,
            "confidence": 1.0,
            "margin": 1.0,
        },
        {
            "frame_index": 61,
            "time_s": 2.0,
            "active_prompt": "scissors",
            "response_prompt": "scissors",
            "decision_state": "wait_counter_paper",
            "confirmed_decision": True,
            "confidence": 1.0,
            "margin": 1.0,
        },
        {
            "frame_index": 65,
            "time_s": 2.133333333,
            "active_prompt": "scissors",
            "response_prompt": "scissors",
            "decision_state": "scissors",
            "confirmed_decision": True,
            "confidence": 1.0,
            "margin": 1.0,
            "p_rock": 0.0,
            "p_paper": 0.0,
            "p_scissors": 1.0,
        },
    ]
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_freeze_final_demo_policy_records_v4_profiles_and_rejects_v7_live_policy(tmp_path: Path) -> None:
    freeze = freeze_final_demo_policy(
        config_path=Path("configs/realtime_two_stage_selector_demo_conditional_scissors_rescue_rock_guard.yaml"),
        output_root=tmp_path,
        project_root=Path.cwd(),
    )

    assert freeze["status"] == "passed"
    assert freeze["live_demo_model_family"] == "v4"
    assert freeze["preserved_diagnostic_model_family"] == "v7e"
    assert freeze["profile_weights"] == [0.25, 0.75, 0.0]
    assert all("v4" in profile["path"] for profile in freeze["profiles"])
    assert (tmp_path / "v4_live_demo_policy_freeze.json").exists()

    bad_config = tmp_path / "bad_v7e.yaml"
    bad_config.write_text(
        "\n".join(
            [
                "profiles:",
                "  - results/model_profiles/real_skeleton_three_class_wait_v7e_stage1_tcn_seed11.json",
                "profile_weights: 1.0",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="v4"):
        freeze_final_demo_policy(config_path=bad_config, output_root=tmp_path / "bad", project_root=Path.cwd())


def test_final_wait_state_keeps_robot_at_rock_until_response_prediction() -> None:
    assert final_robot_pose_for_decision("wait_counter_paper", confirmed=True, in_response_window=True) == "rock"
    assert final_robot_pose_for_decision("paper", confirmed=True, in_response_window=False) == "rock"
    assert final_robot_pose_for_decision("paper", confirmed=False, in_response_window=True) == "rock"
    assert final_robot_pose_for_decision("paper", confirmed=True, in_response_window=True) == "scissors"


def test_extract_response_window_decision_ignores_provisional_wait_state(tmp_path: Path) -> None:
    frame_log = tmp_path / "frames.jsonl"
    _write_fixture_frame_log(frame_log)

    decision = extract_response_window_decision(frame_log, response_prompt="scissors")

    assert decision.predicted_gesture == "scissors"
    assert decision.counter_move == "rock"
    assert decision.decision_frame == 65
    assert decision.decision_latency_s == pytest.approx(0.133333333)


def test_load_kinematic_motion_config_checks_rock_to_counter_feasibility() -> None:
    motion = load_kinematic_motion_config(Path("configs/kinematic_rps.yaml"))

    result = motion.check_from_rock("rock", decision_latency_s=0.133333333)

    assert result.feasible is True
    assert result.remaining_time_s == pytest.approx(0.366666667)
    assert result.actuator_result.failure_reason is None


def test_validate_no_heldout_test_inputs_rejects_test_mp4_paths() -> None:
    with pytest.raises(ValueError, match="validation-only"):
        validate_no_heldout_test_inputs([Path("dataset/heldout/test/example.mp4")])


def test_build_final_counterattack_artifacts_writes_logs_tables_and_previews(tmp_path: Path) -> None:
    frame_log = tmp_path / "frames.jsonl"
    _write_fixture_frame_log(frame_log)

    summary = build_final_counterattack_artifacts(
        overlay_video=None,
        frame_log=frame_log,
        config_path=Path("configs/realtime_two_stage_selector_demo_conditional_scissors_rescue_rock_guard.yaml"),
        pose_config_path=Path("configs/kinematic_rps.yaml"),
        schunk_pose_config_path=Path("configs/schunk_rps_poses.yaml"),
        output_root=tmp_path / "demo",
        project_root=Path.cwd(),
        run_label="fixture",
    )

    assert summary["status"] == "passed"
    assert summary["selected_counter_move"] == "rock"
    assert summary["robot_start_pose"] == "rock"
    assert summary["feasible"] is True
    outputs = summary["outputs"]
    for key in (
        "robot_motion_log_jsonl",
        "feasibility_summary_json",
        "episode_summary_csv",
        "report_metrics_csv",
        "motion_preview_dir",
    ):
        assert Path(outputs[key]).exists()
    assert (Path(outputs["motion_preview_dir"]) / "schunk_rock_yaw0_pitch20.png").exists()
