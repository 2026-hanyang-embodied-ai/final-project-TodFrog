from __future__ import annotations

import json
from pathlib import Path

from embodied_rps.realtime_demo_preflight import (
    RealtimeDemoPreflightConfig,
    run_realtime_demo_preflight,
)
from embodied_rps.tools.run_realtime_demo_preflight import main


def _write_minimal_demo_files(project_root: Path) -> tuple[Path, Path]:
    python_executable = project_root / "python.exe"
    python_executable.write_text("python", encoding="utf-8")
    model_state = project_root / "model.pt"
    model_state.write_bytes(b"model")
    profile = project_root / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "profile_name": "unit_profile",
                "model_state_path": "model.pt",
                "label_names": ["rock", "paper", "scissors"],
                "sequence_length": 72,
                "feature_dim": 126,
            }
        ),
        encoding="utf-8",
    )
    config = project_root / "demo.yaml"
    config.write_text(
        "\n".join(
            [
                "profiles:",
                "  - profile.json",
                "response_prompt: scissors",
                "reset_on_prompt_change: true",
                "prompt_cycle: true",
                "prompt_cycle_s: 1.0",
                "prompt_sequence: rock,paper,scissors",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config, python_executable


def test_realtime_demo_preflight_reports_ready_without_camera_check(tmp_path: Path) -> None:
    config_path, python_executable = _write_minimal_demo_files(tmp_path)
    output_root = tmp_path / "preflight"

    summary = run_realtime_demo_preflight(
        RealtimeDemoPreflightConfig(
            project_root=tmp_path,
            config_path=config_path,
            python_executable=python_executable,
            output_root=output_root,
            check_camera=False,
        )
    )

    assert summary["status"] == "ready_without_camera_check"
    assert summary["ok"] is True
    assert "camera_not_checked" in summary["warnings"]
    assert summary["checks"]["model_state_paths_exist"] is True
    assert summary["profiles"][0]["model_state_exists"] is True
    assert (output_root / "preflight_summary.json").exists()
    assert (output_root / "preflight_summary.md").exists()


def test_realtime_demo_preflight_reports_ready_when_camera_probe_passes(tmp_path: Path) -> None:
    config_path, python_executable = _write_minimal_demo_files(tmp_path)

    summary = run_realtime_demo_preflight(
        RealtimeDemoPreflightConfig(
            project_root=tmp_path,
            config_path=config_path,
            python_executable=python_executable,
            output_root=tmp_path / "preflight",
            check_camera=True,
            camera_index=2,
        ),
        camera_probe=lambda index: {
            "opened": index == 2,
            "frame_read": True,
            "frame_width": 1280,
            "frame_height": 720,
        },
    )

    assert summary["status"] == "ready_for_live_demo"
    assert summary["checks"]["camera_opened"] is True
    assert summary["camera"]["frame_width"] == 1280


def test_realtime_demo_preflight_reports_ready_when_hand_visibility_probe_passes(tmp_path: Path) -> None:
    config_path, python_executable = _write_minimal_demo_files(tmp_path)

    summary = run_realtime_demo_preflight(
        RealtimeDemoPreflightConfig(
            project_root=tmp_path,
            config_path=config_path,
            python_executable=python_executable,
            output_root=tmp_path / "preflight",
            check_camera=True,
            check_hand_visibility=True,
            hand_visibility_min_detection_rate=0.80,
            camera_index=2,
        ),
        camera_probe=lambda index: {
            "opened": index == 2,
            "frame_read": True,
            "frame_width": 1280,
            "frame_height": 720,
        },
        hand_visibility_probe=lambda index, max_frames: {
            "checked": True,
            "frame_count": max_frames,
            "detected_frames": 27,
            "detection_rate": 0.90,
        },
    )

    assert summary["status"] == "ready_for_live_demo"
    assert summary["checks"]["hand_visibility_checked"] is True
    assert summary["checks"]["hand_visibility_ok"] is True
    assert summary["hand_visibility"]["detected_frames"] == 27


def test_realtime_demo_preflight_records_hand_visibility_diagnostic_images(tmp_path: Path) -> None:
    config_path, python_executable = _write_minimal_demo_files(tmp_path)
    output_root = tmp_path / "preflight"
    diagnostic = output_root / "hand_visibility" / "first_detected.png"

    summary = run_realtime_demo_preflight(
        RealtimeDemoPreflightConfig(
            project_root=tmp_path,
            config_path=config_path,
            python_executable=python_executable,
            output_root=output_root,
            check_camera=True,
            check_hand_visibility=True,
            camera_index=2,
        ),
        camera_probe=lambda index: {
            "opened": index == 2,
            "frame_read": True,
            "frame_width": 1280,
            "frame_height": 720,
        },
        hand_visibility_probe=lambda index, max_frames: {
            "checked": True,
            "frame_count": max_frames,
            "detected_frames": 30,
            "detection_rate": 1.0,
            "diagnostic_image_paths": [diagnostic.as_posix()],
        },
    )

    summary_json = json.loads((output_root / "preflight_summary.json").read_text(encoding="utf-8"))
    summary_md = (output_root / "preflight_summary.md").read_text(encoding="utf-8")
    assert summary["hand_visibility"]["diagnostic_image_paths"] == [diagnostic.as_posix()]
    assert summary_json["hand_visibility"]["diagnostic_image_paths"] == [diagnostic.as_posix()]
    assert diagnostic.as_posix() in summary_md


def test_realtime_demo_preflight_blocks_when_hand_visibility_is_too_low(tmp_path: Path) -> None:
    config_path, python_executable = _write_minimal_demo_files(tmp_path)

    summary = run_realtime_demo_preflight(
        RealtimeDemoPreflightConfig(
            project_root=tmp_path,
            config_path=config_path,
            python_executable=python_executable,
            output_root=tmp_path / "preflight",
            check_camera=True,
            check_hand_visibility=True,
            hand_visibility_min_detection_rate=0.80,
        ),
        camera_probe=lambda index: {
            "opened": True,
            "frame_read": True,
            "frame_width": 1280,
            "frame_height": 720,
        },
        hand_visibility_probe=lambda index, max_frames: {
            "checked": True,
            "frame_count": max_frames,
            "detected_frames": 3,
            "detection_rate": 0.10,
        },
    )

    assert summary["status"] == "blocked"
    assert summary["checks"]["hand_visibility_ok"] is False
    assert "hand_visibility_low" in summary["failures"]


def test_realtime_demo_preflight_blocks_policy_mismatch(tmp_path: Path) -> None:
    config_path, python_executable = _write_minimal_demo_files(tmp_path)
    config_path.write_text(
        "\n".join(
            [
                "profiles:",
                "  - profile.json",
                "response_prompt: paper",
                "reset_on_prompt_change: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_realtime_demo_preflight(
        RealtimeDemoPreflightConfig(
            project_root=tmp_path,
            config_path=config_path,
            python_executable=python_executable,
            output_root=tmp_path / "preflight",
        )
    )

    assert summary["status"] == "blocked"
    assert "response_prompt_mismatch" in summary["failures"]
    assert "reset_on_prompt_change_disabled" in summary["failures"]


def test_realtime_demo_preflight_cli_writes_summary(tmp_path: Path) -> None:
    config_path, python_executable = _write_minimal_demo_files(tmp_path)
    output_root = tmp_path / "preflight"

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--python-executable",
            str(python_executable),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_root / "preflight_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ready_without_camera_check"
