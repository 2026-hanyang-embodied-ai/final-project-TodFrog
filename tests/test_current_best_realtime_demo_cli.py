from __future__ import annotations

import json
from pathlib import Path

from embodied_rps.realtime_demo_launcher import load_realtime_demo_config
from embodied_rps.tools.run_current_best_realtime_demo import main


def test_current_best_realtime_demo_dry_run_prints_realtime_argv(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "profiles:",
                "  - results/model_profiles/a.json",
                "  - results/model_profiles/b.json",
                "profile_weights: 0.25,0.75",
                "scissors_rescue_profile_index: 0",
                "confidence_threshold: 0.70",
                "margin_threshold: 0.10",
                "confirmation_count: 2",
                "transition_mass_threshold: 0.05",
                "binary_transition_mass_threshold: 0.60",
                "min_binary_decision_progress: 0.05",
                "prompt_cycle: true",
                "reset_on_prompt_cycle: true",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--video",
            "clip.mp4",
            "--output",
            "overlay.mp4",
            "--frame-log-jsonl",
            "frames.jsonl",
            "--skeleton-npz",
            "skeletons.npz",
            "--max-frames",
            "32",
            "--expected-actual-gesture",
            "rock",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"] == str(config_path)
    assert payload["argv"].count("--profile") == 2
    assert payload["argv"][payload["argv"].index("--video") + 1] == "clip.mp4"
    assert payload["argv"][payload["argv"].index("--output") + 1] == "overlay.mp4"
    assert payload["argv"][payload["argv"].index("--frame-log-jsonl") + 1] == "frames.jsonl"
    assert payload["argv"][payload["argv"].index("--skeleton-npz") + 1] == "skeletons.npz"
    assert payload["argv"][payload["argv"].index("--max-frames") + 1] == "32"
    assert payload["argv"][payload["argv"].index("--expected-actual-gesture") + 1] == "rock"
    assert payload["argv"][payload["argv"].index("--min-binary-decision-progress") + 1] == "0.05"
    assert "--prompt-cycle" in payload["argv"]
    assert "--reset-on-prompt-cycle" in payload["argv"]


def test_current_best_realtime_demo_collection_mode_disables_auto_stop(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "profiles:",
                "  - results/model_profiles/a.json",
                "confidence_threshold: 0.70",
                "margin_threshold: 0.10",
                "confirmation_count: 2",
                "prompt_cycle: true",
                "prompt_sequence: scissors",
                "response_prompt: scissors",
                "hold_response_prompt_until_decision: true",
                "response_hold_max_frames: 0",
                "stop_after_confirmed_response_decision: true",
                "post_decision_hold_frames: 120",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--camera",
            "0",
            "--output",
            "collection.mp4",
            "--frame-log-jsonl",
            "collection.jsonl",
            "--skeleton-npz",
            "collection_skeletons.npz",
            "--max-frames",
            "3600",
            "--expected-actual-gesture",
            "scissors",
            "--collection-mode",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["argv"][payload["argv"].index("--skeleton-npz") + 1] == "collection_skeletons.npz"
    assert payload["argv"][payload["argv"].index("--prompt-sequence") + 1] == "scissors"
    assert payload["argv"][payload["argv"].index("--response-prompt") + 1] == "scissors"
    assert "--stop-after-confirmed-response-decision" not in payload["argv"]
    assert "--post-decision-hold-frames" not in payload["argv"]


def test_scissors_collection_config_uses_repeating_response_window_cycle() -> None:
    config = load_realtime_demo_config(Path("configs/realtime_two_stage_selector_scissors_collection.yaml"))

    assert config.prompt_sequence == "rock,paper,scissors"
    assert config.prompt_cycle_s == 1.0
    assert config.response_prompt == "scissors"
    assert config.hold_response_prompt_until_decision is False
    assert config.reset_on_prompt_change is True
    assert config.reset_on_prompt_cycle is False
    assert config.stop_after_confirmed_response_decision is False


def test_current_best_realtime_demo_default_uses_conditional_candidate(capsys) -> None:
    exit_code = main(
        [
            "--video",
            "clip.mp4",
            "--max-frames",
            "1",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"] == "configs\\realtime_two_stage_selector_demo_conditional_scissors_rescue.yaml"
    assert payload["argv"].count("--profile") == 3
    assert payload["argv"][payload["argv"].index("--response-prompt") + 1] == "scissors"
    assert "--reset-on-prompt-change" in payload["argv"]


def test_current_best_realtime_demo_camera_dry_run_requests_display_window(capsys) -> None:
    exit_code = main(
        [
            "--camera",
            "0",
            "--max-frames",
            "1",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["argv"][payload["argv"].index("--camera") + 1] == "0"
    assert "--display-window" in payload["argv"]


def test_current_best_realtime_demo_dry_run_can_write_output_root(tmp_path: Path, capsys) -> None:
    output_root = tmp_path / "dry_run"

    exit_code = main(
        [
            "--video",
            "clip.mp4",
            "--max-frames",
            "1",
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_root"] == str(output_root)
    written = json.loads((output_root / "current_best_realtime_demo_dry_run.json").read_text(encoding="utf-8"))
    assert written["argv"] == payload["argv"]
