"""PowerShell launch scripts for the prompt-gated realtime demo."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RealtimeDemoLaunchScriptsConfig:
    """Configuration for local realtime demo launch scripts."""

    output_root: Path = Path("artifacts/realtime_demo_launch_20260616")
    project_root: Path = Path(".")
    python_executable: Path = field(default_factory=lambda: Path(sys.executable))
    config_path: Path = Path("configs/realtime_two_stage_selector_demo_conditional_scissors_rescue.yaml")
    sample_video: Path = Path("artifacts/realtime_demo_rehearsal_20260616/sample_input.mp4")
    rehearsal_output: Path = Path("artifacts/realtime_demo_rehearsal_20260616/video_rehearsal_overlay.mp4")
    rehearsal_frame_log: Path = Path("artifacts/realtime_demo_rehearsal_20260616/video_rehearsal_frames.jsonl")
    live_output: Path = Path("artifacts/realtime_demo_rehearsal_20260616/live_camera_overlay.mp4")
    live_frame_log: Path = Path("artifacts/realtime_demo_rehearsal_20260616/live_camera_frames.jsonl")
    preflight_output: Path = Path("artifacts/realtime_demo_rehearsal_20260616/preflight")
    response_preview_image: Path = Path(
        "artifacts/schunk_response_preview_v4_late_geometry_new15_20260616/"
        "frames/frame_0002_test_scissors_WIN_20260611_00000016_00000027_00000032_Pro_-_Trim.png"
    )
    live_composite_output_root: Path = Path("artifacts/realtime_schunk_live_demo_composite_20260616")
    readiness_output_root: Path = Path("artifacts/realtime_demo_readiness_20260616")
    triage_output_root: Path = Path("artifacts/realtime_demo_triage_20260616")
    evidence_bundle_output_root: Path = Path("artifacts/realtime_demo_evidence_bundle_20260616")
    review_packet_output_root: Path = Path("artifacts/realtime_demo_review_packet_20260616")
    acceptance_report_output_root: Path = Path("artifacts/realtime_demo_acceptance_report_20260616")
    operator_report_output_root: Path = Path("artifacts/realtime_demo_operator_outcome_20260616")
    archive_output_root: Path = Path("artifacts/realtime_demo_run_archive_20260616")
    final_candidate_output_root: Path = Path("artifacts/realtime_demo_final_candidate_20260616")
    submission_packet_output_root: Path = Path("artifacts/realtime_demo_submission_packet_20260616")
    live_run_checklist_output_root: Path = Path("artifacts/realtime_demo_live_run_checklist_20260616")
    final_run_card_output_root: Path = Path("artifacts/realtime_demo_final_run_card_20260616")
    learning_queue_output_root: Path = Path("artifacts/realtime_demo_learning_queue_20260616")
    goal_progress_audit_output_root: Path = Path("artifacts/realtime_demo_goal_progress_audit_20260616")
    manual_review_decisions: Path = Path("artifacts/realtime_demo_manual_review_20260616/manual_review_decisions.json")
    live_status_snapshot_output_root: Path = Path("artifacts/realtime_demo_live_status_snapshot_20260616")
    operator_handoff_card_output_root: Path = Path("artifacts/realtime_demo_operator_handoff_card_20260616")
    prelaunch_audit_output_root: Path = Path("artifacts/realtime_demo_prelaunch_audit_20260616")
    wrapper_contract_probe_output_root: Path = Path("artifacts/realtime_demo_wrapper_contract_probe_20260616")
    operator_command_audit_output_root: Path = Path("artifacts/realtime_demo_operator_command_audit_20260616")
    live_artifact_cleanup_output_root: Path = Path("artifacts/realtime_demo_live_artifact_cleanup_20260616")
    guarded_retake_readiness_output_root: Path = Path("artifacts/realtime_demo_guarded_retake_readiness_20260616")
    live_rock_retake_gate_output_root: Path = Path("artifacts/realtime_demo_live_rock_retake_gate_20260616")
    scissors_collection_output_root: Path = Path("artifacts/realtime_scissors_pose_collection_20260617")
    scissors_collection_config_path: Path = Path("configs/realtime_two_stage_selector_scissors_collection.yaml")
    scissors_collection_max_frames: int = 3600
    dry_run_overlay_contract_summary: Path = Path(
        "artifacts/realtime_demo_overlay_contract_dryrun_20260616/overlay_contract_summary.json"
    )
    overlay_contract_output_root: Path = Path("artifacts/realtime_demo_overlay_contract_20260616")
    camera_index: int = 0
    camera_max_frames: int = 900
    hand_visibility_max_frames: int = 60
    hand_visibility_min_detection_rate: float = 0.60


def write_realtime_demo_launch_scripts(config: RealtimeDemoLaunchScriptsConfig) -> dict[str, object]:
    """Write inspectable local PowerShell scripts for final-demo rehearsal."""

    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    scripts = {
        "demo_preflight": output_root / "00_demo_preflight.ps1",
        "dry_run_video_rehearsal": output_root / "01_dry_run_video_rehearsal.ps1",
        "live_camera_demo": output_root / "02_live_camera_demo.ps1",
        "print_live_camera_argv": output_root / "03_print_live_camera_argv.ps1",
        "open_demo_artifacts": output_root / "04_open_demo_artifacts.ps1",
        "create_live_schunk_composite": output_root / "05_create_live_schunk_composite.ps1",
        "verify_live_capture": output_root / "06_verify_live_capture.ps1",
        "run_live_demo_pipeline": output_root / "07_run_live_demo_pipeline.ps1",
        "triage_live_capture": output_root / "08_triage_live_capture.ps1",
        "build_demo_evidence_bundle": output_root / "09_build_demo_evidence_bundle.ps1",
        "build_demo_review_packet": output_root / "10_build_demo_review_packet.ps1",
        "audit_live_overlay_contract": output_root / "11_audit_live_overlay_contract.ps1",
        "build_demo_acceptance_report": output_root / "12_build_demo_acceptance_report.ps1",
        "build_operator_outcome_report": output_root / "13_build_operator_outcome_report.ps1",
        "archive_live_demo_run": output_root / "14_archive_live_demo_run.ps1",
        "summarize_run_archives": output_root / "15_summarize_run_archives.ps1",
        "select_final_demo_candidate": output_root / "16_select_final_demo_candidate.ps1",
        "build_submission_candidate_packet": output_root / "17_build_submission_candidate_packet.ps1",
        "build_live_run_checklist": output_root / "18_build_live_run_checklist.ps1",
        "build_final_run_card": output_root / "19_build_final_run_card.ps1",
        "build_learning_queue": output_root / "20_build_learning_queue.ps1",
        "build_goal_progress_audit": output_root / "21_build_goal_progress_audit.ps1",
        "run_live_demo_operator_confirmed": output_root / "22_run_live_demo_operator_confirmed.ps1",
        "check_goal_progress_strict": output_root / "23_check_goal_progress_strict.ps1",
        "run_live_demo_operator_confirmed_strict": output_root / "24_run_live_demo_operator_confirmed_strict.ps1",
        "build_live_status_snapshot": output_root / "25_build_live_status_snapshot.ps1",
        "build_operator_handoff_card": output_root / "26_build_operator_handoff_card.ps1",
        "check_prelaunch_audit": output_root / "27_check_prelaunch_audit.ps1",
        "check_wrapper_contract": output_root / "28_check_wrapper_contract.ps1",
        "check_operator_command_audit": output_root / "29_check_operator_command_audit.ps1",
        "clear_stale_live_artifacts": output_root / "30_clear_stale_live_artifacts.ps1",
        "record_manual_review_decision": output_root / "31_record_manual_review_decision.ps1",
        "check_guarded_retake_readiness": output_root / "32_check_guarded_retake_readiness.ps1",
        "check_live_rock_retake_gate": output_root / "33_check_live_rock_retake_gate.ps1",
        "run_scissors_pose_collection": output_root / "34_run_scissors_pose_collection.ps1",
    }
    _write_ps1(
        scripts["demo_preflight"],
        _script_header(config)
        + _preflight_command_lines(config),
    )
    _write_ps1(
        scripts["dry_run_video_rehearsal"],
        _script_header(config)
        + _command_lines(
            config,
            [
                "--config",
                config.config_path.as_posix(),
                "--video",
                config.sample_video.as_posix(),
                "--output",
                config.rehearsal_output.as_posix(),
                "--frame-log-jsonl",
                config.rehearsal_frame_log.as_posix(),
            ],
        ),
    )
    _write_ps1(
        scripts["live_camera_demo"],
        _script_header(config)
        + _command_lines(
            config,
            [
                "--config",
                config.config_path.as_posix(),
                "--camera",
                str(int(config.camera_index)),
                "--output",
                config.live_output.as_posix(),
                "--frame-log-jsonl",
                config.live_frame_log.as_posix(),
                "--max-frames",
                str(int(config.camera_max_frames)),
            ],
        ),
    )
    _write_ps1(
        scripts["print_live_camera_argv"],
        _script_header(config)
        + _command_lines(
            config,
            [
                "--config",
                config.config_path.as_posix(),
                "--camera",
                str(int(config.camera_index)),
                "--output",
                config.live_output.as_posix(),
                "--max-frames",
                str(int(config.camera_max_frames)),
                "--dry-run",
            ],
        ),
    )
    _write_ps1(
        scripts["open_demo_artifacts"],
        _script_header(config)
        + "\n"
        + f"New-Item -ItemType Directory -Force -Path '{_ps_quote(config.rehearsal_output.parent.as_posix())}' | Out-Null\n"
        + f"Invoke-Item '{_ps_quote(config.rehearsal_output.parent.as_posix())}'\n",
    )
    _write_ps1(
        scripts["create_live_schunk_composite"],
        _script_header(config)
        + _live_composite_command_lines(config),
    )
    _write_ps1(
        scripts["verify_live_capture"],
        _script_header(config)
        + _verify_live_capture_command_lines(config),
    )
    _write_ps1(
        scripts["run_live_demo_pipeline"],
        _script_header(config)
        + _live_demo_pipeline_command_lines(config),
    )
    _write_ps1(
        scripts["triage_live_capture"],
        _script_header(config)
        + _triage_live_capture_command_lines(config),
    )
    _write_ps1(
        scripts["build_demo_evidence_bundle"],
        _script_header(config)
        + _evidence_bundle_command_lines(config),
    )
    _write_ps1(
        scripts["build_demo_review_packet"],
        _script_header(config)
        + _review_packet_command_lines(config),
    )
    _write_ps1(
        scripts["audit_live_overlay_contract"],
        _script_header(config)
        + _overlay_contract_command_lines(config),
    )
    _write_ps1(
        scripts["build_demo_acceptance_report"],
        _script_header(config)
        + _acceptance_report_command_lines(config),
    )
    _write_ps1(
        scripts["build_operator_outcome_report"],
        _script_header(config)
        + _operator_report_command_lines(config),
    )
    _write_ps1(
        scripts["archive_live_demo_run"],
        _script_header(config)
        + _archive_run_command_lines(config),
    )
    _write_ps1(
        scripts["summarize_run_archives"],
        _script_header(config)
        + _archive_index_command_lines(config),
    )
    _write_ps1(
        scripts["select_final_demo_candidate"],
        _script_header(config)
        + _final_candidate_command_lines(config),
    )
    _write_ps1(
        scripts["build_submission_candidate_packet"],
        _script_header(config)
        + _submission_packet_command_lines(config),
    )
    _write_ps1(
        scripts["build_live_run_checklist"],
        _script_header(config)
        + _live_run_checklist_command_lines(config),
    )
    _write_ps1(
        scripts["build_final_run_card"],
        _script_header(config)
        + _final_run_card_command_lines(config),
    )
    _write_ps1(
        scripts["build_learning_queue"],
        _script_header(config)
        + _learning_queue_command_lines(config),
    )
    _write_ps1(
        scripts["build_goal_progress_audit"],
        _script_header(config)
        + _goal_progress_audit_command_lines(config),
    )
    _write_ps1(
        scripts["run_live_demo_operator_confirmed"],
        _script_header(config)
        + _operator_confirmed_live_pipeline_command_lines(config),
    )
    _write_ps1(
        scripts["check_goal_progress_strict"],
        _script_header(config)
        + _goal_progress_audit_command_lines(config, strict_exit_code=True),
    )
    _write_ps1(
        scripts["run_live_demo_operator_confirmed_strict"],
        _script_header(config)
        + _operator_confirmed_strict_command_lines(config),
    )
    _write_ps1(
        scripts["build_live_status_snapshot"],
        _script_header(config)
        + _live_status_snapshot_command_lines(config),
    )
    _write_ps1(
        scripts["build_operator_handoff_card"],
        _script_header(config)
        + _operator_handoff_card_command_lines(config),
    )
    _write_ps1(
        scripts["check_prelaunch_audit"],
        _script_header(config)
        + _prelaunch_audit_command_lines(config),
    )
    _write_ps1(
        scripts["check_wrapper_contract"],
        _script_header(config)
        + _wrapper_contract_probe_command_lines(config),
    )
    _write_ps1(
        scripts["check_operator_command_audit"],
        _script_header(config)
        + _operator_command_audit_command_lines(config),
    )
    _write_ps1(
        scripts["clear_stale_live_artifacts"],
        _script_header(config)
        + _live_artifact_cleanup_command_lines(config),
    )
    _write_ps1(
        scripts["record_manual_review_decision"],
        _script_header(config)
        + _manual_review_decision_command_lines(config),
    )
    _write_ps1(
        scripts["check_guarded_retake_readiness"],
        _script_header(config)
        + _guarded_retake_readiness_command_lines(config),
    )
    _write_ps1(
        scripts["check_live_rock_retake_gate"],
        _script_header(config)
        + _live_rock_retake_gate_command_lines(config),
    )
    _write_ps1(
        scripts["run_scissors_pose_collection"],
        _script_header(config)
        + _scissors_pose_collection_command_lines(config),
    )
    summary: dict[str, object] = {
        "output_root": output_root.as_posix(),
        "project_root": config.project_root.as_posix(),
        "project_root_absolute": _resolved_posix(config.project_root),
        "python_executable": config.python_executable.as_posix(),
        "config_path": config.config_path.as_posix(),
        "sample_video": config.sample_video.as_posix(),
        "rehearsal_output": config.rehearsal_output.as_posix(),
        "rehearsal_frame_log": config.rehearsal_frame_log.as_posix(),
        "live_output": config.live_output.as_posix(),
        "live_frame_log": config.live_frame_log.as_posix(),
        "preflight_output": config.preflight_output.as_posix(),
        "response_preview_image": config.response_preview_image.as_posix(),
        "live_composite_output_root": config.live_composite_output_root.as_posix(),
        "readiness_output_root": config.readiness_output_root.as_posix(),
        "triage_output_root": config.triage_output_root.as_posix(),
        "evidence_bundle_output_root": config.evidence_bundle_output_root.as_posix(),
        "review_packet_output_root": config.review_packet_output_root.as_posix(),
        "acceptance_report_output_root": config.acceptance_report_output_root.as_posix(),
        "operator_report_output_root": config.operator_report_output_root.as_posix(),
        "archive_output_root": config.archive_output_root.as_posix(),
        "final_candidate_output_root": config.final_candidate_output_root.as_posix(),
        "submission_packet_output_root": config.submission_packet_output_root.as_posix(),
        "live_run_checklist_output_root": config.live_run_checklist_output_root.as_posix(),
        "final_run_card_output_root": config.final_run_card_output_root.as_posix(),
        "learning_queue_output_root": config.learning_queue_output_root.as_posix(),
        "goal_progress_audit_output_root": config.goal_progress_audit_output_root.as_posix(),
        "manual_review_decisions": config.manual_review_decisions.as_posix(),
        "live_status_snapshot_output_root": config.live_status_snapshot_output_root.as_posix(),
        "operator_handoff_card_output_root": config.operator_handoff_card_output_root.as_posix(),
        "prelaunch_audit_output_root": config.prelaunch_audit_output_root.as_posix(),
        "wrapper_contract_probe_output_root": config.wrapper_contract_probe_output_root.as_posix(),
        "operator_command_audit_output_root": config.operator_command_audit_output_root.as_posix(),
        "live_artifact_cleanup_output_root": config.live_artifact_cleanup_output_root.as_posix(),
        "guarded_retake_readiness_output_root": config.guarded_retake_readiness_output_root.as_posix(),
        "live_rock_retake_gate_output_root": config.live_rock_retake_gate_output_root.as_posix(),
        "scissors_collection_output_root": config.scissors_collection_output_root.as_posix(),
        "scissors_collection_config_path": config.scissors_collection_config_path.as_posix(),
        "scissors_collection_max_frames": int(config.scissors_collection_max_frames),
        "dry_run_overlay_contract_summary": config.dry_run_overlay_contract_summary.as_posix(),
        "overlay_contract_output_root": config.overlay_contract_output_root.as_posix(),
        "camera_index": int(config.camera_index),
        "camera_max_frames": int(config.camera_max_frames),
        "hand_visibility_max_frames": int(config.hand_visibility_max_frames),
        "hand_visibility_min_detection_rate": float(config.hand_visibility_min_detection_rate),
        "script_count": len(scripts),
        "scripts": {name: path.as_posix() for name, path in scripts.items()},
        "scripts_absolute": {name: _resolved_posix(path) for name, path in scripts.items()},
    }
    (output_root / "demo_launch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(_readme(config, scripts), encoding="utf-8")
    return summary


def _script_header(config: RealtimeDemoLaunchScriptsConfig) -> str:
    project_root = _resolved_posix(config.project_root)
    return "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"Set-Location '{_ps_quote(project_root)}'",
            "$env:PYTHONPATH = 'src'",
            "",
        ]
    )


def _resolved_posix(path: Path) -> str:
    return path.resolve().as_posix()


def _command_lines(config: RealtimeDemoLaunchScriptsConfig, args: list[str]) -> str:
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    lines = [
        "Write-Host 'Running current-best realtime RPS skeleton demo'",
        *_expected_actual_gesture_arg_lines(),
        (
            f"& '{_ps_quote(config.python_executable.as_posix())}' "
            f"-m embodied_rps.tools.run_current_best_realtime_demo @expectedActualGestureArgs {quoted_args}"
        ),
    ]
    return "\n".join(lines) + "\n"


def _expected_actual_gesture_arg_lines() -> list[str]:
    return [
        "$expectedActualGestureArgs = @()",
        "if ($env:EMBODIED_RPS_EXPECTED_ACTUAL_GESTURE) {",
        "  $expectedActualGestureArgs = @('--expected-actual-gesture', $env:EMBODIED_RPS_EXPECTED_ACTUAL_GESTURE)",
        "}",
    ]


def _preflight_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--project-root",
        config.project_root.as_posix(),
        "--config",
        config.config_path.as_posix(),
        "--python-executable",
        config.python_executable.as_posix(),
        "--output-root",
        config.preflight_output.as_posix(),
        "--camera",
        str(int(config.camera_index)),
        "--check-camera",
        "--check-hand-visibility",
        "--hand-visibility-max-frames",
        str(int(config.hand_visibility_max_frames)),
        "--hand-visibility-min-detection-rate",
        str(float(config.hand_visibility_min_detection_rate)),
        "--require-response-prompt",
        "scissors",
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Running realtime RPS skeleton demo preflight'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.run_realtime_demo_preflight {quoted_args}\n"
    )


def _live_composite_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--overlay-video",
        config.live_output.as_posix(),
        "--response-preview-image",
        config.response_preview_image.as_posix(),
        "--output-root",
        config.live_composite_output_root.as_posix(),
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        f"if (-not (Test-Path '{_ps_quote(config.live_output.as_posix())}')) {{\n"
        "  throw 'Run 02_live_camera_demo.ps1 first so the live camera overlay exists.'\n"
        "}\n"
        "Write-Host 'Creating live realtime-plus-SCHUNK composite demo video'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.create_realtime_schunk_demo_composite {quoted_args}\n"
    )


def _verify_live_capture_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    output_root = config.live_output.parent / "postcapture"
    args = [
        "--overlay-video",
        config.live_output.as_posix(),
        "--output-root",
        output_root.as_posix(),
        "--frame-log-jsonl",
        config.live_frame_log.as_posix(),
        "--response-prompt",
        "scissors",
        "--enforce-demo-success-gate",
        "--max-response-binary-latency-s",
        "0.50",
        "--min-detection-rate",
        "0.80",
        "--response-preview-image",
        config.response_preview_image.as_posix(),
        "--live-composite-output-root",
        config.live_composite_output_root.as_posix(),
        "--min-frame-count",
        "90",
        "--min-duration-s",
        "3.0",
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    lines = [
        f"if (-not (Test-Path '{_ps_quote(config.live_output.as_posix())}')) {{",
        "  throw 'Run 02_live_camera_demo.ps1 first so the live camera overlay exists.'",
        "}",
        "Write-Host 'Verifying live realtime demo overlay capture'",
        *_expected_actual_gesture_arg_lines(),
        (
            f"& '{_ps_quote(config.python_executable.as_posix())}' "
            f"-m embodied_rps.tools.verify_realtime_demo_capture @expectedActualGestureArgs {quoted_args}"
        ),
    ]
    return "\n".join(lines) + "\n"


def _live_demo_pipeline_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    ordered_scripts = [
        "30_clear_stale_live_artifacts.ps1",
        "18_build_live_run_checklist.ps1",
        "00_demo_preflight.ps1",
        "02_live_camera_demo.ps1",
        "06_verify_live_capture.ps1",
        "33_check_live_rock_retake_gate.ps1",
        "05_create_live_schunk_composite.ps1",
        "04_open_demo_artifacts.ps1",
    ]
    lines = [
        "$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path",
        "Write-Host 'Running full live realtime RPS demo pipeline'",
        "$pipelineFailed = $false",
        "$pipelineError = $null",
        "try {",
    ]
    for script_name in ordered_scripts:
        lines.extend(
            [
                f"  Write-Host '--- {script_name} ---'",
                f"  & (Join-Path $scriptRoot '{script_name}')",
            ]
        )
    lines.extend(
        [
            "}",
            "catch {",
            "  $pipelineFailed = $true",
            "  $pipelineError = $_",
            "  Write-Host ('Live demo pipeline failed: ' + $_.Exception.Message)",
            "}",
            "finally {",
            "  Write-Host '--- 08_triage_live_capture.ps1 ---'",
            "  & (Join-Path $scriptRoot '08_triage_live_capture.ps1')",
            "  Write-Host '--- 11_audit_live_overlay_contract.ps1 ---'",
            f"  if (Test-Path '{_ps_quote(config.live_output.as_posix())}') {{",
            "    & (Join-Path $scriptRoot '11_audit_live_overlay_contract.ps1')",
            "  } else {",
            "    Write-Host 'Skipping overlay contract audit because live camera overlay is missing.'",
            "  }",
            "  Write-Host '--- 09_build_demo_evidence_bundle.ps1 ---'",
            "  & (Join-Path $scriptRoot '09_build_demo_evidence_bundle.ps1')",
            "  Write-Host '--- 10_build_demo_review_packet.ps1 ---'",
            "  & (Join-Path $scriptRoot '10_build_demo_review_packet.ps1')",
            "  Write-Host '--- 12_build_demo_acceptance_report.ps1 ---'",
            "  & (Join-Path $scriptRoot '12_build_demo_acceptance_report.ps1')",
            "  Write-Host '--- 13_build_operator_outcome_report.ps1 ---'",
            "  & (Join-Path $scriptRoot '13_build_operator_outcome_report.ps1')",
            "  Write-Host '--- 14_archive_live_demo_run.ps1 ---'",
            "  & (Join-Path $scriptRoot '14_archive_live_demo_run.ps1')",
            "  Write-Host '--- 15_summarize_run_archives.ps1 ---'",
            "  & (Join-Path $scriptRoot '15_summarize_run_archives.ps1')",
            "  Write-Host '--- 16_select_final_demo_candidate.ps1 ---'",
            "  & (Join-Path $scriptRoot '16_select_final_demo_candidate.ps1')",
            "  Write-Host '--- 17_build_submission_candidate_packet.ps1 ---'",
            "  & (Join-Path $scriptRoot '17_build_submission_candidate_packet.ps1')",
            "  Write-Host '--- 19_build_final_run_card.ps1 ---'",
            "  & (Join-Path $scriptRoot '19_build_final_run_card.ps1')",
            "  Write-Host '--- 20_build_learning_queue.ps1 ---'",
            "  & (Join-Path $scriptRoot '20_build_learning_queue.ps1')",
            "  Write-Host '--- 21_build_goal_progress_audit.ps1 ---'",
            "  & (Join-Path $scriptRoot '21_build_goal_progress_audit.ps1')",
            "}",
            "if ($pipelineFailed) {",
            "  throw $pipelineError",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def _triage_live_capture_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    readiness_summary = config.readiness_output_root / "readiness_summary.json"
    preflight_summary = config.preflight_output / "preflight_summary.json"
    live_postcapture_summary = config.live_output.parent / "postcapture" / "postcapture_summary.json"
    live_rock_retake_gate = config.live_rock_retake_gate_output_root / "live_rock_retake_gate.json"
    live_composite_manifest = config.live_composite_output_root / "realtime_schunk_demo_composite_manifest.json"
    readiness_args = [
        "--output-root",
        config.readiness_output_root.as_posix(),
    ]
    triage_args = [
        "--output-root",
        config.triage_output_root.as_posix(),
        "--readiness-summary",
        readiness_summary.as_posix(),
        "--preflight-summary",
        preflight_summary.as_posix(),
        "--postcapture-summary",
        live_postcapture_summary.as_posix(),
        "--composite-manifest",
        live_composite_manifest.as_posix(),
        "--live-rock-retake-gate",
        live_rock_retake_gate.as_posix(),
    ]
    readiness_quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in readiness_args)
    triage_quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in triage_args)
    return (
        "Write-Host 'Refreshing realtime demo readiness summary'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.summarize_realtime_demo_readiness {readiness_quoted_args}\n"
        "Write-Host 'Triaging realtime demo capture state'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.triage_realtime_demo_capture {triage_quoted_args}\n"
    )


def _evidence_bundle_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    readiness_summary = config.readiness_output_root / "readiness_summary.json"
    triage_summary = config.triage_output_root / "triage_summary.json"
    dry_postcapture_summary = config.rehearsal_output.parent / "dry_run_postcapture" / "postcapture_summary.json"
    dry_composite_manifest = (
        Path("artifacts/realtime_schunk_demo_composite_response_frame_20260616")
        / "realtime_schunk_demo_composite_manifest.json"
    )
    live_postcapture_summary = config.live_output.parent / "postcapture" / "postcapture_summary.json"
    live_composite_manifest = config.live_composite_output_root / "realtime_schunk_demo_composite_manifest.json"
    live_overlay_contract_summary = config.overlay_contract_output_root / "overlay_contract_summary.json"
    args = [
        "--output-root",
        config.evidence_bundle_output_root.as_posix(),
        "--readiness-summary",
        readiness_summary.as_posix(),
        "--triage-summary",
        triage_summary.as_posix(),
        "--dry-run-postcapture-summary",
        dry_postcapture_summary.as_posix(),
        "--dry-run-composite-manifest",
        dry_composite_manifest.as_posix(),
        "--dry-run-overlay-contract-summary",
        config.dry_run_overlay_contract_summary.as_posix(),
        "--live-postcapture-summary",
        live_postcapture_summary.as_posix(),
        "--live-composite-manifest",
        live_composite_manifest.as_posix(),
        "--live-overlay-contract-summary",
        live_overlay_contract_summary.as_posix(),
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Building realtime demo evidence bundle summary'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.build_realtime_demo_evidence_bundle {quoted_args}\n"
    )


def _review_packet_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    evidence_bundle = config.evidence_bundle_output_root / "demo_evidence_bundle.json"
    preflight_summary = config.preflight_output / "preflight_summary.json"
    dry_postcapture_summary = config.rehearsal_output.parent / "dry_run_postcapture" / "postcapture_summary.json"
    dry_composite_manifest = (
        Path("artifacts/realtime_schunk_demo_composite_response_frame_20260616")
        / "realtime_schunk_demo_composite_manifest.json"
    )
    args = [
        "--output-root",
        config.review_packet_output_root.as_posix(),
        "--evidence-bundle",
        evidence_bundle.as_posix(),
        "--preflight-summary",
        preflight_summary.as_posix(),
        "--dry-run-postcapture-summary",
        dry_postcapture_summary.as_posix(),
        "--dry-run-composite-manifest",
        dry_composite_manifest.as_posix(),
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Building lightweight realtime demo review packet'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.build_realtime_demo_review_packet {quoted_args}\n"
    )


def _acceptance_report_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    evidence_bundle = config.evidence_bundle_output_root / "demo_evidence_bundle.json"
    review_packet_manifest = config.review_packet_output_root / "review_packet_manifest.json"
    live_overlay_contract_summary = config.overlay_contract_output_root / "overlay_contract_summary.json"
    args = [
        "--output-root",
        config.acceptance_report_output_root.as_posix(),
        "--evidence-bundle",
        evidence_bundle.as_posix(),
        "--review-packet-manifest",
        review_packet_manifest.as_posix(),
        "--dry-run-overlay-contract-summary",
        config.dry_run_overlay_contract_summary.as_posix(),
        "--live-overlay-contract-summary",
        live_overlay_contract_summary.as_posix(),
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Building realtime demo final acceptance report'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.build_realtime_demo_acceptance_report {quoted_args}\n"
    )


def _operator_report_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    readiness_summary = config.readiness_output_root / "readiness_summary.json"
    triage_summary = config.triage_output_root / "triage_summary.json"
    review_packet_manifest = config.review_packet_output_root / "review_packet_manifest.json"
    acceptance_report = config.acceptance_report_output_root / "acceptance_report.json"
    args = [
        "--output-root",
        config.operator_report_output_root.as_posix(),
        "--acceptance-report",
        acceptance_report.as_posix(),
        "--triage-summary",
        triage_summary.as_posix(),
        "--review-packet-manifest",
        review_packet_manifest.as_posix(),
        "--readiness-summary",
        readiness_summary.as_posix(),
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Building realtime demo operator outcome report'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.build_realtime_demo_operator_report {quoted_args}\n"
    )


def _archive_run_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    live_postcapture_summary = config.live_output.parent / "postcapture" / "postcapture_summary.json"
    live_composite_manifest = config.live_composite_output_root / "realtime_schunk_demo_composite_manifest.json"
    live_rock_retake_gate = config.live_rock_retake_gate_output_root / "live_rock_retake_gate.json"
    args = [
        "--output-root",
        config.archive_output_root.as_posix(),
        "--live-overlay-video",
        config.live_output.as_posix(),
        "--live-frame-log",
        config.live_frame_log.as_posix(),
        "--live-postcapture-summary",
        live_postcapture_summary.as_posix(),
        "--live-composite-manifest",
        live_composite_manifest.as_posix(),
        "--operator-outcome",
        (config.operator_report_output_root / "operator_outcome.json").as_posix(),
        "--triage-summary",
        (config.triage_output_root / "triage_summary.json").as_posix(),
        "--acceptance-report",
        (config.acceptance_report_output_root / "acceptance_report.json").as_posix(),
        "--evidence-bundle",
        (config.evidence_bundle_output_root / "demo_evidence_bundle.json").as_posix(),
        "--review-packet-manifest",
        (config.review_packet_output_root / "review_packet_manifest.json").as_posix(),
        "--readiness-summary",
        (config.readiness_output_root / "readiness_summary.json").as_posix(),
        "--preflight-summary",
        (config.preflight_output / "preflight_summary.json").as_posix(),
        "--live-rock-retake-gate",
        live_rock_retake_gate.as_posix(),
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Archiving current realtime demo run artifacts'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.archive_realtime_demo_run {quoted_args}\n"
    )


def _archive_index_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--archive-root",
        config.archive_output_root.as_posix(),
        "--output-root",
        config.archive_output_root.as_posix(),
        "--manual-review-decisions",
        config.manual_review_decisions.as_posix(),
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Summarizing archived realtime demo runs'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.summarize_realtime_demo_run_archives {quoted_args}\n"
    )


def _final_candidate_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--archive-index",
        (config.archive_output_root / "run_archive_index.json").as_posix(),
        "--output-root",
        config.final_candidate_output_root.as_posix(),
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Selecting final realtime demo video candidate from archive index'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.select_realtime_demo_final_candidate {quoted_args}\n"
    )


def _submission_packet_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--final-candidate",
        (config.final_candidate_output_root / "final_demo_candidate.json").as_posix(),
        "--output-root",
        config.submission_packet_output_root.as_posix(),
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Building realtime demo submission candidate packet'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.build_realtime_demo_submission_packet {quoted_args}\n"
    )


def _live_run_checklist_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--output-root",
        config.live_run_checklist_output_root.as_posix(),
        "--readiness-summary",
        (config.readiness_output_root / "readiness_summary.json").as_posix(),
        "--operator-outcome",
        (config.operator_report_output_root / "operator_outcome.json").as_posix(),
        "--preflight-summary",
        (config.preflight_output / "preflight_summary.json").as_posix(),
        "--launch-summary",
        (config.output_root / "demo_launch_summary.json").as_posix(),
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Building pre-run realtime demo operator checklist'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.build_realtime_demo_live_run_checklist {quoted_args}\n"
    )


def _final_run_card_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--output-root",
        config.final_run_card_output_root.as_posix(),
        "--live-run-checklist",
        (config.live_run_checklist_output_root / "live_run_checklist.json").as_posix(),
        "--operator-outcome",
        (config.operator_report_output_root / "operator_outcome.json").as_posix(),
        "--submission-packet",
        (config.submission_packet_output_root / "submission_candidate_packet.json").as_posix(),
        "--launch-summary",
        (config.output_root / "demo_launch_summary.json").as_posix(),
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Building one-card realtime demo final run summary'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.build_realtime_demo_final_run_card {quoted_args}\n"
    )


def _learning_queue_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--output-root",
        config.learning_queue_output_root.as_posix(),
        "--final-run-card",
        (config.final_run_card_output_root / "final_run_card.json").as_posix(),
        "--triage-summary",
        (config.triage_output_root / "triage_summary.json").as_posix(),
        "--operator-outcome",
        (config.operator_report_output_root / "operator_outcome.json").as_posix(),
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Building post-run realtime demo learning queue'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.build_realtime_demo_learning_queue {quoted_args}\n"
    )


def _goal_progress_audit_command_lines(
    config: RealtimeDemoLaunchScriptsConfig,
    *,
    strict_exit_code: bool = False,
) -> str:
    args = [
        "--output-root",
        config.goal_progress_audit_output_root.as_posix(),
        "--readiness-summary",
        (config.readiness_output_root / "readiness_summary.json").as_posix(),
        "--evidence-bundle",
        (config.evidence_bundle_output_root / "demo_evidence_bundle.json").as_posix(),
        "--final-run-card",
        (config.final_run_card_output_root / "final_run_card.json").as_posix(),
        "--learning-queue",
        (config.learning_queue_output_root / "learning_queue.json").as_posix(),
    ]
    if strict_exit_code:
        args.append("--strict-exit-code")
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    lines = [
        "Write-Host 'Building objective-level realtime demo goal progress audit'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.build_realtime_demo_goal_progress_audit {quoted_args}"
    ]
    if strict_exit_code:
        lines.extend(
            [
                "if ($LASTEXITCODE -ne 0) {",
                "  exit $LASTEXITCODE",
                "}",
            ]
        )
    return "\n".join(lines) + "\n"


def _operator_confirmed_live_pipeline_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    checklist_md = config.live_run_checklist_output_root / "live_run_checklist.md"
    lines = [
        "$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path",
        "Write-Host 'Preparing operator-confirmed live realtime RPS demo run'",
        "Write-Host '--- 18_build_live_run_checklist.ps1 ---'",
        "& (Join-Path $scriptRoot '18_build_live_run_checklist.ps1')",
        f"$checklistPath = '{_ps_quote(checklist_md.as_posix())}'",
        "if (Test-Path $checklistPath) {",
        "  Write-Host ''",
        "  Write-Host '--- Live run checklist ---'",
        "  Get-Content $checklistPath | ForEach-Object { Write-Host $_ }",
        "}",
        "Write-Host ''",
        "$expectedActualGesture = Read-Host 'Enter the actual gesture you will make during PROMPT SCISSORS response window (rock/paper/scissors, Enter=rock)'",
        "if ([string]::IsNullOrWhiteSpace($expectedActualGesture)) {",
        "  $expectedActualGesture = 'rock'",
        "  Write-Host 'No expected actual gesture entered; defaulting to rock for the first retake.'",
        "}",
        "$expectedActualGesture = $expectedActualGesture.Trim().ToLowerInvariant()",
        "if (@('rock', 'paper', 'scissors') -notcontains $expectedActualGesture) {",
        "  throw 'Expected actual gesture must be rock, paper, or scissors.'",
        "}",
        "$env:EMBODIED_RPS_EXPECTED_ACTUAL_GESTURE = $expectedActualGesture",
        "Write-Host ('Validation ground truth only: ' + $expectedActualGesture)",
        "Write-Host ''",
        "Write-Host 'Before pressing Enter:'",
        "Write-Host '  1. Keep one hand centered, visible, and well lit.'",
        "Write-Host '  2. Keep rock standby during PROMPT ROCK and PROMPT PAPER.'",
        "Write-Host '  3. Treat PROMPT SCISSORS as RESPONSE WINDOW, then make the selected actual gesture or keep rock.'",
        "Write-Host '  4. Hold the hand until the prediction and robot action are visible.'",
        "Read-Host 'Press Enter when the camera view and hand are ready for live capture'",
        "Write-Host '--- 07_run_live_demo_pipeline.ps1 ---'",
        "& (Join-Path $scriptRoot '07_run_live_demo_pipeline.ps1')",
    ]
    return "\n".join(lines) + "\n"


def _operator_confirmed_strict_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    handoff_markdown = (config.operator_handoff_card_output_root / "operator_handoff_card.md").as_posix()
    lines = [
        "$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path",
        "$pipelineFailed = $false",
        "$pipelineError = $null",
        "$strictExitCode = 0",
        f"$handoffMarkdownPath = '{_ps_quote(handoff_markdown)}'",
        "Write-Host '--- 30_clear_stale_live_artifacts.ps1 ---'",
        "& (Join-Path $scriptRoot '30_clear_stale_live_artifacts.ps1')",
        "$liveArtifactCleanupExitCode = $LASTEXITCODE",
        "if ($liveArtifactCleanupExitCode -ne 0) {",
        "  exit $liveArtifactCleanupExitCode",
        "}",
        "Write-Host '--- 08_triage_live_capture.ps1 ---'",
        "& (Join-Path $scriptRoot '08_triage_live_capture.ps1')",
        "$preCaptureTriageExitCode = $LASTEXITCODE",
        "if ($preCaptureTriageExitCode -ne 0) {",
        "  exit $preCaptureTriageExitCode",
        "}",
        "Write-Host '--- 13_build_operator_outcome_report.ps1 ---'",
        "& (Join-Path $scriptRoot '13_build_operator_outcome_report.ps1')",
        "$preCaptureOperatorOutcomeExitCode = $LASTEXITCODE",
        "if ($preCaptureOperatorOutcomeExitCode -ne 0) {",
        "  exit $preCaptureOperatorOutcomeExitCode",
        "}",
        "Write-Host '--- 18_build_live_run_checklist.ps1 ---'",
        "& (Join-Path $scriptRoot '18_build_live_run_checklist.ps1')",
        "$preCaptureChecklistExitCode = $LASTEXITCODE",
        "if ($preCaptureChecklistExitCode -ne 0) {",
        "  exit $preCaptureChecklistExitCode",
        "}",
        "Write-Host '--- 19_build_final_run_card.ps1 ---'",
        "& (Join-Path $scriptRoot '19_build_final_run_card.ps1')",
        "$preCaptureFinalRunCardExitCode = $LASTEXITCODE",
        "if ($preCaptureFinalRunCardExitCode -ne 0) {",
        "  exit $preCaptureFinalRunCardExitCode",
        "}",
        "Write-Host '--- 20_build_learning_queue.ps1 ---'",
        "& (Join-Path $scriptRoot '20_build_learning_queue.ps1')",
        "$preCaptureLearningQueueExitCode = $LASTEXITCODE",
        "if ($preCaptureLearningQueueExitCode -ne 0) {",
        "  exit $preCaptureLearningQueueExitCode",
        "}",
        "Write-Host '--- 21_build_goal_progress_audit.ps1 ---'",
        "& (Join-Path $scriptRoot '21_build_goal_progress_audit.ps1')",
        "$preCaptureGoalAuditExitCode = $LASTEXITCODE",
        "if ($preCaptureGoalAuditExitCode -ne 0) {",
        "  exit $preCaptureGoalAuditExitCode",
        "}",
        "Write-Host '--- 25_build_live_status_snapshot.ps1 ---'",
        "& (Join-Path $scriptRoot '25_build_live_status_snapshot.ps1')",
        "$preCaptureStatusExitCode = $LASTEXITCODE",
        "if ($preCaptureStatusExitCode -ne 0) {",
        "  exit $preCaptureStatusExitCode",
        "}",
        "Write-Host '--- 26_build_operator_handoff_card.ps1 ---'",
        "& (Join-Path $scriptRoot '26_build_operator_handoff_card.ps1')",
        "$preCaptureHandoffExitCode = $LASTEXITCODE",
        "if ($preCaptureHandoffExitCode -ne 0) {",
        "  exit $preCaptureHandoffExitCode",
        "}",
        "Write-Host '--- 27_check_prelaunch_audit.ps1 ---'",
        "& (Join-Path $scriptRoot '27_check_prelaunch_audit.ps1')",
        "$prelaunchExitCode = $LASTEXITCODE",
        "if ($prelaunchExitCode -ne 0) {",
        "  exit $prelaunchExitCode",
        "}",
        "Write-Host '--- 29_check_operator_command_audit.ps1 ---'",
        "& (Join-Path $scriptRoot '29_check_operator_command_audit.ps1')",
        "$operatorCommandAuditExitCode = $LASTEXITCODE",
        "if ($operatorCommandAuditExitCode -ne 0) {",
        "  exit $operatorCommandAuditExitCode",
        "}",
        "Write-Host '--- 32_check_guarded_retake_readiness.ps1 ---'",
        "& (Join-Path $scriptRoot '32_check_guarded_retake_readiness.ps1')",
        "$guardedRetakeExitCode = $LASTEXITCODE",
        "if ($guardedRetakeExitCode -ne 0) {",
        "  exit $guardedRetakeExitCode",
        "}",
        "try {",
        "  Write-Host '--- 22_run_live_demo_operator_confirmed.ps1 ---'",
        "  & (Join-Path $scriptRoot '22_run_live_demo_operator_confirmed.ps1')",
        "}",
        "catch {",
        "  $pipelineFailed = $true",
        "  $pipelineError = $_",
        "  Write-Host ('Operator-confirmed live demo run failed: ' + $_.Exception.Message)",
        "}",
        "finally {",
        "  Write-Host '--- 23_check_goal_progress_strict.ps1 ---'",
        "  & (Join-Path $scriptRoot '23_check_goal_progress_strict.ps1')",
        "  $strictExitCode = $LASTEXITCODE",
        "  Write-Host '--- 25_build_live_status_snapshot.ps1 ---'",
        "  & (Join-Path $scriptRoot '25_build_live_status_snapshot.ps1')",
        "  Write-Host '--- 26_build_operator_handoff_card.ps1 ---'",
        "  & (Join-Path $scriptRoot '26_build_operator_handoff_card.ps1')",
        "  if (Test-Path $handoffMarkdownPath) {",
        "    Write-Host ''",
        "    Write-Host '--- Operator handoff summary ---'",
        "    Get-Content $handoffMarkdownPath | ForEach-Object { Write-Host $_ }",
        "  }",
        "}",
        "if ($strictExitCode -ne 0) {",
        "  exit $strictExitCode",
        "}",
        "if ($pipelineFailed) {",
        "  throw $pipelineError",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _live_status_snapshot_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--output-root",
        config.live_status_snapshot_output_root.as_posix(),
        "--goal-audit",
        (config.goal_progress_audit_output_root / "goal_progress_audit.json").as_posix(),
        "--final-run-card",
        (config.final_run_card_output_root / "final_run_card.json").as_posix(),
        "--learning-queue",
        (config.learning_queue_output_root / "learning_queue.json").as_posix(),
        "--readiness-summary",
        (config.readiness_output_root / "readiness_summary.json").as_posix(),
        "--evidence-bundle",
        (config.evidence_bundle_output_root / "demo_evidence_bundle.json").as_posix(),
        "--live-overlay",
        config.live_output.as_posix(),
        "--live-frame-log",
        config.live_frame_log.as_posix(),
        "--live-composite",
        (config.live_composite_output_root / "realtime_schunk_demo_composite.mp4").as_posix(),
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Building realtime demo live status snapshot'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.build_realtime_demo_live_status_snapshot {quoted_args}\n"
    )


def _operator_handoff_card_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--output-root",
        config.operator_handoff_card_output_root.as_posix(),
        "--live-status-snapshot",
        (config.live_status_snapshot_output_root / "live_status_snapshot.json").as_posix(),
        "--final-run-card",
        (config.final_run_card_output_root / "final_run_card.json").as_posix(),
        "--launch-summary",
        (config.output_root / "demo_launch_summary.json").as_posix(),
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Building realtime demo operator handoff card'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.build_realtime_demo_operator_handoff_card {quoted_args}\n"
    )


def _prelaunch_audit_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--output-root",
        config.prelaunch_audit_output_root.as_posix(),
        "--live-status-snapshot",
        (config.live_status_snapshot_output_root / "live_status_snapshot.json").as_posix(),
        "--operator-handoff-card",
        (config.operator_handoff_card_output_root / "operator_handoff_card.json").as_posix(),
        "--launch-summary",
        (config.output_root / "demo_launch_summary.json").as_posix(),
        "--strict-exit-code",
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Checking realtime demo prelaunch audit'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.build_realtime_demo_prelaunch_audit {quoted_args}\n"
    )


def _wrapper_contract_probe_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--output-root",
        config.wrapper_contract_probe_output_root.as_posix(),
        "--strict-exit-code",
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Checking realtime demo strict-wrapper PowerShell contract'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.run_realtime_demo_wrapper_contract_probe {quoted_args}\n"
        "exit $LASTEXITCODE\n"
    )


def _operator_command_audit_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--output-root",
        config.operator_command_audit_output_root.as_posix(),
        "--strict-exit-code",
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Checking realtime demo operator command audit'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.audit_realtime_demo_operator_commands {quoted_args}\n"
        "exit $LASTEXITCODE\n"
    )


def _guarded_retake_readiness_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--output-root",
        config.guarded_retake_readiness_output_root.as_posix(),
        "--readiness-summary",
        (config.readiness_output_root / "readiness_summary.json").as_posix(),
        "--live-status-snapshot",
        (config.live_status_snapshot_output_root / "live_status_snapshot.json").as_posix(),
        "--prelaunch-audit",
        (config.prelaunch_audit_output_root / "prelaunch_audit.json").as_posix(),
        "--operator-command-audit",
        (config.operator_command_audit_output_root / "operator_command_audit.json").as_posix(),
        "--live-artifact-cleanup",
        (config.live_artifact_cleanup_output_root / "live_artifact_cleanup.json").as_posix(),
        "--launch-summary",
        (config.output_root / "demo_launch_summary.json").as_posix(),
        "--strict-exit-code",
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Checking guarded realtime demo rock-retake readiness'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.build_realtime_demo_guarded_retake_readiness {quoted_args}\n"
        "exit $LASTEXITCODE\n"
    )


def _live_rock_retake_gate_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    postcapture_summary = config.live_output.parent / "postcapture" / "postcapture_summary.json"
    args = [
        "--output-root",
        config.live_rock_retake_gate_output_root.as_posix(),
        "--frame-log",
        config.live_frame_log.as_posix(),
        "--postcapture-summary",
        postcapture_summary.as_posix(),
        "--response-prompt",
        "scissors",
        "--min-detection-rate",
        "0.80",
        "--strict-exit-code",
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    lines = [
        "Write-Host 'Checking live rock-retake false-trigger gate'",
        "$rockRetakeArgs = @()",
        "if (-not [string]::IsNullOrWhiteSpace($env:EMBODIED_RPS_EXPECTED_ACTUAL_GESTURE)) {",
        "  $rockRetakeArgs += @('--expected-actual-gesture', $env:EMBODIED_RPS_EXPECTED_ACTUAL_GESTURE)",
        "}",
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.build_realtime_demo_live_rock_retake_gate @rockRetakeArgs {quoted_args}",
        "exit $LASTEXITCODE",
    ]
    return "\n".join(lines) + "\n"


def _scissors_pose_collection_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    lines = [
        "Write-Host 'Starting prompt pose collection mode'",
        "$selectedGesture = Read-Host 'Collection gesture (rock/paper/scissors)'",
        "$selectedGesture = $selectedGesture.Trim().ToLowerInvariant()",
        "if ([string]::IsNullOrWhiteSpace($selectedGesture)) {",
        "  throw 'Collection gesture is required: rock, paper, or scissors.'",
        "}",
        "if (@('rock', 'paper', 'scissors') -notcontains $selectedGesture) {",
        "  throw 'Collection gesture must be rock, paper, or scissors.'",
        "}",
        "$runId = 'run_' + (Get-Date -Format 'yyyyMMdd_HHmmss')",
        f"$collectionRootBase = '{_ps_quote(config.scissors_collection_output_root.as_posix())}'",
        "$collectionRootBase = Join-Path $collectionRootBase $selectedGesture",
        "$collectionRoot = Join-Path $collectionRootBase $runId",
        "New-Item -ItemType Directory -Force -Path $collectionRoot | Out-Null",
        "$collectionConfig = Join-Path $collectionRoot 'collection_config.yaml'",
        "$baseConfigLines = Get-Content '" + _ps_quote(config.scissors_collection_config_path.as_posix()) + "'",
        "$baseConfigLines = $baseConfigLines -replace '^response_prompt:.*$', 'response_prompt: scissors'",
        "$baseConfigLines | Set-Content -Path $collectionConfig -Encoding UTF8",
        "$overlayVideo = Join-Path $collectionRoot ($selectedGesture + '_pose_collection_overlay.mp4')",
        "$frameLog = Join-Path $collectionRoot ($selectedGesture + '_pose_collection_frames.jsonl')",
        "$skeletonNpz = Join-Path $collectionRoot ($selectedGesture + '_pose_collection_skeletons.npz')",
        "$summaryRoot = Join-Path $collectionRoot 'summary'",
        "Write-Host ('Collection output root: ' + $collectionRoot)",
        "Write-Host ('Selected actual gesture label: ' + $selectedGesture)",
        "Write-Host 'Collection uses a rock/paper/scissors prompt cycle.'",
        "Write-Host 'PROMPT SCISSORS remains the bounded response window for every collection label.'",
        "Write-Host 'Perform the selected gesture during PROMPT SCISSORS; use standby/ambiguous preparation during PROMPT ROCK and PROMPT PAPER.'",
        "Write-Host 'For rock collection, keep rock/wait through PROMPT SCISSORS; do not open into paper or scissors.'",
        "Write-Host 'For paper collection, allow early ambiguity but resolve to paper inside PROMPT SCISSORS.'",
        "Write-Host 'For scissors collection, move to varied scissors poses during PROMPT SCISSORS.'",
        "Write-Host 'Collect at least 10 prompt cycles per selected gesture when possible.'",
        "Write-Host 'Press q in the OpenCV preview to stop early.'",
        (
            f"& '{_ps_quote(config.python_executable.as_posix())}' "
            "-m embodied_rps.tools.run_current_best_realtime_demo "
            "--config $collectionConfig "
            f"--camera '{int(config.camera_index)}' "
            "--output $overlayVideo "
            "--frame-log-jsonl $frameLog "
            "--skeleton-npz $skeletonNpz "
            f"--max-frames '{int(config.scissors_collection_max_frames)}' "
            "--expected-actual-gesture $selectedGesture "
            "--collection-mode"
        ),
        "$captureExitCode = $LASTEXITCODE",
        "if ($captureExitCode -ne 0) {",
        "  exit $captureExitCode",
        "}",
        "Write-Host 'Summarizing prompt pose collection artifacts'",
        (
            f"& '{_ps_quote(config.python_executable.as_posix())}' "
            "-m embodied_rps.tools.summarize_scissors_pose_collection "
            "--frame-log-jsonl $frameLog "
            "--skeleton-npz $skeletonNpz "
            "--overlay-video $overlayVideo "
            "--output-root $summaryRoot "
            "--collection-label $selectedGesture"
        ),
        "exit $LASTEXITCODE",
    ]
    return "\n".join(lines) + "\n"


def _live_artifact_cleanup_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--output-root",
        config.live_artifact_cleanup_output_root.as_posix(),
        "--workspace-root",
        config.project_root.as_posix(),
        "--live-overlay",
        config.live_output.as_posix(),
        "--live-frame-log",
        config.live_frame_log.as_posix(),
        "--live-postcapture-root",
        (config.live_output.parent / "postcapture").as_posix(),
        "--live-composite-root",
        config.live_composite_output_root.as_posix(),
        "--live-overlay-contract-root",
        config.overlay_contract_output_root.as_posix(),
        "--live-rock-retake-gate-root",
        config.live_rock_retake_gate_output_root.as_posix(),
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        "Write-Host 'Clearing stale realtime demo live artifacts'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.clear_realtime_demo_live_artifacts {quoted_args}\n"
    )


def _manual_review_decision_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    archive_index = (config.archive_output_root / "run_archive_index.json").as_posix()
    output_root = config.manual_review_decisions.parent.as_posix()
    lines = [
        "$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path",
        "Write-Host 'Recording operator manual review for an archived realtime demo run'",
        "Write-Host '--- 15_summarize_run_archives.ps1 ---'",
        "& (Join-Path $scriptRoot '15_summarize_run_archives.ps1')",
        "$reviewStatus = Read-Host 'Review status (approved/rejected_by_manual_review)'",
        "$reviewStatus = $reviewStatus.Trim().ToLowerInvariant()",
        "if (@('approved', 'rejected_by_manual_review') -notcontains $reviewStatus) {",
        "  throw 'Review status must be approved or rejected_by_manual_review.'",
        "}",
        "$reviewRunId = Read-Host 'Archived run ID to review (blank = latest complete archived run)'",
        "$reviewNotes = Read-Host 'Manual review notes'",
        "$reviewArgs = @(",
        f"  '--archive-index', '{_ps_quote(archive_index)}',",
        f"  '--manual-review-decisions', '{_ps_quote(config.manual_review_decisions.as_posix())}',",
        f"  '--output-root', '{_ps_quote(output_root)}',",
        "  '--status', $reviewStatus,",
        "  '--notes', $reviewNotes",
        ")",
        "if ($reviewRunId.Trim()) {",
        "  $reviewArgs += @('--run-id', $reviewRunId.Trim())",
        "}",
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.record_realtime_demo_manual_review @reviewArgs",
        "Write-Host '--- 15_summarize_run_archives.ps1 ---'",
        "& (Join-Path $scriptRoot '15_summarize_run_archives.ps1')",
        "Write-Host '--- 16_select_final_demo_candidate.ps1 ---'",
        "& (Join-Path $scriptRoot '16_select_final_demo_candidate.ps1')",
        "Write-Host '--- 17_build_submission_candidate_packet.ps1 ---'",
        "& (Join-Path $scriptRoot '17_build_submission_candidate_packet.ps1')",
        "Write-Host '--- 19_build_final_run_card.ps1 ---'",
        "& (Join-Path $scriptRoot '19_build_final_run_card.ps1')",
        "Write-Host '--- 20_build_learning_queue.ps1 ---'",
        "& (Join-Path $scriptRoot '20_build_learning_queue.ps1')",
        "Write-Host '--- 21_build_goal_progress_audit.ps1 ---'",
        "& (Join-Path $scriptRoot '21_build_goal_progress_audit.ps1')",
        "Write-Host '--- 25_build_live_status_snapshot.ps1 ---'",
        "& (Join-Path $scriptRoot '25_build_live_status_snapshot.ps1')",
        "Write-Host '--- 26_build_operator_handoff_card.ps1 ---'",
        "& (Join-Path $scriptRoot '26_build_operator_handoff_card.ps1')",
    ]
    return "\n".join(lines) + "\n"


def _overlay_contract_command_lines(config: RealtimeDemoLaunchScriptsConfig) -> str:
    args = [
        "--overlay-video",
        config.live_output.as_posix(),
        "--frame-log-jsonl",
        config.live_frame_log.as_posix(),
        "--output-root",
        config.overlay_contract_output_root.as_posix(),
        "--response-prompt",
        "scissors",
        "--min-detection-rate",
        "0.80",
        "--max-binary-latency-s",
        "0.50",
    ]
    quoted_args = " ".join(f"'{_ps_quote(value)}'" for value in args)
    return (
        f"if (-not (Test-Path '{_ps_quote(config.live_output.as_posix())}')) {{\n"
        "  throw 'Run 02_live_camera_demo.ps1 first so the live camera overlay exists.'\n"
        "}\n"
        f"if (-not (Test-Path '{_ps_quote(config.live_frame_log.as_posix())}')) {{\n"
        "  throw 'Run 02_live_camera_demo.ps1 first so the live frame log exists.'\n"
        "}\n"
        + "\n".join(_expected_actual_gesture_arg_lines())
        + "\n"
        "Write-Host 'Auditing live realtime demo overlay display contract'\n"
        f"& '{_ps_quote(config.python_executable.as_posix())}' -m embodied_rps.tools.audit_realtime_demo_overlay_contract @expectedActualGestureArgs {quoted_args}\n"
    )


def _readme(config: RealtimeDemoLaunchScriptsConfig, scripts: dict[str, Path]) -> str:
    lines = [
        "# Realtime Demo Launch Scripts",
        "",
        "Run these from PowerShell on the local demo computer.",
        "",
        "## Order",
        "",
        "1. `18_build_live_run_checklist.ps1` writes the pre-run operator checklist from the current readiness/preflight/operator artifacts.",
        "2. `00_demo_preflight.ps1` checks config, profiles, model files, policy settings, camera access, and MediaPipe hand visibility.",
        "3. `03_print_live_camera_argv.ps1` prints the delegated realtime predictor arguments.",
        "4. `01_dry_run_video_rehearsal.ps1` verifies the overlay path on a prerecorded MP4.",
        "5. `02_live_camera_demo.ps1` starts the live camera demo and writes the overlay video.",
        "6. `06_verify_live_capture.ps1` checks the live overlay stream and extracts prompt review frames.",
        "7. `33_check_live_rock_retake_gate.ps1` strictly checks expected-rock live retakes for any response-window paper/scissors false-trigger leakage.",
        "8. `05_create_live_schunk_composite.ps1` combines the live overlay with the SCHUNK response preview.",
        "9. `07_run_live_demo_pipeline.ps1` clears stale fixed-path live artifacts, runs the pre-run checklist, preflight, live capture, verification, rock-retake gate, composite creation, artifact-folder opening, and always runs triage, live overlay-contract audit, evidence-bundle refresh, review-packet refresh, final acceptance-report refresh, operator-outcome refresh, run archiving, archive-index refresh, final-candidate selection, submission-candidate packet refresh, final-run card refresh, learning-queue refresh, and goal-progress audit refresh at the end.",
        "10. `22_run_live_demo_operator_confirmed.ps1` is the recommended live-capture entry point; it asks for the actual gesture to validate, prints the checklist, waits for Enter, then runs the full pipeline.",
        "11. `08_triage_live_capture.ps1` refreshes readiness and classifies any capture/model/demo failure when run manually.",
        "12. `09_build_demo_evidence_bundle.ps1` writes the submission-facing demo evidence status.",
        "13. `10_build_demo_review_packet.ps1` collects lightweight review visuals and references the demo videos without duplicating them.",
        "14. `11_audit_live_overlay_contract.ps1` verifies the live overlay/video frame log exposes the required demo information.",
        "15. `12_build_demo_acceptance_report.ps1` writes the final live-demo acceptance verdict for README/report/video claims.",
        "16. `13_build_operator_outcome_report.ps1` writes the one-screen operator decision after a live run.",
        "17. `14_archive_live_demo_run.ps1` preserves the current live/demo reports and any live media into a timestamped archive folder.",
        "18. `15_summarize_run_archives.ps1` summarizes archived runs and identifies the latest final-video candidate.",
        "19. `16_select_final_demo_candidate.ps1` writes the packaging-facing final demo candidate manifest from the archive index.",
        "20. `17_build_submission_candidate_packet.ps1` writes the README/report/YouTube-linking packet for the selected final demo candidate.",
        "21. `19_build_final_run_card.ps1` writes the one-card current state for record, retake, research, or submission handoff.",
        "22. `20_build_learning_queue.ps1` writes the post-run branch for live capture, retake, setup fix, final packaging, or simulation-first augmentation.",
        "23. `21_build_goal_progress_audit.ps1` writes the objective-level requirement audit for the active goal.",
        "24. `23_check_goal_progress_strict.ps1` refreshes the goal audit and returns a strict exit code: `0` complete, `10` waiting for live capture, `30` postprocess repair needed, `40` research iteration needed, or `60` manual review.",
        "25. `24_run_live_demo_operator_confirmed_strict.ps1` clears stale live artifacts, rebuilds pre-capture triage/operator outcome/checklist/final-card/learning-queue/goal-audit/status/handoff artifacts, runs the prelaunch/operator-command/guarded-retake audits, executes the operator-confirmed live capture flow, then refreshes strict goal progress, live status, and the operator handoff card.",
        "26. `25_build_live_status_snapshot.ps1` writes one JSON/Markdown snapshot of live artifacts, goal state, and next action.",
        "27. `26_build_operator_handoff_card.ps1` writes the human-facing command, gesture-timing, exit-code, and artifact-review card.",
        "28. `27_check_prelaunch_audit.ps1` verifies the recommended command, launch scripts, local Python/config/sample assets, and wrapper contract before live capture.",
        "29. `28_check_wrapper_contract.ps1` runs stubbed PowerShell scenarios against the strict wrapper without camera capture.",
        "30. `29_check_operator_command_audit.ps1` verifies all operator-facing artifacts recommend the strict wrapper, not the raw internal pipeline.",
        "31. `30_clear_stale_live_artifacts.ps1` removes fixed-path live overlay, frame-log, postcapture, live-composite, live-overlay-contract, and live-rock-retake gate artifacts before a new capture attempt.",
        "32. `31_record_manual_review_decision.ps1` records operator visual approval or rejection for an archived live retake, then refreshes archive/final-candidate/submission/status artifacts.",
        "33. `32_check_guarded_retake_readiness.ps1` verifies the next rock retake is wired to the live-only rock-hold guard and stale fixed-path live artifacts are absent.",
        "34. `33_check_live_rock_retake_gate.ps1` checks expected-rock live retakes for response-window false triggers.",
        "35. `34_run_scissors_pose_collection.ps1` records a long prompt pose collection session using the rock/paper/scissors prompt cycle. It asks which actual gesture label to collect (`rock`, `paper`, or `scissors`), keeps `PROMPT SCISSORS` as the fixed response window, and stores the run under a gesture-specific folder. Collect at least 10 prompt cycles per selected gesture when possible. It writes overlay MP4, frame log, skeleton NPZ, contact sheet, and crop-range template; it does not run final-demo acceptance gates.",
        "36. `04_open_demo_artifacts.ps1` opens the output artifact folder.",
        "",
        "## Current Demo Policy",
        "",
        f"- Config: `{config.config_path.as_posix()}`",
        "- Uses `response-prompt scissors` through the config.",
        "- Treats `PROMPT ROCK` and `PROMPT PAPER` as rock-standby prompts.",
        "- Treats `PROMPT SCISSORS` as the response window, not as an instruction to show scissors.",
        "- Passes the operator-entered actual gesture through `EMBODIED_RPS_EXPECTED_ACTUAL_GESTURE` for validation metadata only.",
        "- Uses `reset-on-prompt-change` through the config so each prompt segment starts fresh.",
        "- Requires a manual review decision before any archived run can become the final demo candidate.",
        "- The predictor still consumes MediaPipe skeleton landmarks, not RGB frames.",
        "- SCHUNK autonomous response remains blocked outside this prompt-gated demo context.",
        "",
        "## Operator Outcome Exit Codes",
        "",
        "`13_build_operator_outcome_report.ps1` writes the outcome without failing the live pipeline. For scripting, run the underlying CLI with `--strict-exit-code` to return the recommended code:",
        "",
        "```powershell",
        f"& '{config.python_executable.as_posix()}' -m embodied_rps.tools.build_realtime_demo_operator_report --strict-exit-code",
        "```",
        "",
        "- `0`: ready for final video packaging.",
        "- `10`: live capture is still missing.",
        "- `20`: retake the live capture after fixing camera/framing.",
        "- `30`: rerun post-processing or repair the live capture artifacts.",
        "- `40`: run a simulation-first research iteration.",
        "- `50`: fix local setup, config, model profile, or camera access.",
        "- `60`: inspect artifacts manually.",
        "",
        "## Scripts",
        "",
    ]
    for name, path in scripts.items():
        lines.append(f"- `{name}`: `{path.as_posix()}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Preflight summary: `{config.preflight_output.as_posix()}/preflight_summary.json`",
            f"- Prerecorded rehearsal overlay: `{config.rehearsal_output.as_posix()}`",
            f"- Prerecorded rehearsal frame log: `{config.rehearsal_frame_log.as_posix()}`",
            f"- Live camera overlay: `{config.live_output.as_posix()}`",
            f"- Live camera frame log: `{config.live_frame_log.as_posix()}`",
            f"- Hand visibility preflight: `{config.hand_visibility_max_frames}` frames, minimum detection rate `{config.hand_visibility_min_detection_rate}`",
            f"- SCHUNK response preview image: `{config.response_preview_image.as_posix()}`",
            f"- Live composite output root: `{config.live_composite_output_root.as_posix()}`",
            f"- Readiness output root: `{config.readiness_output_root.as_posix()}`",
            f"- Triage output root: `{config.triage_output_root.as_posix()}`",
            f"- Evidence bundle output root: `{config.evidence_bundle_output_root.as_posix()}`",
            f"- Review packet output root: `{config.review_packet_output_root.as_posix()}`",
            f"- Acceptance report output root: `{config.acceptance_report_output_root.as_posix()}`",
            f"- Operator outcome output root: `{config.operator_report_output_root.as_posix()}`",
            f"- Run archive output root: `{config.archive_output_root.as_posix()}`",
            f"- Final candidate output root: `{config.final_candidate_output_root.as_posix()}`",
            f"- Submission packet output root: `{config.submission_packet_output_root.as_posix()}`",
            f"- Live run checklist output root: `{config.live_run_checklist_output_root.as_posix()}`",
            f"- Final run card output root: `{config.final_run_card_output_root.as_posix()}`",
            f"- Learning queue output root: `{config.learning_queue_output_root.as_posix()}`",
            f"- Goal progress audit output root: `{config.goal_progress_audit_output_root.as_posix()}`",
            f"- Live status snapshot output root: `{config.live_status_snapshot_output_root.as_posix()}`",
            f"- Operator handoff card output root: `{config.operator_handoff_card_output_root.as_posix()}`",
            f"- Prelaunch audit output root: `{config.prelaunch_audit_output_root.as_posix()}`",
            f"- Wrapper contract probe output root: `{config.wrapper_contract_probe_output_root.as_posix()}`",
            f"- Operator command audit output root: `{config.operator_command_audit_output_root.as_posix()}`",
            f"- Live artifact cleanup output root: `{config.live_artifact_cleanup_output_root.as_posix()}`",
            f"- Prompt pose collection output root: `{config.scissors_collection_output_root.as_posix()}`",
            f"- Prompt pose collection base config: `{config.scissors_collection_config_path.as_posix()}`",
            f"- Prompt pose collection max frames: `{config.scissors_collection_max_frames}`",
            f"- Dry-run overlay contract summary: `{config.dry_run_overlay_contract_summary.as_posix()}`",
            f"- Overlay contract output root: `{config.overlay_contract_output_root.as_posix()}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _ps_quote(value: str) -> str:
    return value.replace("'", "''")


def _write_ps1(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8-sig")


__all__ = ["RealtimeDemoLaunchScriptsConfig", "write_realtime_demo_launch_scripts"]
