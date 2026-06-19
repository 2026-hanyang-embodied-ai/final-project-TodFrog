from __future__ import annotations

import json
from pathlib import Path

from embodied_rps.realtime_demo_launch_scripts import (
    RealtimeDemoLaunchScriptsConfig,
    write_realtime_demo_launch_scripts,
)
from embodied_rps.tools.write_realtime_demo_launch_scripts import main


def test_realtime_demo_launch_scripts_write_operator_scripts(tmp_path: Path) -> None:
    output_root = tmp_path / "launch"
    project_root = tmp_path / "project"
    sample_video = tmp_path / "텀프영상" / "sample.mp4"
    python_executable = tmp_path / "python.exe"
    response_preview_image = tmp_path / "response_preview.png"
    live_composite_output_root = tmp_path / "live_composite"

    summary = write_realtime_demo_launch_scripts(
        RealtimeDemoLaunchScriptsConfig(
            output_root=output_root,
            project_root=project_root,
            python_executable=python_executable,
            sample_video=sample_video,
            response_preview_image=response_preview_image,
            live_composite_output_root=live_composite_output_root,
            camera_index=1,
            camera_max_frames=300,
        )
    )

    assert summary["script_count"] == 35
    preflight = (output_root / "00_demo_preflight.ps1").read_text(encoding="utf-8")
    dry_video = (output_root / "01_dry_run_video_rehearsal.ps1").read_text(encoding="utf-8")
    live_camera = (output_root / "02_live_camera_demo.ps1").read_text(encoding="utf-8")
    dry_argv = (output_root / "03_print_live_camera_argv.ps1").read_text(encoding="utf-8")
    verify_live_capture = (output_root / "06_verify_live_capture.ps1").read_text(encoding="utf-8")
    live_pipeline = (output_root / "07_run_live_demo_pipeline.ps1").read_text(encoding="utf-8")
    triage_live_capture = (output_root / "08_triage_live_capture.ps1").read_text(encoding="utf-8")
    evidence_bundle = (output_root / "09_build_demo_evidence_bundle.ps1").read_text(encoding="utf-8")
    review_packet = (output_root / "10_build_demo_review_packet.ps1").read_text(encoding="utf-8")
    overlay_contract = (output_root / "11_audit_live_overlay_contract.ps1").read_text(encoding="utf-8")
    acceptance_report = (output_root / "12_build_demo_acceptance_report.ps1").read_text(encoding="utf-8")
    operator_report = (output_root / "13_build_operator_outcome_report.ps1").read_text(encoding="utf-8")
    archive_run = (output_root / "14_archive_live_demo_run.ps1").read_text(encoding="utf-8")
    archive_index = (output_root / "15_summarize_run_archives.ps1").read_text(encoding="utf-8")
    final_candidate = (output_root / "16_select_final_demo_candidate.ps1").read_text(encoding="utf-8")
    submission_packet = (output_root / "17_build_submission_candidate_packet.ps1").read_text(encoding="utf-8")
    live_run_checklist = (output_root / "18_build_live_run_checklist.ps1").read_text(encoding="utf-8")
    final_run_card = (output_root / "19_build_final_run_card.ps1").read_text(encoding="utf-8")
    learning_queue = (output_root / "20_build_learning_queue.ps1").read_text(encoding="utf-8")
    goal_audit = (output_root / "21_build_goal_progress_audit.ps1").read_text(encoding="utf-8")
    operator_confirmed = (output_root / "22_run_live_demo_operator_confirmed.ps1").read_text(encoding="utf-8")
    strict_goal_check = (output_root / "23_check_goal_progress_strict.ps1").read_text(encoding="utf-8")
    operator_confirmed_strict = (output_root / "24_run_live_demo_operator_confirmed_strict.ps1").read_text(
        encoding="utf-8"
    )
    live_status_snapshot = (output_root / "25_build_live_status_snapshot.ps1").read_text(encoding="utf-8")
    operator_handoff_card = (output_root / "26_build_operator_handoff_card.ps1").read_text(encoding="utf-8")
    prelaunch_audit = (output_root / "27_check_prelaunch_audit.ps1").read_text(encoding="utf-8")
    wrapper_contract_probe = (output_root / "28_check_wrapper_contract.ps1").read_text(encoding="utf-8")
    operator_command_audit = (output_root / "29_check_operator_command_audit.ps1").read_text(encoding="utf-8")
    live_artifact_cleanup = (output_root / "30_clear_stale_live_artifacts.ps1").read_text(encoding="utf-8")
    manual_review_decision = (output_root / "31_record_manual_review_decision.ps1").read_text(encoding="utf-8")
    guarded_retake_readiness = (output_root / "32_check_guarded_retake_readiness.ps1").read_text(encoding="utf-8")
    live_rock_retake_gate = (output_root / "33_check_live_rock_retake_gate.ps1").read_text(encoding="utf-8")
    scissors_collection = (output_root / "34_run_scissors_pose_collection.ps1").read_text(encoding="utf-8")
    live_composite = (output_root / "05_create_live_schunk_composite.ps1").read_text(encoding="utf-8")
    readme = (output_root / "README.md").read_text(encoding="utf-8")

    assert "run_realtime_demo_preflight" in preflight
    assert "--check-camera" in preflight
    assert "--check-hand-visibility" in preflight
    assert "--hand-visibility-min-detection-rate" in preflight
    assert "--hand-visibility-max-frames" in preflight
    assert "--require-response-prompt" in preflight
    assert "scissors" in preflight
    assert "run_current_best_realtime_demo" in dry_video
    assert "--video" in dry_video
    assert sample_video.as_posix() in dry_video
    assert "--camera" in live_camera
    assert "'1'" in live_camera
    assert "--max-frames" in live_camera
    assert "'300'" in live_camera
    assert "$env:EMBODIED_RPS_EXPECTED_ACTUAL_GESTURE" in live_camera
    assert "--expected-actual-gesture" in live_camera
    assert "--dry-run" in dry_argv
    assert "verify_realtime_demo_capture" in verify_live_capture
    assert "--overlay-video" in verify_live_capture
    assert "live_camera_overlay.mp4" in verify_live_capture
    assert "live_camera_frames.jsonl" in live_camera
    assert "video_rehearsal_frames.jsonl" in dry_video
    assert "--enforce-demo-success-gate" in verify_live_capture
    assert "--max-response-binary-latency-s" in verify_live_capture
    assert "--min-detection-rate" in verify_live_capture
    assert "--response-preview-image" in verify_live_capture
    assert response_preview_image.as_posix() in verify_live_capture
    assert "--live-composite-output-root" in verify_live_capture
    assert live_composite_output_root.as_posix() in verify_live_capture
    assert "$env:EMBODIED_RPS_EXPECTED_ACTUAL_GESTURE" in verify_live_capture
    assert "--expected-actual-gesture" in verify_live_capture
    assert "00_demo_preflight.ps1" in live_pipeline
    assert "30_clear_stale_live_artifacts.ps1" in live_pipeline
    assert "02_live_camera_demo.ps1" in live_pipeline
    assert "06_verify_live_capture.ps1" in live_pipeline
    assert "33_check_live_rock_retake_gate.ps1" in live_pipeline
    assert "05_create_live_schunk_composite.ps1" in live_pipeline
    assert "04_open_demo_artifacts.ps1" in live_pipeline
    assert "try {" in live_pipeline
    assert "finally {" in live_pipeline
    assert "08_triage_live_capture.ps1" in live_pipeline
    assert "09_build_demo_evidence_bundle.ps1" in live_pipeline
    assert "10_build_demo_review_packet.ps1" in live_pipeline
    assert "11_audit_live_overlay_contract.ps1" in live_pipeline
    assert "12_build_demo_acceptance_report.ps1" in live_pipeline
    assert "13_build_operator_outcome_report.ps1" in live_pipeline
    assert "14_archive_live_demo_run.ps1" in live_pipeline
    assert "15_summarize_run_archives.ps1" in live_pipeline
    assert "16_select_final_demo_candidate.ps1" in live_pipeline
    assert "17_build_submission_candidate_packet.ps1" in live_pipeline
    assert "18_build_live_run_checklist.ps1" in live_pipeline
    assert "19_build_final_run_card.ps1" in live_pipeline
    assert "20_build_learning_queue.ps1" in live_pipeline
    assert "21_build_goal_progress_audit.ps1" in live_pipeline
    assert "triage_realtime_demo_capture" in triage_live_capture
    assert "--readiness-summary" in triage_live_capture
    assert "--preflight-summary" in triage_live_capture
    assert "--postcapture-summary" in triage_live_capture
    assert "--composite-manifest" in triage_live_capture
    assert "--live-rock-retake-gate" in triage_live_capture
    assert "live_rock_retake_gate.json" in triage_live_capture
    assert "build_realtime_demo_evidence_bundle" in evidence_bundle
    assert "--readiness-summary" in evidence_bundle
    assert "--triage-summary" in evidence_bundle
    assert "--live-postcapture-summary" in evidence_bundle
    assert "--live-composite-manifest" in evidence_bundle
    assert "--dry-run-overlay-contract-summary" in evidence_bundle
    assert "--live-overlay-contract-summary" in evidence_bundle
    assert "build_realtime_demo_review_packet" in review_packet
    assert "--evidence-bundle" in review_packet
    assert "--preflight-summary" in review_packet
    assert "--dry-run-postcapture-summary" in review_packet
    assert "--dry-run-composite-manifest" in review_packet
    assert "audit_realtime_demo_overlay_contract" in overlay_contract
    assert "--overlay-video" in overlay_contract
    assert "live_camera_overlay.mp4" in overlay_contract
    assert "--frame-log-jsonl" in overlay_contract
    assert "live_camera_frames.jsonl" in overlay_contract
    assert "$expectedActualGestureArgs = @()" in overlay_contract
    assert "--expected-actual-gesture" in overlay_contract
    assert "@expectedActualGestureArgs" in overlay_contract
    assert "build_realtime_demo_acceptance_report" in acceptance_report
    assert "--evidence-bundle" in acceptance_report
    assert "--review-packet-manifest" in acceptance_report
    assert "--dry-run-overlay-contract-summary" in acceptance_report
    assert "--live-overlay-contract-summary" in acceptance_report
    assert "build_realtime_demo_operator_report" in operator_report
    assert "--acceptance-report" in operator_report
    assert "--triage-summary" in operator_report
    assert "--review-packet-manifest" in operator_report
    assert "--readiness-summary" in operator_report
    assert "archive_realtime_demo_run" in archive_run
    assert "--operator-outcome" in archive_run
    assert "--live-overlay-video" in archive_run
    assert "--live-frame-log" in archive_run
    assert "--live-composite-manifest" in archive_run
    assert "--live-rock-retake-gate" in archive_run
    assert "live_rock_retake_gate.json" in archive_run
    assert "summarize_realtime_demo_run_archives" in archive_index
    assert "--archive-root" in archive_index
    assert "--output-root" in archive_index
    assert "--manual-review-decisions" in archive_index
    assert "select_realtime_demo_final_candidate" in final_candidate
    assert "--archive-index" in final_candidate
    assert "--output-root" in final_candidate
    assert "build_realtime_demo_submission_packet" in submission_packet
    assert "--final-candidate" in submission_packet
    assert "--output-root" in submission_packet
    assert "build_realtime_demo_live_run_checklist" in live_run_checklist
    assert "--readiness-summary" in live_run_checklist
    assert "--operator-outcome" in live_run_checklist
    assert "--preflight-summary" in live_run_checklist
    assert "--launch-summary" in live_run_checklist
    assert "build_realtime_demo_final_run_card" in final_run_card
    assert "--live-run-checklist" in final_run_card
    assert "--operator-outcome" in final_run_card
    assert "--submission-packet" in final_run_card
    assert "--output-root" in final_run_card
    assert "build_realtime_demo_learning_queue" in learning_queue
    assert "--final-run-card" in learning_queue
    assert "--triage-summary" in learning_queue
    assert "--operator-outcome" in learning_queue
    assert "--output-root" in learning_queue
    assert "build_realtime_demo_goal_progress_audit" in goal_audit
    assert "--readiness-summary" in goal_audit
    assert "--evidence-bundle" in goal_audit
    assert "--final-run-card" in goal_audit
    assert "--learning-queue" in goal_audit
    assert "--output-root" in goal_audit
    assert "18_build_live_run_checklist.ps1" in operator_confirmed
    assert "live_run_checklist.md" in operator_confirmed
    assert "Read-Host" in operator_confirmed
    assert "actual gesture" in operator_confirmed
    assert "Enter=rock" in operator_confirmed
    assert "[string]::IsNullOrWhiteSpace($expectedActualGesture)" in operator_confirmed
    assert "defaulting to rock" in operator_confirmed
    assert "$env:EMBODIED_RPS_EXPECTED_ACTUAL_GESTURE = $expectedActualGesture" in operator_confirmed
    assert "PROMPT SCISSORS" in operator_confirmed
    assert "RESPONSE WINDOW" in operator_confirmed
    assert "Keep rock standby during PROMPT ROCK and PROMPT PAPER" in operator_confirmed
    assert "07_run_live_demo_pipeline.ps1" in operator_confirmed
    assert operator_confirmed.index("18_build_live_run_checklist.ps1") < operator_confirmed.index("Read-Host")
    assert operator_confirmed.index("Read-Host") < operator_confirmed.index("07_run_live_demo_pipeline.ps1")
    assert "build_realtime_demo_goal_progress_audit" in strict_goal_check
    assert "--strict-exit-code" in strict_goal_check
    assert "21_build_goal_progress_audit.ps1" not in strict_goal_check
    assert "$LASTEXITCODE" in strict_goal_check
    assert "exit $LASTEXITCODE" in strict_goal_check
    assert "22_run_live_demo_operator_confirmed.ps1" in operator_confirmed_strict
    assert "27_check_prelaunch_audit.ps1" in operator_confirmed_strict
    assert "30_clear_stale_live_artifacts.ps1" in operator_confirmed_strict
    assert "08_triage_live_capture.ps1" in operator_confirmed_strict
    assert "13_build_operator_outcome_report.ps1" in operator_confirmed_strict
    assert "18_build_live_run_checklist.ps1" in operator_confirmed_strict
    assert "19_build_final_run_card.ps1" in operator_confirmed_strict
    assert "20_build_learning_queue.ps1" in operator_confirmed_strict
    assert "21_build_goal_progress_audit.ps1" in operator_confirmed_strict
    assert "32_check_guarded_retake_readiness.ps1" in operator_confirmed_strict
    assert "23_check_goal_progress_strict.ps1" in operator_confirmed_strict
    assert "25_build_live_status_snapshot.ps1" in operator_confirmed_strict
    assert "26_build_operator_handoff_card.ps1" in operator_confirmed_strict
    assert "finally {" in operator_confirmed_strict
    assert "$prelaunchExitCode" in operator_confirmed_strict
    assert "exit $prelaunchExitCode" in operator_confirmed_strict
    assert "$guardedRetakeExitCode" in operator_confirmed_strict
    assert "exit $guardedRetakeExitCode" in operator_confirmed_strict
    assert "$strictExitCode" in operator_confirmed_strict
    assert "exit $strictExitCode" in operator_confirmed_strict
    assert operator_confirmed_strict.index("30_clear_stale_live_artifacts.ps1") < operator_confirmed_strict.index(
        "08_triage_live_capture.ps1"
    )
    assert operator_confirmed_strict.index("08_triage_live_capture.ps1") < operator_confirmed_strict.index(
        "13_build_operator_outcome_report.ps1"
    )
    assert operator_confirmed_strict.index("13_build_operator_outcome_report.ps1") < operator_confirmed_strict.index(
        "18_build_live_run_checklist.ps1"
    )
    assert operator_confirmed_strict.index("18_build_live_run_checklist.ps1") < operator_confirmed_strict.index(
        "19_build_final_run_card.ps1"
    )
    assert operator_confirmed_strict.index("19_build_final_run_card.ps1") < operator_confirmed_strict.index(
        "20_build_learning_queue.ps1"
    )
    assert operator_confirmed_strict.index("20_build_learning_queue.ps1") < operator_confirmed_strict.index(
        "21_build_goal_progress_audit.ps1"
    )
    assert operator_confirmed_strict.index("21_build_goal_progress_audit.ps1") < operator_confirmed_strict.index(
        "25_build_live_status_snapshot.ps1"
    )
    first_status_refresh = operator_confirmed_strict.index("25_build_live_status_snapshot.ps1")
    assert first_status_refresh < operator_confirmed_strict.index("26_build_operator_handoff_card.ps1")
    assert operator_confirmed_strict.index("26_build_operator_handoff_card.ps1") < operator_confirmed_strict.index(
        "27_check_prelaunch_audit.ps1"
    )
    assert operator_confirmed_strict.index("27_check_prelaunch_audit.ps1") < operator_confirmed_strict.index("try {")
    assert operator_confirmed_strict.index("exit $prelaunchExitCode") < operator_confirmed_strict.index("try {")
    assert operator_confirmed_strict.index("27_check_prelaunch_audit.ps1") < operator_confirmed_strict.index(
        "22_run_live_demo_operator_confirmed.ps1"
    )
    assert operator_confirmed_strict.index("29_check_operator_command_audit.ps1") < operator_confirmed_strict.index(
        "32_check_guarded_retake_readiness.ps1"
    )
    assert operator_confirmed_strict.index("32_check_guarded_retake_readiness.ps1") < operator_confirmed_strict.index(
        "22_run_live_demo_operator_confirmed.ps1"
    )
    assert operator_confirmed_strict.index("22_run_live_demo_operator_confirmed.ps1") < operator_confirmed_strict.index(
        "23_check_goal_progress_strict.ps1"
    )
    assert operator_confirmed_strict.index("23_check_goal_progress_strict.ps1") < operator_confirmed_strict.rindex(
        "25_build_live_status_snapshot.ps1"
    )
    assert operator_confirmed_strict.rindex("25_build_live_status_snapshot.ps1") < operator_confirmed_strict.rindex(
        "26_build_operator_handoff_card.ps1"
    )
    assert "operator_handoff_card.md" in operator_confirmed_strict
    assert "--- Operator handoff summary ---" in operator_confirmed_strict
    assert "Get-Content $handoffMarkdownPath" in operator_confirmed_strict
    assert operator_confirmed_strict.index("26_build_operator_handoff_card.ps1") < operator_confirmed_strict.index(
        "--- Operator handoff summary ---"
    )
    assert "build_realtime_demo_live_status_snapshot" in live_status_snapshot
    assert "--goal-audit" in live_status_snapshot
    assert "--final-run-card" in live_status_snapshot
    assert "--learning-queue" in live_status_snapshot
    assert "--live-overlay" in live_status_snapshot
    assert "--live-frame-log" in live_status_snapshot
    assert "--live-composite" in live_status_snapshot
    assert "build_realtime_demo_operator_handoff_card" in operator_handoff_card
    assert "--live-status-snapshot" in operator_handoff_card
    assert "--final-run-card" in operator_handoff_card
    assert "--launch-summary" in operator_handoff_card
    assert "--output-root" in operator_handoff_card
    assert "build_realtime_demo_prelaunch_audit" in prelaunch_audit
    assert "--live-status-snapshot" in prelaunch_audit
    assert "--operator-handoff-card" in prelaunch_audit
    assert "--launch-summary" in prelaunch_audit
    assert "--strict-exit-code" in prelaunch_audit
    assert "run_realtime_demo_wrapper_contract_probe" in wrapper_contract_probe
    assert "--output-root" in wrapper_contract_probe
    assert "realtime_demo_wrapper_contract_probe_20260616" in wrapper_contract_probe
    assert "exit $LASTEXITCODE" in wrapper_contract_probe
    assert "audit_realtime_demo_operator_commands" in operator_command_audit
    assert "--strict-exit-code" in operator_command_audit
    assert "realtime_demo_operator_command_audit_20260616" in operator_command_audit
    assert "exit $LASTEXITCODE" in operator_command_audit
    assert "clear_realtime_demo_live_artifacts" in live_artifact_cleanup
    assert "--workspace-root" in live_artifact_cleanup
    assert "--live-overlay" in live_artifact_cleanup
    assert "live_camera_overlay.mp4" in live_artifact_cleanup
    assert "--live-frame-log" in live_artifact_cleanup
    assert "live_camera_frames.jsonl" in live_artifact_cleanup
    assert "--live-postcapture-root" in live_artifact_cleanup
    assert "--live-composite-root" in live_artifact_cleanup
    assert "--live-overlay-contract-root" in live_artifact_cleanup
    assert "--live-rock-retake-gate-root" in live_artifact_cleanup
    assert "realtime_demo_live_rock_retake_gate_20260616" in live_artifact_cleanup
    assert "record_realtime_demo_manual_review" in manual_review_decision
    assert "Review status" in manual_review_decision
    assert "--manual-review-decisions" in manual_review_decision
    assert "--archive-index" in manual_review_decision
    assert "15_summarize_run_archives.ps1" in manual_review_decision
    assert "16_select_final_demo_candidate.ps1" in manual_review_decision
    assert "17_build_submission_candidate_packet.ps1" in manual_review_decision
    assert "build_realtime_demo_guarded_retake_readiness" in guarded_retake_readiness
    assert "--strict-exit-code" in guarded_retake_readiness
    assert "realtime_demo_guarded_retake_readiness_20260616" in guarded_retake_readiness
    assert "exit $LASTEXITCODE" in guarded_retake_readiness
    assert "build_realtime_demo_live_rock_retake_gate" in live_rock_retake_gate
    assert "--frame-log" in live_rock_retake_gate
    assert "live_camera_frames.jsonl" in live_rock_retake_gate
    assert "--postcapture-summary" in live_rock_retake_gate
    assert "postcapture_summary.json" in live_rock_retake_gate
    assert "--expected-actual-gesture" in live_rock_retake_gate
    assert "$env:EMBODIED_RPS_EXPECTED_ACTUAL_GESTURE" in live_rock_retake_gate
    assert "--strict-exit-code" in live_rock_retake_gate
    assert "exit $LASTEXITCODE" in live_rock_retake_gate
    assert "run_current_best_realtime_demo" in scissors_collection
    assert "prompt pose collection" in scissors_collection.lower()
    assert "Read-Host 'Collection gesture (rock/paper/scissors)'" in scissors_collection
    assert "@('rock', 'paper', 'scissors')" in scissors_collection
    assert "$selectedGesture = $selectedGesture.Trim().ToLowerInvariant()" in scissors_collection
    assert "Join-Path $collectionRootBase $selectedGesture" in scissors_collection
    assert "$baseConfigLines = $baseConfigLines -replace '^response_prompt:.*$', 'response_prompt: scissors'" in scissors_collection
    assert "('response_prompt: ' + $selectedGesture)" not in scissors_collection
    assert "$overlayVideo = Join-Path $collectionRoot ($selectedGesture + '_pose_collection_overlay.mp4')" in scissors_collection
    assert "--expected-actual-gesture $selectedGesture" in scissors_collection
    assert "--collection-label $selectedGesture" in scissors_collection
    assert "rock/paper/scissors prompt cycle" in scissors_collection
    assert "PROMPT SCISSORS remains the bounded response window for every collection label" in scissors_collection
    assert "Perform the selected gesture during PROMPT SCISSORS" in scissors_collection
    assert "For rock collection, keep rock/wait through PROMPT SCISSORS" in scissors_collection
    assert "For paper collection, allow early ambiguity but resolve to paper inside PROMPT SCISSORS" in scissors_collection
    assert "For scissors collection, move to varied scissors poses during PROMPT SCISSORS" in scissors_collection
    assert "--expected-actual-gesture" in scissors_collection
    assert "--skeleton-npz" in scissors_collection
    assert "--collection-mode" in scissors_collection
    assert "--max-frames" in scissors_collection
    assert "3600" in scissors_collection
    assert "summarize_scissors_pose_collection" in scissors_collection
    assert "34_run_scissors_pose_collection.ps1" in readme
    assert live_pipeline.index("00_demo_preflight.ps1") < live_pipeline.index("02_live_camera_demo.ps1")
    assert live_pipeline.index("30_clear_stale_live_artifacts.ps1") < live_pipeline.index("00_demo_preflight.ps1")
    assert live_pipeline.index("02_live_camera_demo.ps1") < live_pipeline.index("06_verify_live_capture.ps1")
    assert live_pipeline.index("06_verify_live_capture.ps1") < live_pipeline.index("33_check_live_rock_retake_gate.ps1")
    assert live_pipeline.index("33_check_live_rock_retake_gate.ps1") < live_pipeline.index("05_create_live_schunk_composite.ps1")
    assert live_pipeline.index("06_verify_live_capture.ps1") < live_pipeline.index("05_create_live_schunk_composite.ps1")
    assert live_pipeline.index("05_create_live_schunk_composite.ps1") < live_pipeline.index("04_open_demo_artifacts.ps1")
    assert live_pipeline.index("08_triage_live_capture.ps1") > live_pipeline.index("04_open_demo_artifacts.ps1")
    assert live_pipeline.index("11_audit_live_overlay_contract.ps1") > live_pipeline.index("08_triage_live_capture.ps1")
    assert live_pipeline.index("09_build_demo_evidence_bundle.ps1") > live_pipeline.index("11_audit_live_overlay_contract.ps1")
    assert live_pipeline.index("10_build_demo_review_packet.ps1") > live_pipeline.index("09_build_demo_evidence_bundle.ps1")
    assert live_pipeline.index("12_build_demo_acceptance_report.ps1") > live_pipeline.index("10_build_demo_review_packet.ps1")
    assert live_pipeline.index("13_build_operator_outcome_report.ps1") > live_pipeline.index("12_build_demo_acceptance_report.ps1")
    assert live_pipeline.index("14_archive_live_demo_run.ps1") > live_pipeline.index("13_build_operator_outcome_report.ps1")
    assert live_pipeline.index("15_summarize_run_archives.ps1") > live_pipeline.index("14_archive_live_demo_run.ps1")
    assert live_pipeline.index("16_select_final_demo_candidate.ps1") > live_pipeline.index("15_summarize_run_archives.ps1")
    assert live_pipeline.index("17_build_submission_candidate_packet.ps1") > live_pipeline.index("16_select_final_demo_candidate.ps1")
    assert live_pipeline.index("18_build_live_run_checklist.ps1") < live_pipeline.index("00_demo_preflight.ps1")
    assert live_pipeline.index("19_build_final_run_card.ps1") > live_pipeline.index("17_build_submission_candidate_packet.ps1")
    assert live_pipeline.index("20_build_learning_queue.ps1") > live_pipeline.index("19_build_final_run_card.ps1")
    assert live_pipeline.index("21_build_goal_progress_audit.ps1") > live_pipeline.index("20_build_learning_queue.ps1")
    assert "create_realtime_schunk_demo_composite" in live_composite
    assert "--overlay-video" in live_composite
    assert "live_camera_overlay.mp4" in live_composite
    assert "--response-preview-image" in live_composite
    assert response_preview_image.as_posix() in live_composite
    assert "--output-root" in live_composite
    assert live_composite_output_root.as_posix() in live_composite
    assert "Run 02_live_camera_demo.ps1 first" in live_composite
    assert "response-prompt" in readme
    assert "reset-on-prompt-change" in readme
    assert "05_create_live_schunk_composite.ps1" in readme
    assert "06_verify_live_capture.ps1" in readme
    assert "07_run_live_demo_pipeline.ps1" in readme
    assert "08_triage_live_capture.ps1" in readme
    assert "09_build_demo_evidence_bundle.ps1" in readme
    assert "10_build_demo_review_packet.ps1" in readme
    assert "11_audit_live_overlay_contract.ps1" in readme
    assert "12_build_demo_acceptance_report.ps1" in readme
    assert "13_build_operator_outcome_report.ps1" in readme
    assert "14_archive_live_demo_run.ps1" in readme
    assert "15_summarize_run_archives.ps1" in readme
    assert "16_select_final_demo_candidate.ps1" in readme
    assert "17_build_submission_candidate_packet.ps1" in readme
    assert "18_build_live_run_checklist.ps1" in readme
    assert "19_build_final_run_card.ps1" in readme
    assert "20_build_learning_queue.ps1" in readme
    assert "21_build_goal_progress_audit.ps1" in readme
    assert "22_run_live_demo_operator_confirmed.ps1" in readme
    assert "23_check_goal_progress_strict.ps1" in readme
    assert "`30` postprocess repair needed" in readme
    assert "24_run_live_demo_operator_confirmed_strict.ps1" in readme
    assert "25_build_live_status_snapshot.ps1" in readme
    assert "26_build_operator_handoff_card.ps1" in readme
    assert "27_check_prelaunch_audit.ps1" in readme
    assert "28_check_wrapper_contract.ps1" in readme
    assert "29_check_operator_command_audit.ps1" in readme
    assert "30_clear_stale_live_artifacts.ps1" in readme
    assert "live-rock-retake gate artifacts" in readme
    assert "32_check_guarded_retake_readiness.ps1" in readme
    assert "33_check_live_rock_retake_gate.ps1" in readme
    assert "rock/paper/scissors prompt cycle" in readme
    assert "asks which actual gesture label to collect" in readme
    assert "keeps `PROMPT SCISSORS` as the fixed response window" in readme
    assert "Collect at least 10 prompt cycles per selected gesture" in readme
    assert "--strict-exit-code" in readme
    assert "00_demo_preflight.ps1" in readme
    assert (output_root / "demo_launch_summary.json").exists()
    summary_text = (output_root / "demo_launch_summary.json").read_text(encoding="utf-8")
    assert "텀프영상" not in summary_text
    assert "\\ud" in summary_text
    summary_json = json.loads(summary_text)
    assert summary_json["response_preview_image"] == response_preview_image.as_posix()
    assert summary_json["live_composite_output_root"] == live_composite_output_root.as_posix()
    assert summary_json["manual_review_decisions"] == (
        "artifacts/realtime_demo_manual_review_20260616/manual_review_decisions.json"
    )
    assert summary_json["project_root_absolute"] == project_root.resolve().as_posix()
    assert "run_live_demo_pipeline" in summary_json["scripts"]
    assert "triage_live_capture" in summary_json["scripts"]
    assert "build_demo_evidence_bundle" in summary_json["scripts"]
    assert "build_demo_review_packet" in summary_json["scripts"]
    assert "audit_live_overlay_contract" in summary_json["scripts"]
    assert "build_demo_acceptance_report" in summary_json["scripts"]
    assert "build_operator_outcome_report" in summary_json["scripts"]
    assert "archive_live_demo_run" in summary_json["scripts"]
    assert "summarize_run_archives" in summary_json["scripts"]
    assert "select_final_demo_candidate" in summary_json["scripts"]
    assert "build_submission_candidate_packet" in summary_json["scripts"]
    assert "build_live_run_checklist" in summary_json["scripts"]
    assert "build_final_run_card" in summary_json["scripts"]
    assert "build_learning_queue" in summary_json["scripts"]
    assert "build_goal_progress_audit" in summary_json["scripts"]
    assert "run_live_demo_operator_confirmed" in summary_json["scripts"]
    assert "check_goal_progress_strict" in summary_json["scripts"]
    assert "run_live_demo_operator_confirmed_strict" in summary_json["scripts"]
    assert "build_live_status_snapshot" in summary_json["scripts"]
    assert "build_operator_handoff_card" in summary_json["scripts"]
    assert "check_prelaunch_audit" in summary_json["scripts"]
    assert "check_wrapper_contract" in summary_json["scripts"]
    assert "check_operator_command_audit" in summary_json["scripts"]
    assert "clear_stale_live_artifacts" in summary_json["scripts"]
    assert "record_manual_review_decision" in summary_json["scripts"]
    assert "check_guarded_retake_readiness" in summary_json["scripts"]
    assert "check_live_rock_retake_gate" in summary_json["scripts"]
    assert "run_scissors_pose_collection" in summary_json["scripts"]
    assert summary_json["scripts_absolute"]["run_live_demo_operator_confirmed_strict"] == (
        output_root / "24_run_live_demo_operator_confirmed_strict.ps1"
    ).resolve().as_posix()
    assert (output_root / "00_demo_preflight.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "01_dry_run_video_rehearsal.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "05_create_live_schunk_composite.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "07_run_live_demo_pipeline.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "08_triage_live_capture.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "09_build_demo_evidence_bundle.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "10_build_demo_review_packet.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "11_audit_live_overlay_contract.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "12_build_demo_acceptance_report.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "13_build_operator_outcome_report.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "14_archive_live_demo_run.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "15_summarize_run_archives.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "16_select_final_demo_candidate.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "17_build_submission_candidate_packet.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "18_build_live_run_checklist.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "19_build_final_run_card.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "20_build_learning_queue.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "21_build_goal_progress_audit.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "22_run_live_demo_operator_confirmed.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "23_check_goal_progress_strict.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "24_run_live_demo_operator_confirmed_strict.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "25_build_live_status_snapshot.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "26_build_operator_handoff_card.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "27_check_prelaunch_audit.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "28_check_wrapper_contract.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "29_check_operator_command_audit.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "30_clear_stale_live_artifacts.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "31_record_manual_review_decision.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output_root / "34_run_scissors_pose_collection.ps1").read_bytes().startswith(b"\xef\xbb\xbf")


def test_realtime_demo_launch_scripts_resolve_relative_project_root_for_any_cwd_execution(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    output_root = tmp_path / "launch"

    summary = write_realtime_demo_launch_scripts(
        RealtimeDemoLaunchScriptsConfig(
            output_root=output_root,
            project_root=Path("."),
            python_executable=tmp_path / "python.exe",
        )
    )

    expected_project_root = tmp_path.resolve().as_posix()
    preflight = (output_root / "00_demo_preflight.ps1").read_text(encoding="utf-8")
    strict_wrapper = (output_root / "24_run_live_demo_operator_confirmed_strict.ps1").read_text(encoding="utf-8")
    assert summary["project_root"] == "."
    assert summary["project_root_absolute"] == expected_project_root
    assert f"Set-Location '{expected_project_root}'" in preflight
    assert f"Set-Location '{expected_project_root}'" in strict_wrapper
    assert summary["scripts_absolute"]["run_live_demo_operator_confirmed_strict"] == (
        output_root / "24_run_live_demo_operator_confirmed_strict.ps1"
    ).resolve().as_posix()


def test_realtime_demo_launch_scripts_cli(tmp_path: Path) -> None:
    output_root = tmp_path / "launch"

    exit_code = main(
        [
            "--output-root",
            str(output_root),
            "--project-root",
            str(tmp_path / "project"),
            "--sample-video",
            str(tmp_path / "sample.mp4"),
            "--camera",
            "2",
            "--camera-max-frames",
            "120",
            "--response-preview-image",
            str(tmp_path / "preview.png"),
            "--live-composite-output-root",
            str(tmp_path / "composite"),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_root / "demo_launch_summary.json").read_text(encoding="utf-8"))
    assert summary["camera_index"] == 2
    assert summary["camera_max_frames"] == 120
    assert summary["response_preview_image"] == (tmp_path / "preview.png").as_posix()
    assert summary["live_composite_output_root"] == (tmp_path / "composite").as_posix()
