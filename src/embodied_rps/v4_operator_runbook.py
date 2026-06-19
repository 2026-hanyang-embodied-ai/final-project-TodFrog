"""Operator runbook writer for the v4 skeleton prediction pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.v4_readiness_dashboard import V4ReadinessDashboardConfig, build_v4_readiness_dashboard

KOREAN_RPS_DATASET_NAME = "\uD140\uD504\uC601\uC0C1"
DEFAULT_LOCAL_DATA_ROOT = Path("D:/dataset") / KOREAN_RPS_DATASET_NAME


@dataclass(frozen=True)
class V4OperatorRunbookConfig:
    """Configuration for the v4 operator runbook."""

    output_root: Path
    staging_root: Path = DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging"
    calibration_root: Path = DEFAULT_LOCAL_DATA_ROOT / "v4_calibration"
    heldout_root: Path = DEFAULT_LOCAL_DATA_ROOT / "test"
    original20_root: Path = DEFAULT_LOCAL_DATA_ROOT
    expected_per_label: int = 20
    end_to_end_summary_path: Path = Path("artifacts/real_skeleton_v4_end_to_end_20260612/end_to_end_summary.json")
    dataset_root: Path = Path("artifacts/real_guided_three_class_wait_expanded_v4_20260611")
    training_config_path: Path = Path("configs/real_skeleton_three_class_wait_prediction_v4.yaml")
    profile_json_path: Path = Path("results/model_profiles/real_skeleton_three_class_wait_v4.json")
    readiness_output_root: Path = Path("artifacts/real_skeleton_v4_readiness_dashboard_20260612")
    staging_audit_output_root: Path = Path("artifacts/real_skeleton_v4_recording_staging_audit_20260612")
    recording_launch_output_root: Path = Path("artifacts/real_skeleton_v4_recording_launch_20260612")
    recording_preflight_output_root: Path = Path("artifacts/real_skeleton_v4_recording_preflight_20260612")
    guided_recording_flow_output_root: Path = Path("artifacts/real_skeleton_v4_guided_recording_flow_20260612")
    guided_recording_output_root: Path = Path("artifacts/real_skeleton_v4_guided_recording_session_20260612")
    recording_postcheck_output_root: Path = Path("artifacts/real_skeleton_v4_recording_postcheck_20260612")
    end_to_end_output_root: Path = Path("artifacts/real_skeleton_v4_end_to_end_20260612")
    training_gate_output_root: Path = Path("artifacts/real_skeleton_v4_training_gate_20260612")
    original20_validation_root: Path = Path("artifacts/real_mp4_prediction_validation_original20_v4_20260612")
    heldout15_validation_root: Path = Path("artifacts/real_mp4_prediction_validation_new15_v4_20260612")
    event_manifest_path: Path = Path("artifacts/real_skeleton_schunk_events_v4_20260612/events.jsonl")


def write_v4_operator_runbook(config: V4OperatorRunbookConfig) -> dict[str, object]:
    """Write the v4 operator runbook and return its JSON payload."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    dashboard = build_v4_readiness_dashboard(
        V4ReadinessDashboardConfig(
            calibration_root=config.calibration_root,
            heldout_root=config.heldout_root,
            expected_per_label=config.expected_per_label,
            output_root=config.output_root / "readiness_snapshot",
            end_to_end_summary_path=config.end_to_end_summary_path,
        )
    )
    phases = _phases(config)
    runbook = {
        "status": dashboard.get("status"),
        "current_gate": dashboard.get("current_gate"),
        "blocking_stage": dashboard.get("blocking_stage"),
        "next_action": dashboard.get("next_action"),
        "calibration_root": config.calibration_root.as_posix(),
        "heldout_root": config.heldout_root.as_posix(),
        "original20_root": config.original20_root.as_posix(),
        "expected_per_label": int(config.expected_per_label),
        "video_count": dashboard.get("video_count"),
        "label_counts": dashboard.get("label_counts"),
        "remaining_counts": dashboard.get("remaining_counts"),
        "operator_shortcuts": _operator_shortcuts(config),
        "hard_stops": _hard_stops(),
        "acceptance_gate": _acceptance_gate(),
        "phases": phases,
        "runbook_markdown": (config.output_root / "runbook.md").as_posix(),
        "runbook_json": (config.output_root / "runbook.json").as_posix(),
    }
    (config.output_root / "runbook.json").write_text(json.dumps(runbook, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.output_root / "runbook.md").write_text(_runbook_markdown(runbook), encoding="utf-8")
    return runbook


def _phases(config: V4OperatorRunbookConfig) -> list[dict[str, object]]:
    safe_end_to_end = _end_to_end_command(config)
    return [
        {
            "order": 1,
            "name": "prepare_v4_recording_staging_folders",
            "when": "Now, before recording or copying raw MP4s.",
            "goal": "Create the staging drop folders for raw non-held-out v4 recordings.",
            "command": _join_command(
                [
                    "python",
                    "-m",
                    "embodied_rps.tools.prepare_v4_recording_staging_folders",
                    "--staging-root",
                    config.staging_root.as_posix(),
                    "--calibration-root",
                    config.calibration_root.as_posix(),
                    "--heldout-root",
                    config.heldout_root.as_posix(),
                    "--expected-per-label",
                    str(config.expected_per_label),
                ]
            ),
            "expected_output": config.staging_root.as_posix(),
        },
        {
            "order": 2,
            "name": "preflight_v4_recording_session",
            "when": "Before running a live camera recording session.",
            "goal": "Check OpenCV availability, optional camera access, roots, slot manifest, and planned recording outputs.",
            "command": _join_command(
                [
                    "python",
                    "-m",
                    "embodied_rps.tools.preflight_v4_recording_session",
                    "--staging-root",
                    config.staging_root.as_posix(),
                    "--calibration-root",
                    config.calibration_root.as_posix(),
                    "--heldout-root",
                    config.heldout_root.as_posix(),
                    "--output-root",
                    config.recording_preflight_output_root.as_posix(),
                    "--slot-manifest",
                    (config.calibration_root / "recording_slot_manifest.json").as_posix(),
                    "--count-per-label",
                    "1",
                    "--pre-roll-s",
                    "1.5",
                    "--duration-s",
                    "3",
                    "--fps",
                    "30",
                ]
            ),
            "expected_output": config.recording_preflight_output_root.as_posix(),
        },
        {
            "order": 3,
            "name": "run_v4_guided_recording_flow",
            "when": "Now, before MediaPipe review or v4 training. This is the preferred one-command operator path.",
            "goal": (
                "Run recording preflight, guide a balanced non-held-out staging session one cue at a time, "
                "and immediately postcheck staged MP4s. Add --execute when ready to record, and repeat until "
                f"each label has at least {config.expected_per_label} clips."
            ),
            "command": _join_command(
                [
                    "python",
                    "-m",
                    "embodied_rps.tools.run_v4_guided_recording_flow",
                    "--staging-root",
                    config.staging_root.as_posix(),
                    "--calibration-root",
                    config.calibration_root.as_posix(),
                    "--heldout-root",
                    config.heldout_root.as_posix(),
                    "--output-root",
                    config.guided_recording_flow_output_root.as_posix(),
                    "--count-per-label",
                    "1",
                    "--pre-roll-s",
                    "1.5",
                    "--duration-s",
                    "3",
                    "--fps",
                    "30",
                    "--slot-manifest",
                    (config.calibration_root / "recording_slot_manifest.json").as_posix(),
                    "--expected-per-label",
                    str(config.expected_per_label),
                    "--check-camera",
                ]
            ),
            "expected_output": config.guided_recording_flow_output_root.as_posix(),
        },
        {
            "order": 4,
            "name": "postcheck_v4_recording_session",
            "when": "Immediately after recording staged MP4s, before copy or calibration-slot assignment.",
            "goal": "Confirm the staged MP4s exist, pass video probe checks, and are ready for assignment review.",
            "command": _join_command(
                [
                    "python",
                    "-m",
                    "embodied_rps.tools.check_v4_recording_postcheck",
                    "--staging-root",
                    config.staging_root.as_posix(),
                    "--calibration-root",
                    config.calibration_root.as_posix(),
                    "--heldout-root",
                    config.heldout_root.as_posix(),
                    "--output-root",
                    config.recording_postcheck_output_root.as_posix(),
                    "--slot-manifest",
                    (config.calibration_root / "recording_slot_manifest.json").as_posix(),
                    "--expected-per-label",
                    str(config.expected_per_label),
                    "--expected-new-per-label",
                    "1",
                    "--pre-roll-s",
                    "1.5",
                    "--duration-s",
                    "3",
                    "--fps",
                    "30",
                ]
            ),
            "expected_output": config.recording_postcheck_output_root.as_posix(),
        },
        {
            "order": 5,
            "name": "monitor_v4_recording_ingest",
            "when": "Optional while recording or copying MP4s into staging.",
            "goal": "Refresh recording ingest status whenever staged MP4 files change without running copy.",
            "command": _join_command(
                [
                    "python",
                    "-m",
                    "embodied_rps.tools.monitor_v4_recording_ingest",
                    "--source-root",
                    config.staging_root.as_posix(),
                    "--calibration-root",
                    config.calibration_root.as_posix(),
                    "--heldout-root",
                    config.heldout_root.as_posix(),
                    "--output-root",
                    "artifacts/real_skeleton_v4_recording_ingest_monitor_20260612",
                    "--ingest-output-root",
                    "artifacts/real_skeleton_v4_recording_ingest_20260612",
                    "--end-to-end-summary",
                    config.end_to_end_summary_path.as_posix(),
                    "--expected-per-label",
                    str(config.expected_per_label),
                    "--iterations",
                    "1",
                    "--poll-interval-s",
                    "0",
                ]
            ),
            "expected_output": "artifacts/real_skeleton_v4_recording_ingest_monitor_20260612",
        },
        {
            "order": 6,
            "name": "audit_v4_recording_staging",
            "when": "After adding raw MP4s into staging and before slot assignment.",
            "goal": "Check staging label counts, invalid label paths, empty MP4s, duplicate filenames, and held-out leakage.",
            "command": _join_command(
                [
                    "python",
                    "-m",
                    "embodied_rps.tools.audit_v4_recording_staging",
                    "--staging-root",
                    config.staging_root.as_posix(),
                    "--calibration-root",
                    config.calibration_root.as_posix(),
                    "--heldout-root",
                    config.heldout_root.as_posix(),
                    "--expected-per-label",
                    str(config.expected_per_label),
                    "--output-root",
                    config.staging_audit_output_root.as_posix(),
                ]
            ),
            "expected_output": config.staging_audit_output_root.as_posix(),
        },
        {
            "order": 7,
            "name": "plan_staging_to_slot_assignment",
            "when": "After recording clips into a separate staging folder, before copying them into calibration slots.",
            "goal": "Dry-run the mapping from staging MP4s to planned v4 slot filenames without modifying source files.",
            "command": _join_command(
                [
                    "python",
                    "-m",
                    "embodied_rps.tools.plan_v4_recording_slot_assignment",
                    "--source-root",
                    config.staging_root.as_posix(),
                    "--calibration-root",
                    config.calibration_root.as_posix(),
                    "--output-root",
                    "artifacts/real_skeleton_v4_recording_slot_assignment_20260612",
                ]
            ),
            "expected_output": "artifacts/real_skeleton_v4_recording_slot_assignment_20260612",
        },
        {
            "order": 8,
            "name": "run_recording_ingest_status",
            "when": "After reviewing the dry-run assignment, or whenever staging/calibration status should be refreshed together.",
            "goal": "Run assignment, slot coverage audit, and readiness dashboard in one safe dry-run command.",
            "command": _join_command(
                [
                    "python",
                    "-m",
                    "embodied_rps.tools.run_v4_recording_ingest",
                    "--source-root",
                    config.staging_root.as_posix(),
                    "--calibration-root",
                    config.calibration_root.as_posix(),
                    "--heldout-root",
                    config.heldout_root.as_posix(),
                    "--output-root",
                    "artifacts/real_skeleton_v4_recording_ingest_20260612",
                    "--end-to-end-summary",
                    config.end_to_end_summary_path.as_posix(),
                ]
            ),
            "expected_output": "artifacts/real_skeleton_v4_recording_ingest_20260612",
        },
        {
            "order": 9,
            "name": "refresh_readiness_dashboard",
            "when": "After adding or replacing calibration MP4s.",
            "goal": "Confirm recording counts and the current blocker.",
            "command": _join_command(
                [
                    "python",
                    "-m",
                    "embodied_rps.tools.report_v4_readiness_dashboard",
                    "--calibration-root",
                    config.calibration_root.as_posix(),
                    "--heldout-root",
                    config.heldout_root.as_posix(),
                    "--expected-per-label",
                    str(config.expected_per_label),
                    "--output-root",
                    config.readiness_output_root.as_posix(),
                    "--end-to-end-summary",
                    config.end_to_end_summary_path.as_posix(),
                ]
            ),
            "expected_output": config.readiness_output_root.as_posix(),
        },
        {
            "order": 10,
            "name": "audit_recording_slot_coverage",
            "when": "After adding or replacing calibration MP4s and before MP4 preflight.",
            "goal": "Confirm every planned slot file exists and no extra MP4s are present.",
            "command": _join_command(
                [
                    "python",
                    "-m",
                    "embodied_rps.tools.audit_v4_recording_slots",
                    "--calibration-root",
                    config.calibration_root.as_posix(),
                    "--output-root",
                    "artifacts/real_skeleton_v4_recording_slot_audit_20260612",
                ]
            ),
            "expected_output": "artifacts/real_skeleton_v4_recording_slot_audit_20260612",
        },
        {
            "order": 11,
            "name": "run_safe_end_to_end_status",
            "when": "After the dashboard has enough recordings.",
            "goal": "Run metadata, preflight, skeleton-review-plan, post-review dry-run, and training-gate status checks.",
            "command": safe_end_to_end,
            "expected_output": config.end_to_end_output_root.as_posix(),
        },
        {
            "order": 12,
            "name": "execute_skeleton_review",
            "when": "Only after MP4 preflight and intake report pass.",
            "goal": "Generate MediaPipe skeleton-only and side-by-side review videos for visual approval.",
            "command": safe_end_to_end + " --execute-skeleton-review",
            "expected_output": "artifacts/real_hand_skeleton_review_v4_calibration_20260611",
        },
        {
            "order": 13,
            "name": "generate_v4_dataset_after_visual_approval",
            "when": "Only after visual skeleton approval.",
            "goal": "Build the approved v4 seed package and generate the v4 skeleton-only training dataset.",
            "command": safe_end_to_end + " --execute-dataset-generation --overwrite-dataset",
            "expected_output": config.dataset_root.as_posix(),
        },
        {
            "order": 14,
            "name": "smoke_train_v4_gru",
            "when": "After the v4 dataset exists and loads.",
            "goal": "Run a one-epoch sanity check before full training.",
            "command": _join_command(
                [
                    "python",
                    "-m",
                    "embodied_rps.tools.train_real_skeleton_predictor",
                    "--config",
                    config.training_config_path.as_posix(),
                    "--model",
                    "gru",
                    "--smoke",
                    "--max-runs",
                    "1",
                ]
            ),
            "expected_output": "results/real_skeleton_three_class_wait_prediction_v4",
        },
        {
            "order": 15,
            "name": "full_train_v4_models",
            "when": "After smoke training passes.",
            "goal": "Train the configured GRU first and TCN/other ablations as configured.",
            "command": _join_command(
                [
                    "python",
                    "-m",
                    "embodied_rps.tools.train_real_skeleton_predictor",
                    "--config",
                    config.training_config_path.as_posix(),
                    "--model",
                    "all",
                ]
            ),
            "expected_output": config.profile_json_path.as_posix(),
        },
        {
            "order": 16,
            "name": "run_training_gate_summary",
            "when": "After a v4 profile is exported.",
            "goal": "Summarize dataset, profile, and strict-validation readiness.",
            "command": _join_command(
                [
                    "python",
                    "-m",
                    "embodied_rps.tools.run_v4_training_gate",
                    "--dataset-root",
                    config.dataset_root.as_posix(),
                    "--training-config",
                    config.training_config_path.as_posix(),
                    "--output-root",
                    config.training_gate_output_root.as_posix(),
                    "--original20-root",
                    config.original20_root.as_posix(),
                    "--heldout15-root",
                    config.heldout_root.as_posix(),
                    "--profile-json",
                    config.profile_json_path.as_posix(),
                    "--event-manifest",
                    config.event_manifest_path.as_posix(),
                ]
            ),
            "expected_output": config.training_gate_output_root.as_posix(),
        },
        {
            "order": 17,
            "name": "validate_original20_gate",
            "when": "After the v4 profile is exported.",
            "goal": "Confirm the original 20 transition MP4s still pass.",
            "command": _validation_command(
                profile=config.profile_json_path,
                input_root=config.original20_root,
                output_root=config.original20_validation_root,
                event_output=config.event_manifest_path,
                expected_count=20,
                label_mode="transition",
            ),
            "expected_output": config.original20_validation_root.as_posix(),
        },
        {
            "order": 18,
            "name": "validate_heldout15_gate",
            "when": "After the original 20 gate passes.",
            "goal": "Confirm paper/scissors predictions and rock wait_counter_paper behavior on the held-out 15 MP4s.",
            "command": _validation_command(
                profile=config.profile_json_path,
                input_root=config.heldout_root,
                output_root=config.heldout15_validation_root,
                event_output=config.event_manifest_path,
                expected_count=15,
                label_mode="final-label",
            ),
            "expected_output": config.heldout15_validation_root.as_posix(),
        },
    ]


def _operator_shortcuts(config: V4OperatorRunbookConfig) -> list[dict[str, object]]:
    return [
        {
            "name": "write_v4_recording_launch_scripts",
            "goal": "Write local PowerShell scripts for dry-run camera check, live guided recording, and status refresh.",
            "command": _join_command(
                [
                    "python",
                    "-m",
                    "embodied_rps.tools.write_v4_recording_launch_scripts",
                    "--output-root",
                    config.recording_launch_output_root.as_posix(),
                    "--expected-per-label",
                    str(config.expected_per_label),
                ]
            ),
            "expected_output": config.recording_launch_output_root.as_posix(),
        }
    ]


def _end_to_end_command(config: V4OperatorRunbookConfig) -> str:
    return _join_command(
        [
            "python",
            "-m",
            "embodied_rps.tools.run_v4_end_to_end",
            "--calibration-input-root",
            config.calibration_root.as_posix(),
            "--heldout-root",
            config.heldout_root.as_posix(),
            "--original20-root",
            config.original20_root.as_posix(),
            "--expected-min-per-label",
            str(config.expected_per_label),
            "--base-dataset-root",
            "artifacts/real_guided_large_sharded_20260610",
            "--training-config",
            config.training_config_path.as_posix(),
            "--output-root",
            config.end_to_end_output_root.as_posix(),
        ]
    )


def _validation_command(
    *,
    profile: Path,
    input_root: Path,
    output_root: Path,
    event_output: Path,
    expected_count: int,
    label_mode: str,
) -> str:
    return _join_command(
        [
            "python",
            "-m",
            "embodied_rps.tools.evaluate_real_skeleton_video_predictions",
            "--profile",
            profile.as_posix(),
            "--input-root",
            input_root.as_posix(),
            "--output-root",
            output_root.as_posix(),
            "--event-output",
            event_output.as_posix(),
            "--expected-count",
            str(expected_count),
            "--label-mode",
            label_mode,
        ]
    )


def _hard_stops() -> list[str]:
    return [
        "Do not use the held-out test folder as training data.",
        "Do not run dataset generation before visual skeleton approval.",
        "Do not export a replacement v4 profile unless strict validation improves.",
        "Do not run SCHUNK or Isaac rendering until the original 20 and held-out 15 strict gates both pass.",
    ]


def _acceptance_gate() -> dict[str, object]:
    return {
        "original20": {"required_passed": 20, "required_total": 20},
        "heldout15": {
            "paper_scissors_required_correct": 10,
            "paper_scissors_required_total": 10,
            "rock_required_wait_counter_paper": 5,
            "rock_required_total": 5,
            "no_rock_binary_false_trigger": True,
        },
        "strict_decision": {
            "confidence_threshold": 0.85,
            "margin_threshold": 0.20,
            "confirmation_count": 3,
            "max_decision_progress": 0.50,
        },
    }


def _runbook_markdown(runbook: Mapping[str, object]) -> str:
    lines = [
        "# V4 Operator Runbook",
        "",
        "## Current Status",
        "",
        f"- Status: `{runbook.get('status')}`",
        f"- Current gate: `{runbook.get('current_gate')}`",
        f"- Blocking stage: `{runbook.get('blocking_stage')}`",
        f"- Next action: `{runbook.get('next_action')}`",
        f"- Calibration root: `{runbook.get('calibration_root')}`",
        f"- Held-out root: `{runbook.get('heldout_root')}`",
        f"- Original 20 root: `{runbook.get('original20_root')}`",
        "",
        "## Recording Counts",
        "",
        "| Label | Current | Remaining |",
        "|---|---:|---:|",
    ]
    label_counts = runbook.get("label_counts")
    remaining_counts = runbook.get("remaining_counts")
    if isinstance(label_counts, Mapping) and isinstance(remaining_counts, Mapping):
        for label in ("rock", "paper", "scissors"):
            lines.append(f"| `{label}` | `{label_counts.get(label, 0)}` | `{remaining_counts.get(label, 0)}` |")
    lines.extend(["", "## Hard Stops", ""])
    hard_stops = runbook.get("hard_stops")
    if isinstance(hard_stops, Sequence) and not isinstance(hard_stops, (str, bytes)):
        for stop in hard_stops:
            lines.append(f"- {stop}")
    lines.extend(["", "## Operator Shortcuts", ""])
    shortcuts = runbook.get("operator_shortcuts")
    if isinstance(shortcuts, Sequence) and not isinstance(shortcuts, (str, bytes)):
        for shortcut in shortcuts:
            if not isinstance(shortcut, Mapping):
                continue
            lines.extend(
                [
                    f"### {shortcut.get('name')}",
                    "",
                    f"- Goal: {shortcut.get('goal')}",
                    f"- Expected output: `{shortcut.get('expected_output')}`",
                    "",
                    "```powershell",
                    str(shortcut.get("command", "")),
                    "```",
                    "",
                ]
            )
    lines.extend(["", "## Ordered Phases", ""])
    phases = runbook.get("phases")
    if isinstance(phases, Sequence) and not isinstance(phases, (str, bytes)):
        for phase in phases:
            if not isinstance(phase, Mapping):
                continue
            lines.extend(
                [
                    f"### {phase.get('order')}. {phase.get('name')}",
                    "",
                    f"- When: {phase.get('when')}",
                    f"- Goal: {phase.get('goal')}",
                    f"- Expected output: `{phase.get('expected_output')}`",
                ]
            )
            command = phase.get("command")
            if isinstance(command, str) and command:
                lines.extend(["", "```powershell", command, "```"])
            lines.append("")
    lines.extend(
        [
            "## Acceptance Gate",
            "",
            "- Original 20 MP4s must pass `20 / 20`.",
            "- Held-out 15 MP4s require `paper/scissors 10 / 10`, `rock wait_counter_paper 5 / 5`, and no rock false triggers.",
            "- Strict decision thresholds remain confidence `>= 0.85`, margin `>= 0.20`, three-frame confirmation, and progress `<= 0.50`.",
            "- Only after both gates pass should SCHUNK response-event integration resume; Isaac rendering still remains a separate later action.",
            "",
        ]
    )
    return "\n".join(lines)


def _join_command(parts: Sequence[str]) -> str:
    return " ".join(_quote_part(str(part)) for part in parts)


def _quote_part(part: str) -> str:
    if not part or any(char.isspace() for char in part) or any(ord(char) > 127 for char in part) or "\\" in part or ":" in part:
        escaped = part.replace("'", "''")
        return f"'{escaped}'"
    return part


__all__ = [
    "DEFAULT_LOCAL_DATA_ROOT",
    "V4OperatorRunbookConfig",
    "write_v4_operator_runbook",
]
