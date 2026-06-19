"""PowerShell launch scripts for the local v4 recording workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT

PS_SCRIPT_ENCODING: Final[str] = "utf-8-sig"


@dataclass(frozen=True)
class V4RecordingLaunchScriptsConfig:
    """Configuration for local operator launch scripts."""

    output_root: Path = Path("artifacts/real_skeleton_v4_recording_launch_20260612")
    project_root: Path | None = None
    staging_root: Path = DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging"
    calibration_root: Path = DEFAULT_LOCAL_DATA_ROOT / "v4_calibration"
    heldout_root: Path = DEFAULT_LOCAL_DATA_ROOT / "test"
    flow_output_root: Path = Path("artifacts/real_skeleton_v4_guided_recording_flow_20260612_camera_check")
    ingest_output_root: Path = Path("artifacts/real_skeleton_v4_recording_ingest_20260612")
    readiness_output_root: Path = Path("artifacts/real_skeleton_v4_readiness_dashboard_20260612")
    end_to_end_output_root: Path = Path("artifacts/real_skeleton_v4_end_to_end_20260612")
    end_to_end_summary_path: Path = Path("artifacts/real_skeleton_v4_end_to_end_20260612/end_to_end_summary.json")
    skeleton_review_plan_path: Path = Path("artifacts/real_skeleton_v4_calibration_review_plan_20260611/skeleton_review_plan.json")
    skeleton_review_execution_output_root: Path = Path("artifacts/real_skeleton_v4_review_execution_20260612")
    count_per_label: int = 1
    camera_index: int = 0
    pre_roll_s: float = 1.5
    duration_s: float = 3.0
    fps: float = 30.0
    expected_per_label: int = 20


def write_v4_recording_launch_scripts(config: V4RecordingLaunchScriptsConfig) -> dict[str, object]:
    """Write local PowerShell scripts for the v4 recording operator."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    project_root = config.project_root.resolve() if config.project_root is not None else None
    slot_manifest = config.calibration_root / "recording_slot_manifest.json"
    scripts = {
        "dry_run_camera_check": config.output_root / "01_dry_run_camera_check.ps1",
        "live_guided_recording": config.output_root / "02_live_guided_recording.ps1",
        "refresh_recording_status": config.output_root / "03_refresh_recording_status.ps1",
        "open_recording_folders": config.output_root / "04_open_recording_folders.ps1",
        "review_assignment_table": config.output_root / "05_review_assignment_table.ps1",
        "execute_copy_after_review": config.output_root / "06_execute_copy_after_review.ps1",
        "prepare_skeleton_review_plan": config.output_root / "07_prepare_skeleton_review_plan.ps1",
        "dry_run_skeleton_review": config.output_root / "08_dry_run_skeleton_review.ps1",
        "execute_skeleton_review": config.output_root / "09_execute_skeleton_review.ps1",
    }
    scripts["dry_run_camera_check"].write_text(
        _flow_script(config, project_root=project_root, slot_manifest=slot_manifest, execute=False),
        encoding=PS_SCRIPT_ENCODING,
    )
    scripts["live_guided_recording"].write_text(
        _flow_script(config, project_root=project_root, slot_manifest=slot_manifest, execute=True),
        encoding=PS_SCRIPT_ENCODING,
    )
    scripts["refresh_recording_status"].write_text(
        _refresh_script(config, project_root=project_root),
        encoding=PS_SCRIPT_ENCODING,
    )
    scripts["open_recording_folders"].write_text(_open_folders_script(config), encoding=PS_SCRIPT_ENCODING)
    scripts["review_assignment_table"].write_text(_review_assignment_table_script(config), encoding=PS_SCRIPT_ENCODING)
    scripts["execute_copy_after_review"].write_text(
        _ingest_script(config, project_root=project_root, execute_copy=True),
        encoding=PS_SCRIPT_ENCODING,
    )
    scripts["prepare_skeleton_review_plan"].write_text(_end_to_end_script(config, project_root=project_root), encoding=PS_SCRIPT_ENCODING)
    scripts["dry_run_skeleton_review"].write_text(
        _skeleton_review_script(config, project_root=project_root, dry_run=True),
        encoding=PS_SCRIPT_ENCODING,
    )
    scripts["execute_skeleton_review"].write_text(
        _skeleton_review_script(config, project_root=project_root, dry_run=False),
        encoding=PS_SCRIPT_ENCODING,
    )
    summary = {
        "status": "ready_for_operator_recording",
        "project_root": project_root.as_posix() if project_root is not None else "script_relative_project_root",
        "staging_root": config.staging_root.as_posix(),
        "calibration_root": config.calibration_root.as_posix(),
        "heldout_root": config.heldout_root.as_posix(),
        "count_per_label": int(config.count_per_label),
        "expected_per_label": int(config.expected_per_label),
        "scripts": {name: path.as_posix() for name, path in scripts.items()},
        "operator_order": [
            "01_dry_run_camera_check.ps1",
            "02_live_guided_recording.ps1",
            "03_refresh_recording_status.ps1",
            "05_review_assignment_table.ps1",
            "06_execute_copy_after_review.ps1",
            "07_prepare_skeleton_review_plan.ps1",
            "08_dry_run_skeleton_review.ps1",
            "09_execute_skeleton_review.ps1",
        ],
        "hard_stops": [
            "Do not copy held-out test MP4s into staging or calibration.",
            "Do not run skeleton review until MP4 preflight and the skeleton-review plan are ready.",
            "Do not run dataset generation, training, SCHUNK, or Isaac from these launch scripts.",
            "Only 02_live_guided_recording.ps1 records MP4s because it passes --execute.",
            "Only 06_execute_copy_after_review.ps1 copies staged MP4s because it passes --execute-copy.",
            "Only 09_execute_skeleton_review.ps1 runs MediaPipe skeleton review.",
        ],
    }
    (config.output_root / "recording_launch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (config.output_root / "README.md").write_text(_readme(summary), encoding="utf-8")
    return summary


def _flow_script(
    config: V4RecordingLaunchScriptsConfig,
    *,
    project_root: Path | None,
    slot_manifest: Path,
    execute: bool,
) -> str:
    parts = [
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
        config.flow_output_root.as_posix(),
        "--count-per-label",
        str(config.count_per_label),
        "--camera",
        str(config.camera_index),
        "--pre-roll-s",
        _format_number(config.pre_roll_s),
        "--duration-s",
        _format_number(config.duration_s),
        "--fps",
        _format_number(config.fps),
        "--slot-manifest",
        slot_manifest.as_posix(),
        "--expected-per-label",
        str(config.expected_per_label),
        "--check-camera",
    ]
    if execute:
        parts.append("--execute")
    command = "& " + " ".join(_ps_quote(part) for part in parts)
    title = "Live guided recording" if execute else "Dry-run camera check"
    return "\n".join(
        [
            "Set-StrictMode -Version Latest",
            "$ErrorActionPreference = 'Stop'",
            *_set_location_lines(project_root),
            "$env:PYTHONPATH = 'src'",
            f"Write-Host {_ps_quote(title)}",
            command,
            "",
        ]
    )


def _refresh_script(config: V4RecordingLaunchScriptsConfig, *, project_root: Path | None) -> str:
    ingest = _ingest_command(config, execute_copy=False)
    readiness = "& " + " ".join(
        _ps_quote(part)
        for part in [
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
    )
    return "\n".join(
        [
            "Set-StrictMode -Version Latest",
            "$ErrorActionPreference = 'Stop'",
            *_set_location_lines(project_root),
            "$env:PYTHONPATH = 'src'",
            "Write-Host 'Refreshing v4 recording status'",
            ingest,
            readiness,
            "",
        ]
    )


def _ingest_script(config: V4RecordingLaunchScriptsConfig, *, project_root: Path | None, execute_copy: bool) -> str:
    title = "Executing reviewed v4 recording copy" if execute_copy else "Refreshing v4 recording ingest"
    return "\n".join(
        [
            "Set-StrictMode -Version Latest",
            "$ErrorActionPreference = 'Stop'",
            *_set_location_lines(project_root),
            "$env:PYTHONPATH = 'src'",
            f"Write-Host {_ps_quote(title)}",
            _ingest_command(config, execute_copy=execute_copy),
            "",
        ]
    )


def _ingest_command(config: V4RecordingLaunchScriptsConfig, *, execute_copy: bool) -> str:
    parts = [
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
        config.ingest_output_root.as_posix(),
        "--end-to-end-summary",
        config.end_to_end_summary_path.as_posix(),
        "--expected-per-label",
        str(config.expected_per_label),
    ]
    if execute_copy:
        parts.append("--execute-copy")
    return "& " + " ".join(_ps_quote(part) for part in parts)


def _review_assignment_table_script(config: V4RecordingLaunchScriptsConfig) -> str:
    assignment_table = config.ingest_output_root / "assignment" / "recording_slot_assignment_table.csv"
    staging_audit_table = config.ingest_output_root / "staging_audit" / "recording_staging_audit_table.csv"
    return "\n".join(
        [
            "Set-StrictMode -Version Latest",
            "$ErrorActionPreference = 'Stop'",
            f"$AssignmentTable = {_ps_quote(str(assignment_table))}",
            f"$StagingAuditTable = {_ps_quote(str(staging_audit_table))}",
            "if (Test-Path -LiteralPath $StagingAuditTable) { Start-Process -FilePath $StagingAuditTable }",
            "if (Test-Path -LiteralPath $AssignmentTable) { Start-Process -FilePath $AssignmentTable }",
            "if (-not (Test-Path -LiteralPath $AssignmentTable)) { Write-Host 'Assignment table not found yet. Run 03_refresh_recording_status.ps1 first.' }",
            "",
        ]
    )


def _end_to_end_script(config: V4RecordingLaunchScriptsConfig, *, project_root: Path | None) -> str:
    command = "& " + " ".join(
        _ps_quote(part)
        for part in [
            "python",
            "-m",
            "embodied_rps.tools.run_v4_end_to_end",
            "--calibration-input-root",
            config.calibration_root.as_posix(),
            "--heldout-root",
            config.heldout_root.as_posix(),
            "--original20-root",
            DEFAULT_LOCAL_DATA_ROOT.as_posix(),
            "--expected-min-per-label",
            str(config.expected_per_label),
            "--output-root",
            config.end_to_end_output_root.as_posix(),
            "--recording-ingest-summary",
            (config.ingest_output_root / "recording_ingest_summary.json").as_posix(),
        ]
    )
    return "\n".join(
        [
            "Set-StrictMode -Version Latest",
            "$ErrorActionPreference = 'Stop'",
            *_set_location_lines(project_root),
            "$env:PYTHONPATH = 'src'",
            "Write-Host 'Preparing v4 skeleton-review plan'",
            command,
            "",
        ]
    )


def _skeleton_review_script(config: V4RecordingLaunchScriptsConfig, *, project_root: Path | None, dry_run: bool) -> str:
    parts = [
        "python",
        "-m",
        "embodied_rps.tools.run_v4_skeleton_review_from_plan",
        "--skeleton-review-plan",
        config.skeleton_review_plan_path.as_posix(),
        "--output-root",
        config.skeleton_review_execution_output_root.as_posix(),
    ]
    if dry_run:
        parts.append("--dry-run")
    command = "& " + " ".join(_ps_quote(part) for part in parts)
    title = "Dry-running v4 skeleton review plan" if dry_run else "Executing v4 MediaPipe skeleton review"
    return "\n".join(
        [
            "Set-StrictMode -Version Latest",
            "$ErrorActionPreference = 'Stop'",
            *_set_location_lines(project_root),
            "$env:PYTHONPATH = 'src'",
            f"Write-Host {_ps_quote(title)}",
            command,
            "",
        ]
    )


def _set_location_lines(project_root: Path | None) -> list[str]:
    if project_root is not None:
        return [f"Set-Location -LiteralPath {_ps_quote(str(project_root))}"]
    return [
        "$ScriptRoot = Split-Path -Parent $PSCommandPath",
        "$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptRoot '..\\..')).Path",
        "Set-Location -LiteralPath $ProjectRoot",
    ]


def _open_folders_script(config: V4RecordingLaunchScriptsConfig) -> str:
    return "\n".join(
        [
            "Set-StrictMode -Version Latest",
            "$ErrorActionPreference = 'Stop'",
            f"Start-Process explorer.exe -ArgumentList {_ps_quote(str(config.staging_root))}",
            f"Start-Process explorer.exe -ArgumentList {_ps_quote(str(config.calibration_root))}",
            f"Start-Process explorer.exe -ArgumentList {_ps_quote(str(config.heldout_root))}",
            "",
        ]
    )


def _readme(summary: dict[str, object]) -> str:
    scripts = summary.get("scripts")
    lines = [
        "# V4 Recording Launch Scripts",
        "",
        "Run these from PowerShell on the local recording computer.",
        "",
        "1. `01_dry_run_camera_check.ps1` checks the camera and writes the next cue sheet.",
        "2. `02_live_guided_recording.ps1` records one balanced `rock`, `paper`, and `scissors` batch.",
        "3. `03_refresh_recording_status.ps1` refreshes ingest and readiness artifacts after recording.",
        "4. `05_review_assignment_table.ps1` opens the staging audit and assignment tables for review.",
        "5. `06_execute_copy_after_review.ps1` copies staged MP4s only after the assignment table is approved.",
        "6. `07_prepare_skeleton_review_plan.ps1` prepares the next safe skeleton-review gate.",
        "7. `08_dry_run_skeleton_review.ps1` validates the skeleton-review plan without MediaPipe extraction.",
        "8. `09_execute_skeleton_review.ps1` runs MediaPipe skeleton review after the plan is ready.",
        "",
        "The held-out `test` folder remains validation-only and must not be copied into staging or calibration.",
        "",
        "## Scripts",
        "",
    ]
    if isinstance(scripts, dict):
        for name, path in scripts.items():
            lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def _format_number(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _ps_quote(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


__all__ = ["V4RecordingLaunchScriptsConfig", "write_v4_recording_launch_scripts"]
