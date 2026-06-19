"""PowerShell contract probe for the realtime demo strict wrapper."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_rps.realtime_demo_launch_scripts import RealtimeDemoLaunchScriptsConfig, write_realtime_demo_launch_scripts


@dataclass(frozen=True)
class RealtimeDemoWrapperContractProbeConfig:
    """Output path for the strict-wrapper PowerShell contract probe."""

    output_root: Path = Path("artifacts/realtime_demo_wrapper_contract_probe_20260616")


_PRE_CAPTURE_REFRESH_SCRIPTS = [
    "30_clear_stale_live_artifacts.ps1",
    "08_triage_live_capture.ps1",
    "13_build_operator_outcome_report.ps1",
    "18_build_live_run_checklist.ps1",
    "19_build_final_run_card.ps1",
    "20_build_learning_queue.ps1",
    "21_build_goal_progress_audit.ps1",
    "25_build_live_status_snapshot.ps1",
    "26_build_operator_handoff_card.ps1",
]

_AUDIT_SCRIPTS = [
    "27_check_prelaunch_audit.ps1",
    "29_check_operator_command_audit.ps1",
    "32_check_guarded_retake_readiness.ps1",
]

_CAPTURE_SCRIPT = "22_run_live_demo_operator_confirmed.ps1"

_POST_CAPTURE_REFRESH_SCRIPTS = [
    "23_check_goal_progress_strict.ps1",
    "25_build_live_status_snapshot.ps1",
    "26_build_operator_handoff_card.ps1",
]


def run_realtime_demo_wrapper_contract_probe(config: RealtimeDemoWrapperContractProbeConfig) -> dict[str, object]:
    """Run stubbed PowerShell scenarios against the generated strict live wrapper."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    workspace = config.output_root / "workspace"
    if workspace.exists():
        _remove_tree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    scenarios = [
        _run_scenario(
            workspace=workspace,
            scenario_id="prelaunch_failure_short_circuit",
            exits=_scenario_exits({"27_check_prelaunch_audit.ps1": 50, "23_check_goal_progress_strict.ps1": 10}),
            expected_exit_code=50,
            expected_ran_scripts=[*_PRE_CAPTURE_REFRESH_SCRIPTS, "27_check_prelaunch_audit.ps1"],
        ),
        _run_scenario(
            workspace=workspace,
            scenario_id="operator_command_audit_failure_short_circuit",
            exits=_scenario_exits({"29_check_operator_command_audit.ps1": 70, "23_check_goal_progress_strict.ps1": 10}),
            expected_exit_code=70,
            expected_ran_scripts=[
                *_PRE_CAPTURE_REFRESH_SCRIPTS,
                "27_check_prelaunch_audit.ps1",
                "29_check_operator_command_audit.ps1",
            ],
        ),
        _run_scenario(
            workspace=workspace,
            scenario_id="guarded_retake_failure_short_circuit",
            exits=_scenario_exits({"32_check_guarded_retake_readiness.ps1": 55, "23_check_goal_progress_strict.ps1": 10}),
            expected_exit_code=55,
            expected_ran_scripts=[*_PRE_CAPTURE_REFRESH_SCRIPTS, *_AUDIT_SCRIPTS],
        ),
        _run_scenario(
            workspace=workspace,
            scenario_id="goal_check_live_capture_missing",
            exits=_scenario_exits({"23_check_goal_progress_strict.ps1": 10}),
            expected_exit_code=10,
            expected_ran_scripts=[
                *_PRE_CAPTURE_REFRESH_SCRIPTS,
                *_AUDIT_SCRIPTS,
                _CAPTURE_SCRIPT,
                *_POST_CAPTURE_REFRESH_SCRIPTS,
            ],
        ),
        _run_scenario(
            workspace=workspace,
            scenario_id="live_capture_success",
            exits=_scenario_exits({}),
            expected_exit_code=0,
            expected_ran_scripts=[
                *_PRE_CAPTURE_REFRESH_SCRIPTS,
                *_AUDIT_SCRIPTS,
                _CAPTURE_SCRIPT,
                *_POST_CAPTURE_REFRESH_SCRIPTS,
            ],
        ),
    ]
    passed = all(scenario["passed"] is True for scenario in scenarios)
    output_json = config.output_root / "wrapper_contract_probe.json"
    output_md = config.output_root / "wrapper_contract_probe.md"
    summary: dict[str, object] = {
        "contract_status": "passed" if passed else "failed",
        "scenario_count": len(scenarios),
        "passed_count": sum(1 for scenario in scenarios if scenario["passed"] is True),
        "scenarios": scenarios,
        "outputs": {
            "wrapper_contract_probe_json": output_json.as_posix(),
            "wrapper_contract_probe_md": output_md.as_posix(),
            "workspace": workspace.as_posix(),
        },
        "claim_scope": (
            "stubbed PowerShell contract probe for the strict live wrapper; does not run camera capture, "
            "MediaPipe inference, rendering, training, upload, or repository edits"
        ),
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(_probe_markdown(summary), encoding="utf-8")
    return summary


def _scenario_exits(overrides: dict[str, int]) -> dict[str, int]:
    scripts = [
        *_PRE_CAPTURE_REFRESH_SCRIPTS,
        *_AUDIT_SCRIPTS,
        _CAPTURE_SCRIPT,
        *_POST_CAPTURE_REFRESH_SCRIPTS,
    ]
    exits = {script: 0 for script in scripts}
    exits.update(overrides)
    return exits


def _run_scenario(
    *,
    workspace: Path,
    scenario_id: str,
    exits: dict[str, int],
    expected_exit_code: int,
    expected_ran_scripts: list[str],
) -> dict[str, object]:
    scenario_root = workspace / scenario_id
    launch_root = scenario_root / "launch"
    project_root = scenario_root / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    write_realtime_demo_launch_scripts(
        RealtimeDemoLaunchScriptsConfig(
            output_root=launch_root,
            project_root=project_root,
            python_executable=Path(sys.executable),
            sample_video=scenario_root / "sample.mp4",
            response_preview_image=scenario_root / "preview.png",
            live_composite_output_root=scenario_root / "composite",
        )
    )
    run_log = scenario_root / "run_log.txt"
    for script_name, exit_code in exits.items():
        _write_stub_script(
            launch_root / script_name,
            run_log=run_log,
            script_name=script_name,
            exit_code=exit_code,
            scenario_id=scenario_id,
            project_root=project_root,
        )
    strict_wrapper = launch_root / "24_run_live_demo_operator_confirmed_strict.ps1"
    result = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(strict_wrapper)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        text=True,
    )
    ran_scripts = _read_run_log(run_log)
    handoff_summary_printed = _handoff_summary_printed(result.stdout, scenario_id=scenario_id)
    should_print_handoff = "23_check_goal_progress_strict.ps1" in expected_ran_scripts
    passed = (
        result.returncode == expected_exit_code
        and ran_scripts == expected_ran_scripts
        and handoff_summary_printed is should_print_handoff
    )
    return {
        "id": scenario_id,
        "passed": passed,
        "exit_code": int(result.returncode),
        "expected_exit_code": int(expected_exit_code),
        "ran_scripts": ran_scripts,
        "expected_ran_scripts": expected_ran_scripts,
        "handoff_summary_printed": handoff_summary_printed,
        "expected_handoff_summary_printed": should_print_handoff,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _write_stub_script(
    path: Path,
    *,
    run_log: Path,
    script_name: str,
    exit_code: int,
    scenario_id: str,
    project_root: Path,
) -> None:
    absolute_run_log = run_log.resolve()
    lines = [
        "$ErrorActionPreference = 'Stop'",
        f"Add-Content -Path '{_ps_quote(absolute_run_log.as_posix())}' -Value '{_ps_quote(script_name)}'",
    ]
    if script_name == "26_build_operator_handoff_card.ps1":
        handoff_path = (
            project_root.resolve() / "artifacts/realtime_demo_operator_handoff_card_20260616/operator_handoff_card.md"
        )
        lines.extend(
            [
                f"$handoffPath = '{_ps_quote(handoff_path.as_posix())}'",
                "New-Item -ItemType Directory -Force -Path (Split-Path -Parent $handoffPath) | Out-Null",
                f"Set-Content -Encoding UTF8 -Path $handoffPath -Value '# Stub handoff for {_ps_quote(scenario_id)}'",
            ]
        )
    lines.extend([f"exit {int(exit_code)}", ""])
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def _read_run_log(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _handoff_summary_printed(stdout: str, *, scenario_id: str) -> bool:
    return "--- Operator handoff summary ---" in stdout and f"Stub handoff for {scenario_id}" in stdout


def _remove_tree(path: Path) -> None:
    def _make_writable_and_retry(function: Any, failed_path: str, _exc_info: object) -> None:
        Path(failed_path).chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        function(failed_path)

    shutil.rmtree(path, onerror=_make_writable_and_retry)


def _probe_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Realtime Demo Wrapper Contract Probe",
        "",
        f"- Contract status: `{summary.get('contract_status')}`",
        f"- Scenario count: `{summary.get('scenario_count')}`",
        f"- Passed count: `{summary.get('passed_count')}`",
        "",
        "## Scenarios",
        "",
    ]
    scenarios = summary.get("scenarios", [])
    if isinstance(scenarios, list):
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            lines.extend(
                [
                    f"### {scenario.get('id')}",
                    "",
                    f"- Passed: `{scenario.get('passed')}`",
                    f"- Exit code: `{scenario.get('exit_code')}`",
                    f"- Expected exit code: `{scenario.get('expected_exit_code')}`",
                    f"- Ran scripts: `{scenario.get('ran_scripts')}`",
                    f"- Expected scripts: `{scenario.get('expected_ran_scripts')}`",
                    "",
                ]
            )
    return "\n".join(lines)


def _ps_quote(value: str) -> str:
    return value.replace("'", "''")


__all__ = ["RealtimeDemoWrapperContractProbeConfig", "run_realtime_demo_wrapper_contract_probe"]
