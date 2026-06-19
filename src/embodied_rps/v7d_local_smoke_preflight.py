"""Fail-closed local smoke preflight for the v7d two-stage branch."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from embodied_rps.v7d_post_approval_preflight import V7DPostApprovalPreflightConfig

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"

V7DLocalSmokePreflightStatus = Literal[
    "blocked_datasets_missing",
    "blocked_training_configs_missing",
    "ready_for_local_smoke",
]


@dataclass(frozen=True)
class V7DLocalSmokePreflightConfig:
    """Inputs for the v7d local smoke preflight."""

    project_root: Path = field(default_factory=Path.cwd)
    output_root: Path = Path("artifacts/real_skeleton_v7d_local_smoke_preflight_20260618")
    stage1_dataset_root: Path = V7DPostApprovalPreflightConfig.stage1_dataset_root
    stage2_dataset_root: Path = V7DPostApprovalPreflightConfig.stage2_dataset_root
    stage1_training_config: Path = V7DPostApprovalPreflightConfig.stage1_training_config
    stage2_training_config: Path = V7DPostApprovalPreflightConfig.stage2_training_config


def write_v7d_local_smoke_preflight(config: V7DLocalSmokePreflightConfig) -> dict[str, object]:
    """Write a non-training local smoke readiness artifact for v7d."""

    project_root = config.project_root.resolve()
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    artifact_status = _artifact_status(project_root=project_root, config=config)
    status = _status(artifact_status)
    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "stage1_dataset_root": _display_path(_resolve_path(project_root, config.stage1_dataset_root), base=project_root),
        "stage2_dataset_root": _display_path(_resolve_path(project_root, config.stage2_dataset_root), base=project_root),
        "stage1_training_config": _display_path(
            _resolve_path(project_root, config.stage1_training_config),
            base=project_root,
        ),
        "stage2_training_config": _display_path(
            _resolve_path(project_root, config.stage2_training_config),
            base=project_root,
        ),
        "artifact_status": artifact_status,
        "planned_commands": _planned_commands(config),
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test MP4s remain validation-only and are not used by local smoke training",
        "next_action": _next_action(status),
        "config": _config_summary(project_root=project_root, config=config),
    }
    (output_root / "v7d_local_smoke_preflight_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7d_local_smoke_preflight_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _artifact_status(*, project_root: Path, config: V7DLocalSmokePreflightConfig) -> dict[str, bool]:
    stage1_root = _resolve_path(project_root, config.stage1_dataset_root)
    stage2_root = _resolve_path(project_root, config.stage2_dataset_root)
    return {
        "stage1_dataset_root_exists": stage1_root.exists(),
        "stage1_remap_summary_exists": (stage1_root / "remap_summary.json").exists(),
        "stage2_dataset_root_exists": stage2_root.exists(),
        "stage2_remap_summary_exists": (stage2_root / "remap_summary.json").exists(),
        "stage1_training_config_exists": _resolve_path(project_root, config.stage1_training_config).exists(),
        "stage2_training_config_exists": _resolve_path(project_root, config.stage2_training_config).exists(),
    }


def _status(artifact_status: Mapping[str, bool]) -> V7DLocalSmokePreflightStatus:
    if not artifact_status.get("stage1_remap_summary_exists") or not artifact_status.get("stage2_remap_summary_exists"):
        return "blocked_datasets_missing"
    if not artifact_status.get("stage1_training_config_exists") or not artifact_status.get("stage2_training_config_exists"):
        return "blocked_training_configs_missing"
    return "ready_for_local_smoke"


def _planned_commands(config: V7DLocalSmokePreflightConfig) -> dict[str, str]:
    return {
        "local_gru_smoke_stage1": _smoke_command(config.stage1_training_config, model="gru"),
        "local_tcn_smoke_stage1": _smoke_command(config.stage1_training_config, model="tcn"),
        "local_gru_smoke_stage2": _smoke_command(config.stage2_training_config, model="gru"),
        "local_tcn_smoke_stage2": _smoke_command(config.stage2_training_config, model="tcn"),
    }


def _smoke_command(config_path: Path, *, model: str) -> str:
    return (
        "python -m embodied_rps.tools.train_real_skeleton_predictor "
        f"--config {config_path.as_posix()} --model {model} --smoke --max-runs 1 --skip-export"
    )


def _next_action(status: V7DLocalSmokePreflightStatus) -> str:
    if status == "blocked_datasets_missing":
        return "generate v7d two-stage remap datasets before local smoke training"
    if status == "blocked_training_configs_missing":
        return "restore v7d two-stage training configs before local smoke training"
    return "run local GRU and TCN smoke for both v7d two-stage configs"


def _summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V7d Local Smoke Preflight",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Training started: `{summary.get('training_started')}`",
        f"- Remote training started: `{summary.get('remote_training_started')}`",
        f"- Validation started: `{summary.get('validation_started')}`",
        f"- Promotion eligible: `{summary.get('promotion_eligible')}`",
        f"- Next action: `{summary.get('next_action')}`",
        "",
        "## Planned Commands",
        "",
    ]
    commands = summary.get("planned_commands", {})
    if isinstance(commands, Mapping):
        for name, command in commands.items():
            lines.extend([f"### {name}", "", "```text", str(command), "```", ""])
    return "\n".join(lines)


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _config_summary(*, project_root: Path, config: V7DLocalSmokePreflightConfig) -> dict[str, object]:
    summary = asdict(config)
    summary["project_root"] = "."
    for key, value in list(summary.items()):
        if isinstance(value, Path):
            summary[key] = _display_path(_resolve_path(project_root, value), base=project_root)
    return _json_ready(summary)  # type: ignore[return-value]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["V7DLocalSmokePreflightConfig", "write_v7d_local_smoke_preflight"]
