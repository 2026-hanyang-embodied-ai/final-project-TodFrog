"""Fail-closed local smoke preflight for the v7e stage1 branch."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from embodied_rps.v7e_stage1_dataset_handoff import (
    DEFAULT_STAGE1_DATASET_ROOT,
    DEFAULT_STAGE1_TRAINING_CONFIG,
)


BRANCH_LABEL = "v7e_stage1_paper_transition_rescue"
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7e_stage1_local_smoke_preflight_20260619")

V7EStage1LocalSmokePreflightStatus = Literal[
    "blocked_stage1_dataset_missing",
    "blocked_stage1_remap_contract_invalid",
    "blocked_stage1_training_config_missing",
    "ready_for_stage1_local_smoke",
]


@dataclass(frozen=True)
class V7EStage1LocalSmokePreflightConfig:
    """Inputs for the v7e stage1-only local smoke preflight."""

    project_root: Path = field(default_factory=Path.cwd)
    output_root: Path = DEFAULT_OUTPUT_ROOT
    stage1_dataset_root: Path = DEFAULT_STAGE1_DATASET_ROOT
    stage1_training_config: Path = DEFAULT_STAGE1_TRAINING_CONFIG


def write_v7e_stage1_local_smoke_preflight(config: V7EStage1LocalSmokePreflightConfig) -> dict[str, Any]:
    """Write a non-training local smoke readiness artifact for v7e stage1."""

    project_root = config.project_root.resolve()
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    artifact_status = _artifact_status(project_root=project_root, config=config)
    status = _status(artifact_status)
    summary: dict[str, Any] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "stage1_dataset_root": _display_path(_resolve_path(project_root, config.stage1_dataset_root), base=project_root),
        "stage1_training_config": _display_path(
            _resolve_path(project_root, config.stage1_training_config),
            base=project_root,
        ),
        "artifact_status": artifact_status,
        "stage1_training_scope": "retrain_rock_vs_transition_only",
        "stage2_policy": "reuse_v7d_paper_scissors_stage2_unless_new_evidence_requires_change",
        "planned_commands": _planned_commands(config),
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "heldout15_started": False,
        "promotion_eligible": False,
        "fallback_policy": "keep_v4_live_demo_fallback",
        "heldout_policy": "heldout */test MP4s remain validation-only and are not used by local smoke training",
        "next_action": _next_action(status),
        "config": _config_summary(project_root=project_root, config=config),
    }
    (output_root / "v7e_stage1_local_smoke_preflight_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7e_stage1_local_smoke_preflight_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _artifact_status(*, project_root: Path, config: V7EStage1LocalSmokePreflightConfig) -> dict[str, Any]:
    stage1_root = _resolve_path(project_root, config.stage1_dataset_root)
    remap_summary = _read_json_if_exists(stage1_root / "remap_summary.json")
    return {
        "stage1_dataset_root_exists": stage1_root.exists(),
        "stage1_remap_summary_exists": bool(remap_summary),
        "stage1_remap_status": str(remap_summary.get("status", "missing")) if remap_summary else "missing",
        "stage1_remap_mode": str(remap_summary.get("mode", "missing")) if remap_summary else "missing",
        "stage1_training_config_exists": _resolve_path(project_root, config.stage1_training_config).exists(),
    }


def _status(artifact_status: Mapping[str, Any]) -> V7EStage1LocalSmokePreflightStatus:
    if not artifact_status.get("stage1_remap_summary_exists"):
        return "blocked_stage1_dataset_missing"
    if artifact_status.get("stage1_remap_status") != "passed" or artifact_status.get("stage1_remap_mode") != "rock_vs_transition":
        return "blocked_stage1_remap_contract_invalid"
    if not artifact_status.get("stage1_training_config_exists"):
        return "blocked_stage1_training_config_missing"
    return "ready_for_stage1_local_smoke"


def _planned_commands(config: V7EStage1LocalSmokePreflightConfig) -> dict[str, str]:
    return {
        "local_gru_smoke_stage1": _smoke_command(config.stage1_training_config, model="gru"),
        "local_tcn_smoke_stage1": _smoke_command(config.stage1_training_config, model="tcn"),
    }


def _smoke_command(config_path: Path, *, model: str) -> str:
    return (
        "python -m embodied_rps.tools.train_real_skeleton_predictor "
        f"--config {config_path.as_posix()} --model {model} --smoke --max-runs 1 --skip-export"
    )


def _next_action(status: V7EStage1LocalSmokePreflightStatus) -> str:
    if status == "blocked_stage1_dataset_missing":
        return "generate the v7e rock_vs_transition stage1 remap before local smoke training"
    if status == "blocked_stage1_remap_contract_invalid":
        return "regenerate the v7e stage1 remap with mode rock_vs_transition before local smoke training"
    if status == "blocked_stage1_training_config_missing":
        return "restore the v7e stage1 training config before local smoke training"
    return "run local v7e stage1 GRU and TCN smoke before remote sync or training"


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# V7e Stage1 Local Smoke Preflight",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Stage1 scope: `{summary.get('stage1_training_scope')}`",
        f"- Stage2 policy: `{summary.get('stage2_policy')}`",
        f"- Training started: `{summary.get('training_started')}`",
        f"- Remote training started: `{summary.get('remote_training_started')}`",
        f"- Validation started: `{summary.get('validation_started')}`",
        f"- Heldout15 started: `{summary.get('heldout15_started')}`",
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


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object in {path}")
    return dict(value)


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _config_summary(*, project_root: Path, config: V7EStage1LocalSmokePreflightConfig) -> dict[str, Any]:
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


__all__ = [
    "BRANCH_LABEL",
    "DEFAULT_OUTPUT_ROOT",
    "V7EStage1LocalSmokePreflightConfig",
    "write_v7e_stage1_local_smoke_preflight",
]
