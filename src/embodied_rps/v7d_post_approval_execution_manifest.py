"""Execution manifest for the v7d post-approval path."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from embodied_rps.v7d_post_approval_preflight import (
    V7DPostApprovalPreflightConfig,
    write_v7d_post_approval_preflight,
)

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_EVIDENCE_INTEGRITY_ROOT = Path("artifacts/real_skeleton_v7d_review_evidence_integrity_20260618")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_post_approval_execution_manifest_20260618")
LOCAL_EXECUTION_STEPS: tuple[str, ...] = (
    "build_seed_package",
    "generate_three_class_dataset",
    "remap_stage1_rock_transition",
    "remap_stage2_paper_scissors",
    "local_smoke_preflight",
)
STEP_FIELDS: tuple[str, ...] = (
    "step_index",
    "name",
    "phase",
    "run_now",
    "gate",
    "command",
)


@dataclass(frozen=True)
class V7DPostApprovalExecutionManifestConfig:
    """Inputs for writing a status-only v7d post-approval execution manifest."""

    project_root: Path = field(default_factory=Path.cwd)
    review_root: Path = V7DPostApprovalPreflightConfig.review_root
    shortlist_root: Path = V7DPostApprovalPreflightConfig.shortlist_root
    selection_root: Path = V7DPostApprovalPreflightConfig.selection_root
    selection_decision_materialization_root: Path = V7DPostApprovalPreflightConfig.selection_decision_materialization_root
    preflight_output_root: Path = V7DPostApprovalPreflightConfig.output_root
    evidence_integrity_root: Path = DEFAULT_EVIDENCE_INTEGRITY_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    readiness_root: Path = V7DPostApprovalPreflightConfig.readiness_root
    seed_package_root: Path = V7DPostApprovalPreflightConfig.seed_package_root
    three_class_dataset_root: Path = V7DPostApprovalPreflightConfig.three_class_dataset_root
    stage1_dataset_root: Path = V7DPostApprovalPreflightConfig.stage1_dataset_root
    stage2_dataset_root: Path = V7DPostApprovalPreflightConfig.stage2_dataset_root
    base_dataset_root: Path = V7DPostApprovalPreflightConfig.base_dataset_root
    calibration_seed_package_root: Path = V7DPostApprovalPreflightConfig.calibration_seed_package_root
    live_rock_seed_package_root: Path = V7DPostApprovalPreflightConfig.live_rock_seed_package_root
    stage1_training_config: Path = V7DPostApprovalPreflightConfig.stage1_training_config
    stage2_training_config: Path = V7DPostApprovalPreflightConfig.stage2_training_config
    post_approval_pipeline_root: Path = V7DPostApprovalPreflightConfig.post_approval_pipeline_root
    review_decision_validation_root: Path = V7DPostApprovalPreflightConfig.review_decision_validation_root
    local_smoke_preflight_root: Path = V7DPostApprovalPreflightConfig.local_smoke_preflight_root
    remote_training_preflight_root: Path = V7DPostApprovalPreflightConfig.remote_training_preflight_root
    strict_validation_preflight_root: Path = V7DPostApprovalPreflightConfig.strict_validation_preflight_root
    stage1_profile_json: Path = V7DPostApprovalPreflightConfig.stage1_profile_json
    stage2_profile_json: Path = V7DPostApprovalPreflightConfig.stage2_profile_json


def write_v7d_post_approval_execution_manifest(
    config: V7DPostApprovalExecutionManifestConfig,
) -> dict[str, object]:
    """Write the current v7d post-approval execution manifest without running any stage."""

    project_root = config.project_root.resolve()
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    evidence = _read_evidence_integrity_summary(project_root, config.evidence_integrity_root)
    materialization = _read_selection_materialization_summary(
        project_root,
        config.selection_decision_materialization_root,
    )
    pipeline = _read_post_approval_pipeline_summary(project_root, config.post_approval_pipeline_root)
    approval_handoff = _approval_handoff_status(materialization=materialization, pipeline=pipeline)
    preflight = write_v7d_post_approval_preflight(_preflight_config(config))
    commands = _commands(preflight)
    for command in commands.values():
        _reject_heldout_path(command, context="planned command")

    preflight_status = str(preflight.get("status", ""))
    evidence_status = str(evidence.get("status", "missing"))
    status = _manifest_status(
        evidence_status=evidence_status,
        preflight_status=preflight_status,
        approval_handoff=approval_handoff,
    )
    steps = _execution_steps(commands=commands, manifest_status=status)
    phases = _phases(evidence_status=evidence_status, preflight_status=preflight_status, manifest_status=status)
    commands_to_run_now = [step for step in steps if bool(step.get("run_now"))]
    artifact_status = preflight.get("artifact_status", {})

    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "preflight_status": preflight_status,
        "evidence_integrity_status": evidence_status,
        "approval_handoff_status": approval_handoff,
        "missing_required_approved_roles": preflight.get("missing_required_approved_roles", []),
        "phases": phases,
        "all_steps": steps,
        "commands_to_run_now": commands_to_run_now,
        "artifact_status": artifact_status,
        "seed_package_created": bool(_mapping_get(artifact_status, "seed_package_root_exists")),
        "dataset_generated": bool(_mapping_get(artifact_status, "three_class_generation_summary_exists")),
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "promotion_eligible": False,
        "heldout_policy": "heldout */test paths are rejected from v7d execution manifest commands and metadata",
        "next_action": _next_action(status),
        "config": _config_summary(project_root=project_root, config=config),
    }
    _write_csv(output_root / "post_approval_execution_steps.csv", STEP_FIELDS, steps)
    (output_root / "post_approval_execution_manifest_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "post_approval_execution_manifest_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _preflight_config(config: V7DPostApprovalExecutionManifestConfig) -> V7DPostApprovalPreflightConfig:
    return V7DPostApprovalPreflightConfig(
        project_root=config.project_root,
        review_root=config.review_root,
        shortlist_root=config.shortlist_root,
        selection_root=config.selection_root,
        selection_decision_materialization_root=config.selection_decision_materialization_root,
        readiness_root=config.readiness_root,
        seed_package_root=config.seed_package_root,
        three_class_dataset_root=config.three_class_dataset_root,
        stage1_dataset_root=config.stage1_dataset_root,
        stage2_dataset_root=config.stage2_dataset_root,
        base_dataset_root=config.base_dataset_root,
        calibration_seed_package_root=config.calibration_seed_package_root,
        live_rock_seed_package_root=config.live_rock_seed_package_root,
        stage1_training_config=config.stage1_training_config,
        stage2_training_config=config.stage2_training_config,
        output_root=config.preflight_output_root,
        post_approval_pipeline_root=config.post_approval_pipeline_root,
        review_decision_validation_root=config.review_decision_validation_root,
        local_smoke_preflight_root=config.local_smoke_preflight_root,
        remote_training_preflight_root=config.remote_training_preflight_root,
        strict_validation_preflight_root=config.strict_validation_preflight_root,
        stage1_profile_json=config.stage1_profile_json,
        stage2_profile_json=config.stage2_profile_json,
    )


def _read_evidence_integrity_summary(project_root: Path, evidence_root: Path) -> dict[str, object]:
    path = _resolve_path(project_root, evidence_root) / "review_evidence_integrity_summary.json"
    if not path.exists():
        return {"status": "missing", "summary_path": _display_path(path, base=project_root)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"status": "invalid"}


def _read_selection_materialization_summary(project_root: Path, materialization_root: Path) -> dict[str, object]:
    path = _resolve_path(project_root, materialization_root) / "selection_decision_materialization_summary.json"
    if not path.exists():
        return {"status": "missing", "summary_path": _display_path(path, base=project_root)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"status": "invalid"}


def _read_post_approval_pipeline_summary(project_root: Path, pipeline_root: Path) -> dict[str, object]:
    path = _resolve_path(project_root, pipeline_root) / "v7d_post_approval_pipeline_summary.json"
    if not path.exists():
        return {"status": "missing", "summary_path": _display_path(path, base=project_root)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"status": "invalid"}


def _approval_handoff_status(
    *,
    materialization: Mapping[str, object],
    pipeline: Mapping[str, object],
) -> dict[str, object]:
    materialization_status = str(materialization.get("status", "missing"))
    dry_run_status = str(pipeline.get("status", "missing"))
    materialized_apply_safe = (
        materialization_status == "ready_for_review_decision_apply"
        and materialization.get("decisions_apply_safe") is True
    )
    dry_run_apply_ready = (
        dry_run_status == "ready_for_review_decision_apply"
        and str(pipeline.get("review_decision_mode", "")) == "dry-run"
    )
    if dry_run_apply_ready:
        status = "ready_for_confirmed_apply"
        next_approval_command = "apply_decisions"
    elif materialized_apply_safe:
        status = "ready_for_dry_run"
        next_approval_command = "apply_decisions_dry_run"
    else:
        status = "awaiting_selection_materialization"
        next_approval_command = ""
    return {
        "status": status,
        "materialization_status": materialization_status,
        "materialized_decisions_apply_safe": materialized_apply_safe,
        "dry_run_status": dry_run_status,
        "dry_run_apply_ready": dry_run_apply_ready,
        "next_approval_command": next_approval_command,
    }


def _commands(preflight: Mapping[str, object]) -> dict[str, str]:
    value = preflight.get("planned_commands", {})
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _manifest_status(
    *,
    evidence_status: str,
    preflight_status: str,
    approval_handoff: Mapping[str, object],
) -> str:
    if evidence_status != "ready_for_manual_temporal_approval":
        return "blocked_review_evidence_integrity"
    if preflight_status == "blocked_manual_approval_required":
        handoff_status = str(approval_handoff.get("status", ""))
        if handoff_status == "ready_for_confirmed_apply":
            return "ready_for_approval_apply"
        if handoff_status == "ready_for_dry_run":
            return "ready_for_approval_apply_dry_run"
        return "awaiting_manual_approval"
    if preflight_status in {
        "ready_for_seed_package_build",
        "ready_for_three_class_dataset_generation",
        "ready_for_two_stage_remap",
    }:
        return "ready_for_local_seed_dataset_execution"
    if preflight_status == "ready_for_local_smoke_training":
        return "ready_for_local_smoke"
    return "blocked_post_approval_preflight"


def _execution_steps(*, commands: Mapping[str, str], manifest_status: str) -> list[dict[str, object]]:
    steps: list[dict[str, object]] = []
    for index, (name, command) in enumerate(commands.items(), start=1):
        phase = _step_phase(name)
        run_now = (
            (manifest_status == "ready_for_approval_apply_dry_run" and name == "apply_decisions_dry_run")
            or (manifest_status == "ready_for_approval_apply" and name == "apply_decisions")
            or (manifest_status == "ready_for_local_seed_dataset_execution" and name in LOCAL_EXECUTION_STEPS)
        )
        steps.append(
            {
                "step_index": index,
                "name": name,
                "phase": phase,
                "run_now": run_now,
                "gate": _step_gate(name),
                "command": command,
            }
        )
    return steps


def _step_phase(name: str) -> str:
    if name in {"materialize_selection_decisions", "validate_review_decisions", "apply_decisions_dry_run", "apply_decisions"}:
        return "approval_apply"
    if name in {
        "check_seed_readiness",
        "build_seed_package",
        "generate_three_class_dataset",
        "remap_stage1_rock_transition",
        "remap_stage2_paper_scissors",
    }:
        return "local_seed_dataset_remap"
    if name.startswith("local_"):
        return "local_smoke"
    if name.startswith("remote_"):
        return "remote_training"
    return "strict_validation"


def _step_gate(name: str) -> str:
    if name in {"materialize_selection_decisions", "validate_review_decisions", "apply_decisions_dry_run", "apply_decisions"}:
        return "requires human-approved prompt-window rows"
    if name in LOCAL_EXECUTION_STEPS or name == "check_seed_readiness":
        return "requires applied decisions and ready_for_v7d_seed_package"
    if name.startswith("local_"):
        return "requires generated two-stage datasets"
    if name.startswith("remote_"):
        return "requires local smoke readiness"
    return "requires remote TCN profiles and strict MP4 gates"


def _phases(*, evidence_status: str, preflight_status: str, manifest_status: str) -> list[dict[str, object]]:
    manual_status = "ready_for_human_review" if evidence_status == "ready_for_manual_temporal_approval" else "blocked_evidence_integrity"
    if manifest_status == "ready_for_approval_apply_dry_run":
        approval_status = "ready_for_dry_run"
    elif manifest_status == "ready_for_approval_apply":
        approval_status = "ready_for_confirmed_apply"
    elif preflight_status == "blocked_manual_approval_required":
        approval_status = "blocked_manual_approval_required"
    else:
        approval_status = "manual_approval_applied_or_ready"
    local_status = (
        "ready_after_manual_approval"
        if preflight_status in {
            "ready_for_seed_package_build",
            "ready_for_three_class_dataset_generation",
            "ready_for_two_stage_remap",
        }
        else "blocked_until_manual_approval"
    )
    smoke_status = "ready_after_local_datasets" if preflight_status == "ready_for_local_smoke_training" else "blocked_until_datasets_exist"
    return [
        {"name": "manual_temporal_review", "status": manual_status},
        {"name": "approval_apply", "status": approval_status},
        {"name": "local_seed_dataset_remap", "status": local_status},
        {"name": "local_smoke", "status": smoke_status},
        {"name": "remote_training", "status": "blocked_until_local_smoke_passes"},
        {"name": "strict_validation", "status": "blocked_until_remote_training_passes"},
    ]


def _next_action(status: str) -> str:
    if status == "ready_for_approval_apply_dry_run":
        return "run the v7d post-approval pipeline dry-run against the materialized decision CSV before any mutating apply"
    if status == "ready_for_approval_apply":
        return "run the confirmed v7d apply command with the temporal-review confirmation phrase, then refresh seed readiness"
    if status == "awaiting_manual_approval":
        return "inspect temporal evidence, then fill or guarded-apply the three required approval selection rows"
    if status == "ready_for_local_seed_dataset_execution":
        return "run local seed-package, three-class dataset, and two-stage remap steps before local smoke"
    if status == "ready_for_local_smoke":
        return "run local GRU/TCN smoke before remote sync and A6000 TCN training"
    if status == "blocked_review_evidence_integrity":
        return "repair missing review evidence before manual approval"
    return "repair post-approval preflight blockers"


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _reject_heldout_path(value: str, *, context: str) -> None:
    normalized = value.replace("\\", "/").lower()
    if "/test/" in normalized or normalized.rstrip("/").endswith("/test"):
        raise ValueError(f"{context} contains held-out test path: {value}")


def _mapping_get(value: object, key: str) -> object:
    return value.get(key) if isinstance(value, Mapping) else None


def _summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V7d Post-Approval Execution Manifest",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Evidence integrity status: `{summary.get('evidence_integrity_status')}`",
        f"- Preflight status: `{summary.get('preflight_status')}`",
        f"- Approval handoff: `{summary.get('approval_handoff_status')}`",
        f"- Commands to run now: `{len(summary.get('commands_to_run_now', []))}`",
        "- This manifest does not run approval, seed, dataset, smoke, remote, validation, or promotion commands.",
        f"- Next action: `{summary.get('next_action')}`",
        "",
    ]
    steps = summary.get("all_steps", [])
    if isinstance(steps, list):
        lines.extend(["## Steps", ""])
        for step in steps:
            if isinstance(step, Mapping):
                lines.append(f"- `{step.get('name')}`: phase `{step.get('phase')}`, run_now `{step.get('run_now')}`")
    return "\n".join(lines) + "\n"


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(base.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _config_summary(*, project_root: Path, config: V7DPostApprovalExecutionManifestConfig) -> dict[str, object]:
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
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["V7DPostApprovalExecutionManifestConfig", "write_v7d_post_approval_execution_manifest"]
