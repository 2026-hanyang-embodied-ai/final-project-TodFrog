"""Local fail-closed pipeline for the v7e stage1 paper-transition rescue branch."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from embodied_rps.real_skeleton_dataset_remap import remap_real_skeleton_dataset
from embodied_rps.three_class_wait_skeletons import (
    ThreeClassWaitExpansionConfig,
    generate_three_class_wait_dataset,
)
from embodied_rps.v7e_seed_package_builder import (
    V7ESeedPackageBuilderConfig,
    blocked_v7e_seed_package_summary,
    build_v7e_stage1_paper_transition_rescue_seed_package,
)
from embodied_rps.v7e_stage1_dataset_handoff import (
    DEFAULT_BASE_DATASET_ROOT,
    DEFAULT_CALIBRATION_SEED_PACKAGE_ROOT,
    DEFAULT_LIVE_ROCK_SEED_PACKAGE_ROOT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_HANDOFF_OUTPUT_ROOT,
    DEFAULT_STAGE1_DATASET_ROOT,
    DEFAULT_STAGE1_TRAINING_CONFIG,
    DEFAULT_THREE_CLASS_DATASET_ROOT,
    V7E_AUGMENTATION_PROFILE,
    V7EStage1DatasetHandoffConfig,
    write_v7e_stage1_dataset_handoff,
)
from embodied_rps.v7e_seed_package_preflight import DEFAULT_V7E_SEED_PACKAGE_ROOT
from embodied_rps.v7e_stage1_local_smoke_preflight import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_LOCAL_SMOKE_PREFLIGHT_ROOT,
    V7EStage1LocalSmokePreflightConfig,
    write_v7e_stage1_local_smoke_preflight,
)


BRANCH_LABEL = "v7e_stage1_paper_transition_rescue"
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7e_stage1_local_pipeline_20260619")

V7EStage1LocalPipelineStatus = Literal[
    "official_full_scale_required",
    "blocked_seed_package_missing",
    "blocked_seed_package_not_passed",
    "blocked_paper_seed_review_required",
    "ready_for_stage1_local_pipeline_execution",
    "local_v7e_stage1_dataset_ready",
    "three_class_dataset_generation_failed",
    "stage1_remap_failed",
]


@dataclass(frozen=True)
class V7EStage1LocalPipelineConfig:
    """Inputs for the v7e local stage1 data pipeline."""

    project_root: Path = field(default_factory=Path.cwd)
    seed_package_root: Path = DEFAULT_V7E_SEED_PACKAGE_ROOT
    three_class_dataset_root: Path = DEFAULT_THREE_CLASS_DATASET_ROOT
    stage1_dataset_root: Path = DEFAULT_STAGE1_DATASET_ROOT
    base_dataset_root: Path = DEFAULT_BASE_DATASET_ROOT
    calibration_seed_package_root: Path = DEFAULT_CALIBRATION_SEED_PACKAGE_ROOT
    live_rock_seed_package_root: Path = DEFAULT_LIVE_ROCK_SEED_PACKAGE_ROOT
    stage1_training_config: Path = DEFAULT_STAGE1_TRAINING_CONFIG
    output_root: Path = DEFAULT_OUTPUT_ROOT
    handoff_output_root: Path = DEFAULT_HANDOFF_OUTPUT_ROOT
    local_smoke_preflight_root: Path = DEFAULT_LOCAL_SMOKE_PREFLIGHT_ROOT
    generated_per_target: int = 10000
    sequence_length: int = 72
    min_length: int = 48
    shard_size: int = 512
    base_rock_stride: int = 2
    seed: int = 20260619
    execute_local: bool = False
    overwrite_outputs: bool = False


def run_v7e_stage1_local_pipeline(config: V7EStage1LocalPipelineConfig) -> dict[str, Any]:
    """Run or dry-run the v7e stage1 local data pipeline."""

    project_root = config.project_root.resolve()
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if config.execute_local and _uses_official_dataset_roots(config) and int(config.generated_per_target) != 10000:
        summary = _base_summary(
            project_root=project_root,
            config=config,
            output_root=output_root,
            status="official_full_scale_required",
            completed_stages=[],
        )
        summary["next_action"] = "rerun the official v7e pipeline with generated_per_target=10000"
        return _write_summary(output_root, summary)

    completed_stages: list[str] = []
    seed_build_summary: dict[str, Any] | None = None
    handoff = _write_handoff(project_root=project_root, config=config)

    if config.execute_local and handoff["status"] == "blocked_seed_package_missing":
        seed_build_summary = _build_seed_package(project_root=project_root, config=config)
        if seed_build_summary.get("status") != "passed":
            summary = _base_summary(
                project_root=project_root,
                config=config,
                output_root=output_root,
                status=_status_from_seed_build(seed_build_summary),
                completed_stages=completed_stages,
                handoff=handoff,
            )
            summary["seed_package_build_summary"] = _relativize_paths(seed_build_summary, project_root=project_root)
            summary["next_action"] = str(seed_build_summary.get("next_action", "approve v7e paper seeds before building the seed package"))
            return _write_summary(output_root, summary)
        completed_stages.append("seed_package_build")
        handoff = _write_handoff(project_root=project_root, config=config)

    completed_stages.append("stage1_dataset_handoff")

    if not config.execute_local:
        status = (
            "ready_for_stage1_local_pipeline_execution"
            if str(handoff.get("status", "")).startswith("ready_for_")
            else str(handoff.get("status", "blocked_seed_package_missing"))
        )
        summary = _base_summary(
            project_root=project_root,
            config=config,
            output_root=output_root,
            status=status,
            completed_stages=completed_stages,
            handoff=handoff,
        )
        summary["next_action"] = _dry_run_next_action(handoff_status=str(handoff.get("status", "")))
        return _write_summary(output_root, summary)

    handoff_status = str(handoff.get("status", ""))
    generation_summary: dict[str, Any] | None = None
    remap_summary: dict[str, Any] | None = None
    local_smoke_preflight_summary: dict[str, Any] | None = None

    if handoff_status == "ready_for_three_class_dataset_generation":
        generation_summary = _generate_three_class_dataset(project_root=project_root, config=config)
        if generation_summary.get("status") != "passed":
            summary = _base_summary(
                project_root=project_root,
                config=config,
                output_root=output_root,
                status="three_class_dataset_generation_failed",
                completed_stages=completed_stages,
                handoff=handoff,
            )
            summary["generation_summary"] = _relativize_paths(generation_summary, project_root=project_root)
            return _write_summary(output_root, summary)
        completed_stages.append("three_class_dataset_generation")
        handoff = _write_handoff(project_root=project_root, config=config)
        handoff_status = str(handoff.get("status", ""))

    if handoff_status == "ready_for_stage1_remap":
        remap_summary = _remap_stage1_dataset(project_root=project_root, config=config)
        if remap_summary.get("status") != "passed":
            summary = _base_summary(
                project_root=project_root,
                config=config,
                output_root=output_root,
                status="stage1_remap_failed",
                completed_stages=completed_stages,
                handoff=handoff,
            )
            summary["stage1_remap_summary"] = _relativize_paths(remap_summary, project_root=project_root)
            return _write_summary(output_root, summary)
        completed_stages.append("stage1_rock_transition_remap")
        handoff = _write_handoff(project_root=project_root, config=config)

    if str(handoff.get("status", "")) == "ready_for_local_stage1_smoke":
        local_smoke_preflight_summary = _write_local_smoke_preflight(project_root=project_root, config=config)
        completed_stages.append("local_smoke_preflight")

    summary = _base_summary(
        project_root=project_root,
        config=config,
        output_root=output_root,
        status="local_v7e_stage1_dataset_ready",
        completed_stages=completed_stages,
        handoff=handoff,
    )
    if seed_build_summary is not None:
        summary["seed_package_build_summary"] = _relativize_paths(seed_build_summary, project_root=project_root)
    if generation_summary is not None:
        summary["generation_summary"] = _relativize_paths(generation_summary, project_root=project_root)
    if remap_summary is not None:
        summary["stage1_remap_summary"] = _relativize_paths(remap_summary, project_root=project_root)
    if local_smoke_preflight_summary is not None:
        summary["local_smoke_preflight_summary"] = _relativize_paths(local_smoke_preflight_summary, project_root=project_root)
    summary["dataset_generated"] = True
    summary["stage1_dataset_generated"] = True
    summary["next_action"] = "run local v7e stage1 GRU/TCN smoke before remote sync"
    return _write_summary(output_root, summary)


def _write_handoff(*, project_root: Path, config: V7EStage1LocalPipelineConfig) -> dict[str, Any]:
    return write_v7e_stage1_dataset_handoff(
        V7EStage1DatasetHandoffConfig(
            project_root=project_root,
            seed_package_root=config.seed_package_root,
            three_class_dataset_root=config.three_class_dataset_root,
            stage1_dataset_root=config.stage1_dataset_root,
            base_dataset_root=config.base_dataset_root,
            calibration_seed_package_root=config.calibration_seed_package_root,
            live_rock_seed_package_root=config.live_rock_seed_package_root,
            stage1_training_config=config.stage1_training_config,
            output_root=config.handoff_output_root,
            generated_per_target=int(config.generated_per_target),
            sequence_length=int(config.sequence_length),
            min_length=int(config.min_length),
            shard_size=int(config.shard_size),
            base_rock_stride=int(config.base_rock_stride),
            seed=int(config.seed),
        )
    )


def _build_seed_package(*, project_root: Path, config: V7EStage1LocalPipelineConfig) -> dict[str, Any]:
    builder_config = V7ESeedPackageBuilderConfig(
        project_root=project_root,
        output_root=config.seed_package_root,
        sequence_length=int(config.sequence_length),
        overwrite=bool(config.overwrite_outputs),
    )
    try:
        return build_v7e_stage1_paper_transition_rescue_seed_package(builder_config)
    except ValueError as exc:
        if "v7e seed-package preflight is not ready" not in str(exc):
            raise
        return blocked_v7e_seed_package_summary(builder_config)


def _generate_three_class_dataset(*, project_root: Path, config: V7EStage1LocalPipelineConfig) -> dict[str, Any]:
    output_root = _resolve_path(project_root, config.three_class_dataset_root)
    summary = dict(
        generate_three_class_wait_dataset(
            ThreeClassWaitExpansionConfig(
                output_root=output_root,
                base_dataset_root=_resolve_path(project_root, config.base_dataset_root),
                generated_per_target=int(config.generated_per_target),
                sequence_length=int(config.sequence_length),
                min_length=int(config.min_length),
                shard_size=int(config.shard_size),
                seed=int(config.seed),
                base_rock_stride=int(config.base_rock_stride),
                augmentation_profile=V7E_AUGMENTATION_PROFILE,
                calibration_seed_package_root=_resolve_optional_path(project_root, config.calibration_seed_package_root),
                live_rock_seed_package_root=_resolve_optional_path(project_root, config.live_rock_seed_package_root),
                v7_seed_package_root=_resolve_optional_path(project_root, config.seed_package_root),
                overwrite=bool(config.overwrite_outputs),
            )
        )
    )
    _sanitize_text_metadata_paths(project_root=project_root, root=output_root)
    return summary


def _remap_stage1_dataset(*, project_root: Path, config: V7EStage1LocalPipelineConfig) -> dict[str, Any]:
    output_root = _resolve_path(project_root, config.stage1_dataset_root)
    summary = dict(
        remap_real_skeleton_dataset(
            _resolve_path(project_root, config.three_class_dataset_root),
            output_root,
            mode="rock_vs_transition",
        )
    )
    summary.setdefault("status", "passed")
    (output_root / "remap_summary.json").write_text(
        json.dumps(_json_ready(_relativize_paths(summary, project_root=project_root)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _write_local_smoke_preflight(*, project_root: Path, config: V7EStage1LocalPipelineConfig) -> dict[str, Any]:
    return write_v7e_stage1_local_smoke_preflight(
        V7EStage1LocalSmokePreflightConfig(
            project_root=project_root,
            output_root=config.local_smoke_preflight_root,
            stage1_dataset_root=config.stage1_dataset_root,
            stage1_training_config=config.stage1_training_config,
        )
    )


def _base_summary(
    *,
    project_root: Path,
    config: V7EStage1LocalPipelineConfig,
    output_root: Path,
    status: str,
    completed_stages: list[str],
    handoff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    handoff_status = str(handoff.get("status", "not_written")) if handoff is not None else "not_written"
    return {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "handoff_status": handoff_status,
        "completed_stages": list(completed_stages),
        "augmentation_profile": V7E_AUGMENTATION_PROFILE,
        "generated_per_target": int(config.generated_per_target),
        "execute_local": bool(config.execute_local),
        "planned_execution": _planned_execution(handoff_status=handoff_status),
        "seed_package_created": False,
        "dataset_generated": False,
        "stage1_dataset_generated": False,
        "stage2_dataset_generated": False,
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "heldout15_started": False,
        "promotion_eligible": False,
        "stage1_training_scope": "retrain_rock_vs_transition_only",
        "stage2_policy": "reuse_v7d_paper_scissors_stage2_unless_new_evidence_requires_change",
        "fallback_policy": "keep_v4_live_demo_fallback",
        "heldout_policy": "heldout */test MP4s remain validation-only and are not copied into v7e training metadata",
        "config": _config_summary(project_root=project_root, config=config),
        "handoff_summary": _relativize_paths(dict(handoff), project_root=project_root) if handoff is not None else None,
        "next_action": "inspect v7e pipeline summary",
    }


def _planned_execution(*, handoff_status: str) -> dict[str, bool]:
    return {
        "build_seed_package": handoff_status == "blocked_seed_package_missing",
        "generate_three_class_dataset": handoff_status == "ready_for_three_class_dataset_generation",
        "remap_stage1_rock_transition": handoff_status in {"ready_for_three_class_dataset_generation", "ready_for_stage1_remap"},
        "remap_stage2_paper_scissors": False,
        "local_stage1_smoke": handoff_status == "ready_for_local_stage1_smoke",
        "remote_training": False,
        "strict_original20_validation": False,
        "heldout15_validation": False,
    }


def _dry_run_next_action(*, handoff_status: str) -> str:
    if handoff_status == "blocked_seed_package_missing":
        return "approve v7e paper seeds and build the v7e seed package before dataset generation"
    if handoff_status == "ready_for_three_class_dataset_generation":
        return "run this pipeline with --execute-local to generate the v7e three-class dataset and stage1 remap"
    if handoff_status == "ready_for_stage1_remap":
        return "run this pipeline with --execute-local to remap rock_vs_transition stage1"
    if handoff_status == "ready_for_local_stage1_smoke":
        return "run local v7e stage1 GRU/TCN smoke before remote sync"
    return "resolve the blocked v7e handoff before local execution"


def _status_from_seed_build(seed_build_summary: Mapping[str, Any]) -> str:
    status = str(seed_build_summary.get("status", "blocked_paper_seed_review_required"))
    if status.startswith("blocked_"):
        return status
    return "blocked_paper_seed_review_required"


def _uses_official_dataset_roots(config: V7EStage1LocalPipelineConfig) -> bool:
    return (
        config.seed_package_root == DEFAULT_V7E_SEED_PACKAGE_ROOT
        and config.three_class_dataset_root == DEFAULT_THREE_CLASS_DATASET_ROOT
        and config.stage1_dataset_root == DEFAULT_STAGE1_DATASET_ROOT
    )


def _config_summary(*, project_root: Path, config: V7EStage1LocalPipelineConfig) -> dict[str, Any]:
    return {
        "seed_package_root": _display_path(_resolve_path(project_root, config.seed_package_root), base=project_root),
        "three_class_dataset_root": _display_path(_resolve_path(project_root, config.three_class_dataset_root), base=project_root),
        "stage1_dataset_root": _display_path(_resolve_path(project_root, config.stage1_dataset_root), base=project_root),
        "stage1_training_config": _display_path(_resolve_path(project_root, config.stage1_training_config), base=project_root),
        "handoff_output_root": _display_path(_resolve_path(project_root, config.handoff_output_root), base=project_root),
        "local_smoke_preflight_root": _display_path(_resolve_path(project_root, config.local_smoke_preflight_root), base=project_root),
        "generated_per_target": int(config.generated_per_target),
        "execute_local": bool(config.execute_local),
        "overwrite_outputs": bool(config.overwrite_outputs),
    }


def _write_summary(output_root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    json_ready = _json_ready(summary)
    (output_root / "v7e_stage1_local_pipeline_summary.json").write_text(
        json.dumps(json_ready, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7e_stage1_local_pipeline_summary.md").write_text(_markdown(summary), encoding="utf-8")
    return json_ready


def _sanitize_text_metadata_paths(*, project_root: Path, root: Path) -> None:
    filenames = (
        "dataset_card.md",
        "generation_config.json",
        "run_summary.json",
        "sample_metadata.jsonl",
        "shard_index.csv",
        "validation_summary.json",
    )
    replacements = _metadata_path_replacements(project_root)
    for filename in filenames:
        path = root / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        sanitized = text
        for old, new in replacements:
            sanitized = sanitized.replace(old, new)
        if sanitized != text:
            path.write_text(sanitized, encoding="utf-8")


def _metadata_path_replacements(project_root: Path) -> tuple[tuple[str, str], ...]:
    root_posix = project_root.resolve().as_posix().rstrip("/")
    root_native = str(project_root.resolve()).rstrip("\\/")
    root_native_posix = root_native.replace("\\", "/")
    roots = []
    for value in (root_posix, root_native, root_native_posix):
        if value and value not in roots:
            roots.append(value)
    replacements: list[tuple[str, str]] = []
    for value in roots:
        replacements.extend(
            [
                (value + "/", ""),
                (value + "\\", ""),
                (value, "."),
            ]
        )
    replacements.extend(
        [
            ("D:/dataset/", "dataset:/"),
            ("D:\\dataset\\", "dataset:/"),
            ("D:/dataset", "dataset:/"),
            ("D:\\dataset", "dataset:/"),
            ("d:/dataset/", "dataset:/"),
            ("d:\\dataset\\", "dataset:/"),
            ("d:/dataset", "dataset:/"),
            ("d:\\dataset", "dataset:/"),
        ]
    )
    return tuple(replacements)


def _markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# V7e Stage1 Local Pipeline",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Branch: `{summary.get('branch_label')}`",
            f"- Handoff status: `{summary.get('handoff_status')}`",
            f"- Completed stages: `{summary.get('completed_stages')}`",
            f"- Dataset generated: `{summary.get('dataset_generated')}`",
            f"- Stage1 dataset generated: `{summary.get('stage1_dataset_generated')}`",
            "- Stage2 dataset generated: `False`",
            "- Training started: `False`",
            "- Heldout15 started: `False`",
            "- Promotion eligible: `False`",
            f"- Next action: {summary.get('next_action')}",
            "",
        ]
    )


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _resolve_optional_path(project_root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return _resolve_path(project_root, path)


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _relativize_paths(value: Any, *, project_root: Path) -> Any:
    if isinstance(value, Path):
        return _display_path(value, base=project_root)
    if isinstance(value, Mapping):
        return {str(key): _relativize_paths(item, project_root=project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_relativize_paths(item, project_root=project_root) for item in value]
    if isinstance(value, str):
        try:
            path = Path(value)
        except ValueError:
            return value
        if path.is_absolute():
            return _display_path(path, base=project_root)
    return value


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
    "V7EStage1LocalPipelineConfig",
    "run_v7e_stage1_local_pipeline",
]
