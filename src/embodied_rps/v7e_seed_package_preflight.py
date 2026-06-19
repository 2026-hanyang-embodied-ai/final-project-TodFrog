"""Fail-closed seed-package preflight for the v7e paper rescue branch."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodied_rps.v7e_paper_seed_review_validator import (
    BRANCH_LABEL,
    DEFAULT_MINIMUM_APPROVED_PAPER_SEED_COUNT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_PAPER_REVIEW_VALIDATION_ROOT,
)


DEFAULT_V7D_SEED_PACKAGE_ROOT = Path("artifacts/real_skeleton_v7d_prompt_pose_seed_package_20260618")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7e_seed_package_preflight_20260619")
DEFAULT_V7E_SEED_PACKAGE_ROOT = Path("artifacts/real_skeleton_v7e_stage1_paper_transition_rescue_seed_package_20260619")
DEFAULT_V7E_STAGE1_DATASET_ROOT = Path("artifacts/real_guided_two_stage_rock_transition_v7e_stage1_paper_transition_rescue_20260619")
DEFAULT_V7E_STAGE1_RESULTS_ROOT = Path("results/real_skeleton_two_stage_rock_transition_v7e_stage1_paper_transition_rescue_tcn_ensemble")

V7D_SEED_SUMMARY_FILENAME = "v7d_seed_package_summary.json"
PAPER_REVIEW_SUMMARY_FILENAME = "v7e_paper_seed_review_validation_summary.json"


@dataclass(frozen=True)
class V7ESeedPackagePreflightConfig:
    """Inputs for checking whether v7e seed-package work may start."""

    project_root: Path = field(default_factory=Path.cwd)
    v7d_seed_package_root: Path = DEFAULT_V7D_SEED_PACKAGE_ROOT
    paper_review_validation_root: Path = DEFAULT_PAPER_REVIEW_VALIDATION_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    minimum_approved_paper_seed_count: int = DEFAULT_MINIMUM_APPROVED_PAPER_SEED_COUNT
    planned_v7e_seed_package_root: Path = DEFAULT_V7E_SEED_PACKAGE_ROOT
    planned_v7e_stage1_dataset_root: Path = DEFAULT_V7E_STAGE1_DATASET_ROOT
    planned_v7e_stage1_results_root: Path = DEFAULT_V7E_STAGE1_RESULTS_ROOT


def write_v7e_seed_package_preflight(config: V7ESeedPackagePreflightConfig) -> dict[str, Any]:
    """Write a non-mutating v7e seed-package preflight summary."""

    project_root = config.project_root.resolve()
    v7d_seed_package_root = _resolve_path(project_root, config.v7d_seed_package_root)
    paper_review_validation_root = _resolve_path(project_root, config.paper_review_validation_root)
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_seed_summary = _read_json_if_exists(v7d_seed_package_root / V7D_SEED_SUMMARY_FILENAME)
    paper_review_summary = _read_json_if_exists(paper_review_validation_root / PAPER_REVIEW_SUMMARY_FILENAME)
    base_seed_status = _base_seed_status(
        project_root=project_root,
        v7d_seed_package_root=v7d_seed_package_root,
        base_seed_summary=base_seed_summary,
    )
    paper_review_status = str(paper_review_summary.get("status", "missing")) if paper_review_summary else "missing"
    approve_count = int(paper_review_summary.get("approve_count", 0)) if paper_review_summary else 0
    approved_ids = _string_list(paper_review_summary.get("approved_paper_segment_ids", [])) if paper_review_summary else []
    minimum_approved_count = int(
        paper_review_summary.get("minimum_approved_paper_seed_count", config.minimum_approved_paper_seed_count)
    ) if paper_review_summary else int(config.minimum_approved_paper_seed_count)
    heldout_path_failures = [
        *_heldout_path_failures(base_seed_summary, prefix="v7d_seed_package"),
        *_heldout_path_failures(paper_review_summary, prefix="paper_review_validation"),
    ]
    status = _status(
        base_seed_status=base_seed_status,
        paper_review_status=paper_review_status,
        approve_count=approve_count,
        minimum_approved_count=minimum_approved_count,
        heldout_path_failures=heldout_path_failures,
    )

    summary: dict[str, Any] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "v7d_seed_package_root": _display_path(v7d_seed_package_root, base=project_root),
        "paper_review_validation_root": _display_path(paper_review_validation_root, base=project_root),
        "output_root": _display_path(output_root, base=project_root),
        "base_seed_package_status": base_seed_status,
        "paper_review_status": paper_review_status,
        "approved_paper_seed_count": approve_count,
        "minimum_approved_paper_seed_count": minimum_approved_count,
        "approved_paper_segment_ids": approved_ids,
        "heldout_path_failures": heldout_path_failures,
        "stage1_training_scope": "retrain_rock_vs_transition_only",
        "stage2_policy": "reuse_v7d_paper_scissors_stage2_unless_new_evidence_requires_change",
        "planned_roots": {
            "v7e_seed_package_root": _display_path(_resolve_path(project_root, config.planned_v7e_seed_package_root), base=project_root),
            "v7e_stage1_dataset_root": _display_path(_resolve_path(project_root, config.planned_v7e_stage1_dataset_root), base=project_root),
            "v7e_stage1_results_root": _display_path(_resolve_path(project_root, config.planned_v7e_stage1_results_root), base=project_root),
        },
        "planned_commands": _planned_commands(
            status=status,
            base_seed_status=base_seed_status,
            paper_review_status=paper_review_status,
            config=config,
            project_root=project_root,
        ),
        "seed_package_created": False,
        "dataset_generated": False,
        "training_started": False,
        "remote_training_started": False,
        "validation_started": False,
        "heldout15_started": False,
        "promotion_eligible": False,
        "fallback_policy": "keep_v4_live_demo_fallback",
        "heldout_policy": "heldout */test MP4s remain validation-only and are rejected from v7e seed-package preflight inputs",
        "next_action": _next_action(status),
    }
    (output_root / "v7e_seed_package_preflight_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7e_seed_package_preflight_summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _base_seed_status(*, project_root: Path, v7d_seed_package_root: Path, base_seed_summary: Mapping[str, Any]) -> str:
    if not base_seed_summary:
        return "missing"
    status = str(base_seed_summary.get("status", "missing"))
    if status != "passed":
        return status
    builder = base_seed_summary.get("builder_summary", {})
    if not isinstance(builder, Mapping):
        return "missing_builder_summary"
    seed_npz = _resolve_summary_path(project_root, v7d_seed_package_root, builder.get("seed_npz"), fallback="v7_rps_seed_dataset.npz")
    seed_metadata = _resolve_summary_path(project_root, v7d_seed_package_root, builder.get("seed_metadata"), fallback="seed_metadata.jsonl")
    if not seed_npz.exists() or not seed_metadata.exists():
        return "missing_seed_artifacts"
    return "passed"


def _status(
    *,
    base_seed_status: str,
    paper_review_status: str,
    approve_count: int,
    minimum_approved_count: int,
    heldout_path_failures: Sequence[Mapping[str, Any]],
) -> str:
    if heldout_path_failures:
        return "blocked_heldout_leakage_detected"
    if base_seed_status == "missing":
        return "blocked_missing_v7d_base_seed_package"
    if base_seed_status != "passed":
        return "blocked_v7d_base_seed_package_not_passed"
    if paper_review_status != "ready_for_v7e_seed_package_inputs":
        return "blocked_paper_seed_review_required"
    if approve_count < minimum_approved_count:
        return "blocked_paper_seed_review_required"
    return "ready_for_v7e_seed_package_build"


def _planned_commands(
    *,
    status: str,
    base_seed_status: str,
    paper_review_status: str,
    config: V7ESeedPackagePreflightConfig,
    project_root: Path,
) -> dict[str, str]:
    validate_command = "python -m embodied_rps.tools.validate_v7e_paper_seed_review"
    if config.paper_review_validation_root != DEFAULT_PAPER_REVIEW_VALIDATION_ROOT:
        validate_command += f" --output-root {_display_path(_resolve_path(project_root, config.paper_review_validation_root), base=project_root)}"
    if base_seed_status != "passed":
        build_command = "blocked until v7d base seed package summary status is passed"
    elif status == "blocked_heldout_leakage_detected":
        build_command = "blocked until heldout path failures are removed from seed-package preflight inputs"
    elif paper_review_status != "ready_for_v7e_seed_package_inputs":
        build_command = "blocked until paper review status is ready_for_v7e_seed_package_inputs"
    elif status != "ready_for_v7e_seed_package_build":
        build_command = "blocked until v7e seed-package preflight reaches ready_for_v7e_seed_package_build"
    else:
        build_command = "python -m embodied_rps.tools.build_v7e_stage1_paper_transition_rescue_seed_package"
    return {
        "validate_paper_review": validate_command,
        "seed_package_preflight": "python -m embodied_rps.tools.write_v7e_seed_package_preflight",
        "build_seed_package": build_command,
        "generate_stage1_dataset": "not planned until v7e seed package exists",
        "local_smoke_stage1": "not planned until v7e stage1 dataset exists",
        "train_remote_stage1": "not planned until local v7e stage1 dataset and smoke checks pass",
        "strict_original20_validation": "not planned until remote stage1 profile returns locally",
        "heldout15_validation": "not planned unless original20 reaches 20/20",
    }


def _next_action(status: str) -> str:
    if status == "ready_for_v7e_seed_package_build":
        return "build the v7e paper-expanded seed package, then create the stage1-only dataset and local smoke gates"
    if status == "blocked_missing_v7d_base_seed_package":
        return "restore or rebuild the passed v7d base seed package before v7e seed-package work"
    if status == "blocked_v7d_base_seed_package_not_passed":
        return "fix the v7d base seed package status before v7e seed-package work"
    if status == "blocked_heldout_leakage_detected":
        return "remove heldout */test paths from v7e seed-package preflight inputs before continuing"
    return "approve reviewed v7e paper prompt-window seeds, rerun the paper review validator, then rerun this preflight"


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# V7e Seed-Package Preflight",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Branch: `{summary.get('branch_label')}`",
        f"- Base seed package status: `{summary.get('base_seed_package_status')}`",
        f"- Paper review status: `{summary.get('paper_review_status')}`",
        f"- Approved paper seeds: `{summary.get('approved_paper_seed_count')}` / `{summary.get('minimum_approved_paper_seed_count')}`",
        "- Seed package created: `False`",
        "- Dataset generated: `False`",
        "- Training started: `False`",
        "- Heldout15 started: `False`",
        f"- Next action: {summary.get('next_action')}",
        "",
        "## Planned Scope",
        "",
        f"- Stage 1: `{summary.get('stage1_training_scope')}`",
        f"- Stage 2: `{summary.get('stage2_policy')}`",
        "- v4 remains the live/demo fallback until strict promotion gates pass.",
        "",
    ]
    failures = summary.get("heldout_path_failures", [])
    if failures:
        lines.extend(["## Heldout Path Failures", ""])
        for failure in failures:
            if isinstance(failure, Mapping):
                lines.append(f"- `{failure.get('field')}`: `{failure.get('value')}`")
        lines.append("")
    return "\n".join(lines)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object in {path}")
    return dict(value)


def _resolve_summary_path(project_root: Path, root: Path, value: Any, *, fallback: str) -> Path:
    text = str(value or "").strip()
    if not text:
        return root / fallback
    path = Path(text)
    if path.is_absolute():
        return path
    project_relative = project_root / path
    if project_relative.exists():
        return project_relative
    return root / path


def _heldout_path_failures(value: Any, *, prefix: str) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    _walk_for_heldout_paths(value, prefix=prefix, failures=failures)
    return failures


def _walk_for_heldout_paths(value: Any, *, prefix: str, failures: list[dict[str, str]]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _walk_for_heldout_paths(item, prefix=f"{prefix}.{key}", failures=failures)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk_for_heldout_paths(item, prefix=f"{prefix}[{index}]", failures=failures)
        return
    if isinstance(value, str) and _is_heldout_test_path(value):
        failures.append({"field": prefix, "value": value})


def _is_heldout_test_path(value: str) -> bool:
    normalized = value.replace("\\", "/").lower().rstrip("/")
    return "/test/" in normalized or normalized.endswith("/test")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_V7D_SEED_PACKAGE_ROOT",
    "DEFAULT_V7E_SEED_PACKAGE_ROOT",
    "DEFAULT_V7E_STAGE1_DATASET_ROOT",
    "DEFAULT_V7E_STAGE1_RESULTS_ROOT",
    "V7ESeedPackagePreflightConfig",
    "write_v7e_seed_package_preflight",
]
