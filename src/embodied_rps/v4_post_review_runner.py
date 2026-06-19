"""Post-review runner for v4 seed-package and dataset-generation readiness."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.three_class_wait_skeletons import ThreeClassWaitExpansionConfig, generate_three_class_wait_dataset
from embodied_rps.v4_calibration_intake import build_v4_dataset_generation_plan
from embodied_rps.v4_calibration_seed_package import V4SeedPackageConfig, build_v4_calibration_seed_package


@dataclass(frozen=True)
class V4PostReviewRunConfig:
    """Configuration for advancing v4 after skeleton-review artifact approval."""

    skeleton_review_plan_path: Path
    review_manifest_path: Path
    seed_package_root: Path
    dataset_plan_output_root: Path
    dataset_output_root: Path
    base_dataset_root: Path
    pipeline_output_root: Path
    min_detection_coverage: float = 0.98
    generated_per_target: int = 10000
    augmentation_profile: str = "v3_targeted"
    seed: int = 20260611
    overwrite_dataset: bool = False
    dry_run: bool = True


def run_v4_post_review_pipeline(config: V4PostReviewRunConfig) -> dict[str, object]:
    """Advance seed-package and dataset-generation gates after v4 skeleton review."""

    config.pipeline_output_root.mkdir(parents=True, exist_ok=True)
    completed_stages: list[str] = []
    stage_outputs: dict[str, object] = {}

    if not config.skeleton_review_plan_path.exists():
        stage_outputs["v4_dataset_generation_readiness"] = {
            "status": "awaiting_skeleton_review_plan",
            "failure_count": 1,
            "failures": [
                {
                    "code": "missing_skeleton_review_plan",
                    "path": config.skeleton_review_plan_path.as_posix(),
                }
            ],
        }
        summary = _summary(
            config=config,
            status="awaiting_skeleton_review_plan",
            next_action="run_v4_pipeline_or_prepare_v4_skeleton_review_plan",
            blocking_stage="skeleton_review_plan",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
        )
        _write_summary(config.pipeline_output_root, summary)
        return summary

    dataset_plan = build_v4_dataset_generation_plan(
        skeleton_review_plan_path=config.skeleton_review_plan_path,
        output_root=config.dataset_plan_output_root,
        dataset_output_root=config.dataset_output_root,
        base_dataset_root=config.base_dataset_root,
        calibration_seed_package_root=config.seed_package_root,
        review_manifest_path=config.review_manifest_path,
        min_detection_coverage=config.min_detection_coverage,
    )
    stage_outputs["v4_dataset_generation_readiness"] = dataset_plan
    if dataset_plan["status"] != "ready_for_v4_dataset_generation":
        summary = _summary(
            config=config,
            status="awaiting_approved_skeleton_review",
            next_action="run_or_approve_v4_skeleton_review",
            blocking_stage="skeleton_review_manifest",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
        )
        _write_summary(config.pipeline_output_root, summary)
        return summary

    completed_stages.append("v4_dataset_generation_readiness")
    seed_summary = build_v4_calibration_seed_package(
        V4SeedPackageConfig(
            review_manifest_path=config.review_manifest_path,
            skeleton_review_plan_path=config.skeleton_review_plan_path,
            output_root=config.seed_package_root,
            min_detection_coverage=config.min_detection_coverage,
        )
    )
    stage_outputs["v4_calibration_seed_package"] = _compact_seed_summary(seed_summary)
    if seed_summary["status"] != "passed":
        summary = _summary(
            config=config,
            status="seed_package_failed",
            next_action="inspect_v4_seed_package_failures",
            blocking_stage="v4_calibration_seed_package",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
        )
        _write_summary(config.pipeline_output_root, summary)
        return summary

    completed_stages.append("v4_calibration_seed_package")
    if config.dry_run:
        summary = _summary(
            config=config,
            status="ready_for_v4_dataset_generation",
            next_action="rerun_without_dry_run_to_generate_v4_dataset",
            blocking_stage="manual_dataset_generation",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
        )
        _write_summary(config.pipeline_output_root, summary)
        return summary

    dataset_summary = generate_three_class_wait_dataset(
        ThreeClassWaitExpansionConfig(
            output_root=config.dataset_output_root,
            base_dataset_root=config.base_dataset_root,
            generated_per_target=config.generated_per_target,
            seed=config.seed,
            augmentation_profile=config.augmentation_profile,
            calibration_seed_package_root=config.seed_package_root,
            overwrite=config.overwrite_dataset,
        )
    )
    stage_outputs["v4_dataset_generation"] = _compact_dataset_summary(dataset_summary)
    if dataset_summary["status"] == "passed":
        completed_stages.append("v4_dataset_generation")
        status = "v4_dataset_generated"
        next_action = "train_v4_gru_tcn_then_run_strict_video_gates"
        blocking_stage = "v4_model_training"
    else:
        status = "v4_dataset_generation_failed"
        next_action = "inspect_v4_dataset_generation_failures"
        blocking_stage = "v4_dataset_generation"

    summary = _summary(
        config=config,
        status=status,
        next_action=next_action,
        blocking_stage=blocking_stage,
        completed_stages=completed_stages,
        stage_outputs=stage_outputs,
    )
    _write_summary(config.pipeline_output_root, summary)
    return summary


def _summary(
    *,
    config: V4PostReviewRunConfig,
    status: str,
    next_action: str,
    blocking_stage: str,
    completed_stages: list[str],
    stage_outputs: Mapping[str, object],
) -> dict[str, object]:
    return {
        "status": status,
        "next_action": next_action,
        "blocking_stage": blocking_stage,
        "dry_run": config.dry_run,
        "skeleton_review_plan": config.skeleton_review_plan_path.as_posix(),
        "review_manifest": config.review_manifest_path.as_posix(),
        "seed_package_root": config.seed_package_root.as_posix(),
        "dataset_plan_output_root": config.dataset_plan_output_root.as_posix(),
        "dataset_output_root": config.dataset_output_root.as_posix(),
        "base_dataset_root": config.base_dataset_root.as_posix(),
        "completed_stages": completed_stages,
        "stage_outputs": dict(stage_outputs),
        "notes": [
            "This runner starts after v4 skeleton-review artifacts exist.",
            "It does not train a model.",
            "It does not run SCHUNK or Isaac rendering.",
        ],
    }


def _compact_seed_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": summary.get("status"),
        "sample_count": summary.get("sample_count"),
        "target_counts": summary.get("target_counts"),
        "seed_npz": summary.get("seed_npz"),
        "validation": summary.get("validation"),
        "failures": summary.get("failures"),
    }


def _compact_dataset_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": summary.get("status"),
        "output_root": summary.get("output_root"),
        "sample_count": summary.get("sample_count"),
        "split_counts": summary.get("split_counts"),
        "target_counts": summary.get("target_counts"),
        "validation": summary.get("validation"),
    }


def _write_summary(output_root: Path, summary: Mapping[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "post_review_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    (output_root / "post_review_run_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")


def _summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V4 Post-Review Run Summary",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Next action: `{summary.get('next_action')}`",
        f"- Blocking stage: `{summary.get('blocking_stage')}`",
        f"- Dry run: `{summary.get('dry_run')}`",
        f"- Review manifest: `{summary.get('review_manifest')}`",
        f"- Seed package root: `{summary.get('seed_package_root')}`",
        f"- Dataset output root: `{summary.get('dataset_output_root')}`",
        "",
        "## Completed Stages",
        "",
    ]
    completed = summary.get("completed_stages")
    if isinstance(completed, list) and completed:
        for stage in completed:
            lines.append(f"- `{stage}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Stage Outputs", ""])
    stage_outputs = summary.get("stage_outputs")
    if isinstance(stage_outputs, Mapping) and stage_outputs:
        for stage, output in stage_outputs.items():
            if isinstance(output, Mapping):
                status = output.get("status")
                details = [f"status=`{status}`"]
                if output.get("failure_count") is not None:
                    details.append(f"failure_count=`{output.get('failure_count')}`")
                if output.get("sample_count") is not None:
                    details.append(f"sample_count=`{output.get('sample_count')}`")
                lines.append(f"- `{stage}`: {', '.join(details)}")
                failures = output.get("failures")
                if isinstance(failures, list) and failures:
                    for failure in failures:
                        if isinstance(failure, Mapping):
                            lines.append(f"  - failure `{failure.get('code')}`")
            else:
                lines.append(f"- `{stage}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Notes", ""])
    notes = summary.get("notes")
    if isinstance(notes, list):
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _json_default(value: object) -> object:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


__all__ = ["V4PostReviewRunConfig", "run_v4_post_review_pipeline"]
