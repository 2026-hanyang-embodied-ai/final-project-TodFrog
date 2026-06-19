"""Gated v7 post-review seed-package and dataset-generation runner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from embodied_rps.three_class_wait_skeletons import (
    ThreeClassWaitExpansionConfig,
    generate_three_class_wait_dataset,
)
from embodied_rps.v7_rps_seed_package import (
    audit_v7_segment_review_readiness,
    build_v7_rps_seed_package,
)

V7PostReviewStatus = Literal[
    "awaiting_manual_segment_approval",
    "ready_for_v7_dataset_generation",
    "seed_package_build_failed",
    "v7_dataset_generated",
    "dataset_generation_failed",
    "invalid_review_manifest",
]


@dataclass(frozen=True)
class V7PostReviewRunConfig:
    """Configuration for the review-gated v7 post-review runner."""

    review_root: Path = Path("artifacts/real_skeleton_v7_rps_seed_package_20260617")
    dataset_output_root: Path = Path("artifacts/real_guided_three_class_wait_expanded_v7_rps_pose_20260617")
    base_dataset_root: Path = Path("artifacts/real_guided_large_sharded_20260610")
    pipeline_output_root: Path = Path("artifacts/real_skeleton_v7_post_review_run_20260617")
    calibration_seed_package_root: Path | None = Path("artifacts/real_skeleton_v4_calibration_seed_package_fewshot_20260615")
    live_rock_seed_package_root: Path | None = Path("artifacts/live_rock_false_trigger_overlay_seed_20260616")
    generated_per_target: int = 10000
    sequence_length: int = 72
    min_length: int = 48
    shard_size: int = 512
    base_rock_stride: int = 2
    seed: int = 20260617
    overwrite_dataset: bool = False
    dry_run: bool = True


def run_v7_post_review_pipeline(config: V7PostReviewRunConfig) -> dict[str, object]:
    """Run the post-review v7 gates without bypassing manual approval."""

    config.pipeline_output_root.mkdir(parents=True, exist_ok=True)
    review = audit_v7_segment_review_readiness(output_root=config.review_root)
    completed_stages = ["review_readiness_audit"]
    planned_commands = _planned_commands(config)

    if review["status"] == "awaiting_manual_segment_approval":
        summary = _summary(
            config=config,
            status="awaiting_manual_segment_approval",
            review=review,
            completed_stages=completed_stages,
            planned_commands=planned_commands,
            dataset_summary=None,
            seed_package_summary=None,
        )
        _write_outputs(config.pipeline_output_root, summary)
        return summary

    if review["status"] != "ready_for_seed_package_build":
        summary = _summary(
            config=config,
            status="invalid_review_manifest",
            review=review,
            completed_stages=completed_stages,
            planned_commands=planned_commands,
            dataset_summary=None,
            seed_package_summary=None,
        )
        _write_outputs(config.pipeline_output_root, summary)
        return summary

    if config.dry_run:
        summary = _summary(
            config=config,
            status="ready_for_v7_dataset_generation",
            review=review,
            completed_stages=completed_stages,
            planned_commands=planned_commands,
            dataset_summary=None,
            seed_package_summary=None,
        )
        _write_outputs(config.pipeline_output_root, summary)
        return summary

    seed_summary = build_v7_rps_seed_package(output_root=config.review_root, sequence_length=config.sequence_length)
    completed_stages.append("seed_package_build")
    if seed_summary["status"] != "passed":
        summary = _summary(
            config=config,
            status="seed_package_build_failed",
            review=review,
            completed_stages=completed_stages,
            planned_commands=planned_commands,
            dataset_summary=None,
            seed_package_summary=seed_summary,
        )
        _write_outputs(config.pipeline_output_root, summary)
        return summary
    seed_contract = _seed_package_artifact_contract(config.review_root)
    if seed_contract["status"] != "passed":
        seed_summary = dict(seed_summary)
        seed_summary["contract_validation"] = seed_contract
        summary = _summary(
            config=config,
            status="seed_package_build_failed",
            review=review,
            completed_stages=completed_stages,
            planned_commands=planned_commands,
            dataset_summary=None,
            seed_package_summary=seed_summary,
        )
        _write_outputs(config.pipeline_output_root, summary)
        return summary
    seed_summary = dict(seed_summary)
    seed_summary["contract_validation"] = seed_contract

    try:
        dataset_summary = generate_three_class_wait_dataset(
            ThreeClassWaitExpansionConfig(
                output_root=config.dataset_output_root,
                base_dataset_root=config.base_dataset_root,
                generated_per_target=config.generated_per_target,
                sequence_length=config.sequence_length,
                min_length=config.min_length,
                shard_size=config.shard_size,
                seed=config.seed,
                base_rock_stride=config.base_rock_stride,
                augmentation_profile="v7_rps_pose",
                calibration_seed_package_root=config.calibration_seed_package_root,
                live_rock_seed_package_root=config.live_rock_seed_package_root,
                v7_seed_package_root=config.review_root,
                overwrite=config.overwrite_dataset,
            )
        )
    except Exception as exc:
        dataset_summary = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        summary = _summary(
            config=config,
            status="dataset_generation_failed",
            review=review,
            completed_stages=completed_stages,
            planned_commands=planned_commands,
            dataset_summary=dataset_summary,
            seed_package_summary=seed_summary,
        )
        _write_outputs(config.pipeline_output_root, summary)
        return summary

    if dataset_summary["status"] != "passed":
        summary = _summary(
            config=config,
            status="dataset_generation_failed",
            review=review,
            completed_stages=completed_stages,
            planned_commands=planned_commands,
            dataset_summary=dataset_summary,
            seed_package_summary=seed_summary,
        )
        _write_outputs(config.pipeline_output_root, summary)
        return summary

    completed_stages.append("dataset_generation")
    summary = _summary(
        config=config,
        status="v7_dataset_generated",
        review=review,
        completed_stages=completed_stages,
        planned_commands=planned_commands,
        dataset_summary=dataset_summary,
        seed_package_summary=seed_summary,
    )
    _write_outputs(config.pipeline_output_root, summary)
    return summary


def _summary(
    *,
    config: V7PostReviewRunConfig,
    status: V7PostReviewStatus,
    review: dict[str, object],
    completed_stages: list[str],
    planned_commands: dict[str, str],
    dataset_summary: dict[str, object] | None,
    seed_package_summary: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "status": status,
        "dry_run": config.dry_run,
        "completed_stages": completed_stages,
        "review_root": config.review_root.as_posix(),
        "dataset_output_root": config.dataset_output_root.as_posix(),
        "base_dataset_root": config.base_dataset_root.as_posix(),
        "pipeline_output_root": config.pipeline_output_root.as_posix(),
        "review_readiness": review,
        "planned_commands": planned_commands,
        "seed_package_summary": seed_package_summary,
        "dataset_summary": dataset_summary,
        "next_action": _next_action(status),
        "config": _json_ready(asdict(config)),
    }


def _planned_commands(config: V7PostReviewRunConfig) -> dict[str, str]:
    build_cmd = (
        "python -m embodied_rps.tools.build_v7_rps_seed_package "
        f"--output-root {config.review_root.as_posix()} --sequence-length {config.sequence_length}"
    )
    generate_cmd = (
        "python -m embodied_rps.tools.generate_three_class_wait_skeleton_dataset "
        f"--base-dataset-root {config.base_dataset_root.as_posix()} "
        f"--output-root {config.dataset_output_root.as_posix()} "
        f"--generated-per-target {config.generated_per_target} "
        f"--sequence-length {config.sequence_length} "
        f"--min-length {config.min_length} "
        f"--shard-size {config.shard_size} "
        f"--base-rock-stride {config.base_rock_stride} "
        f"--seed {config.seed} "
        "--augmentation-profile v7_rps_pose "
        f"--v7-seed-package-root {config.review_root.as_posix()}"
    )
    if config.calibration_seed_package_root is not None:
        generate_cmd += f" --calibration-seed-package-root {config.calibration_seed_package_root.as_posix()}"
    if config.live_rock_seed_package_root is not None:
        generate_cmd += f" --live-rock-seed-package-root {config.live_rock_seed_package_root.as_posix()}"
    if config.overwrite_dataset:
        generate_cmd += " --overwrite"
    return {
        "audit_review": f"python -m embodied_rps.tools.audit_v7_segment_review_readiness --output-root {config.review_root.as_posix()}",
        "build_seed_package": build_cmd,
        "generate_dataset": generate_cmd,
    }


def _next_action(status: V7PostReviewStatus) -> str:
    if status == "awaiting_manual_segment_approval":
        return "review auto-quality-passed segments and approve selected rows before rerunning this pipeline"
    if status == "ready_for_v7_dataset_generation":
        return "rerun with execution enabled to build the seed package and generate the v7 dataset"
    if status == "v7_dataset_generated":
        return "run local smoke training, then remote TCN ensemble training and strict validation gates"
    if status == "seed_package_build_failed":
        return "repair the reviewed v7 seed package artifacts before generating the expanded v7 dataset"
    if status == "dataset_generation_failed":
        return "fix dataset generation failure while preserving the approved seed package and review diagnostics"
    return "fix review manifest or dataset generation failures"


def _seed_package_artifact_contract(review_root: Path) -> dict[str, object]:
    required_files = {
        "missing_seed_npz": review_root / "v7_rps_seed_dataset.npz",
        "missing_seed_metadata": review_root / "seed_metadata.jsonl",
        "missing_seed_quality_summary": review_root / "seed_quality_summary.csv",
        "missing_seed_package_summary": review_root / "seed_package_summary.json",
    }
    failures = [
        {"code": code, "path": path.as_posix()}
        for code, path in required_files.items()
        if not path.exists()
    ]
    return {
        "status": "failed" if failures else "passed",
        "required_files": {code: path.as_posix() for code, path in required_files.items()},
        "failures": failures,
    }


def _write_outputs(output_root: Path, summary: dict[str, object]) -> None:
    json_path = output_root / "v7_post_review_summary.json"
    md_path = output_root / "v7_post_review_summary.md"
    json_path.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_summary_markdown(summary), encoding="utf-8")


def _summary_markdown(summary: dict[str, object]) -> str:
    review = summary.get("review_readiness")
    review_map = review if isinstance(review, dict) else {}
    lines = [
        "# V7 Post-Review Pipeline Summary",
        "",
        f"- status: `{summary.get('status')}`",
        f"- dry run: `{summary.get('dry_run')}`",
        f"- completed stages: `{summary.get('completed_stages')}`",
        f"- approved segments: `{review_map.get('approved_segment_count')}`",
        f"- seed NPZ exists: `{review_map.get('seed_npz_exists')}`",
        f"- next action: `{summary.get('next_action')}`",
        "",
        "## Planned Commands",
        "",
    ]
    planned = summary.get("planned_commands")
    if isinstance(planned, dict):
        for name, command in planned.items():
            lines.extend([f"### {name}", "", "```text", str(command), "```", ""])
    return "\n".join(lines)


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["V7PostReviewRunConfig", "run_v7_post_review_pipeline"]
