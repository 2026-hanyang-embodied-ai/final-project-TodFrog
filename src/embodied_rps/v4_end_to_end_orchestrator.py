"""Safe end-to-end orchestration for the v4 skeleton prediction pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.v4_pipeline_runner import V4PipelineRunConfig, run_v4_pipeline
from embodied_rps.v4_post_review_runner import V4PostReviewRunConfig, run_v4_post_review_pipeline
from embodied_rps.v4_skeleton_review_executor import V4SkeletonReviewExecutionConfig, run_v4_skeleton_review_from_plan
from embodied_rps.v4_training_gate_runner import V4TrainingGateConfig, run_v4_training_gate


@dataclass(frozen=True)
class V4EndToEndConfig:
    """Configuration for one safe v4 pipeline sweep."""

    calibration_input_root: Path
    heldout_root: Path
    expected_min_per_label: int
    base_dataset_root: Path
    training_config_path: Path
    output_root: Path
    original20_root: Path
    execute_skeleton_review: bool = False
    execute_dataset_generation: bool = False
    overwrite_dataset: bool = False
    intake_output_root: Path = Path("artifacts/real_skeleton_v4_calibration_intake_20260611")
    skeleton_review_plan_output_root: Path = Path("artifacts/real_skeleton_v4_calibration_review_plan_20260611")
    skeleton_review_output_root: Path = Path("artifacts/real_hand_skeleton_review_v4_calibration_20260611")
    skeleton_review_execution_output_root: Path = Path("artifacts/real_skeleton_v4_review_execution_20260612")
    seed_package_root: Path = Path("artifacts/real_skeleton_v4_calibration_seed_package_20260612")
    recording_slot_audit_output_root: Path = Path("artifacts/real_skeleton_v4_recording_slot_audit_20260612")
    mp4_preflight_output_root: Path = Path("artifacts/real_skeleton_v4_mp4_preflight_20260612")
    dataset_generation_plan_output_root: Path = Path("artifacts/real_skeleton_v4_dataset_generation_plan_20260611")
    dataset_output_root: Path = Path("artifacts/real_guided_three_class_wait_expanded_v4_20260611")
    pipeline_output_root: Path = Path("artifacts/real_skeleton_v4_pipeline_run_20260612")
    post_review_output_root: Path = Path("artifacts/real_skeleton_v4_post_review_run_20260612")
    training_gate_output_root: Path = Path("artifacts/real_skeleton_v4_training_gate_20260612")
    profile_json_path: Path = Path("results/model_profiles/real_skeleton_three_class_wait_v4.json")
    original20_validation_root: Path = Path("artifacts/real_mp4_prediction_validation_original20_v4_20260612")
    heldout15_validation_root: Path = Path("artifacts/real_mp4_prediction_validation_new15_v4_20260612")
    event_manifest_path: Path = Path("artifacts/real_skeleton_schunk_events_v4_20260612/events.jsonl")
    v3_summary_path: Path | None = Path("artifacts/real_guided_three_class_wait_expanded_v3_20260611/training_and_validation_summary.json")
    recording_ingest_summary_path: Path | None = None


def run_v4_end_to_end(config: V4EndToEndConfig) -> dict[str, object]:
    """Run all safe v4 gates and summarize the first blocking stage."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    stage_outputs: dict[str, object] = {}

    pipeline_summary = run_v4_pipeline(
        V4PipelineRunConfig(
            calibration_input_root=config.calibration_input_root,
            heldout_roots=(config.heldout_root,),
            expected_min_per_label=config.expected_min_per_label,
            intake_output_root=config.intake_output_root,
            skeleton_review_plan_output_root=config.skeleton_review_plan_output_root,
            skeleton_review_output_root=config.skeleton_review_output_root,
            seed_package_root=config.seed_package_root,
            recording_slot_audit_output_root=config.recording_slot_audit_output_root,
            mp4_preflight_output_root=config.mp4_preflight_output_root,
            dataset_generation_plan_output_root=config.dataset_generation_plan_output_root,
            dataset_output_root=config.dataset_output_root,
            base_dataset_root=config.base_dataset_root,
            training_config_path=config.training_config_path,
            pipeline_output_root=config.pipeline_output_root,
            v3_summary_path=config.v3_summary_path,
            recording_ingest_summary_path=config.recording_ingest_summary_path,
        )
    )
    stage_outputs["metadata_pipeline"] = _compact_stage(pipeline_summary)

    review_execution_summary = run_v4_skeleton_review_from_plan(
        V4SkeletonReviewExecutionConfig(
            skeleton_review_plan_path=config.skeleton_review_plan_output_root / "skeleton_review_plan.json",
            output_root=config.skeleton_review_execution_output_root,
            dry_run=not config.execute_skeleton_review,
        )
    )
    stage_outputs["skeleton_review_execution"] = _compact_stage(review_execution_summary)

    post_review_summary = run_v4_post_review_pipeline(
        V4PostReviewRunConfig(
            skeleton_review_plan_path=config.skeleton_review_plan_output_root / "skeleton_review_plan.json",
            review_manifest_path=config.skeleton_review_output_root / "manifest.json",
            seed_package_root=config.seed_package_root,
            dataset_plan_output_root=config.dataset_generation_plan_output_root,
            dataset_output_root=config.dataset_output_root,
            base_dataset_root=config.base_dataset_root,
            pipeline_output_root=config.post_review_output_root,
            dry_run=not config.execute_dataset_generation,
            overwrite_dataset=config.overwrite_dataset,
        )
    )
    stage_outputs["post_review"] = _compact_stage(post_review_summary)

    training_gate_summary = run_v4_training_gate(
        V4TrainingGateConfig(
            dataset_root=config.dataset_output_root,
            training_config_path=config.training_config_path,
            output_root=config.training_gate_output_root,
            original20_root=config.original20_root,
            heldout15_root=config.heldout_root,
            profile_json_path=config.profile_json_path,
            original20_validation_root=config.original20_validation_root,
            heldout15_validation_root=config.heldout15_validation_root,
            event_manifest_path=config.event_manifest_path,
        )
    )
    stage_outputs["training_gate"] = _compact_stage(training_gate_summary)

    first_blocker = _first_blocker(stage_outputs)
    summary = {
        "status": "strict_gates_passed" if stage_outputs["training_gate"]["status"] == "strict_gates_passed" else "blocked_at_current_gate",
        "current_gate": first_blocker["stage"],
        "blocking_stage": first_blocker["blocking_stage"],
        "next_action": first_blocker["next_action"],
        "execute_skeleton_review": config.execute_skeleton_review,
        "execute_dataset_generation": config.execute_dataset_generation,
        "calibration_input_root": config.calibration_input_root.as_posix(),
        "heldout_root": config.heldout_root.as_posix(),
        "stage_outputs": stage_outputs,
        "notes": [
            "Default mode does not run MediaPipe extraction, dataset generation, model training, SCHUNK, or Isaac rendering.",
            "Use execute_skeleton_review only after calibration preflight and intake are ready.",
            "Use execute_dataset_generation only after visual skeleton approval.",
        ],
    }
    _write_summary(config.output_root, summary)
    return summary


def _compact_stage(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": summary.get("status"),
        "next_action": summary.get("next_action"),
        "blocking_stage": summary.get("blocking_stage"),
        "completed_stages": summary.get("completed_stages", []),
        "stage_outputs": summary.get("stage_outputs", {}),
    }


def _first_blocker(stage_outputs: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    for stage in ("metadata_pipeline", "skeleton_review_execution", "post_review", "training_gate"):
        output = stage_outputs[stage]
        status = output.get("status")
        if status in {"strict_gates_passed", "skeleton_review_passed", "v4_dataset_generated"}:
            continue
        return {
            "stage": stage,
            "blocking_stage": output.get("blocking_stage"),
            "next_action": output.get("next_action"),
        }
    return {"stage": None, "blocking_stage": None, "next_action": "connect_schunk_response_evaluation"}


def _write_summary(output_root: Path, summary: Mapping[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "end_to_end_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "end_to_end_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")


def _summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V4 End-to-End Orchestration Summary",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Current gate: `{summary.get('current_gate')}`",
        f"- Blocking stage: `{summary.get('blocking_stage')}`",
        f"- Next action: `{summary.get('next_action')}`",
        f"- Calibration input root: `{summary.get('calibration_input_root')}`",
        f"- Held-out root: `{summary.get('heldout_root')}`",
        f"- Execute skeleton review: `{summary.get('execute_skeleton_review')}`",
        f"- Execute dataset generation: `{summary.get('execute_dataset_generation')}`",
        "",
        "## Stages",
        "",
    ]
    stage_outputs = summary.get("stage_outputs")
    if isinstance(stage_outputs, Mapping):
        for stage, output in stage_outputs.items():
            if isinstance(output, Mapping):
                lines.append(
                    f"- `{stage}`: status=`{output.get('status')}`, "
                    f"blocking_stage=`{output.get('blocking_stage')}`, next_action=`{output.get('next_action')}`"
                )
    lines.extend(["", "## Notes", ""])
    notes = summary.get("notes")
    if isinstance(notes, list):
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


__all__ = ["V4EndToEndConfig", "run_v4_end_to_end"]
