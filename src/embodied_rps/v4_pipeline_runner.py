"""Metadata-safe orchestration for the v4 calibration pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.v4_calibration_intake import (
    build_v4_calibration_intake_report,
    build_v4_dataset_generation_plan,
    build_v4_skeleton_review_plan,
)
from embodied_rps.v4_mp4_preflight import V4Mp4PreflightConfig, VideoProbe, audit_v4_calibration_mp4s
from embodied_rps.v4_pipeline_status import V4PipelineStatusConfig, build_v4_pipeline_status
from embodied_rps.v4_recording_slot_audit import V4RecordingSlotAuditConfig, audit_v4_recording_slots


@dataclass(frozen=True)
class V4PipelineRunConfig:
    """Configuration for a metadata-safe v4 pipeline advance."""

    calibration_input_root: Path
    heldout_roots: tuple[Path, ...]
    expected_min_per_label: int
    intake_output_root: Path
    skeleton_review_plan_output_root: Path
    skeleton_review_output_root: Path
    seed_package_root: Path
    recording_slot_audit_output_root: Path
    mp4_preflight_output_root: Path
    dataset_generation_plan_output_root: Path
    dataset_output_root: Path
    base_dataset_root: Path
    training_config_path: Path
    pipeline_output_root: Path
    v3_summary_path: Path | None = None
    recording_ingest_summary_path: Path | None = None
    min_detection_coverage: float = 0.98
    min_frame_count: int = 5
    min_fps: float = 1.0
    min_width: int = 1
    min_height: int = 1


def run_v4_pipeline(config: V4PipelineRunConfig, *, video_probe: VideoProbe | None = None) -> dict[str, object]:
    """Advance v4 metadata gates until the next data/approval boundary."""

    config.pipeline_output_root.mkdir(parents=True, exist_ok=True)
    completed_stages: list[str] = []
    stage_outputs: dict[str, object] = {}
    initial_status = _status(config)

    slot_audit_summary = _run_slot_audit(config)
    stage_outputs["v4_recording_slot_audit"] = _slot_audit_stage_summary(slot_audit_summary)
    if slot_audit_summary["status"] == "ready_for_mp4_preflight":
        completed_stages.append("v4_recording_slot_audit")
    else:
        summary = _summary_from_status(config, initial_status, completed_stages=completed_stages, stage_outputs=stage_outputs)
        _write_run_summary(config.pipeline_output_root, summary)
        return summary

    calibration_input = initial_status.get("calibration_input")
    if not isinstance(calibration_input, dict) or calibration_input.get("status") != "ready_for_skeleton_review":
        preflight_summary = _run_preflight(config, video_probe=video_probe)
        stage_outputs["v4_mp4_preflight"] = _preflight_stage_summary(preflight_summary)
        summary = _summary_from_status(config, initial_status, completed_stages=completed_stages, stage_outputs=stage_outputs)
        _write_run_summary(config.pipeline_output_root, summary)
        return summary

    preflight_summary = _run_preflight(config, video_probe=video_probe)
    stage_outputs["v4_mp4_preflight"] = _preflight_stage_summary(preflight_summary)
    if preflight_summary["status"] == "passed":
        completed_stages.append("v4_mp4_preflight")
    else:
        summary = _summary_from_status(config, initial_status, completed_stages=completed_stages, stage_outputs=stage_outputs)
        _write_run_summary(config.pipeline_output_root, summary)
        return summary

    intake_summary = build_v4_calibration_intake_report(
        input_root=config.calibration_input_root,
        output_root=config.intake_output_root,
        heldout_roots=config.heldout_roots,
        v3_summary_path=config.v3_summary_path,
        expected_min_per_label=config.expected_min_per_label,
        allow_missing_input=False,
    )
    stage_outputs["v4_calibration_intake"] = intake_summary
    if intake_summary["status"] == "ready_for_skeleton_review":
        completed_stages.append("v4_calibration_intake")
    else:
        summary = _summary_from_status(config, _status(config), completed_stages=completed_stages, stage_outputs=stage_outputs)
        _write_run_summary(config.pipeline_output_root, summary)
        return summary

    review_plan_summary = build_v4_skeleton_review_plan(
        intake_manifest_path=Path(str(intake_summary["intake_manifest"])),
        output_root=config.skeleton_review_plan_output_root,
        review_output_root=config.skeleton_review_output_root,
    )
    stage_outputs["v4_skeleton_review_plan"] = review_plan_summary
    if review_plan_summary["status"] == "ready_for_skeleton_review":
        completed_stages.append("v4_skeleton_review_plan")

    dataset_plan_summary = build_v4_dataset_generation_plan(
        skeleton_review_plan_path=Path(str(review_plan_summary["skeleton_review_plan"])),
        output_root=config.dataset_generation_plan_output_root,
        dataset_output_root=config.dataset_output_root,
        base_dataset_root=config.base_dataset_root,
        calibration_seed_package_root=config.seed_package_root,
        min_detection_coverage=config.min_detection_coverage,
    )
    stage_outputs["v4_dataset_generation_readiness"] = dataset_plan_summary
    if dataset_plan_summary["status"] in {"ready_for_v4_dataset_generation", "awaiting_skeleton_review"}:
        completed_stages.append("v4_dataset_generation_readiness")

    summary = _summary_from_status(config, _status(config), completed_stages=completed_stages, stage_outputs=stage_outputs)
    _write_run_summary(config.pipeline_output_root, summary)
    return summary


def _run_slot_audit(config: V4PipelineRunConfig) -> dict[str, object]:
    return audit_v4_recording_slots(
        V4RecordingSlotAuditConfig(
            calibration_root=config.calibration_input_root,
            output_root=config.recording_slot_audit_output_root,
        )
    )


def _slot_audit_stage_summary(slot_audit_summary: dict[str, object]) -> dict[str, object]:
    return {
        "status": slot_audit_summary["status"],
        "slot_count": slot_audit_summary["slot_count"],
        "filled_slot_count": slot_audit_summary["filled_slot_count"],
        "missing_slot_count": slot_audit_summary["missing_slot_count"],
        "extra_mp4_count": slot_audit_summary["extra_mp4_count"],
        "audit_table": slot_audit_summary.get("audit_table"),
    }


def _run_preflight(config: V4PipelineRunConfig, *, video_probe: VideoProbe | None) -> dict[str, object]:
    return audit_v4_calibration_mp4s(
        V4Mp4PreflightConfig(
            input_root=config.calibration_input_root,
            heldout_roots=config.heldout_roots,
            output_root=config.mp4_preflight_output_root,
            expected_min_per_label=config.expected_min_per_label,
            min_frame_count=config.min_frame_count,
            min_fps=config.min_fps,
            min_width=config.min_width,
            min_height=config.min_height,
        ),
        video_probe=video_probe,
    )


def _preflight_stage_summary(preflight_summary: dict[str, object]) -> dict[str, object]:
    return {
        "status": preflight_summary["status"],
        "video_count": preflight_summary["video_count"],
        "label_counts": preflight_summary["label_counts"],
        "failed_video_count": preflight_summary["failed_video_count"],
        "preflight_table": preflight_summary["preflight_table"],
    }


def _status(config: V4PipelineRunConfig) -> dict[str, object]:
    return build_v4_pipeline_status(
        V4PipelineStatusConfig(
            calibration_input_root=config.calibration_input_root,
            heldout_roots=config.heldout_roots,
            expected_min_per_label=config.expected_min_per_label,
            intake_manifest_path=config.intake_output_root / "intake_manifest.json",
            skeleton_review_plan_path=config.skeleton_review_plan_output_root / "skeleton_review_plan.json",
            skeleton_review_manifest_path=config.skeleton_review_output_root / "manifest.json",
            seed_package_root=config.seed_package_root,
            dataset_generation_plan_path=config.dataset_generation_plan_output_root / "dataset_generation_plan.json",
            dataset_root=config.dataset_output_root,
            training_config_path=config.training_config_path,
            recording_ingest_summary_path=config.recording_ingest_summary_path,
            output_root=config.pipeline_output_root,
        )
    )


def _summary_from_status(
    config: V4PipelineRunConfig,
    status: dict[str, object],
    *,
    completed_stages: list[str],
    stage_outputs: dict[str, object],
) -> dict[str, object]:
    return {
        "status": status["status"],
        "next_action": status["next_action"],
        "blocking_stage": status["blocking_stage"],
        "calibration_input_root": config.calibration_input_root.as_posix(),
        "completed_stages": completed_stages,
        "stage_outputs": stage_outputs,
        "pipeline_status": (config.pipeline_output_root / "pipeline_status.json").as_posix(),
        "notes": [
            "This runner does not run MediaPipe skeleton extraction.",
            "This runner does not build a seed package before visual skeleton approval.",
            "This runner does not train a model or launch SCHUNK/Isaac rendering.",
        ],
    }


def _write_run_summary(output_root: Path, summary: dict[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "pipeline_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "pipeline_run_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")


def _summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# V4 Pipeline Run Summary",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Next action: `{summary.get('next_action')}`",
        f"- Blocking stage: `{summary.get('blocking_stage')}`",
        f"- Calibration input root: `{summary.get('calibration_input_root')}`",
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
    if isinstance(stage_outputs, dict) and stage_outputs:
        for stage_name, output in stage_outputs.items():
            if isinstance(output, dict):
                details: list[str] = []
                for key in (
                    "status",
                    "video_count",
                    "failed_video_count",
                    "slot_count",
                    "filled_slot_count",
                    "missing_slot_count",
                    "extra_mp4_count",
                    "preflight_table",
                    "audit_table",
                ):
                    if key in output:
                        details.append(f"{key}=`{output[key]}`")
                label_counts = output.get("label_counts")
                if isinstance(label_counts, dict):
                    details.append(f"label_counts=`{json.dumps(label_counts, ensure_ascii=False, sort_keys=True)}`")
                if details:
                    lines.append(f"- `{stage_name}`: {', '.join(details)}")
                else:
                    lines.append(f"- `{stage_name}`")
            else:
                lines.append(f"- `{stage_name}`: `{output}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Notes", ""])
    notes = summary.get("notes")
    if isinstance(notes, list):
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


__all__ = ["V4PipelineRunConfig", "run_v4_pipeline"]
