"""Status reporting for the v4 real-calibration skeleton pipeline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER
from embodied_rps.v4_calibration_intake import discover_calibration_videos, validate_calibration_discovery


@dataclass(frozen=True)
class V4PipelineStatusConfig:
    """Inputs for the v4 pipeline status reporter."""

    calibration_input_root: Path
    heldout_roots: tuple[Path, ...]
    expected_min_per_label: int
    intake_manifest_path: Path
    skeleton_review_plan_path: Path
    skeleton_review_manifest_path: Path
    seed_package_root: Path
    dataset_generation_plan_path: Path
    dataset_root: Path
    training_config_path: Path
    output_root: Path | None = None
    recording_ingest_summary_path: Path | None = None


def build_v4_pipeline_status(config: V4PipelineStatusConfig) -> dict[str, object]:
    """Build and optionally write a current v4 pipeline status report."""

    calibration_input = _calibration_input_status(config)
    intake_manifest = _json_file_status(config.intake_manifest_path)
    skeleton_review_plan = _json_file_status(config.skeleton_review_plan_path)
    skeleton_review_manifest = _json_file_status(config.skeleton_review_manifest_path)
    seed_package = _seed_package_status(config.seed_package_root)
    dataset_generation_plan = _json_file_status(config.dataset_generation_plan_path)
    dataset = _dataset_status(config.dataset_root)
    recording_ingest = _recording_ingest_status(config.recording_ingest_summary_path)
    training = {
        "config_path": config.training_config_path.as_posix(),
        "config_exists": config.training_config_path.exists(),
    }
    decision = _decide_next_action(
        calibration_input=calibration_input,
        recording_ingest=recording_ingest,
        intake_manifest=intake_manifest,
        skeleton_review_plan=skeleton_review_plan,
        skeleton_review_manifest=skeleton_review_manifest,
        seed_package=seed_package,
        dataset_generation_plan=dataset_generation_plan,
        dataset=dataset,
        training=training,
    )
    status = {
        **decision,
        "calibration_input": calibration_input,
        "recording_ingest": recording_ingest,
        "intake_manifest": intake_manifest,
        "skeleton_review_plan": skeleton_review_plan,
        "skeleton_review_manifest": skeleton_review_manifest,
        "seed_package": seed_package,
        "dataset_generation_plan": dataset_generation_plan,
        "dataset": dataset,
        "training": training,
        "strict_gate": {
            "original_20_required_passes": "20/20",
            "heldout_15_required_paper_scissors": "10/10",
            "heldout_15_required_rock_wait": "5/5",
            "schunk_state": "blocked_until_strict_gates_pass",
        },
    }
    if config.output_root is not None:
        _write_status(config.output_root, status)
    return status


def _calibration_input_status(config: V4PipelineStatusConfig) -> dict[str, object]:
    if not config.calibration_input_root.exists():
        return {
            "path": config.calibration_input_root.as_posix(),
            "exists": False,
            "status": "awaiting_calibration_videos",
            "label_counts": {},
            "expected_min_per_label": config.expected_min_per_label,
            "missing_or_low_labels": {label: {"actual": 0, "minimum": config.expected_min_per_label} for label in REVIEW_LABEL_ORDER},
        }
    try:
        videos = discover_calibration_videos(config.calibration_input_root, heldout_roots=config.heldout_roots)
        validation = validate_calibration_discovery(videos, expected_min_per_label=config.expected_min_per_label)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "path": config.calibration_input_root.as_posix(),
            "exists": True,
            "status": "invalid_calibration_input",
            "error": str(exc),
            "label_counts": {},
            "expected_min_per_label": config.expected_min_per_label,
            "missing_or_low_labels": {},
        }
    return {
        "path": config.calibration_input_root.as_posix(),
        "exists": True,
        "status": validation["status"],
        "passed": bool(validation["passed"]),
        "video_count": int(validation["video_count"]),
        "label_counts": dict(validation["label_counts"]),
        "expected_min_per_label": int(validation["expected_min_per_label"]),
        "duplicate_count": int(validation["duplicate_count"]),
        "missing_or_low_labels": dict(validation["missing_or_low_labels"]),
    }


def _json_file_status(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": path.as_posix(), "exists": False, "status": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"path": path.as_posix(), "exists": True, "status": "invalid_json", "error": str(exc)}
    return {
        "path": path.as_posix(),
        "exists": True,
        "status": str(payload.get("status", "unknown")),
        "failure_codes": _failure_codes(payload),
    }


def _recording_ingest_status(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"path": None, "exists": False, "status": "not_configured"}
    base = _json_file_status(path)
    if not bool(base.get("exists")) or base.get("status") == "invalid_json":
        return base
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"path": path.as_posix(), "exists": True, "status": "invalid_payload"}
    gate_decision = payload.get("gate_decision")
    assignment = payload.get("assignment")
    staging_audit = payload.get("staging_audit")
    return {
        "path": path.as_posix(),
        "exists": True,
        "status": str(payload.get("status", "unknown")),
        "next_action": payload.get("next_action"),
        "gate_decision": gate_decision if isinstance(gate_decision, dict) else {},
        "assignment_status": assignment.get("status") if isinstance(assignment, dict) else None,
        "staging_audit_status": staging_audit.get("status") if isinstance(staging_audit, dict) else None,
    }


def _seed_package_status(root: Path) -> dict[str, object]:
    summary_path = root / "seed_package_summary.json"
    seed_npz = root / "v4_calibration_seed_dataset.npz"
    summary = _json_file_status(summary_path)
    return {
        "root": root.as_posix(),
        "summary": summary,
        "summary_status": summary["status"],
        "seed_npz": seed_npz.as_posix(),
        "seed_npz_exists": seed_npz.exists(),
    }


def _dataset_status(root: Path) -> dict[str, object]:
    summary_path = root / "run_summary.json"
    summary = _json_file_status(summary_path)
    return {
        "root": root.as_posix(),
        "exists": root.exists(),
        "run_summary": summary,
        "run_summary_status": summary["status"],
    }


def _decide_next_action(
    *,
    calibration_input: dict[str, object],
    recording_ingest: dict[str, object],
    intake_manifest: dict[str, object],
    skeleton_review_plan: dict[str, object],
    skeleton_review_manifest: dict[str, object],
    seed_package: dict[str, object],
    dataset_generation_plan: dict[str, object],
    dataset: dict[str, object],
    training: dict[str, object],
) -> dict[str, object]:
    calibration_status = str(calibration_input["status"])
    if calibration_status != "ready_for_skeleton_review":
        ingest_status = str(recording_ingest.get("status"))
        if ingest_status not in {"not_configured", "missing", "invalid_json", "invalid_payload"}:
            next_action = str(recording_ingest.get("next_action") or "record_or_add_mp4s_to_v4_recording_staging")
            return _decision("awaiting_recording_ingest", next_action, "recording_ingest")
        return _decision("awaiting_calibration_videos", "record_or_add_v4_calibration_videos", "calibration_input")
    if intake_manifest["status"] != "ready_for_skeleton_review":
        return _decision("awaiting_intake_manifest", "run_prepare_v4_calibration_intake", "intake_manifest")
    if skeleton_review_plan["status"] != "ready_for_skeleton_review":
        return _decision("awaiting_skeleton_review_plan", "run_prepare_v4_skeleton_review_plan", "skeleton_review_plan")
    if skeleton_review_manifest["status"] != "passed":
        return _decision("awaiting_skeleton_review", "run_v4_skeleton_review_and_visual_approval", "skeleton_review_manifest")
    if seed_package["summary_status"] != "passed" or not bool(seed_package["seed_npz_exists"]):
        return _decision("awaiting_seed_package", "build_v4_calibration_seed_package", "seed_package")
    if dataset_generation_plan["status"] != "ready_for_v4_dataset_generation":
        return _decision("awaiting_dataset_generation_readiness", "run_v4_dataset_generation_readiness", "dataset_generation_plan")
    if dataset["run_summary_status"] != "passed":
        return _decision("awaiting_v4_dataset", "generate_v4_skeleton_dataset", "dataset")
    if not bool(training["config_exists"]):
        return _decision("awaiting_training_config", "create_v4_training_config", "training")
    return _decision("ready_for_v4_training", "train_and_validate_v4_models", None)


def _decision(status: str, next_action: str, blocking_stage: str | None) -> dict[str, object]:
    return {"status": status, "next_action": next_action, "blocking_stage": blocking_stage}


def _failure_codes(payload: dict[str, Any]) -> list[str]:
    failures = payload.get("failures")
    if not isinstance(failures, Sequence) or isinstance(failures, (str, bytes)):
        return []
    codes: list[str] = []
    for failure in failures:
        if isinstance(failure, dict) and "code" in failure:
            codes.append(str(failure["code"]))
    return codes


def _write_status(output_root: Path, status: dict[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "pipeline_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "pipeline_status.md").write_text(_status_markdown(status), encoding="utf-8")


def _status_markdown(status: dict[str, object]) -> str:
    calibration = status["calibration_input"] if isinstance(status["calibration_input"], dict) else {}
    recording_ingest = status["recording_ingest"] if isinstance(status["recording_ingest"], dict) else {}
    gate_decision = recording_ingest.get("gate_decision") if isinstance(recording_ingest.get("gate_decision"), dict) else {}
    seed = status["seed_package"] if isinstance(status["seed_package"], dict) else {}
    dataset = status["dataset"] if isinstance(status["dataset"], dict) else {}
    lines = [
        "# V4 Pipeline Status",
        "",
        "## Current Gate",
        "",
        f"- Status: `{status.get('status')}`",
        f"- Next action: `{status.get('next_action')}`",
        f"- Blocking stage: `{status.get('blocking_stage')}`",
        "",
        "## Inputs",
        "",
        f"- Calibration root: `{calibration.get('path')}`",
        f"- Calibration status: `{calibration.get('status')}`",
        f"- Label counts: `{calibration.get('label_counts')}`",
        f"- Recording ingest status: `{recording_ingest.get('status')}`",
        f"- Recording ingest next action: `{recording_ingest.get('next_action')}`",
        f"- Copy execution allowed: `{gate_decision.get('copy_execution_allowed')}`",
        f"- MP4 preflight allowed: `{gate_decision.get('mp4_preflight_allowed')}`",
        "",
        "## Seed And Dataset",
        "",
        f"- Seed package status: `{seed.get('summary_status')}`",
        f"- Seed NPZ exists: `{seed.get('seed_npz_exists')}`",
        f"- Dataset status: `{dataset.get('run_summary_status')}`",
        "",
        "## Strict Gate",
        "",
        "- Original 20 MP4s must pass `20/20`.",
        "- Held-out 15 MP4s must pass paper/scissors `10/10` and rock wait `5/5`.",
        "- SCHUNK remains blocked until those gates pass.",
        "",
    ]
    return "\n".join(lines)


__all__ = ["V4PipelineStatusConfig", "build_v4_pipeline_status"]
