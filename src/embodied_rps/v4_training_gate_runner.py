"""Training and strict-validation gate runner for v4 skeleton predictor."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.real_skeleton_training import load_real_skeleton_dataset, load_sweep_config, real_dataset_summary


@dataclass(frozen=True)
class V4TrainingGateConfig:
    """Configuration for v4 training and strict validation gate status."""

    dataset_root: Path
    training_config_path: Path
    output_root: Path
    original20_root: Path
    heldout15_root: Path
    profile_json_path: Path
    original20_validation_root: Path
    heldout15_validation_root: Path
    event_manifest_path: Path
    model: str = "all"


def run_v4_training_gate(config: V4TrainingGateConfig) -> dict[str, object]:
    """Report or advance the v4 model-training gate without launching SCHUNK/Isaac."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    completed_stages: list[str] = []
    stage_outputs: dict[str, object] = {}
    if not config.training_config_path.exists():
        summary = _summary(
            config=config,
            status="awaiting_training_config",
            next_action="create_v4_training_config",
            blocking_stage="training_config",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
        )
        _write_summary(config.output_root, summary)
        return summary

    sweep_config = load_sweep_config(config.training_config_path)
    configured_dataset_root = Path(str(sweep_config.get("dataset_root", config.dataset_root.as_posix())))
    if configured_dataset_root != config.dataset_root:
        stage_outputs["training_config"] = {
            "status": "dataset_root_mismatch",
            "configured_dataset_root": configured_dataset_root.as_posix(),
            "runner_dataset_root": config.dataset_root.as_posix(),
        }
        summary = _summary(
            config=config,
            status="training_config_dataset_mismatch",
            next_action="align_v4_training_config_dataset_root",
            blocking_stage="training_config",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
        )
        _write_summary(config.output_root, summary)
        return summary

    dataset_status = _dataset_status(config.dataset_root)
    stage_outputs["v4_dataset"] = dataset_status
    if dataset_status["status"] != "passed":
        summary = _summary(
            config=config,
            status="awaiting_v4_dataset",
            next_action="run_v4_post_review_pipeline_dataset_generation",
            blocking_stage="dataset",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
            sweep_config=sweep_config,
        )
        _write_summary(config.output_root, summary)
        return summary
    completed_stages.append("v4_dataset")

    profile_status = _profile_status(config.profile_json_path)
    stage_outputs["v4_profile"] = profile_status
    if profile_status["status"] != "passed":
        summary = _summary(
            config=config,
            status="ready_for_v4_training",
            next_action="train_v4_gru_tcn_then_export_profile",
            blocking_stage="model_training",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
            sweep_config=sweep_config,
        )
        _write_summary(config.output_root, summary)
        return summary
    completed_stages.append("v4_profile")

    original_status = _validation_status(config.original20_validation_root / "validation_summary.json")
    heldout_status = _validation_status(config.heldout15_validation_root / "validation_summary.json")
    stage_outputs["original20_strict_validation"] = original_status
    stage_outputs["heldout15_strict_validation"] = heldout_status
    if original_status["status"] == "missing" or heldout_status["status"] == "missing":
        summary = _summary(
            config=config,
            status="ready_for_strict_video_validation",
            next_action="run_original20_and_heldout15_v4_validation",
            blocking_stage="strict_video_validation",
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
            sweep_config=sweep_config,
        )
        _write_summary(config.output_root, summary)
        return summary

    if bool(original_status.get("passed")) and bool(heldout_status.get("passed")):
        status = "strict_gates_passed"
        next_action = "write_or_verify_schunk_event_manifest_then_evaluate_response_timing"
        blocking_stage = None
        completed_stages.extend(["original20_strict_validation", "heldout15_strict_validation"])
    else:
        status = "strict_gates_failed"
        next_action = "inspect_failures_and_expand_v4_dataset"
        blocking_stage = "strict_video_validation"
    summary = _summary(
        config=config,
        status=status,
        next_action=next_action,
        blocking_stage=blocking_stage,
        completed_stages=completed_stages,
        stage_outputs=stage_outputs,
        sweep_config=sweep_config,
    )
    _write_summary(config.output_root, summary)
    return summary


def _dataset_status(dataset_root: Path) -> dict[str, object]:
    if not dataset_root.exists():
        return {"status": "missing", "dataset_root": dataset_root.as_posix(), "failures": [{"code": "missing_dataset_root"}]}
    try:
        dataset = load_real_skeleton_dataset(dataset_root)
        return {"status": "passed", "summary": real_dataset_summary(dataset, dataset_root)}
    except (FileNotFoundError, ValueError) as exc:
        return {"status": "invalid", "dataset_root": dataset_root.as_posix(), "failures": [{"code": "dataset_load_failed", "message": str(exc)}]}


def _profile_status(profile_json_path: Path) -> dict[str, object]:
    profile_pt_path = profile_json_path.with_suffix(".pt")
    if not profile_json_path.exists() or not profile_pt_path.exists():
        return {
            "status": "missing",
            "profile_json": profile_json_path.as_posix(),
            "profile_pt": profile_pt_path.as_posix(),
            "json_exists": profile_json_path.exists(),
            "pt_exists": profile_pt_path.exists(),
        }
    try:
        profile = json.loads(profile_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "invalid", "profile_json": profile_json_path.as_posix(), "failure": str(exc)}
    return {
        "status": "passed",
        "profile_json": profile_json_path.as_posix(),
        "profile_pt": profile_pt_path.as_posix(),
        "profile_name": profile.get("profile_name"),
        "model": profile.get("model"),
        "label_names": profile.get("label_names"),
        "selected_accuracy": profile.get("selected_accuracy"),
        "selected_macro_f1": profile.get("selected_macro_f1"),
    }


def _validation_status(summary_path: Path) -> dict[str, object]:
    if not summary_path.exists():
        return {"status": "missing", "summary_path": summary_path.as_posix()}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "invalid", "summary_path": summary_path.as_posix(), "failure": str(exc), "passed": False}
    return {
        "status": "passed" if bool(summary.get("passed")) else "failed",
        "summary_path": summary_path.as_posix(),
        "passed": bool(summary.get("passed")),
        "passed_count": summary.get("passed_count"),
        "failed_count": summary.get("failed_count"),
        "clip_count": summary.get("clip_count"),
        "per_class": summary.get("per_class"),
        "event_manifest_written": summary.get("event_manifest_written"),
    }


def _summary(
    *,
    config: V4TrainingGateConfig,
    status: str,
    next_action: str,
    blocking_stage: str | None,
    completed_stages: list[str],
    stage_outputs: Mapping[str, object],
    sweep_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "next_action": next_action,
        "blocking_stage": blocking_stage,
        "dataset_root": config.dataset_root.as_posix(),
        "training_config": config.training_config_path.as_posix(),
        "profile_json": config.profile_json_path.as_posix(),
        "original20_root": config.original20_root.as_posix(),
        "heldout15_root": config.heldout15_root.as_posix(),
        "original20_validation_root": config.original20_validation_root.as_posix(),
        "heldout15_validation_root": config.heldout15_validation_root.as_posix(),
        "event_manifest_path": config.event_manifest_path.as_posix(),
        "completed_stages": completed_stages,
        "stage_outputs": dict(stage_outputs),
        "commands": _commands(config, sweep_config=sweep_config),
        "notes": [
            "This runner does not launch SCHUNK or Isaac rendering.",
            "Strict validation must pass both the original 20 MP4s and the held-out 15 MP4s before SCHUNK integration resumes.",
        ],
    }


def _commands(config: V4TrainingGateConfig, *, sweep_config: Mapping[str, object] | None) -> dict[str, object]:
    training_config = config.training_config_path.as_posix()
    profile_json = config.profile_json_path.as_posix()
    commands = {
        "smoke_train_gru": [
            "python",
            "-m",
            "embodied_rps.tools.train_real_skeleton_predictor",
            "--config",
            training_config,
            "--model",
            "gru",
            "--smoke",
            "--max-runs",
            "1",
        ],
        "full_train": [
            "python",
            "-m",
            "embodied_rps.tools.train_real_skeleton_predictor",
            "--config",
            training_config,
            "--model",
            config.model,
        ],
        "validate_original20": [
            "python",
            "-m",
            "embodied_rps.tools.evaluate_real_skeleton_video_predictions",
            "--profile",
            profile_json,
            "--input-root",
            config.original20_root.as_posix(),
            "--output-root",
            config.original20_validation_root.as_posix(),
            "--event-output",
            config.event_manifest_path.as_posix(),
            "--expected-count",
            "20",
            "--label-mode",
            "transition",
        ],
        "validate_heldout15": [
            "python",
            "-m",
            "embodied_rps.tools.evaluate_real_skeleton_video_predictions",
            "--profile",
            profile_json,
            "--input-root",
            config.heldout15_root.as_posix(),
            "--output-root",
            config.heldout15_validation_root.as_posix(),
            "--expected-count",
            "15",
            "--label-mode",
            "final-label",
        ],
    }
    if sweep_config is not None:
        commands["runs_dir"] = str(sweep_config.get("runs_dir", ""))
        commands["comparison_path"] = str(sweep_config.get("comparison_path", ""))
    return commands


def _write_summary(output_root: Path, summary: Mapping[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "training_gate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "training_gate_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")


def _summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V4 Training Gate Summary",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Next action: `{summary.get('next_action')}`",
        f"- Blocking stage: `{summary.get('blocking_stage')}`",
        f"- Dataset root: `{summary.get('dataset_root')}`",
        f"- Training config: `{summary.get('training_config')}`",
        f"- Profile JSON: `{summary.get('profile_json')}`",
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
                lines.append(f"- `{stage}`: status=`{output.get('status')}`")
                failures = output.get("failures")
                if isinstance(failures, list):
                    for failure in failures:
                        if isinstance(failure, Mapping):
                            lines.append(f"  - failure `{failure.get('code')}`")
            else:
                lines.append(f"- `{stage}`")
    else:
        lines.append("- None.")
    commands = summary.get("commands")
    if isinstance(commands, Mapping):
        lines.extend(["", "## Commands", ""])
        for name in ("smoke_train_gru", "full_train", "validate_original20", "validate_heldout15"):
            command = commands.get(name)
            if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
                lines.extend([f"### {name}", "", "```powershell", _quote_command([str(part) for part in command]), "```", ""])
    lines.extend(["## Notes", ""])
    notes = summary.get("notes")
    if isinstance(notes, list):
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _quote_command(parts: Sequence[str]) -> str:
    quoted: list[str] = []
    for part in parts:
        if not part or any(char.isspace() for char in part) or any(ord(char) > 127 for char in part) or "/" in part or "\\" in part or ":" in part:
            quoted.append(f"'{part}'")
        else:
            quoted.append(part)
    return " ".join(quoted)


__all__ = ["V4TrainingGateConfig", "run_v4_training_gate"]
