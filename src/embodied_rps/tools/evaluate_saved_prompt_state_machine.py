"""Evaluate prompt-scoped state-machine policy on saved validation rows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from embodied_rps.real_skeleton_policy_sweep import SavedValidationClip, load_saved_validation_clips
from embodied_rps.real_skeleton_prompt_state_machine import (
    PromptStateMachineConfig,
    summarize_prompt_state_machine_clip,
)
from embodied_rps.real_skeleton_video_eval import build_validation_summary, write_clip_metrics_csv


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for prompt-scoped state-machine saved-output evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate saved validation outputs with a prompt-scoped state machine.")
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--margin-threshold", type=float, default=0.10)
    parser.add_argument("--confirmation-count", type=int, default=2)
    parser.add_argument("--max-commit-progress", type=float, default=0.50)
    parser.add_argument("--transition-mass-threshold", type=float, default=0.05)
    parser.add_argument("--binary-transition-mass-threshold", type=float, default=0.60)
    parser.add_argument("--rock-pre-wait-grace-progress", type=float, default=0.0)
    parser.add_argument(
        "--transition-selection-mode",
        choices=("latest", "highest_confidence"),
        default="highest_confidence",
    )
    parser.add_argument(
        "--progress-mode",
        choices=("clip", "motion", "observed", "model"),
        default="clip",
    )
    args = parser.parse_args(argv)

    config = PromptStateMachineConfig(
        confidence_threshold=float(args.confidence_threshold),
        margin_threshold=float(args.margin_threshold),
        confirmation_count=int(args.confirmation_count),
        max_commit_progress=float(args.max_commit_progress),
        transition_mass_threshold=float(args.transition_mass_threshold),
        binary_transition_mass_threshold=float(args.binary_transition_mass_threshold),
        rock_pre_wait_grace_progress=float(args.rock_pre_wait_grace_progress),
        progress_key=_progress_key_for_mode(str(args.progress_mode)),
        transition_selection_mode=str(args.transition_selection_mode),  # type: ignore[arg-type]
    )
    clips = load_saved_validation_clips(args.validation_root)
    clip_metrics = [_summarize_clip(clip, config=config) for clip in clips]
    discovery_summary = _load_discovery_summary(args.validation_root)
    summary = build_validation_summary(
        clip_metrics=clip_metrics,
        discovery_summary=discovery_summary,
        config=config.to_strict_config(),
        event_manifest_path=None,
    )
    summary["prompt_state_machine"] = {
        "mode": "transition_commit_wait_lock",
        "transition_selection_mode": config.transition_selection_mode,
        "max_commit_progress": config.max_commit_progress,
        "progress_mode": str(args.progress_mode),
        "transition_policy": "commit_highest_confidence_stable_transition",
        "rock_policy": "lock_first_wait_then_ignore_later_binary",
        "rock_pre_wait_grace_progress": config.rock_pre_wait_grace_progress,
    }
    summary["ignored_transition_after_wait_lock_count"] = sum(
        int(metric.get("ignored_transition_after_wait_lock_count", 0) or 0) for metric in clip_metrics
    )
    summary["ignored_transition_before_wait_lock_count"] = sum(
        int(metric.get("ignored_transition_before_wait_lock_count", 0) or 0) for metric in clip_metrics
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = args.output_root / "clip_metrics.csv"
    metrics_json_path = args.output_root / "clip_metrics.json"
    summary_path = args.output_root / "prompt_state_machine_summary.json"
    write_clip_metrics_csv(metrics_csv_path, clip_metrics)
    metrics_json_path.write_text(json.dumps(clip_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["clip_metrics_csv"] = metrics_csv_path.as_posix()
    summary["clip_metrics_json"] = metrics_json_path.as_posix()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": summary_path.as_posix(), "passed_clip_count": summary["passed_clip_count"]}, indent=2))
    return 0


def _summarize_clip(clip: SavedValidationClip, *, config: PromptStateMachineConfig) -> dict[str, object]:
    return summarize_prompt_state_machine_clip(
        clip.rows,
        true_gesture=clip.true_gesture,
        transition_label=clip.transition_label,
        source_path=clip.source_path,
        clip_id=clip.clip_id,
        frame_count=clip.frame_count,
        fps=clip.fps,
        width=clip.width,
        height=clip.height,
        config=config,
        overlay_path=clip.overlay_path,
        frame_csv_path=clip.frame_csv_path,
        frame_jsonl_path=clip.frame_jsonl_path,
    )


def _load_discovery_summary(validation_root: Path) -> dict[str, object]:
    summary_path = validation_root / "validation_summary.json"
    if summary_path.exists():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        discovery = loaded.get("discovery")
        if isinstance(discovery, Mapping):
            return dict(discovery)
    return {"passed": True, "label_mode": "saved-validation"}


def _progress_key_for_mode(mode: str) -> str:
    if mode == "model":
        return "model_progress"
    if mode == "observed":
        return "observed_progress"
    if mode == "motion":
        return "motion_progress"
    return "clip_progress"


if __name__ == "__main__":
    raise SystemExit(main())
