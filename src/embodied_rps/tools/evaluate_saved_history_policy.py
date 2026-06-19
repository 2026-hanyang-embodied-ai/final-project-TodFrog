"""Evaluate a few-shot probability-history policy on saved validation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from embodied_rps.real_skeleton_history_policy import (
    HistoryPolicyConfig,
    HistoryTrainingClip,
    fit_history_centroid_policy,
    summarize_history_clip,
)
from embodied_rps.real_skeleton_policy_sweep import load_saved_validation_clips
from embodied_rps.real_skeleton_video_eval import EvaluationGesture


def main(argv: Sequence[str] | None = None) -> int:
    """Run saved-output history-policy training and evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a saved-output probability-history policy.")
    parser.add_argument("--train-root", required=True, type=Path, action="append", help="Saved validation root used for few-shot calibration. Repeat to add roots.")
    parser.add_argument("--eval-root", required=True, type=Path, help="Saved validation root to evaluate.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--observation-progress", type=float, default=0.50)
    parser.add_argument("--max-decision-progress", type=float, default=0.50)
    parser.add_argument("--confidence-temperature", type=float, default=0.25)
    args = parser.parse_args(argv)

    config = HistoryPolicyConfig(
        observation_progress=float(args.observation_progress),
        max_decision_progress=float(args.max_decision_progress),
        confidence_temperature=float(args.confidence_temperature),
    )
    train_clips = _load_history_clips(args.train_root)
    eval_clips = _load_history_clips([args.eval_root])
    policy = fit_history_centroid_policy(train_clips, config=config)
    clip_metrics = [_with_clip_metadata(clip, summarize_history_clip(clip, policy=policy)) for clip in eval_clips]
    summary = _build_summary(
        train_roots=args.train_root,
        eval_root=args.eval_root,
        train_clips=train_clips,
        clip_metrics=clip_metrics,
        config=config,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_root / "clip_metrics.csv"
    summary_path = args.output_root / "history_policy_summary.json"
    profile_path = args.output_root / "history_policy_profile.json"
    _write_clip_metrics(metrics_path, clip_metrics)
    summary["clip_metrics_csv"] = metrics_path.as_posix()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    profile_path.write_text(
        json.dumps(
            {
                "policy_type": "nearest_centroid_probability_history",
                "labels": list(policy.labels),
                "feature_names": list(policy.feature_names),
                "feature_mean": policy.feature_mean.tolist(),
                "feature_scale": policy.feature_scale.tolist(),
                "centroids": {label: centroid.tolist() for label, centroid in policy.centroids.items()},
                "config": {
                    "observation_progress": policy.config.observation_progress,
                    "max_decision_progress": policy.config.max_decision_progress,
                    "confidence_temperature": policy.config.confidence_temperature,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary["history_policy_profile"] = profile_path.as_posix()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": summary_path.as_posix(), "passed_clip_count": summary["passed_clip_count"]}, indent=2))
    return 0


def _load_history_clips(roots: Sequence[Path]) -> list[HistoryTrainingClip]:
    clips: list[HistoryTrainingClip] = []
    for root in roots:
        for clip in load_saved_validation_clips(root):
            clips.append(
                HistoryTrainingClip(
                    clip_id=clip.clip_id,
                    true_gesture=cast(EvaluationGesture, clip.true_gesture),
                    rows=clip.rows,
                )
            )
    return clips


def _with_clip_metadata(clip: HistoryTrainingClip, metrics: Mapping[str, object]) -> dict[str, object]:
    parsed = dict(metrics)
    parsed["transition_label"] = f"test_{clip.true_gesture}" if not str(clip.clip_id).startswith("rock_to_") else str(clip.clip_id).rsplit("_", 1)[0]
    return parsed


def _build_summary(
    *,
    train_roots: Sequence[Path],
    eval_root: Path,
    train_clips: Sequence[HistoryTrainingClip],
    clip_metrics: Sequence[Mapping[str, object]],
    config: HistoryPolicyConfig,
) -> dict[str, object]:
    passed = [metric for metric in clip_metrics if bool(metric.get("passed"))]
    failed = [metric for metric in clip_metrics if not bool(metric.get("passed"))]
    per_class = {}
    for label in ("rock", "paper", "scissors"):
        class_metrics = [metric for metric in clip_metrics if metric.get("true_gesture") == label]
        class_passed = [metric for metric in class_metrics if bool(metric.get("passed"))]
        per_class[label] = {
            "clip_count": len(class_metrics),
            "passed_count": len(class_passed),
            "accuracy": _safe_rate(len(class_passed), len(class_metrics)),
        }
    binary_metrics = [metric for metric in clip_metrics if metric.get("true_gesture") in {"paper", "scissors"}]
    binary_passed = [metric for metric in binary_metrics if bool(metric.get("passed"))]
    rock_metrics = [metric for metric in clip_metrics if metric.get("true_gesture") == "rock"]
    rock_wait = [metric for metric in rock_metrics if bool(metric.get("passed")) and metric.get("decision_state") == "wait_counter_paper"]
    rock_false = [metric for metric in rock_metrics if metric.get("failure_reason") == "false_trigger"]
    return {
        "policy_type": "nearest_centroid_probability_history",
        "train_roots": [root.as_posix() for root in train_roots],
        "eval_root": eval_root.as_posix(),
        "train_clip_count": len(train_clips),
        "train_label_counts": dict(sorted(Counter(clip.true_gesture for clip in train_clips).items())),
        "eval_clip_count": len(clip_metrics),
        "passed_clip_count": len(passed),
        "failed_clip_count": len(failed),
        "accuracy": _safe_rate(len(passed), len(clip_metrics)),
        "paper_scissors_accuracy": _safe_rate(len(binary_passed), len(binary_metrics)),
        "rock_wait_success_count": len(rock_wait),
        "rock_false_trigger_count": len(rock_false),
        "per_class": per_class,
        "failure_reason_counts": dict(sorted(Counter(str(metric.get("failure_reason")) for metric in failed).items())),
        "failed_clips": [
            {
                "clip_id": metric.get("clip_id"),
                "true_gesture": metric.get("true_gesture"),
                "predicted_gesture": metric.get("predicted_gesture"),
                "failure_reason": metric.get("failure_reason"),
            }
            for metric in failed
        ],
        "config": {
            "observation_progress": config.observation_progress,
            "max_decision_progress": config.max_decision_progress,
            "confidence_temperature": config.confidence_temperature,
        },
    }


def _write_clip_metrics(path: Path, metrics: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "clip_id",
        "transition_label",
        "true_gesture",
        "passed",
        "failure_reason",
        "predicted_gesture",
        "decision_state",
        "selected_robot_action",
        "decision_progress",
        "decision_confidence",
        "decision_confidence_margin",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
