"""Synthetic observation-ratio metrics summary for the v7d two-stage branch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_OBSERVATION_RATIOS: tuple[str, ...] = ("0.10", "0.20", "0.30", "0.40", "0.50", "0.75", "1.00")


@dataclass(frozen=True)
class V7DSyntheticMetricsConfig:
    project_root: Path = Path.cwd()
    output_root: Path = Path("results/real_skeleton_v7d_two_stage_prompt_window_guard_synthetic_metrics")
    stage1_results_root: Path = Path(
        "results/real_skeleton_two_stage_rock_transition_v7d_real_seeded_prompt_window_guard_tcn_ensemble"
    )
    stage2_results_root: Path = Path(
        "results/real_skeleton_two_stage_paper_scissors_v7d_real_seeded_prompt_window_guard_tcn_ensemble"
    )
    quality_ratio: str = "0.50"
    min_accuracy: float = 0.90
    observation_ratios: tuple[str, ...] = DEFAULT_OBSERVATION_RATIOS


def write_v7d_synthetic_metrics(config: V7DSyntheticMetricsConfig) -> dict[str, Any]:
    project_root = config.project_root.resolve()
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage1 = _stage_status(
        project_root=project_root,
        results_root=config.stage1_results_root,
        stage_name="stage1",
        expected_labels=("rock", "transition"),
        quality_ratio=config.quality_ratio,
        min_accuracy=config.min_accuracy,
        expected_ratios=config.observation_ratios,
    )
    stage2 = _stage_status(
        project_root=project_root,
        results_root=config.stage2_results_root,
        stage_name="stage2",
        expected_labels=("paper", "scissors"),
        quality_ratio=config.quality_ratio,
        min_accuracy=config.min_accuracy,
        expected_ratios=config.observation_ratios,
    )

    status = "passed" if stage1["status"] == "passed" and stage2["status"] == "passed" else "failed"
    summary: dict[str, Any] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "summary_path": _display_path(output_root / "synthetic_metrics_summary.json", base=project_root),
        "quality_ratio": config.quality_ratio,
        "min_accuracy": config.min_accuracy,
        "observation_ratios": [float(ratio) for ratio in config.observation_ratios],
        "stage1": stage1,
        "stage2": stage2,
        "failures": [*stage1["failures"], *stage2["failures"]],
        "promotion_eligible": False,
        "heldout_policy": "synthetic metrics are generated from training comparison artifacts only; heldout */test MP4s remain validation-only",
    }
    (output_root / "synthetic_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _stage_status(
    *,
    project_root: Path,
    results_root: Path,
    stage_name: str,
    expected_labels: tuple[str, ...],
    quality_ratio: str,
    min_accuracy: float,
    expected_ratios: tuple[str, ...],
) -> dict[str, Any]:
    root = _resolve_path(project_root, results_root)
    comparison_path = root / "model_comparison.json"
    failures: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}

    if not comparison_path.exists():
        failures.append({"code": f"missing_{stage_name}_model_comparison", "path": _display_path(comparison_path, base=project_root)})
    else:
        try:
            payload = json.loads(comparison_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(
                {
                    "code": f"invalid_{stage_name}_model_comparison",
                    "path": _display_path(comparison_path, base=project_root),
                    "error": str(exc),
                }
            )

    label_names = tuple(str(label) for label in payload.get("label_names", ()))
    if payload and label_names != expected_labels:
        failures.append(
            {
                "code": f"unexpected_{stage_name}_labels",
                "label_names": list(label_names),
                "expected_label_names": list(expected_labels),
            }
        )
    if payload and payload.get("model_ready") is not True:
        failures.append({"code": f"{stage_name}_model_not_ready", "model_ready": payload.get("model_ready")})
    if payload and int(payload.get("num_runs") or 0) < 3:
        failures.append({"code": f"insufficient_{stage_name}_tcn_runs", "num_runs": payload.get("num_runs")})

    best_by_ratio = payload.get("best_by_ratio") if isinstance(payload.get("best_by_ratio"), dict) else {}
    compact_by_ratio: dict[str, dict[str, Any]] = {}
    for ratio in expected_ratios:
        metric = best_by_ratio.get(ratio)
        if not isinstance(metric, dict):
            failures.append({"code": f"missing_{stage_name}_ratio_metric", "ratio": ratio})
            continue
        compact_by_ratio[ratio] = {
            "run_id": metric.get("run_id"),
            "model": metric.get("model"),
            "accuracy": _optional_float(metric.get("accuracy")),
            "macro_f1": _optional_float(metric.get("macro_f1")),
            "mean_confidence": _optional_float(metric.get("mean_confidence")),
            "latency_ms": _optional_float(metric.get("latency_ms")),
        }

    quality_metric = compact_by_ratio.get(quality_ratio, {})
    quality_accuracy = _optional_float(quality_metric.get("accuracy"))
    if quality_accuracy is None:
        failures.append({"code": f"missing_{stage_name}_quality_accuracy", "ratio": quality_ratio})
    elif quality_accuracy < min_accuracy:
        failures.append(
            {
                "code": f"{stage_name}_quality_accuracy_below_threshold",
                "ratio": quality_ratio,
                "accuracy": quality_accuracy,
                "min_accuracy": min_accuracy,
            }
        )

    return {
        "status": "passed" if not failures else "failed",
        "stage_name": stage_name,
        "results_root": _display_path(root, base=project_root),
        "model_comparison": _display_path(comparison_path, base=project_root),
        "num_runs": payload.get("num_runs"),
        "model_ready": payload.get("model_ready"),
        "label_names": list(label_names),
        "quality_ratio": quality_ratio,
        "quality_accuracy": quality_accuracy,
        "min_accuracy": min_accuracy,
        "best_by_ratio": compact_by_ratio,
        "failures": failures,
    }


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
