"""Fail-closed strict validation and promotion preflight for v7e stage1."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from embodied_rps.real_skeleton_two_stage import validate_two_stage_label_names
from embodied_rps.v7_training_gate_runner import discover_v7_validation_roots


BRANCH_LABEL = "v7e_stage1_paper_transition_rescue"

V7EStage1StrictValidationPreflightStatus = Literal[
    "blocked_remote_training_not_ready",
    "blocked_profiles_missing",
    "ready_for_synthetic_metrics",
    "v7e_strict_gates_failed",
    "ready_for_strict_mp4_validation",
    "ready_for_replay_diagnostics",
    "ready_for_fresh_live_retakes",
    "v7e_promotion_candidate",
]


@dataclass(frozen=True)
class V7EStage1StrictValidationPreflightConfig:
    """Inputs for the v7e strict validation and promotion preflight."""

    project_root: Path = field(default_factory=Path.cwd)
    output_root: Path = Path("artifacts/real_skeleton_v7e_stage1_strict_validation_preflight_20260619")
    remote_training_preflight_root: Path = Path("artifacts/real_skeleton_v7e_stage1_remote_training_preflight_20260619")
    stage1_profile_json: Path = Path(
        "results/model_profiles/real_skeleton_two_stage_rock_transition_v7e_stage1_paper_transition_rescue_tcn_ensemble.json"
    )
    stage2_reuse_profile_json: Path = Path(
        "results/model_profiles/real_skeleton_two_stage_paper_scissors_v7d_real_seeded_prompt_window_guard_tcn_ensemble.json"
    )
    synthetic_metrics_root: Path = Path("results/real_skeleton_v7e_stage1_paper_transition_rescue_synthetic_metrics")
    original20_validation_root: Path = Path(
        "artifacts/real_mp4_prediction_validation_original20_v7e_stage1_paper_transition_rescue_20260619"
    )
    heldout15_validation_root: Path = Path(
        "artifacts/real_mp4_prediction_validation_heldout15_v7e_stage1_paper_transition_rescue_20260619"
    )
    replay_diagnostics_root: Path = Path("artifacts/real_skeleton_v7e_stage1_replay_diagnostics_20260619")
    fresh_live_root: Path = Path("artifacts/real_skeleton_v7e_stage1_fresh_live_retakes_20260619")
    event_manifest_path: Path = Path("artifacts/real_skeleton_schunk_events_v7e_stage1_paper_transition_rescue_20260619/events.jsonl")
    dataset_search_root: Path = Path("D:/dataset")


def write_v7e_stage1_strict_validation_preflight(config: V7EStage1StrictValidationPreflightConfig) -> dict[str, Any]:
    """Write a non-mutating strict validation/promotion preflight for v7e."""

    project_root = config.project_root.resolve()
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    validation_root_discovery = discover_v7_validation_roots(config.dataset_search_root)
    remote_training = _remote_training_status(project_root=project_root, config=config)
    profiles = _profile_pair_status(project_root=project_root, config=config)
    synthetic = _synthetic_metrics_status(project_root=project_root, config=config)
    original20 = _strict_validation_status(
        root=_resolve_path(project_root, config.original20_validation_root),
        project_root=project_root,
        expected_clip_count=20,
        expected_passed_count=20,
        require_zero_rock_false_triggers=False,
        label="original20",
    )
    heldout15 = _strict_validation_status(
        root=_resolve_path(project_root, config.heldout15_validation_root),
        project_root=project_root,
        expected_clip_count=15,
        expected_passed_count=15,
        require_zero_rock_false_triggers=True,
        label="heldout15",
    )
    replay = _activity_status(
        root=_resolve_path(project_root, config.replay_diagnostics_root),
        project_root=project_root,
        filenames=("replay_summary.json", "validation_summary.json"),
        label="replay_diagnostics",
    )
    live = _activity_status(
        root=_resolve_path(project_root, config.fresh_live_root),
        project_root=project_root,
        filenames=("live_summary.json", "retake_summary.json", "validation_summary.json"),
        label="fresh_live_retakes",
    )
    status, blocking_stage, next_action, completed = _overall_status(
        remote_training=remote_training,
        profiles=profiles,
        synthetic=synthetic,
        original20=original20,
        heldout15=heldout15,
        replay=replay,
        live=live,
    )

    summary: dict[str, Any] = {
        "status": status,
        "blocking_stage": blocking_stage,
        "next_action": next_action,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "completed_stages": completed,
        "stage1_training_scope": "retrain_rock_vs_transition_only",
        "stage2_policy": "reuse_v7d_paper_scissors_stage2_unless_new_evidence_requires_change",
        "stage_outputs": {
            "remote_training_preflight": remote_training,
            "stage1_and_reused_stage2_profiles": profiles,
            "synthetic_metrics": synthetic,
            "original20_strict_validation": original20,
            "heldout15_strict_validation": heldout15,
            "replay_diagnostics": replay,
            "fresh_live_retakes": live,
        },
        "validation_root_discovery": _sanitize_discovery(validation_root_discovery),
        "planned_commands": _planned_commands(project_root=project_root, config=config),
        "training_started": False,
        "validation_started": False,
        "remote_training_started": False,
        "heldout15_started": False,
        "promotion_decision": {
            "may_promote_v7e": status == "v7e_promotion_candidate",
            "fallback_policy": None if status == "v7e_promotion_candidate" else "keep_v4_live_demo_fallback",
            "promotion_rule": (
                "promote v7e only after synthetic metrics, original20 20/20, heldout15 15/15, "
                "heldout rock false triggers 0, no paper/scissors regression, replay diagnostics, and fresh live retakes pass"
            ),
        },
        "heldout_policy": "heldout */test MP4s remain validation-only and must not enter seed packages or training metadata",
        "notes": [
            "This preflight does not train, validate MP4s, run replay/live capture, promote profiles, edit protected PDFs, or package final outputs.",
            "Heldout15, replay diagnostics, and fresh live retakes cannot override a failed original20 strict gate.",
            "V7e retrains only stage1 and reuses the v7d paper/scissors stage2 profile unless diagnostics require a separate stage2 branch.",
        ],
        "config": _config_summary(project_root=project_root, config=config),
    }
    (output_root / "v7e_stage1_strict_validation_preflight_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "v7e_stage1_strict_validation_preflight_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return _json_ready(summary)


def _remote_training_status(*, project_root: Path, config: V7EStage1StrictValidationPreflightConfig) -> dict[str, Any]:
    summary_path = _resolve_path(project_root, config.remote_training_preflight_root) / "v7e_stage1_remote_training_preflight_summary.json"
    if not summary_path.exists():
        return {
            "status": "missing",
            "summary_path": _display_path(summary_path, base=project_root),
            "failures": [{"code": "missing_remote_training_preflight_summary"}],
        }
    payload = _read_json(summary_path)
    if not isinstance(payload, Mapping):
        return {
            "status": "invalid",
            "summary_path": _display_path(summary_path, base=project_root),
            "failures": [{"code": "invalid_remote_training_preflight_summary"}],
        }
    status = str(payload.get("status", "unknown"))
    failures: list[dict[str, Any]] = []
    if status != "ready_for_remote_stage1_tcn_training":
        failures.append({"code": "remote_training_not_ready", "remote_training_status": status})
    if payload.get("validation_started") is True:
        failures.append({"code": "unexpected_validation_started_flag"})
    return {
        "status": status,
        "summary_path": _display_path(summary_path, base=project_root),
        "failures": failures,
    }


def _profile_pair_status(*, project_root: Path, config: V7EStage1StrictValidationPreflightConfig) -> dict[str, Any]:
    stage1 = _profile_status(
        _resolve_path(project_root, config.stage1_profile_json),
        project_root=project_root,
        expected_labels=("rock", "transition"),
        stage_name="stage1_rock_transition_v7e",
    )
    stage2 = _profile_status(
        _resolve_path(project_root, config.stage2_reuse_profile_json),
        project_root=project_root,
        expected_labels=("paper", "scissors"),
        stage_name="stage2_paper_scissors_v7d_reuse",
    )
    failures = list(stage1.get("failures", [])) + list(stage2.get("failures", []))
    if not failures:
        try:
            validate_two_stage_label_names(
                [str(label) for label in stage1.get("label_names", [])],
                [str(label) for label in stage2.get("label_names", [])],
            )
        except ValueError as exc:
            failures.append({"code": "invalid_two_stage_label_pair", "message": str(exc)})
    return {
        "status": "passed" if not failures else "missing_or_invalid",
        "stage1": stage1,
        "stage2_reuse": stage2,
        "failures": failures,
    }


def _profile_status(path: Path, *, project_root: Path, expected_labels: Sequence[str], stage_name: str) -> dict[str, Any]:
    pt_path = path.with_suffix(".pt")
    if not path.exists() or not pt_path.exists():
        return {
            "status": "missing",
            "stage_name": stage_name,
            "profile_json": _display_path(path, base=project_root),
            "profile_pt": _display_path(pt_path, base=project_root),
            "label_names": [],
            "failures": [{"code": "missing_profile_or_checkpoint", "stage_name": stage_name}],
        }
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        return {
            "status": "invalid",
            "stage_name": stage_name,
            "profile_json": _display_path(path, base=project_root),
            "profile_pt": _display_path(pt_path, base=project_root),
            "label_names": [],
            "failures": [{"code": "invalid_profile_json", "stage_name": stage_name}],
        }
    label_names = [str(label) for label in _sequence(payload.get("label_names"))]
    failures: list[dict[str, Any]] = []
    if payload.get("model") != "tcn":
        failures.append({"code": "profile_model_not_tcn", "stage_name": stage_name, "model": payload.get("model")})
    if set(label_names) != set(expected_labels):
        failures.append(
            {
                "code": "unexpected_profile_labels",
                "stage_name": stage_name,
                "label_names": label_names,
                "expected_labels": list(expected_labels),
            }
        )
    return {
        "status": "passed" if not failures else "invalid",
        "stage_name": stage_name,
        "profile_json": _display_path(path, base=project_root),
        "profile_pt": _display_path(pt_path, base=project_root),
        "model": payload.get("model"),
        "label_names": label_names,
        "profile_name": payload.get("profile_name"),
        "failures": failures,
    }


def _synthetic_metrics_status(*, project_root: Path, config: V7EStage1StrictValidationPreflightConfig) -> dict[str, Any]:
    root = _resolve_path(project_root, config.synthetic_metrics_root)
    summary_path = root / "synthetic_metrics_summary.json"
    if not summary_path.exists():
        return {
            "status": "missing",
            "summary_path": _display_path(summary_path, base=project_root),
            "failures": [{"code": "missing_synthetic_metrics_summary"}],
        }
    payload = _read_json(summary_path)
    if not isinstance(payload, Mapping):
        return {
            "status": "invalid",
            "summary_path": _display_path(summary_path, base=project_root),
            "failures": [{"code": "invalid_synthetic_metrics_summary"}],
        }
    failures: list[dict[str, Any]] = []
    if payload.get("status") != "passed" and payload.get("passed") is not True:
        failures.append({"code": "synthetic_metrics_not_passed", "status": payload.get("status")})
    for stage_name in ("stage1", "stage2_reuse"):
        stage = payload.get(stage_name)
        if isinstance(stage, Mapping) and stage.get("status") not in ("passed", None):
            failures.append({"code": f"{stage_name}_synthetic_metrics_not_passed", "status": stage.get("status")})
    return {
        "status": "passed" if not failures else "failed",
        "summary_path": _display_path(summary_path, base=project_root),
        "observation_ratios": payload.get("observation_ratios"),
        "failures": failures,
    }


def _strict_validation_status(
    *,
    root: Path,
    project_root: Path,
    expected_clip_count: int,
    expected_passed_count: int,
    require_zero_rock_false_triggers: bool,
    label: str,
) -> dict[str, Any]:
    summary_path = root / "validation_summary.json"
    if not summary_path.exists():
        return {
            "status": "missing",
            "summary_path": _display_path(summary_path, base=project_root),
            "expected_clip_count": expected_clip_count,
            "expected_passed_count": expected_passed_count,
            "failures": [{"code": f"missing_{label}_validation_summary"}],
        }
    payload = _read_json(summary_path)
    if not isinstance(payload, Mapping):
        return {
            "status": "invalid",
            "summary_path": _display_path(summary_path, base=project_root),
            "expected_clip_count": expected_clip_count,
            "expected_passed_count": expected_passed_count,
            "failures": [{"code": f"invalid_{label}_validation_summary"}],
        }
    clip_count = _optional_int(payload.get("clip_count"))
    passed_count = _first_int(payload, ("passed_count", "passed_clip_count"))
    failed_count = _first_int(payload, ("failed_count", "failed_clip_count"))
    rock_false_trigger_count = _rock_false_trigger_count(root=root, summary=payload)
    paper_scissors_confusion_count = _paper_scissors_confusion_count(root / "clip_metrics.csv")
    failures: list[dict[str, Any]] = []
    if payload.get("passed") is not True:
        failures.append({"code": f"{label}_validation_not_passed", "passed": payload.get("passed")})
    if clip_count != expected_clip_count:
        failures.append({"code": f"{label}_clip_count_mismatch", "clip_count": clip_count, "expected_clip_count": expected_clip_count})
    if passed_count != expected_passed_count:
        failures.append(
            {
                "code": f"{label}_passed_count_mismatch",
                "passed_count": passed_count,
                "expected_passed_count": expected_passed_count,
            }
        )
    if failed_count not in (None, 0):
        failures.append({"code": f"{label}_nonzero_failed_count", "failed_count": failed_count})
    if require_zero_rock_false_triggers and rock_false_trigger_count != 0:
        failures.append({"code": "heldout_rock_false_triggers", "rock_false_trigger_count": rock_false_trigger_count})
    if paper_scissors_confusion_count != 0:
        failures.append({"code": f"{label}_paper_scissors_confusion", "paper_scissors_confusion_count": paper_scissors_confusion_count})
    return {
        "status": "passed" if not failures else "failed",
        "summary_path": _display_path(summary_path, base=project_root),
        "clip_count": clip_count,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "expected_clip_count": expected_clip_count,
        "expected_passed_count": expected_passed_count,
        "rock_false_trigger_count": rock_false_trigger_count,
        "paper_scissors_confusion_count": paper_scissors_confusion_count,
        "failures": failures,
    }


def _rock_false_trigger_count(*, root: Path, summary: Mapping[str, Any]) -> int:
    for key in ("rock_false_trigger_count", "heldout_rock_false_trigger_count", "false_trigger_count"):
        value = _optional_int(summary.get(key))
        if value is not None:
            return value
    report_path = root / "rock_false_trigger_report.json"
    report = _read_json(report_path)
    if isinstance(report, Mapping):
        for key in ("false_trigger_count", "rock_false_trigger_count", "heldout_rock_false_trigger_count"):
            value = _optional_int(report.get(key))
            if value is not None:
                return value
    return 0


def _paper_scissors_confusion_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                truth = str(row.get("true_gesture") or row.get("target_name") or row.get("label") or "").strip().lower()
                prediction = str(row.get("prediction") or row.get("predicted_gesture") or row.get("decision") or "").strip().lower()
                if (truth, prediction) in {("paper", "scissors"), ("scissors", "paper")}:
                    count += 1
    except csv.Error:
        return 1
    return count


def _activity_status(*, root: Path, project_root: Path, filenames: Sequence[str], label: str) -> dict[str, Any]:
    for filename in filenames:
        path = root / filename
        if not path.exists():
            continue
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            return {
                "status": "invalid",
                "summary_path": _display_path(path, base=project_root),
                "failures": [{"code": f"invalid_{label}_summary"}],
            }
        failures: list[dict[str, Any]] = []
        if payload.get("passed") is not True and payload.get("status") != "passed":
            failures.append({"code": f"{label}_not_passed", "status": payload.get("status"), "passed": payload.get("passed")})
        failed_count = _optional_int(payload.get("failed_count"))
        if failed_count not in (None, 0):
            failures.append({"code": f"{label}_nonzero_failed_count", "failed_count": failed_count})
        return {
            "status": "passed" if not failures else "failed",
            "summary_path": _display_path(path, base=project_root),
            "event_count": _first_int(payload, ("event_count", "replay_count", "retake_count", "clip_count", "sample_count")),
            "passed_count": _first_int(payload, ("passed_count", "success_count", "correct_count")),
            "failed_count": failed_count,
            "failures": failures,
        }
    return {
        "status": "missing",
        "root": _display_path(root, base=project_root),
        "expected_filenames": list(filenames),
        "failures": [{"code": f"missing_{label}_summary"}],
    }


def _overall_status(
    *,
    remote_training: Mapping[str, Any],
    profiles: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    original20: Mapping[str, Any],
    heldout15: Mapping[str, Any],
    replay: Mapping[str, Any],
    live: Mapping[str, Any],
) -> tuple[V7EStage1StrictValidationPreflightStatus, str | None, str, list[str]]:
    completed: list[str] = []
    if remote_training.get("status") != "ready_for_remote_stage1_tcn_training" or _nonempty(remote_training.get("failures")):
        return "blocked_remote_training_not_ready", "remote_training", "complete local smoke and remote stage1 training preflight before strict validation", completed
    completed.append("remote_training_preflight")
    if profiles.get("status") != "passed":
        return "blocked_profiles_missing", "stage1_and_reused_stage2_profiles", "copy back valid v7e stage1 and v7d reused stage2 TCN profiles before strict validation", completed
    completed.append("stage1_and_reused_stage2_profiles")
    if synthetic.get("status") == "missing":
        return "ready_for_synthetic_metrics", "synthetic_metrics", "run v7e stage1 observation-ratio synthetic metrics before MP4 validation", completed
    if synthetic.get("status") != "passed":
        return "v7e_strict_gates_failed", "synthetic_metrics", _fallback_next_action(), completed
    completed.append("synthetic_metrics")
    if original20.get("status") == "missing":
        return "ready_for_strict_mp4_validation", "original20_strict_validation", "validate original20 first; do not run heldout, replay, or live yet", completed
    if original20.get("status") != "passed":
        return "v7e_strict_gates_failed", "original20_strict_validation", _fallback_next_action(), completed
    completed.append("original20_strict_validation")
    if heldout15.get("status") == "missing":
        return "ready_for_strict_mp4_validation", "heldout15_strict_validation", "validate heldout15 only after original20 passes; do not replay or retake live yet", completed
    if heldout15.get("status") != "passed":
        return "v7e_strict_gates_failed", "heldout15_strict_validation", _fallback_next_action(), completed
    completed.append("heldout15_strict_validation")
    if replay.get("status") == "missing":
        return "ready_for_replay_diagnostics", "replay_diagnostics", "run archived/approved replay diagnostics before fresh live retakes", completed
    if replay.get("status") != "passed":
        return "v7e_strict_gates_failed", "replay_diagnostics", _fallback_next_action(), completed
    completed.append("replay_diagnostics")
    if live.get("status") == "missing":
        return "ready_for_fresh_live_retakes", "fresh_live_retakes", "run fresh live retakes only after strict MP4 and replay gates pass", completed
    if live.get("status") != "passed":
        return "v7e_strict_gates_failed", "fresh_live_retakes", _fallback_next_action(), completed
    completed.append("fresh_live_retakes")
    return "v7e_promotion_candidate", None, "manually promote v7e only after reviewing preserved strict-gate diagnostics", completed


def _fallback_next_action() -> str:
    return "keep_v4_fallback_preserve_v7e_diagnostics_and_document_next_targeted_branch"


def _planned_commands(*, project_root: Path, config: V7EStage1StrictValidationPreflightConfig) -> dict[str, list[str]]:
    stage1_profile = _display_path(_resolve_path(project_root, config.stage1_profile_json), base=project_root)
    stage2_profile = _display_path(_resolve_path(project_root, config.stage2_reuse_profile_json), base=project_root)
    original20_output = _display_path(_resolve_path(project_root, config.original20_validation_root), base=project_root)
    heldout15_output = _display_path(_resolve_path(project_root, config.heldout15_validation_root), base=project_root)
    event_manifest = _display_path(_resolve_path(project_root, config.event_manifest_path), base=project_root)
    base = [
        "python",
        "-m",
        "embodied_rps.tools.evaluate_two_stage_skeleton_video_predictions",
        "--stage1-profile",
        stage1_profile,
        "--stage2-profile",
        stage2_profile,
        "--paper-wait-nonterminal-for-transitions",
    ]
    return {
        "write_strict_validation_preflight": ["python", "-m", "embodied_rps.tools.write_v7e_stage1_strict_validation_preflight"],
        "validate_original20": [
            *base,
            "--input-root",
            "<original20-from-dataset-glob>",
            "--output-root",
            original20_output,
            "--event-output",
            event_manifest,
            "--expected-count",
            "20",
            "--label-mode",
            "transition",
        ],
        "validate_heldout15": [
            *base,
            "--input-root",
            "<heldout15-from-dataset-glob>",
            "--output-root",
            heldout15_output,
            "--expected-count",
            "15",
            "--label-mode",
            "final-label",
        ],
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# V7e Stage1 Strict Validation Preflight",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Blocking stage: `{summary.get('blocking_stage')}`",
        f"- Next action: `{summary.get('next_action')}`",
        f"- Promotion eligible: `{_mapping(summary.get('promotion_decision')).get('may_promote_v7e')}`",
        f"- Stage2 policy: `{summary.get('stage2_policy')}`",
        "",
        "## Completed Stages",
        "",
    ]
    completed = summary.get("completed_stages")
    if isinstance(completed, Sequence) and not isinstance(completed, (str, bytes)) and completed:
        for stage in completed:
            lines.append(f"- `{stage}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Stage Outputs", ""])
    outputs = summary.get("stage_outputs")
    if isinstance(outputs, Mapping):
        for name, value in outputs.items():
            status = value.get("status") if isinstance(value, Mapping) else None
            lines.append(f"- `{name}`: status=`{status}`")
    return "\n".join(lines)


def _sanitize_discovery(discovery: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": discovery.get("status"),
        "dataset_search_root": _display_dataset_path(discovery.get("dataset_search_root")),
        "original20_mp4_count": discovery.get("original20_mp4_count"),
        "heldout15_mp4_count": discovery.get("heldout15_mp4_count"),
        "heldout_label_counts": discovery.get("heldout_label_counts"),
    }


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(base.resolve(strict=False)).as_posix()
    except ValueError:
        dataset_path = _display_dataset_path(path.as_posix())
        if dataset_path is not None:
            return dataset_path
        return path.name


def _display_dataset_path(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).replace("\\", "/")
    lower = normalized.lower()
    prefix = "d:/dataset"
    if lower == prefix:
        return "dataset:/"
    if lower.startswith(f"{prefix}/"):
        return f"dataset:/{normalized[len(prefix) + 1:]}"
    return None


def _config_summary(*, project_root: Path, config: V7EStage1StrictValidationPreflightConfig) -> dict[str, Any]:
    summary = asdict(config)
    summary["project_root"] = "."
    for key, value in list(summary.items()):
        if isinstance(value, Path):
            summary[key] = _display_path(_resolve_path(project_root, value), base=project_root)
    return _json_ready(summary)  # type: ignore[return-value]


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return []


def _optional_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _first_int(mapping: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = _optional_int(mapping.get(key))
        if value is not None:
            return value
    return None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nonempty(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) > 0


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "BRANCH_LABEL",
    "V7EStage1StrictValidationPreflightConfig",
    "write_v7e_stage1_strict_validation_preflight",
]
