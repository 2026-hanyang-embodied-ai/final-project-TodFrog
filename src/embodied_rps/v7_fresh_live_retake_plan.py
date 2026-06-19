"""Status-only fresh live retake planning for the v7 RPS branch."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


COUNTER_ACTIONS = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
GESTURES: tuple[str, ...] = ("rock", "paper", "scissors")


@dataclass(frozen=True)
class V7FreshLiveRetakePlanConfig:
    """Inputs for writing the v7 fresh-live retake plan without capturing video."""

    seed_package_root: Path = Path("artifacts/real_skeleton_v7_rps_seed_package_20260617")
    dataset_root: Path = Path("artifacts/real_guided_three_class_wait_expanded_v7_rps_pose_20260617")
    profile_json_path: Path = Path("results/model_profiles/real_skeleton_three_class_wait_v7_rps_pose_tcn_ensemble.json")
    approved_segment_replay_root: Path = Path("artifacts/real_skeleton_v7_approved_segment_replay_20260617")
    output_root: Path = Path("artifacts/real_skeleton_v7_fresh_live_retakes_20260617")
    strict_live_wrapper: Path = Path("artifacts/realtime_demo_launch_20260616/24_run_live_demo_operator_confirmed_strict.ps1")


def write_v7_fresh_live_retake_plan(config: V7FreshLiveRetakePlanConfig) -> dict[str, object]:
    """Write fresh-live retake plan artifacts without launching capture."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    seed_status = _seed_status(config.seed_package_root)
    dataset_status = _dataset_status(config.dataset_root)
    profile_status = _profile_status(config.profile_json_path)
    approved_segment_status = _approved_segment_replay_status(config.approved_segment_replay_root)
    wrapper_status = _strict_wrapper_status(config.strict_live_wrapper)
    manifest_rows = _retake_rows(config.strict_live_wrapper)
    manifest_path = config.output_root / "fresh_live_retake_manifest.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    blocking_stage, status, next_action = _status(
        seed_status=seed_status,
        dataset_status=dataset_status,
        profile_status=profile_status,
        approved_segment_status=approved_segment_status,
        wrapper_status=wrapper_status,
    )
    summary = {
        "status": status,
        "blocking_stage": blocking_stage,
        "next_action": next_action,
        "output_root": config.output_root.as_posix(),
        "seed_package": seed_status,
        "dataset": dataset_status,
        "profile": profile_status,
        "approved_segment_replay": approved_segment_status,
        "strict_live_wrapper": wrapper_status,
        "retake_manifest": manifest_path.as_posix(),
        "retake_count": len(manifest_rows),
        "expected_gestures": list(GESTURES),
        "commands": {
            "status_only": "python -m embodied_rps.tools.write_v7_fresh_live_retake_plan",
            "strict_live_wrapper": _quote_powershell_command(config.strict_live_wrapper),
        },
        "notes": [
            "This plan does not launch camera capture, run inference, write retake_summary.json, or promote v7.",
            "Fresh retakes are only valid after v7 profile training, original20/heldout15 validation, archived replay, and approved segment replay pass.",
            "Run one ground-truthed live attempt each for rock, paper, and scissors; archive and review every attempt before promotion.",
        ],
    }
    _write_summary(config.output_root, summary)
    return summary


def _retake_rows(strict_wrapper: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for order, gesture in enumerate(GESTURES, start=1):
        rows.append(
            {
                "order": order,
                "expected_actual_gesture": gesture,
                "expected_robot_action": COUNTER_ACTIONS[gesture],
                "operator_command": _quote_powershell_command(strict_wrapper),
                "operator_entry": gesture,
                "acceptance_gate": _acceptance_gate_for_gesture(gesture),
                "ground_truth_required": True,
                "manual_visual_review_required": True,
                "archive_required": True,
                "profile_required": "results/model_profiles/real_skeleton_three_class_wait_v7_rps_pose_tcn_ensemble.json",
            }
        )
    return rows


def _acceptance_gate_for_gesture(gesture: str) -> str:
    if gesture == "rock":
        return "ground_truth_passed_and_live_rock_retake_gate_passed_without_binary_false_trigger"
    if gesture == "paper":
        return "ground_truth_passed_with_user_paper_prediction_and_robot_scissors_response"
    return "ground_truth_passed_with_user_scissors_prediction_and_robot_rock_response"


def _seed_status(seed_package_root: Path) -> dict[str, object]:
    seed_npz = seed_package_root / "v7_rps_seed_dataset.npz"
    seed_metadata = seed_package_root / "seed_metadata.jsonl"
    seed_quality_summary = seed_package_root / "seed_quality_summary.csv"
    summary_path = seed_package_root / "seed_package_summary.json"
    required_files = {
        "missing_seed_npz": seed_npz,
        "missing_seed_metadata": seed_metadata,
        "missing_seed_quality_summary": seed_quality_summary,
        "missing_seed_package_summary": summary_path,
    }
    existing_count = sum(1 for path in required_files.values() if path.exists())
    failures = [
        {"code": code, "path": path.as_posix()}
        for code, path in required_files.items()
        if not path.exists()
    ]
    status_base = {
        "seed_npz": seed_npz.as_posix(),
        "seed_metadata": seed_metadata.as_posix(),
        "seed_quality_summary": seed_quality_summary.as_posix(),
        "summary": summary_path.as_posix(),
    }
    if existing_count == 0:
        return {"status": "missing", **status_base, "failures": failures}
    if failures:
        return {"status": "invalid", **status_base, "failures": failures}
    summary = _read_json_object(summary_path)
    if summary is None or summary.get("status") != "passed":
        return {
            "status": "invalid",
            **status_base,
            "seed_package_status": None if summary is None else summary.get("status"),
            "failures": [{"code": "seed_package_summary_not_passed", "path": summary_path.as_posix()}],
        }
    return {"status": "passed", **status_base}


def _dataset_status(dataset_root: Path) -> dict[str, object]:
    validation_path = dataset_root / "validation_summary.json"
    if not dataset_root.exists():
        return {"status": "missing", "dataset_root": dataset_root.as_posix()}
    validation = _read_json_object(validation_path)
    if validation is None:
        return {"status": "invalid", "dataset_root": dataset_root.as_posix(), "validation_summary": validation_path.as_posix()}
    return {
        "status": "passed" if validation.get("status") == "passed" else "invalid",
        "dataset_root": dataset_root.as_posix(),
        "validation_summary": validation_path.as_posix(),
        "validation_status": validation.get("status"),
        "target_counts": validation.get("target_counts"),
    }


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
    profile = _read_json_object(profile_json_path)
    if profile is None:
        return {"status": "invalid", "profile_json": profile_json_path.as_posix(), "profile_pt": profile_pt_path.as_posix()}
    if profile.get("model") != "tcn":
        return {"status": "wrong_model", "profile_json": profile_json_path.as_posix(), "model": profile.get("model")}
    if profile.get("label_names") != ["rock", "paper", "scissors"]:
        return {"status": "invalid_labels", "profile_json": profile_json_path.as_posix(), "label_names": profile.get("label_names")}
    return {"status": "passed", "profile_json": profile_json_path.as_posix(), "profile_pt": profile_pt_path.as_posix()}


def _approved_segment_replay_status(root: Path) -> dict[str, object]:
    for filename in ("replay_summary.json", "validation_summary.json"):
        path = root / filename
        summary = _read_json_object(path)
        if summary is None:
            continue
        passed = bool(summary.get("passed")) or summary.get("status") == "passed"
        return {"status": "passed" if passed else "failed", "summary_path": path.as_posix(), "passed": passed}
    return {"status": "missing", "root": root.as_posix(), "expected_filenames": ["replay_summary.json", "validation_summary.json"]}


def _strict_wrapper_status(path: Path) -> dict[str, object]:
    exists = path.exists()
    return {"status": "present" if exists else "missing", "path": path.as_posix(), "exists": exists}


def _status(
    *,
    seed_status: Mapping[str, object],
    dataset_status: Mapping[str, object],
    profile_status: Mapping[str, object],
    approved_segment_status: Mapping[str, object],
    wrapper_status: Mapping[str, object],
) -> tuple[str | None, str, str]:
    if seed_status.get("status") != "passed":
        return "seed_package", "awaiting_v7_seed_package", "approve reviewed segments and build the v7 seed package first"
    if dataset_status.get("status") != "passed":
        return "dataset", "awaiting_v7_dataset", "generate the approved balanced v7_rps_pose dataset first"
    if profile_status.get("status") != "passed":
        return "profile", "awaiting_v7_profile", "train and export the v7 TCN profile before live retakes"
    if approved_segment_status.get("status") != "passed":
        return "approved_segment_replay", "awaiting_approved_segment_replay", "run approved segment replay before fresh live retakes"
    if wrapper_status.get("status") != "present":
        return "strict_live_wrapper", "missing_strict_live_wrapper", "restore the operator-confirmed strict live wrapper"
    return None, "ready_for_fresh_live_retakes", "run ground-truthed rock, paper, and scissors retakes with the strict wrapper"


def _quote_powershell_command(path: Path) -> str:
    return f"powershell -ExecutionPolicy Bypass -File {path.as_posix()}"


def _read_json_object(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_summary(output_root: Path, summary: Mapping[str, object]) -> None:
    (output_root / "v7_fresh_live_retake_plan.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "v7_fresh_live_retake_plan.md").write_text(_summary_markdown(summary), encoding="utf-8")


def _summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V7 Fresh Live Retake Plan",
        "",
        f"- status: `{summary.get('status')}`",
        f"- blocking stage: `{summary.get('blocking_stage')}`",
        f"- next action: `{summary.get('next_action')}`",
        f"- retake count: `{summary.get('retake_count')}`",
        f"- manifest: `{summary.get('retake_manifest')}`",
        "",
        "## Required Retakes",
        "",
        "| Gesture | Expected robot action | Acceptance gate |",
        "|---|---|---|",
    ]
    for row in _retake_rows(Path("artifacts/realtime_demo_launch_20260616/24_run_live_demo_operator_confirmed_strict.ps1")):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["expected_actual_gesture"]),
                    str(row["expected_robot_action"]),
                    str(row["acceptance_gate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    notes = summary.get("notes", [])
    if isinstance(notes, Sequence) and not isinstance(notes, (str, bytes)):
        for note in notes:
            lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


__all__ = ["V7FreshLiveRetakePlanConfig", "write_v7_fresh_live_retake_plan"]
