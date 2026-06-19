"""Build SCHUNK response-plan artifacts from validated skeleton prediction events."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from embodied_rps.domain import Gesture
from embodied_rps.real_skeleton_video_eval import COUNTER_MOVES
from embodied_rps.schunk import load_schunk_pose_config


def build_schunk_response_plan(
    events: Sequence[Mapping[str, object]],
    *,
    pose_config_path: Path,
    response_window_s: float = 1.0,
    deadline_progress: float = 0.50,
    wait_pose: Gesture = "paper",
) -> dict[str, object]:
    """Map validated prediction events to prompt-cycle SCHUNK response commands."""

    if response_window_s <= 0.0:
        raise ValueError("response_window_s must be positive")
    if not 0.0 < deadline_progress <= 1.0:
        raise ValueError("deadline_progress must be in (0, 1]")

    pose_config = load_schunk_pose_config(pose_config_path)
    if wait_pose not in pose_config.gestures:
        raise ValueError(f"wait_pose {wait_pose!r} is not defined in {pose_config_path}")

    plan_events: list[dict[str, object]] = []
    invalid_events: list[dict[str, object]] = []
    action_counts: Counter[str] = Counter()
    start_times: list[float] = []
    deadline_s = response_window_s * deadline_progress

    for index, event in enumerate(events):
        clip_id = _required_string(event, "clip_id")
        predicted = _required_string(event, "predicted_final_gesture")
        counter_move = _required_string(event, "selected_counter_move")
        robot_action = _required_string(event, "selected_robot_action")
        decision_progress = _required_float(event, "decision_progress")
        expected_counter = COUNTER_MOVES.get(cast(object, predicted))
        event_invalid: list[str] = []
        if expected_counter is None:
            event_invalid.append("invalid_predicted_final_gesture")
        elif counter_move != expected_counter:
            event_invalid.append("selected_counter_move_does_not_counter_prediction")
        if robot_action not in pose_config.gestures:
            event_invalid.append("selected_robot_action_missing_from_pose_config")
        if not 0.0 <= decision_progress <= 1.0:
            event_invalid.append("decision_progress_out_of_range")

        demo_start = max(0.0, min(response_window_s, decision_progress * response_window_s))
        within_deadline = demo_start <= deadline_s
        if not within_deadline:
            event_invalid.append("demo_response_starts_after_deadline")

        if event_invalid:
            invalid_events.append({"clip_id": clip_id, "reasons": event_invalid})

        command_pose = robot_action if robot_action in pose_config.gestures else wait_pose
        command_targets = _joint_targets(pose_config.gestures[cast(Gesture, command_pose)])
        wait_targets = _joint_targets(pose_config.gestures[wait_pose])
        item = {
            "event_index": index,
            "clip_id": clip_id,
            "source_path": _optional_string(event.get("source_path")),
            "transition_label": _optional_string(event.get("transition_label")),
            "true_final_gesture": _optional_string(event.get("true_final_gesture")),
            "predicted_final_gesture": predicted,
            "selected_counter_move": counter_move,
            "selected_robot_action": robot_action,
            "wait_pose": wait_pose,
            "source_decision_time_s": _required_float(event, "decision_time_s"),
            "source_recommended_response_start_time_s": _optional_float(event.get("recommended_response_start_time_s")),
            "decision_frame": int(_required_float(event, "decision_frame")),
            "decision_progress": decision_progress,
            "demo_response_start_time_s": demo_start,
            "demo_response_deadline_s": deadline_s,
            "response_window_s": response_window_s,
            "within_demo_deadline": within_deadline,
            "confidence": _optional_float(event.get("confidence")),
            "confidence_margin": _optional_float(event.get("confidence_margin")),
            "overlay_path": _optional_string(event.get("overlay_path")),
            "phases": [
                {"phase": "wait", "time_s": 0.0, "pose": wait_pose, "joint_targets": wait_targets},
                {"phase": "counter_command", "time_s": demo_start, "pose": command_pose, "joint_targets": command_targets},
                {"phase": "hold", "time_s": response_window_s, "pose": command_pose, "joint_targets": command_targets},
            ],
        }
        plan_events.append(item)
        action_counts[robot_action] += 1
        start_times.append(demo_start)

    summary = {
        "passed": len(invalid_events) == 0 and len(plan_events) > 0,
        "event_count": len(plan_events),
        "pose_config_path": pose_config_path.as_posix(),
        "response_window_s": response_window_s,
        "deadline_progress": deadline_progress,
        "demo_response_deadline_s": deadline_s,
        "wait_pose": wait_pose,
        "action_counts": dict(sorted(action_counts.items())),
        "min_demo_response_start_time_s": min(start_times) if start_times else None,
        "max_demo_response_start_time_s": max(start_times) if start_times else None,
        "mean_demo_response_start_time_s": (sum(start_times) / len(start_times)) if start_times else None,
        "all_within_demo_deadline": all(bool(item["within_demo_deadline"]) for item in plan_events),
        "invalid_events": invalid_events,
    }
    return {"summary": summary, "events": plan_events}


def write_schunk_response_plan_artifacts(
    *,
    event_manifest_path: Path,
    pose_config_path: Path,
    output_root: Path,
    expected_count: int | None = None,
    response_window_s: float = 1.0,
    deadline_progress: float = 0.50,
    wait_pose: Gesture = "paper",
) -> dict[str, object]:
    """Write response-plan JSONL, timeline CSV, markdown, and validation summary."""

    events = _load_jsonl_records(event_manifest_path)
    plan = build_schunk_response_plan(
        events,
        pose_config_path=pose_config_path,
        response_window_s=response_window_s,
        deadline_progress=deadline_progress,
        wait_pose=wait_pose,
    )
    summary = dict(cast(Mapping[str, object], plan["summary"]))
    if expected_count is not None:
        summary["expected_count"] = expected_count
        if summary.get("event_count") != expected_count:
            summary["passed"] = False
            invalid = list(cast(Sequence[object], summary.get("invalid_events", [])))
            invalid.append({"clip_id": None, "reasons": ["event_count_mismatch"]})
            summary["invalid_events"] = invalid

    output_root.mkdir(parents=True, exist_ok=True)
    response_plan_path = output_root / "response_plan.jsonl"
    timeline_csv_path = output_root / "response_timeline.csv"
    summary_json_path = output_root / "validation_summary.json"
    summary_md_path = output_root / "response_plan_summary.md"
    plan_events = [cast(Mapping[str, object], item) for item in cast(Sequence[object], plan["events"])]
    _write_plan_jsonl(response_plan_path, plan_events)
    _write_timeline_csv(timeline_csv_path, plan_events)
    summary.update(
        {
            "event_manifest_path": event_manifest_path.as_posix(),
            "response_plan_jsonl": response_plan_path.as_posix(),
            "response_timeline_csv": timeline_csv_path.as_posix(),
            "response_plan_summary_md": summary_md_path.as_posix(),
        }
    )
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary_md(summary_md_path, summary)
    return summary


def _load_jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        loaded: object = json.loads(line)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} contains a non-object JSONL row")
        records.append({str(key): value for key, value in loaded.items()})
    return records


def _write_plan_jsonl(path: Path, events: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")


def _write_timeline_csv(path: Path, events: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "clip_id",
        "phase",
        "time_s",
        "pose",
        "selected_robot_action",
        "predicted_final_gesture",
        "true_final_gesture",
        "decision_progress",
        "source_decision_time_s",
        "within_demo_deadline",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            phases = event.get("phases")
            if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
                continue
            for phase in phases:
                if not isinstance(phase, Mapping):
                    continue
                writer.writerow(
                    {
                        "clip_id": event.get("clip_id"),
                        "phase": phase.get("phase"),
                        "time_s": phase.get("time_s"),
                        "pose": phase.get("pose"),
                        "selected_robot_action": event.get("selected_robot_action"),
                        "predicted_final_gesture": event.get("predicted_final_gesture"),
                        "true_final_gesture": event.get("true_final_gesture"),
                        "decision_progress": event.get("decision_progress"),
                        "source_decision_time_s": event.get("source_decision_time_s"),
                        "within_demo_deadline": event.get("within_demo_deadline"),
                    }
                )


def _write_summary_md(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "# SCHUNK Response Event Bridge",
        "",
        f"Status: `{summary.get('passed')}`",
        f"Event count: `{summary.get('event_count')}`",
        f"Wait pose: `{summary.get('wait_pose')}`",
        f"Response window: `{summary.get('response_window_s')}` s",
        f"Deadline progress: `{summary.get('deadline_progress')}`",
        f"Max demo response start: `{summary.get('max_demo_response_start_time_s')}` s",
        "",
        "## Outputs",
        "",
        f"- Response plan JSONL: `{summary.get('response_plan_jsonl')}`",
        f"- Response timeline CSV: `{summary.get('response_timeline_csv')}`",
        "",
        "## Scope",
        "",
        "This bridge validates metadata and SCHUNK pose targets only. It does not launch Isaac rendering.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _joint_targets(values: Mapping[str, float]) -> dict[str, float]:
    return {str(name): float(value) for name, value in values.items()}


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"event field {key} must be a non-empty string")
    return value


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _required_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"event field {key} must be numeric")
    return float(value)


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
