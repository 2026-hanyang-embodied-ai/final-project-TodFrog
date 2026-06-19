"""Index archived realtime demo rehearsal runs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeDemoRunArchiveIndexConfig:
    """Configuration for summarizing archived realtime demo runs."""

    archive_root: Path = Path("artifacts/realtime_demo_run_archive_20260616")
    output_root: Path = Path("artifacts/realtime_demo_run_archive_20260616")
    manual_review_decisions: Path | None = Path(
        "artifacts/realtime_demo_manual_review_20260616/manual_review_decisions.json"
    )


def summarize_realtime_demo_run_archives(config: RealtimeDemoRunArchiveIndexConfig) -> dict[str, object]:
    """Build an index over all archive manifests under the archive root."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    index_json = config.output_root / "run_archive_index.json"
    index_md = config.output_root / "run_archive_index.md"
    runs = _load_runs(config.archive_root)
    manual_reviews = _load_manual_reviews(config.manual_review_decisions)
    run_records = [_run_record(run, manual_reviews=manual_reviews) for run in runs]
    status_counts = Counter(str(record.get("archive_status")) for record in run_records)
    operator_counts = Counter(str(record.get("operator_state")) for record in run_records)
    candidates = [record for record in run_records if _is_final_video_candidate(record)]
    latest_run = run_records[-1] if run_records else None
    latest_candidate = candidates[-1] if candidates else None
    summary: dict[str, object] = {
        "index_status": _index_status(runs=runs, candidates=candidates),
        "archive_root": config.archive_root.as_posix(),
        "manual_review_decisions": config.manual_review_decisions.as_posix()
        if config.manual_review_decisions is not None
        else None,
        "run_count": len(runs),
        "latest_run_id": latest_run.get("run_id") if latest_run else None,
        "latest_final_video_candidate": latest_candidate,
        "status_counts": dict(sorted(status_counts.items())),
        "operator_state_counts": dict(sorted(operator_counts.items())),
        "runs": run_records,
        "outputs": {
            "index_json": index_json.as_posix(),
            "index_md": index_md.as_posix(),
        },
        "claim_scope": "index over archived run manifests; does not run camera capture, model inference, verification, or rendering",
    }
    index_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    index_md.write_text(_index_markdown(summary), encoding="utf-8")
    return summary


def _load_runs(archive_root: Path) -> list[dict[str, Any]]:
    if not archive_root.exists():
        return []
    runs: list[dict[str, Any]] = []
    for manifest_path in sorted(archive_root.glob("*/archive_manifest.json")):
        loaded = _read_json_if_exists(manifest_path)
        if loaded is None:
            continue
        loaded["_manifest_path"] = manifest_path.as_posix()
        runs.append(loaded)
    return sorted(runs, key=lambda run: str(run.get("run_id") or ""))


def _is_final_video_candidate(run: dict[str, Any]) -> bool:
    return (
        bool(run.get("primary_video"))
        and bool(run.get("live_response_decision_frame"))
        and run.get("ground_truth_passed") is True
        and run.get("manual_review_status") == "approved"
        and _live_rock_gate_allows_candidate(run)
        and (
            run.get("archive_status") == "archived_live_demo_candidate"
            or run.get("operator_state") == "ready_for_final_video"
        )
    )


def _run_record(run: dict[str, Any], *, manual_reviews: dict[str, dict[str, Any]]) -> dict[str, object]:
    if run is None:
        raise ValueError("run record cannot be None")
    copied = _dict_value(run, "copied_files")
    missing = _dict_value(run, "missing_files")
    run_id = str(run.get("run_id") or "")
    archive_dir = Path(str(run.get("archive_dir") or ""))
    postcapture_summary = _read_json_if_exists(archive_dir / "reports" / "postcapture_summary.json")
    demo_gate = _dict_value(postcapture_summary or {}, "demo_success_gate")
    live_rock_gate = _read_json_if_exists(archive_dir / "reports" / "live_rock_retake_gate.json") or {}
    live_rock_response = _dict_value(live_rock_gate, "response_window")
    live_rock_first_binary = _dict_value(live_rock_response, "first_binary_decision")
    manual_review = manual_reviews.get(run_id, {})
    expected_actual_gesture = _first_non_empty(
        demo_gate.get("expected_actual_gesture"),
        run.get("expected_actual_gesture"),
    )
    ground_truth_match = _optional_bool(_first_non_none(demo_gate.get("ground_truth_match"), run.get("ground_truth_match")))
    robot_action_match = _optional_bool(_first_non_none(demo_gate.get("robot_action_match"), run.get("robot_action_match")))
    gate_passed = _optional_bool(_first_non_none(demo_gate.get("passed"), run.get("ground_truth_passed")))
    if expected_actual_gesture is not None:
        ground_truth_passed = gate_passed is True and ground_truth_match is True and robot_action_match is True
    else:
        ground_truth_passed = _optional_bool(run.get("ground_truth_passed"))
    manual_review_status = _first_non_empty(
        manual_review.get("manual_review_status"),
        run.get("manual_review_status"),
    )
    manual_review_notes = _first_non_empty(
        manual_review.get("manual_review_notes"),
        run.get("manual_review_notes"),
    )
    return {
        "run_id": run_id,
        "archive_status": run.get("archive_status"),
        "operator_state": run.get("operator_state"),
        "operator_recommended_exit_code": run.get("operator_recommended_exit_code"),
        "archive_dir": run.get("archive_dir"),
        "primary_video": copied.get("live_composite_mp4"),
        "live_overlay_video": copied.get("live_overlay_video"),
        "live_response_decision_frame": copied.get("live_response_decision_frame"),
        "live_response_prompt_diagnostic_frame": copied.get("live_response_prompt_diagnostic_frame"),
        "expected_actual_gesture": expected_actual_gesture,
        "ground_truth_passed": ground_truth_passed,
        "ground_truth_match": ground_truth_match,
        "robot_action_match": robot_action_match,
        "live_rock_retake_gate_status": _first_non_empty(
            live_rock_gate.get("gate_status"),
            run.get("live_rock_retake_gate_status"),
        ),
        "live_rock_retake_gate_passed": _optional_bool(
            _first_non_none(live_rock_gate.get("passed"), run.get("live_rock_retake_gate_passed"))
        ),
        "live_rock_retake_binary_decision_frame_count": _optional_int(
            _first_non_none(
                live_rock_response.get("binary_decision_frame_count"),
                run.get("live_rock_retake_binary_decision_frame_count"),
            )
        ),
        "live_rock_retake_confirmed_binary_decision_count": _optional_int(
            _first_non_none(
                live_rock_response.get("confirmed_binary_decision_count"),
                run.get("live_rock_retake_confirmed_binary_decision_count"),
            )
        ),
        "live_rock_retake_first_binary_decision": _first_non_empty(
            live_rock_first_binary.get("decision_state"),
            run.get("live_rock_retake_first_binary_decision"),
        ),
        "live_rock_retake_first_binary_robot_action": _first_non_empty(
            live_rock_first_binary.get("robot_action"),
            run.get("live_rock_retake_first_binary_robot_action"),
        ),
        "manual_review_status": manual_review_status,
        "manual_review_notes": manual_review_notes,
        "copied_file_count": run.get("copied_file_count", len(copied)),
        "missing_file_count": run.get("missing_file_count", len(missing)),
        "manifest_path": run.get("_manifest_path"),
    }


def _live_rock_gate_allows_candidate(run: dict[str, Any]) -> bool:
    if run.get("expected_actual_gesture") != "rock":
        return True
    return run.get("live_rock_retake_gate_passed") is True


def _index_status(*, runs: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> str:
    if candidates:
        return "has_final_video_candidate"
    if runs:
        return "has_archived_runs"
    return "no_archived_runs"


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _load_manual_reviews(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    loaded = _read_json_if_exists(path)
    if loaded is None:
        return {}
    reviews = _dict_value(loaded, "run_reviews")
    return {str(run_id): review for run_id, review in reviews.items() if isinstance(review, dict)}


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _first_non_empty(*values: object) -> str | None:
    for value in values:
        if value is None:
            continue
        parsed = str(value).strip()
        if parsed:
            return parsed
    return None


def _first_non_none(*values: object) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


def _optional_bool(value: object | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    parsed = str(value).strip().lower()
    if parsed in {"1", "true", "yes", "y"}:
        return True
    if parsed in {"0", "false", "no", "n"}:
        return False
    return None


def _optional_int(value: object | None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except ValueError:
        return None


def _index_markdown(summary: dict[str, object]) -> str:
    runs = summary.get("runs", [])
    lines = [
        "# Realtime Demo Run Archive Index",
        "",
        f"- Index status: `{summary.get('index_status')}`",
        f"- Run count: `{summary.get('run_count')}`",
        f"- Latest run ID: `{summary.get('latest_run_id')}`",
        "",
        "## Latest Final Video Candidate",
        "",
    ]
    candidate = summary.get("latest_final_video_candidate")
    if isinstance(candidate, dict) and candidate:
        for key in sorted(candidate):
            lines.append(f"- `{key}`: `{candidate[key]}`")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Run ID | Archive status | Operator state | Exit code | Primary video |",
            "|---|---|---|---:|---|",
        ]
    )
    if isinstance(runs, list):
        for raw_run in runs:
            if not isinstance(raw_run, dict):
                continue
            lines.append(
                "| "
                f"`{raw_run.get('run_id')}` | "
                f"`{raw_run.get('archive_status')}` | "
                f"`{raw_run.get('operator_state')}` | "
                f"`{raw_run.get('operator_recommended_exit_code')}` | "
                f"`{raw_run.get('primary_video')}` |"
            )
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoRunArchiveIndexConfig", "summarize_realtime_demo_run_archives"]
