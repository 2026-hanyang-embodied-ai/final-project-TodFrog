"""Build a submission-facing packet from the selected realtime demo candidate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeDemoSubmissionPacketConfig:
    """Input and output paths for the final demo submission candidate packet."""

    output_root: Path = Path("artifacts/realtime_demo_submission_packet_20260616")
    final_candidate: Path = Path("artifacts/realtime_demo_final_candidate_20260616/final_demo_candidate.json")


def build_realtime_demo_submission_packet(config: RealtimeDemoSubmissionPacketConfig) -> dict[str, object]:
    """Write a submission-facing manifest for the current final demo candidate."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    output_json = config.output_root / "submission_candidate_packet.json"
    output_md = config.output_root / "submission_candidate_packet.md"
    candidate = _read_json_if_exists(config.final_candidate) or {}

    summary = _packet_summary(candidate=candidate, config=config, output_json=output_json, output_md=output_md)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(_packet_markdown(summary), encoding="utf-8")
    return summary


def _packet_summary(
    *,
    candidate: dict[str, Any],
    config: RealtimeDemoSubmissionPacketConfig,
    output_json: Path,
    output_md: Path,
) -> dict[str, object]:
    candidate_ready = candidate.get("ready_for_video_packaging") is True
    candidate_status = str(candidate.get("candidate_status") or "missing_final_candidate")
    final_video_path = _resolve_candidate_path(candidate, "primary_video")
    response_decision_frame_path = _resolve_candidate_path(candidate, "live_response_decision_frame")
    missing = _missing_required_artifacts(
        candidate=candidate,
        candidate_ready=candidate_ready,
        final_video_path=final_video_path,
        response_decision_frame_path=response_decision_frame_path,
    )
    ready = not missing
    if ready:
        packet_status = "ready_for_manual_submission_review"
        next_action = (
            "Inspect the final demo video, upload it to YouTube if accepted, then replace README/report placeholders "
            "with the public demo-video link."
        )
    elif "final_video_candidate" in missing:
        packet_status = "awaiting_final_video_candidate"
        next_action = "Run the live demo pipeline until the final candidate selector finds an archived live composite."
        if candidate.get("missing_reason") == "manual_review_rejected":
            packet_status = "awaiting_operator_ground_truth_retake"
            next_action = (
                "Retake the live demo with operator ground truth; the previous complete media was rejected by "
                "manual visual review."
            )
    else:
        packet_status = "candidate_media_missing"
        next_action = "Inspect the archive folder and rerun archiving or the live demo pipeline so the selected video file exists."

    return {
        "packet_status": packet_status,
        "ready_for_submission_linking": ready,
        "selected_run_id": candidate.get("selected_run_id"),
        "candidate_status": candidate_status,
        "candidate_missing_reason": candidate.get("missing_reason"),
        "blocked_run_id": candidate.get("blocked_run_id"),
        "blocked_primary_video": candidate.get("blocked_primary_video"),
        "final_video_path": final_video_path.as_posix() if final_video_path else None,
        "final_video_exists": bool(final_video_path and final_video_path.exists()),
        "response_decision_frame_path": response_decision_frame_path.as_posix() if response_decision_frame_path else None,
        "response_decision_frame_exists": bool(
            response_decision_frame_path and response_decision_frame_path.exists()
        ),
        "live_overlay_video": candidate.get("live_overlay_video"),
        "archive_dir": candidate.get("archive_dir"),
        "missing_required_artifacts": missing,
        "submission_next_steps": _submission_next_steps(ready),
        "next_action": next_action,
        "inputs": {
            "final_candidate": config.final_candidate.as_posix(),
        },
        "outputs": {
            "packet_json": output_json.as_posix(),
            "packet_md": output_md.as_posix(),
        },
        "claim_scope": "submission candidate packet over existing final-candidate artifacts; does not upload videos or edit README/report links",
    }


def _missing_required_artifacts(
    *,
    candidate: dict[str, Any],
    candidate_ready: bool,
    final_video_path: Path | None,
    response_decision_frame_path: Path | None,
) -> list[str]:
    missing: list[str] = []
    if not candidate or not candidate_ready:
        missing.append("final_video_candidate")
    if candidate_ready and (final_video_path is None or not final_video_path.exists()):
        missing.append("final_candidate_primary_video_file")
    if candidate_ready and (response_decision_frame_path is None or not response_decision_frame_path.exists()):
        missing.append("response_decision_frame_file")
    return missing


def _resolve_candidate_path(candidate: dict[str, Any], key: str) -> Path | None:
    value = str(candidate.get(key) or "")
    if not value:
        return None
    candidate_path = Path(value)
    if candidate_path.is_absolute() or candidate_path.exists():
        return candidate_path
    archive_dir = str(candidate.get("archive_dir") or "")
    if archive_dir:
        archive_path = Path(archive_dir)
        if _path_is_relative_to(candidate_path, archive_path):
            return candidate_path
        return archive_path / candidate_path
    return candidate_path


def _path_is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _submission_next_steps(ready: bool) -> list[str]:
    if not ready:
        return [
            "live_demo_pipeline",
            "archive_index_refresh",
            "final_candidate_selection",
        ]
    return [
        "manual_video_review",
        "youtube_demo_video_upload",
        "readme_demo_video_link",
        "report_demo_video_link",
    ]


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _packet_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Realtime Demo Submission Candidate Packet",
        "",
        f"- Packet status: `{summary.get('packet_status')}`",
        f"- Ready for submission linking: `{summary.get('ready_for_submission_linking')}`",
        f"- Selected run ID: `{summary.get('selected_run_id')}`",
        f"- Candidate status: `{summary.get('candidate_status')}`",
        f"- Final video path: `{summary.get('final_video_path')}`",
        f"- Final video exists: `{summary.get('final_video_exists')}`",
        f"- Response decision frame path: `{summary.get('response_decision_frame_path')}`",
        f"- Response decision frame exists: `{summary.get('response_decision_frame_exists')}`",
        f"- Next action: {summary.get('next_action')}",
        "",
        "## Missing Required Artifacts",
        "",
    ]
    missing = summary.get("missing_required_artifacts", [])
    if isinstance(missing, list) and missing:
        lines.extend(f"- `{item}`" for item in missing)
    else:
        lines.append("- None")
    lines.extend(["", "## Submission Next Steps", ""])
    steps = summary.get("submission_next_steps", [])
    if isinstance(steps, list):
        lines.extend(f"- `{step}`" for step in steps)
    lines.extend(["", "## Inputs", ""])
    inputs = summary.get("inputs", {})
    if isinstance(inputs, dict):
        for key in sorted(inputs):
            lines.append(f"- `{key}`: `{inputs[key]}`")
    lines.extend(["", "## Outputs", ""])
    outputs = summary.get("outputs", {})
    if isinstance(outputs, dict):
        for key in sorted(outputs):
            lines.append(f"- `{key}`: `{outputs[key]}`")
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoSubmissionPacketConfig", "build_realtime_demo_submission_packet"]
