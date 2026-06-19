"""Archive fixed-path realtime demo artifacts into timestamped run folders."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeDemoRunArchiveConfig:
    """Artifact paths used to preserve one realtime demo rehearsal run."""

    output_root: Path
    run_id: str | None = None
    live_overlay_video: Path = Path("artifacts/realtime_demo_rehearsal_20260616/live_camera_overlay.mp4")
    live_frame_log: Path = Path("artifacts/realtime_demo_rehearsal_20260616/live_camera_frames.jsonl")
    live_postcapture_summary: Path = Path("artifacts/realtime_demo_rehearsal_20260616/postcapture/postcapture_summary.json")
    live_composite_manifest: Path = Path(
        "artifacts/realtime_schunk_live_demo_composite_20260616/realtime_schunk_demo_composite_manifest.json"
    )
    operator_outcome: Path = Path("artifacts/realtime_demo_operator_outcome_20260616/operator_outcome.json")
    triage_summary: Path = Path("artifacts/realtime_demo_triage_20260616/triage_summary.json")
    acceptance_report: Path = Path("artifacts/realtime_demo_acceptance_report_20260616/acceptance_report.json")
    evidence_bundle: Path = Path("artifacts/realtime_demo_evidence_bundle_20260616/demo_evidence_bundle.json")
    review_packet_manifest: Path = Path("artifacts/realtime_demo_review_packet_20260616/review_packet_manifest.json")
    readiness_summary: Path = Path("artifacts/realtime_demo_readiness_20260616/readiness_summary.json")
    preflight_summary: Path = Path("artifacts/realtime_demo_rehearsal_20260616/preflight/preflight_summary.json")
    live_rock_retake_gate: Path = Path(
        "artifacts/realtime_demo_live_rock_retake_gate_20260616/live_rock_retake_gate.json"
    )


def archive_realtime_demo_run(config: RealtimeDemoRunArchiveConfig) -> dict[str, object]:
    """Copy the current fixed-path live-demo artifacts into a named archive folder."""

    run_id = _run_id(config.run_id)
    archive_dir = config.output_root / run_id
    media_dir = archive_dir / "media"
    reports_dir = archive_dir / "reports"
    archive_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    copied_files: dict[str, str] = {}
    missing_files: dict[str, str] = {}
    _copy_named(config.live_overlay_video, destination=media_dir / "live_camera_overlay.mp4", key="live_overlay_video", copied=copied_files, missing=missing_files)
    _copy_named(config.live_frame_log, destination=media_dir / "live_camera_frames.jsonl", key="live_frame_log", copied=copied_files, missing=missing_files)
    _copy_named(config.live_postcapture_summary, destination=reports_dir / "postcapture_summary.json", key="live_postcapture_summary", copied=copied_files, missing=missing_files)
    _copy_response_decision_frame(config.live_postcapture_summary, media_dir=media_dir, copied=copied_files, missing=missing_files)
    _copy_response_prompt_diagnostic_frame(config.live_postcapture_summary, media_dir=media_dir, copied=copied_files, missing=missing_files)
    _copy_named(config.live_composite_manifest, destination=reports_dir / "realtime_schunk_demo_composite_manifest.json", key="live_composite_manifest", copied=copied_files, missing=missing_files)
    _copy_composite_outputs(config.live_composite_manifest, media_dir=media_dir, copied=copied_files, missing=missing_files)
    for key, source in {
        "operator_outcome": config.operator_outcome,
        "triage_summary": config.triage_summary,
        "acceptance_report": config.acceptance_report,
        "evidence_bundle": config.evidence_bundle,
        "review_packet_manifest": config.review_packet_manifest,
        "readiness_summary": config.readiness_summary,
        "preflight_summary": config.preflight_summary,
        "live_rock_retake_gate": config.live_rock_retake_gate,
    }.items():
        _copy_named(source, destination=reports_dir / source.name, key=key, copied=copied_files, missing=missing_files)

    operator = _read_json_if_exists(config.operator_outcome) or {}
    archive_status = _archive_status(copied_files=copied_files, operator=operator)
    manifest_path = archive_dir / "archive_manifest.json"
    readme_path = archive_dir / "README.md"
    summary: dict[str, object] = {
        "archive_status": archive_status,
        "run_id": run_id,
        "archive_dir": archive_dir.as_posix(),
        "operator_state": operator.get("operator_state"),
        "operator_recommended_exit_code": operator.get("recommended_exit_code"),
        "copied_files": copied_files,
        "missing_files": missing_files,
        "copied_file_count": len(copied_files),
        "missing_file_count": len(missing_files),
        "outputs": {
            "archive_manifest": manifest_path.as_posix(),
            "readme_md": readme_path.as_posix(),
            "media_dir": media_dir.as_posix(),
            "reports_dir": reports_dir.as_posix(),
        },
        "claim_scope": "archive of already-generated realtime demo artifacts; does not run capture, model inference, verification, or rendering",
    }
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme_path.write_text(_readme_markdown(summary), encoding="utf-8")
    return summary


def _copy_composite_outputs(
    manifest_path: Path,
    *,
    media_dir: Path,
    copied: dict[str, str],
    missing: dict[str, str],
) -> None:
    manifest = _read_json_if_exists(manifest_path) or {}
    outputs = _dict_value(manifest, "outputs")
    _copy_named(
        _path_or_none(outputs.get("mp4")),
        destination=media_dir / "realtime_schunk_demo_composite.mp4",
        key="live_composite_mp4",
        copied=copied,
        missing=missing,
    )
    _copy_named(
        _path_or_none(outputs.get("poster_png")),
        destination=media_dir / "realtime_schunk_demo_composite_poster.png",
        key="live_composite_poster",
        copied=copied,
        missing=missing,
    )


def _copy_response_decision_frame(
    postcapture_summary: Path,
    *,
    media_dir: Path,
    copied: dict[str, str],
    missing: dict[str, str],
) -> None:
    summary = _read_json_if_exists(postcapture_summary) or {}
    outputs = _dict_value(summary, "outputs")
    response_decision_frame = _dict_value(summary, "response_decision_frame")
    source = _path_or_none(
        _first_present(outputs.get("response_decision_frame_png"), response_decision_frame.get("path"))
    )
    if source is not None and not source.is_absolute() and not source.exists():
        source = postcapture_summary.parent / source
    _copy_named(
        source,
        destination=media_dir / "live_response_decision_frame.png",
        key="live_response_decision_frame",
        copied=copied,
        missing=missing,
    )


def _copy_response_prompt_diagnostic_frame(
    postcapture_summary: Path,
    *,
    media_dir: Path,
    copied: dict[str, str],
    missing: dict[str, str],
) -> None:
    summary = _read_json_if_exists(postcapture_summary) or {}
    outputs = _dict_value(summary, "outputs")
    diagnostic_frame = _dict_value(summary, "response_prompt_diagnostic_frame")
    source = _path_or_none(
        _first_present(outputs.get("response_prompt_diagnostic_frame_png"), diagnostic_frame.get("path"))
    )
    if source is not None and not source.is_absolute() and not source.exists():
        source = postcapture_summary.parent / source
    _copy_named(
        source,
        destination=media_dir / "live_response_prompt_diagnostic_frame.png",
        key="live_response_prompt_diagnostic_frame",
        copied=copied,
        missing=missing,
    )


def _copy_named(
    source: Path | None,
    *,
    destination: Path,
    key: str,
    copied: dict[str, str],
    missing: dict[str, str],
) -> None:
    if source is None or not source.exists() or not source.is_file():
        missing[key] = source.as_posix() if isinstance(source, Path) else ""
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied[key] = destination.as_posix()


def _archive_status(*, copied_files: dict[str, str], operator: dict[str, Any]) -> str:
    if "live_overlay_video" in copied_files and "live_composite_mp4" in copied_files:
        if operator.get("operator_state") == "ready_for_final_video":
            return "archived_live_demo_candidate"
        return "archived_live_capture"
    return "archived_report_only"


def _run_id(raw_run_id: str | None) -> str:
    if raw_run_id:
        return _safe_name(raw_run_id)
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(value: str) -> str:
    return "_".join(part for part in "".join(char if char.isalnum() else "_" for char in value).split("_") if part)


def _path_or_none(value: object) -> Path | None:
    return Path(value) if isinstance(value, str) and value else None


def _first_present(*values: object) -> object:
    for value in values:
        if value is not None:
            return value
    return None


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _readme_markdown(summary: dict[str, object]) -> str:
    copied = summary.get("copied_files", {})
    missing = summary.get("missing_files", {})
    lines = [
        "# Realtime Demo Run Archive",
        "",
        f"- Archive status: `{summary.get('archive_status')}`",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Operator state: `{summary.get('operator_state')}`",
        f"- Operator recommended exit code: `{summary.get('operator_recommended_exit_code')}`",
        "",
        "## Copied Files",
        "",
    ]
    if isinstance(copied, dict) and copied:
        for key in sorted(copied):
            lines.append(f"- `{key}`: `{copied[key]}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Missing Files", ""])
    if isinstance(missing, dict) and missing:
        for key in sorted(missing):
            lines.append(f"- `{key}`: `{missing[key]}`")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoRunArchiveConfig", "archive_realtime_demo_run"]
