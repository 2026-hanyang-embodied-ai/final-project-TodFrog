"""Lightweight review packet for realtime RPS demo evidence."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STRICT_LIVE_DEMO_COMMAND = (
    "powershell -ExecutionPolicy Bypass -File "
    "artifacts\\realtime_demo_launch_20260616\\24_run_live_demo_operator_confirmed_strict.ps1"
)


@dataclass(frozen=True)
class RealtimeDemoReviewPacketConfig:
    """Configuration for building a human-reviewable demo evidence packet."""

    output_root: Path
    evidence_bundle: Path = Path("artifacts/realtime_demo_evidence_bundle_20260616/demo_evidence_bundle.json")
    dry_run_postcapture_summary: Path = Path(
        "artifacts/realtime_demo_rehearsal_20260616/dry_run_postcapture/postcapture_summary.json"
    )
    dry_run_composite_manifest: Path = Path(
        "artifacts/realtime_schunk_demo_composite_response_frame_20260616/"
        "realtime_schunk_demo_composite_manifest.json"
    )
    preflight_summary: Path = Path("artifacts/realtime_demo_rehearsal_20260616/preflight/preflight_summary.json")


def build_realtime_demo_review_packet(config: RealtimeDemoReviewPacketConfig) -> dict[str, object]:
    """Collect small visual evidence and write a review manifest/README."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    visuals_root = config.output_root / "visuals"
    visuals_root.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_root / "review_packet_manifest.json"
    readme_path = config.output_root / "README.md"

    evidence_bundle = _read_json_if_exists(config.evidence_bundle) or {}
    dry_postcapture = _read_json_if_exists(config.dry_run_postcapture_summary) or {}
    dry_composite = _read_json_if_exists(config.dry_run_composite_manifest) or {}
    preflight = _read_json_if_exists(config.preflight_summary) or {}
    evidence = _dict_value(evidence_bundle, "evidence")
    dry_evidence = _dict_value(evidence, "dry_run")
    live_evidence = _dict_value(evidence, "live")
    live_artifact_freshness = _dict_value(evidence, "live_artifact_freshness")
    copied_visuals = _copy_visuals(
        output_root=visuals_root,
        dry_postcapture=dry_postcapture,
        live_evidence=live_evidence,
        live_artifact_freshness=live_artifact_freshness,
        dry_composite=dry_composite,
        dry_run_composite_manifest=config.dry_run_composite_manifest,
        preflight=preflight,
    )
    referenced_videos = {
        "dry_overlay_video": dry_evidence.get("overlay_video"),
        "dry_composite_mp4": _first_present(
            dry_evidence.get("composite_mp4"),
            _dict_value(dry_composite, "outputs").get("mp4"),
        ),
        "live_overlay_video": live_evidence.get("overlay_video"),
        "live_composite_mp4": live_evidence.get("composite_mp4"),
    }
    ready = evidence_bundle.get("ready_for_submission_demo") is True
    recommended_action = (
        "Use the live composite video for the final demo recording package."
        if ready
        else "Run the live demo pipeline, then rebuild this packet to include live capture evidence."
    )
    summary: dict[str, object] = {
        "status": evidence_bundle.get("status", "unknown"),
        "ready_for_submission_demo": ready,
        "demo_evidence_level": evidence_bundle.get("demo_evidence_level"),
        "missing_required_evidence": _list_value(evidence_bundle, "missing_required_evidence"),
        "recommended_review_action": recommended_action,
        "operator_commands": {
            "live_pipeline": STRICT_LIVE_DEMO_COMMAND,
            "refresh_packet": "powershell -ExecutionPolicy Bypass -File artifacts\\realtime_demo_launch_20260616\\09_build_demo_evidence_bundle.ps1",
        },
        "copied_visuals": copied_visuals,
        "referenced_videos": referenced_videos,
        "preflight_hand_visibility": _preflight_hand_visibility_summary(preflight),
        "source_artifacts": {
            "evidence_bundle": config.evidence_bundle.as_posix(),
            "dry_run_postcapture_summary": config.dry_run_postcapture_summary.as_posix(),
            "dry_run_composite_manifest": config.dry_run_composite_manifest.as_posix(),
            "preflight_summary": config.preflight_summary.as_posix(),
        },
        "outputs": {
            "manifest_json": manifest_path.as_posix(),
            "readme_md": readme_path.as_posix(),
            "visuals_dir": visuals_root.as_posix(),
        },
        "claim_scope": "human review packet over existing artifacts; does not copy large videos or run camera capture",
    }
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme_path.write_text(_readme_markdown(summary), encoding="utf-8")
    return summary


def _copy_visuals(
    *,
    output_root: Path,
    dry_postcapture: dict[str, Any],
    live_evidence: dict[str, Any],
    live_artifact_freshness: dict[str, Any],
    dry_composite: dict[str, Any],
    dry_run_composite_manifest: Path,
    preflight: dict[str, Any],
) -> dict[str, str | None]:
    postcapture_outputs = _dict_value(dry_postcapture, "outputs")
    composite_outputs = _dict_value(dry_composite, "outputs")
    candidates = {
        "dry_prompt_contact_sheet": postcapture_outputs.get("contact_sheet_png"),
        "dry_response_decision_frame": postcapture_outputs.get("response_decision_frame_png"),
        "dry_response_prompt_diagnostic_frame": postcapture_outputs.get("response_prompt_diagnostic_frame_png"),
        "live_response_decision_frame": _first_present(
            live_artifact_freshness.get("response_decision_frame_path"),
            live_evidence.get("response_decision_frame_png"),
        ),
        "live_response_prompt_diagnostic_frame": live_evidence.get("response_prompt_diagnostic_frame_png"),
        "dry_composite_poster": composite_outputs.get("poster_png"),
        "dry_response_prompt_frame": _first_present(
            dry_composite.get("response_prompt_frame"),
            dry_run_composite_manifest.with_name("realtime_schunk_demo_composite_response_prompt_frame.png").as_posix(),
        ),
    }
    candidates.update(_preflight_diagnostic_image_candidates(preflight))
    copied: dict[str, str | None] = {}
    for name, raw_path in candidates.items():
        copied[name] = _copy_if_exists(raw_path, output_root=output_root, stem=name)
    return copied


def _preflight_diagnostic_image_candidates(preflight: dict[str, Any]) -> dict[str, str]:
    hand_visibility = _dict_value(preflight, "hand_visibility")
    candidates: dict[str, str] = {}
    for index, raw_path in enumerate(_list_value(hand_visibility, "diagnostic_image_paths")):
        stem = _safe_stem(Path(raw_path).stem) or f"diagnostic_{index:02d}"
        key = f"preflight_hand_visibility_{stem}"
        if key in candidates:
            key = f"{key}_{index:02d}"
        candidates[key] = raw_path
    return candidates


def _preflight_hand_visibility_summary(preflight: dict[str, Any]) -> dict[str, object]:
    hand_visibility = _dict_value(preflight, "hand_visibility")
    return {
        "status": preflight.get("status"),
        "detection_rate": hand_visibility.get("detection_rate"),
        "detected_frames": hand_visibility.get("detected_frames"),
        "frame_count": hand_visibility.get("frame_count"),
        "diagnostic_image_paths": _list_value(hand_visibility, "diagnostic_image_paths"),
    }


def _safe_stem(stem: str) -> str:
    return "_".join(part for part in "".join(char if char.isalnum() else "_" for char in stem).split("_") if part)


def _copy_if_exists(raw_path: object, *, output_root: Path, stem: str) -> str | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    source = Path(raw_path)
    if not source.exists() or not source.is_file():
        return None
    suffix = source.suffix or ".bin"
    destination = output_root / f"{stem}{suffix}"
    shutil.copy2(source, destination)
    return destination.as_posix()


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _list_value(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _first_present(*values: object) -> object:
    for value in values:
        if value is not None:
            return value
    return None


def _readme_markdown(summary: dict[str, object]) -> str:
    missing = summary.get("missing_required_evidence", [])
    visuals = summary.get("copied_visuals", {})
    videos = summary.get("referenced_videos", {})
    commands = summary.get("operator_commands", {})
    lines = [
        "# Realtime Demo Review Packet",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Ready for submission demo: `{summary.get('ready_for_submission_demo')}`",
        f"- Demo evidence level: `{summary.get('demo_evidence_level')}`",
        f"- Recommended action: {summary.get('recommended_review_action')}",
        "",
        "## Missing Required Evidence",
        "",
    ]
    if isinstance(missing, list) and missing:
        lines.extend(f"- `{item}`" for item in missing)
    else:
        lines.append("- None")
    lines.extend(["", "## Commands", ""])
    if isinstance(commands, dict):
        for key in sorted(commands):
            lines.append(f"- `{key}`: `{commands[key]}`")
    lines.extend(["", "## Copied Visuals", ""])
    if isinstance(visuals, dict):
        for key in sorted(visuals):
            lines.append(f"- `{key}`: `{visuals[key]}`")
    lines.extend(["", "## Referenced Videos", ""])
    if isinstance(videos, dict):
        for key in sorted(videos):
            lines.append(f"- `{key}`: `{videos[key]}`")
    lines.append("")
    return "\n".join(lines)


__all__ = ["STRICT_LIVE_DEMO_COMMAND", "RealtimeDemoReviewPacketConfig", "build_realtime_demo_review_packet"]
