"""Cleanup for stale live-demo artifacts before a new camera attempt."""

from __future__ import annotations

import json
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeDemoLiveArtifactCleanupConfig:
    """Configured live-demo artifact paths that may be cleared before capture."""

    output_root: Path = Path("artifacts/realtime_demo_live_artifact_cleanup_20260616")
    workspace_root: Path = Path(".")
    live_overlay: Path = Path("artifacts/realtime_demo_rehearsal_20260616/live_camera_overlay.mp4")
    live_frame_log: Path = Path("artifacts/realtime_demo_rehearsal_20260616/live_camera_frames.jsonl")
    live_postcapture_root: Path = Path("artifacts/realtime_demo_rehearsal_20260616/postcapture")
    live_composite_root: Path = Path("artifacts/realtime_schunk_live_demo_composite_20260616")
    live_overlay_contract_root: Path = Path("artifacts/realtime_demo_overlay_contract_20260616")
    live_rock_retake_gate_root: Path = Path("artifacts/realtime_demo_live_rock_retake_gate_20260616")


def clear_realtime_demo_live_artifacts(config: RealtimeDemoLiveArtifactCleanupConfig) -> dict[str, object]:
    """Remove stale fixed-path live outputs while preserving dry-run and archive artifacts."""

    workspace_root = config.workspace_root.resolve()
    output_root = _resolve_inside_workspace(config.output_root, workspace_root=workspace_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_json = output_root / "live_artifact_cleanup.json"
    output_md = output_root / "live_artifact_cleanup.md"

    targets = {
        "live_overlay": config.live_overlay,
        "live_frame_log": config.live_frame_log,
        "live_postcapture_root": config.live_postcapture_root,
        "live_composite_root": config.live_composite_root,
        "live_overlay_contract_root": config.live_overlay_contract_root,
        "live_rock_retake_gate_root": config.live_rock_retake_gate_root,
    }
    records: list[dict[str, object]] = []
    for key, target in targets.items():
        resolved = _resolve_inside_workspace(target, workspace_root=workspace_root)
        existed = resolved.exists()
        error: str | None = None
        if existed:
            try:
                if resolved.is_dir():
                    shutil.rmtree(resolved, onerror=_make_writable_and_retry)
                else:
                    _make_writable(resolved)
                    resolved.unlink()
            except OSError as exc:
                error = f"{type(exc).__name__}: {exc}"
        records.append(
            {
                "key": key,
                "path": _display_path(resolved, workspace_root=workspace_root),
                "existed": existed,
                "removed": existed and not resolved.exists(),
                "error": error,
            }
        )

    removed_count = sum(1 for record in records if record["removed"] is True)
    failed_count = sum(1 for record in records if record["error"] is not None)
    summary: dict[str, object] = {
        "cleanup_status": "partial_cleanup" if failed_count else "cleared",
        "removed_count": removed_count,
        "failed_count": failed_count,
        "target_count": len(records),
        "targets": records,
        "outputs": {
            "summary_json": _display_path(output_json, workspace_root=workspace_root),
            "summary_md": _display_path(output_md, workspace_root=workspace_root),
        },
        "claim_scope": "removes only configured fixed-path live demo artifacts inside the workspace",
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _resolve_inside_workspace(path: Path, *, workspace_root: Path) -> Path:
    candidate = path if path.is_absolute() else workspace_root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to clean path outside workspace: {path}") from exc
    if resolved == workspace_root:
        raise ValueError("Refusing to clean the workspace root")
    return resolved


def _display_path(path: Path, *, workspace_root: Path) -> str:
    try:
        return path.relative_to(workspace_root).as_posix()
    except ValueError:
        return path.as_posix()


def _make_writable(path: Path) -> None:
    try:
        path.chmod(stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    except OSError:
        pass


def _make_writable_and_retry(function: Any, path: str, exc_info: object) -> None:
    """Retry deleting read-only Windows files or directories."""

    del exc_info
    target = Path(path)
    _make_writable(target)
    try:
        function(path)
    except OSError:
        if target.is_dir():
            os.rmdir(path)
        else:
            os.unlink(path)


def _summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Realtime Demo Live Artifact Cleanup",
        "",
        f"- Cleanup status: `{summary.get('cleanup_status')}`",
        f"- Removed count: `{summary.get('removed_count')}`",
        f"- Failed count: `{summary.get('failed_count')}`",
        f"- Target count: `{summary.get('target_count')}`",
        "",
        "## Targets",
        "",
        "| Key | Path | Existed | Removed | Error |",
        "|---|---|---:|---:|---|",
    ]
    targets = summary.get("targets", [])
    if isinstance(targets, list):
        for target in targets:
            if isinstance(target, dict):
                lines.append(
                    f"| `{target.get('key')}` | `{target.get('path')}` | `{target.get('existed')}` | `{target.get('removed')}` | `{target.get('error')}` |"
                )
    lines.append("")
    return "\n".join(lines)


__all__ = ["RealtimeDemoLiveArtifactCleanupConfig", "clear_realtime_demo_live_artifacts"]
