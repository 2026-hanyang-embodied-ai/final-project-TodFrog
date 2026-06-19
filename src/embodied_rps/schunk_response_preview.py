"""Create visual previews from SCHUNK response-plan metadata."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-untyped]

from embodied_rps.schunk import Vector3, generate_pose_skeleton


def create_schunk_response_preview(
    *,
    response_plan_jsonl: Path,
    out_dir: Path,
    max_events: int = 6,
    fps: int = 2,
) -> dict[str, object]:
    """Create a contact sheet, GIF, event frames, and manifest from response-plan JSONL."""

    if max_events <= 0:
        raise ValueError("max_events must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    events = _load_jsonl_records(response_plan_jsonl)
    if len(events) == 0:
        raise ValueError("response plan must contain at least one event")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.png"):
        old_frame.unlink()

    selected = _select_representative_events(events, max_events=max_events)
    font = ImageFont.load_default()
    event_frames: list[Image.Image] = []
    frame_paths: list[Path] = []
    for index, event in enumerate(selected):
        frame = _compose_event_frame(event, font=font)
        frame_path = frames_dir / f"frame_{index:04d}_{_safe_name(str(event.get('clip_id', 'event')))}.png"
        frame.save(frame_path)
        event_frames.append(frame)
        frame_paths.append(frame_path)

    contact_sheet = out_dir / "schunk_response_preview_contact_sheet.png"
    gif_path = out_dir / "schunk_response_preview.gif"
    manifest_path = out_dir / "schunk_response_preview_manifest.json"
    _write_contact_sheet(event_frames, contact_sheet, font=font)
    _write_gif(event_frames, gif_path, fps=fps)

    manifest: dict[str, object] = {
        "status": "passed",
        "response_plan_jsonl": response_plan_jsonl.as_posix(),
        "out_dir": out_dir.as_posix(),
        "selected_event_count": len(selected),
        "source_event_count": len(events),
        "fps": fps,
        "selected_events": [
            {
                "clip_id": event.get("clip_id"),
                "predicted_final_gesture": event.get("predicted_final_gesture"),
                "selected_robot_action": event.get("selected_robot_action"),
                "demo_response_start_time_s": event.get("demo_response_start_time_s"),
                "within_demo_deadline": event.get("within_demo_deadline"),
                "frame_path": frame_paths[index].as_posix(),
            }
            for index, event in enumerate(selected)
        ],
        "outputs": {
            "contact_sheet_png": contact_sheet.as_posix(),
            "gif": gif_path.as_posix(),
            "frames_dir": frames_dir.as_posix(),
        },
        "claim_scope": "metadata-driven SCHUNK response preview; not Isaac physics validation",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _compose_event_frame(event: Mapping[str, object], *, font: ImageFont.ImageFont) -> Image.Image:
    width, height = 1200, 420
    canvas = Image.new("RGB", (width, height), (244, 246, 248))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 58), fill=(31, 36, 43))
    clip_id = str(event.get("clip_id", "event"))
    action = str(event.get("selected_robot_action", "unknown"))
    predicted = str(event.get("predicted_final_gesture", "unknown"))
    start = _optional_float(event.get("demo_response_start_time_s"))
    draw.text((24, 19), "SCHUNK response preview", fill=(245, 247, 250), font=font)
    draw.text((330, 19), f"clip {clip_id[:70]}", fill=(216, 231, 255), font=font)
    draw.text((24, 70), f"predicted user: {predicted}   robot action: {action}   start: {_fmt(start)}s", fill=(30, 34, 40), font=font)

    phases = _phase_mappings(event)
    panel_w = 372
    panel_h = 290
    gap = 18
    top = 108
    for index, phase in enumerate(phases[:3]):
        left = 24 + index * (panel_w + gap)
        _draw_phase_panel(draw, left, top, panel_w, panel_h, phase, font=font)
    return canvas


def _draw_phase_panel(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    width: int,
    height: int,
    phase: Mapping[str, object],
    *,
    font: ImageFont.ImageFont,
) -> None:
    draw.rectangle((left, top, left + width, top + height), fill=(255, 255, 255), outline=(202, 207, 214), width=1)
    phase_name = str(phase.get("phase", "phase"))
    pose = str(phase.get("pose", "pose"))
    time_s = _optional_float(phase.get("time_s"))
    draw.text((left + 12, top + 12), f"{phase_name}  t={_fmt(time_s)}s", fill=(20, 24, 30), font=font)
    draw.text((left + 12, top + 32), f"pose {pose}", fill=(65, 72, 82), font=font)
    joint_targets = _joint_target_mapping(phase.get("joint_targets"))
    skeleton = generate_pose_skeleton(joint_targets)
    projected = _project_points(skeleton, left=left + 22, top=top + 58, width=width - 44, height=height - 72)
    for chain in _finger_chains():
        color = _chain_color(chain[0])
        for start_name, end_name in zip(chain, chain[1:]):
            _draw_line(draw, projected[start_name], projected[end_name], color=color, width=5)
        for name in chain:
            _draw_circle(draw, projected[name], radius=7, fill=(15, 23, 42))
            _draw_circle(draw, projected[name], radius=4, fill=(255, 255, 255))
    _draw_circle(draw, projected["palm"], radius=18, fill=(203, 213, 225))
    _draw_circle(draw, projected["palm"], radius=12, fill=(232, 237, 244))


def _write_contact_sheet(frames: Sequence[Image.Image], path: Path, *, font: ImageFont.ImageFont) -> None:
    thumb_w, thumb_h = 900, 315
    margin = 20
    title_h = 34
    cols = min(2, len(frames))
    rows = math.ceil(len(frames) / cols)
    canvas = Image.new("RGB", (margin * 2 + cols * thumb_w, margin * 2 + title_h + rows * thumb_h), (246, 247, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), "SCHUNK response preview contact sheet", fill=(24, 28, 34), font=font)
    for index, frame in enumerate(frames):
        thumb = frame.copy()
        thumb.thumbnail((thumb_w, thumb_h), _resampling_lanczos())
        x = margin + (index % cols) * thumb_w
        y = margin + title_h + (index // cols) * thumb_h
        tile = Image.new("RGB", (thumb_w, thumb_h), (235, 237, 240))
        tile.paste(thumb, ((thumb_w - thumb.width) // 2, (thumb_h - thumb.height) // 2))
        canvas.paste(tile, (x, y))
        draw.rectangle((x, y, x + thumb_w - 1, y + thumb_h - 1), outline=(204, 209, 216), width=1)
    canvas.save(path)


def _write_gif(frames: Sequence[Image.Image], path: Path, *, fps: int) -> None:
    duration_ms = max(1, int(1000 / fps))
    gif_frames: list[Image.Image] = []
    for frame in frames:
        copied = frame.copy()
        copied.thumbnail((900, 315), _resampling_lanczos())
        tile = Image.new("RGB", (900, 315), (244, 246, 248))
        tile.paste(copied, ((900 - copied.width) // 2, (315 - copied.height) // 2))
        gif_frames.append(tile)
    gif_frames[0].save(path, save_all=True, append_images=gif_frames[1:], duration=duration_ms, loop=0, optimize=True)


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


def _select_representative_events(events: Sequence[dict[str, object]], *, max_events: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    selected_ids: set[int] = set()
    seen_actions: set[str] = set()
    for index, event in enumerate(events):
        action = str(event.get("selected_robot_action", ""))
        if action and action not in seen_actions:
            selected.append(event)
            selected_ids.add(index)
            seen_actions.add(action)
            if len(selected) >= max_events:
                return selected
    for index, event in enumerate(events):
        if index in selected_ids:
            continue
        selected.append(event)
        if len(selected) >= max_events:
            return selected
    return selected


def _phase_mappings(event: Mapping[str, object]) -> list[Mapping[str, object]]:
    phases = event.get("phases")
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
        raise ValueError("response plan event must include phases")
    parsed = [phase for phase in phases if isinstance(phase, Mapping)]
    if len(parsed) == 0:
        raise ValueError("response plan event phases must not be empty")
    return parsed


def _joint_target_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("phase joint_targets must be a mapping")
    targets: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError("phase joint_targets must map strings to numbers")
        targets[key] = float(item)
    return targets


def _project_points(
    points: Mapping[str, Vector3],
    *,
    left: int,
    top: int,
    width: int,
    height: int,
) -> dict[str, tuple[int, int]]:
    yaw = math.radians(35.0)
    pitch = math.radians(18.0)
    rotated: dict[str, tuple[float, float]] = {}
    for name, (x, y, z) in points.items():
        xr = math.cos(yaw) * x + math.sin(yaw) * z
        zr = -math.sin(yaw) * x + math.cos(yaw) * z
        yr = math.cos(pitch) * y - math.sin(pitch) * zr
        rotated[name] = (xr, yr)
    xs = [point[0] for point in rotated.values()]
    ys = [point[1] for point in rotated.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    scale = min(width * 0.78, height * 0.78) / span
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    return {
        name: (
            int(round(left + width / 2.0 + (x - cx) * scale)),
            int(round(top + height / 2.0 - (y - cy) * scale)),
        )
        for name, (x, y) in rotated.items()
    }


def _draw_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)


def _draw_circle(draw: ImageDraw.ImageDraw, center: tuple[int, int], *, radius: int, fill: tuple[int, int, int]) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def _finger_chains() -> tuple[tuple[str, ...], ...]:
    return (
        ("thumb_base", "thumb_mid", "thumb_tip"),
        ("index_base", "index_mid", "index_tip"),
        ("middle_base", "middle_mid", "middle_tip"),
        ("ring_base", "ring_mid", "ring_tip"),
        ("pinky_base", "pinky_mid", "pinky_tip"),
    )


def _chain_color(first: str) -> tuple[int, int, int]:
    if first.startswith("thumb"):
        return (37, 99, 235)
    if first.startswith("index"):
        return (22, 163, 74)
    if first.startswith("middle"):
        return (220, 38, 38)
    if first.startswith("ring"):
        return (147, 51, 234)
    return (234, 88, 12)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe[:80] or "event"


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _resampling_lanczos() -> int:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return int(resampling.LANCZOS)
    return int(Image.LANCZOS)


__all__ = ["create_schunk_response_preview"]
