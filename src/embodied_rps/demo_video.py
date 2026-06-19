"""Compose actuator-constrained SCHUNK RPS demo videos from render frames."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-untyped]


TRANSITION_EPISODE_REQUIREMENTS: dict[str, tuple[str, str, str]] = {
    "rock_to_paper": ("rock", "rock", "paper"),
    "paper_to_scissors": ("paper", "paper", "scissors"),
    "scissors_to_rock": ("scissors", "scissors", "rock"),
}


@dataclass(frozen=True)
class SelectedEpisode:
    """Episode selected for one rendered counter transition."""

    transition_name: str
    record: dict[str, object]


def create_actuator_demo_video(
    *,
    motion_dir: Path,
    episode_log: Path,
    max_win_summary: Path,
    loss_free_summary: Path,
    out_dir: Path,
    fps: int = 24,
    ffmpeg_path: Path | None = None,
    skip_mp4: bool = False,
) -> dict[str, object]:
    """Create storyboard, GIF fallback, optional MP4, and a manifest."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame_*.png"):
        old_frame.unlink()

    max_summary = _load_json_mapping(max_win_summary)
    loss_free = _load_json_mapping(loss_free_summary)
    records = _load_jsonl_records(episode_log)
    selected = _select_demo_episodes(records)
    rendered_frames = _compose_video_frames(
        motion_dir=motion_dir,
        selected=selected,
        max_summary=max_summary,
        loss_free_summary=loss_free,
        fps=fps,
    )
    if not rendered_frames:
        raise ValueError("demo video frame list must not be empty")

    storyboard_path = out_dir / "actuator_constrained_rps_demo_storyboard.png"
    gif_path = out_dir / "actuator_constrained_rps_demo.gif"
    mp4_path = out_dir / "actuator_constrained_rps_demo.mp4"
    manifest_path = out_dir / "actuator_constrained_rps_demo_manifest.json"

    _write_storyboard(
        motion_dir=motion_dir,
        selected=selected,
        max_summary=max_summary,
        loss_free_summary=loss_free,
        out_path=storyboard_path,
    )
    _write_gif(rendered_frames, gif_path, fps=fps)
    for index, frame in enumerate(rendered_frames):
        frame.save(frame_dir / f"frame_{index:04d}.png")

    mp4_status = "skipped"
    ffmpeg_used: str | None = None
    if not skip_mp4:
        ffmpeg_executable = _resolve_ffmpeg(ffmpeg_path)
        if ffmpeg_executable is not None:
            ffmpeg_used = ffmpeg_executable.as_posix()
            command = [
                ffmpeg_executable.as_posix(),
                "-y",
                "-framerate",
                str(fps),
                "-i",
                (frame_dir / "frame_%04d.png").as_posix(),
                "-vf",
                "format=yuv420p",
                "-movflags",
                "+faststart",
                mp4_path.as_posix(),
            ]
            subprocess.run(command, check=True)
            mp4_status = "written"
        else:
            mp4_status = "ffmpeg_not_found"

    manifest: dict[str, object] = {
        "status": "passed",
        "motion_dir": motion_dir.as_posix(),
        "episode_log": episode_log.as_posix(),
        "max_win_summary": max_win_summary.as_posix(),
        "loss_free_summary": loss_free_summary.as_posix(),
        "fps": fps,
        "frame_count": len(rendered_frames),
        "selected_episodes": [
            {
                "transition_name": item.transition_name,
                "episode_id": item.record.get("episode_id"),
                "dataset_index": item.record.get("dataset_index"),
                "true_gesture": item.record.get("true_gesture"),
                "predicted_gesture": item.record.get("predicted_gesture"),
                "selected_counter_move": item.record.get("selected_counter_move"),
                "observation_ratio": item.record.get("observation_ratio"),
                "confidence": item.record.get("confidence"),
                "remaining_time_s": item.record.get("remaining_time_s"),
                "actuator_response_time_s": item.record.get("actuator_response_time_s"),
                "result": item.record.get("result"),
            }
            for item in selected
        ],
        "outputs": {
            "storyboard_png": storyboard_path.as_posix(),
            "gif": gif_path.as_posix(),
            "mp4": mp4_path.as_posix() if mp4_status == "written" else None,
            "frames_dir": frame_dir.as_posix(),
        },
        "mp4_status": mp4_status,
        "ffmpeg": ffmpeg_used,
        "claim_scope": (
            "Presentation demo from validated connected SCHUNK mesh-plus-link-skeleton "
            "motion frames and real actuator-feasible episode records."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _select_demo_episodes(records: Sequence[Mapping[str, object]]) -> list[SelectedEpisode]:
    selected: list[SelectedEpisode] = []
    for transition_name, (true_gesture, predicted_gesture, counter_move) in TRANSITION_EPISODE_REQUIREMENTS.items():
        match = next(
            (
                dict(record)
                for record in records
                if record.get("true_gesture") == true_gesture
                and record.get("predicted_gesture") == predicted_gesture
                and record.get("selected_counter_move") == counter_move
                and record.get("result") == "win"
            ),
            None,
        )
        if match is None:
            raise ValueError(f"episode log lacks a win example for transition {transition_name}")
        selected.append(SelectedEpisode(transition_name=transition_name, record=match))
    return selected


def _compose_video_frames(
    *,
    motion_dir: Path,
    selected: Sequence[SelectedEpisode],
    max_summary: Mapping[str, object],
    loss_free_summary: Mapping[str, object],
    fps: int,
) -> list[Image.Image]:
    font = ImageFont.load_default()
    repeated_frames = max(2, fps // 6)
    frames: list[Image.Image] = []
    frames.extend(_title_frames(font=font, max_summary=max_summary, loss_free_summary=loss_free_summary, fps=fps))
    for item in selected:
        source_paths = _transition_frame_paths(motion_dir, item.transition_name)
        crop_box = _combined_content_bbox(source_paths)
        for source_path in source_paths:
            for _ in range(repeated_frames):
                frames.append(
                    _compose_demo_frame(
                        source_path=source_path,
                        crop_box=crop_box,
                        selected=item,
                        max_summary=max_summary,
                        loss_free_summary=loss_free_summary,
                        font=font,
                    )
                )
        frames.extend([frames[-1].copy() for _ in range(max(1, fps // 3))])
    return frames


def _title_frames(
    *,
    font: ImageFont.ImageFont,
    max_summary: Mapping[str, object],
    loss_free_summary: Mapping[str, object],
    fps: int,
) -> list[Image.Image]:
    frame = Image.new("RGB", (1280, 720), (244, 246, 248))
    draw = ImageDraw.Draw(frame)
    draw.text((70, 90), "Actuator-Constrained Early Intention Prediction", fill=(20, 24, 30), font=font)
    draw.text((70, 135), "RPS robot-hand demo with connected SCHUNK visual evidence", fill=(60, 66, 76), font=font)
    _draw_metric_card(draw, max_summary=max_summary, loss_free_summary=loss_free_summary, x=70, y=220, font=font)
    draw.text((70, 585), "The video uses real max-win episode records and validated connected motion renders.", fill=(70, 76, 86), font=font)
    return [frame.copy() for _ in range(max(1, fps))]


def _compose_demo_frame(
    *,
    source_path: Path,
    crop_box: tuple[int, int, int, int],
    selected: SelectedEpisode,
    max_summary: Mapping[str, object],
    loss_free_summary: Mapping[str, object],
    font: ImageFont.ImageFont,
) -> Image.Image:
    canvas = Image.new("RGB", (1280, 720), (242, 243, 245))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1280, 70), fill=(30, 34, 40))
    draw.text((28, 24), "Actuator-constrained RPS response", fill=(245, 245, 245), font=font)
    draw.text((820, 24), selected.transition_name.replace("_", " -> "), fill=(215, 230, 255), font=font)

    with Image.open(source_path) as source:
        visual = source.convert("RGB").crop(crop_box)
    visual.thumbnail((760, 560), _resampling_lanczos())
    visual_box = Image.new("RGB", (790, 580), (228, 230, 232))
    visual_box.paste(visual, ((790 - visual.width) // 2, (580 - visual.height) // 2))
    canvas.paste(visual_box, (24, 96))
    draw.rectangle((24, 96, 814, 676), outline=(195, 199, 205), width=2)

    panel_x = 850
    draw.rectangle((panel_x, 96, 1248, 676), fill=(255, 255, 255), outline=(205, 208, 214), width=1)
    y = 124
    y = _draw_lines(
        draw,
        x=panel_x + 24,
        y=y,
        lines=[
            "Episode decision",
            f"True opponent: {_value(selected.record, 'true_gesture')}",
            f"Predicted: {_value(selected.record, 'predicted_gesture')}",
            f"Counter move: {_value(selected.record, 'selected_counter_move')}",
            f"Observation ratio: {_float_value(selected.record, 'observation_ratio'):.2f}",
            f"Confidence: {_float_value(selected.record, 'confidence'):.3f}",
            f"Remaining time: {_float_value(selected.record, 'remaining_time_s'):.3f}s",
            f"Required response: {_float_value(selected.record, 'actuator_response_time_s'):.3f}s",
            f"Result: {_value(selected.record, 'result').upper()}",
        ],
        font=font,
        fill=(30, 34, 40),
        line_height=30,
    )
    _draw_metric_card(draw, max_summary=max_summary, loss_free_summary=loss_free_summary, x=panel_x + 24, y=y + 24, font=font)
    return canvas


def _draw_metric_card(
    draw: ImageDraw.ImageDraw,
    *,
    max_summary: Mapping[str, object],
    loss_free_summary: Mapping[str, object],
    x: int,
    y: int,
    font: ImageFont.ImageFont,
) -> None:
    draw.rectangle((x, y, x + 360, y + 220), fill=(238, 242, 246), outline=(205, 210, 216), width=1)
    lines = [
        "Operating points",
        (
            "Max-win: threshold "
            f"{_float_value(max_summary, 'confidence_threshold'):.2f}, "
            f"clear win {_format_rate(max_summary.get('clear_actuator_feasible_win_rate'))}, "
            f"loss {_format_rate(max_summary.get('loss_rate'))}"
        ),
        (
            "Loss-free: threshold "
            f"{_float_value(loss_free_summary, 'confidence_threshold'):.2f}, "
            f"clear win {_format_rate(loss_free_summary.get('clear_actuator_feasible_win_rate'))}, "
            f"loss {_format_rate(loss_free_summary.get('loss_rate'))}"
        ),
        "Metric: actuator-feasible win rate",
    ]
    _draw_lines(draw, x=x + 16, y=y + 18, lines=lines, font=font, fill=(35, 39, 46), line_height=34)


def _write_storyboard(
    *,
    motion_dir: Path,
    selected: Sequence[SelectedEpisode],
    max_summary: Mapping[str, object],
    loss_free_summary: Mapping[str, object],
    out_path: Path,
) -> None:
    font = ImageFont.load_default()
    cell_w, cell_h = 260, 146
    label_h = 28
    margin = 18
    header_h = 76
    rows = len(selected)
    canvas = Image.new("RGB", (margin * 2 + cell_w * 3 + 420, margin * 2 + header_h + rows * (cell_h + label_h)), (246, 247, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), "Connected SCHUNK actuator-constrained demo storyboard", fill=(24, 28, 34), font=font)
    _draw_metric_card(draw, max_summary=max_summary, loss_free_summary=loss_free_summary, x=margin + cell_w * 3 + 40, y=margin + 8, font=font)
    top = margin + header_h
    for row_index, item in enumerate(selected):
        paths = _transition_frame_paths(motion_dir, item.transition_name)
        picks = [paths[0], paths[len(paths) // 2], paths[-1]]
        crop_box = _combined_content_bbox(paths)
        y = top + row_index * (cell_h + label_h)
        for col, path in enumerate(picks):
            with Image.open(path) as image:
                source = image.convert("RGB").crop(crop_box)
            source.thumbnail((cell_w, cell_h), _resampling_lanczos())
            x = margin + col * cell_w
            tile = Image.new("RGB", (cell_w, cell_h), (230, 232, 235))
            tile.paste(source, ((cell_w - source.width) // 2, (cell_h - source.height) // 2))
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(205, 208, 214), width=1)
            draw.text((x + 6, y + cell_h + 7), f"{item.transition_name} f{col}", fill=(34, 38, 44), font=font)
    canvas.save(out_path)


def _write_gif(frames: Sequence[Image.Image], out_path: Path, *, fps: int) -> None:
    duration_ms = max(1, int(1000 / fps))
    gif_frames: list[Image.Image] = []
    for frame in frames:
        resized = frame.copy()
        resized.thumbnail((640, 360), _resampling_lanczos())
        gif_frame = Image.new("RGB", (640, 360), (242, 243, 245))
        gif_frame.paste(resized, ((640 - resized.width) // 2, (360 - resized.height) // 2))
        gif_frames.append(gif_frame)
    gif_frames[0].save(out_path, save_all=True, append_images=gif_frames[1:], duration=duration_ms, loop=0, optimize=True)


def _transition_frame_paths(motion_dir: Path, transition_name: str) -> list[Path]:
    pattern = re.compile(
        rf"^{re.escape(transition_name)}_sequence_frame(?P<frame>\d+)_progress(?P<progress>[0-9.]+)_view_yaw45_pitch20\.png$"
    )
    indexed: list[tuple[int, Path]] = []
    for path in motion_dir.glob(f"{transition_name}_sequence_frame*_view_yaw45_pitch20.png"):
        match = pattern.match(path.name)
        if match is not None:
            indexed.append((int(match.group("frame")), path))
    if not indexed:
        raise FileNotFoundError(f"missing motion frames for transition {transition_name} in {motion_dir}")
    return [path for _, path in sorted(indexed, key=lambda item: item[0])]


def _combined_content_bbox(paths: Sequence[Path]) -> tuple[int, int, int, int]:
    left: int | None = None
    top: int | None = None
    right: int | None = None
    bottom: int | None = None
    for path in paths:
        with Image.open(path) as image:
            bbox = _content_bbox(image.convert("RGB"))
        left = bbox[0] if left is None else min(left, bbox[0])
        top = bbox[1] if top is None else min(top, bbox[1])
        right = bbox[2] if right is None else max(right, bbox[2])
        bottom = bbox[3] if bottom is None else max(bottom, bbox[3])
    if left is None or top is None or right is None or bottom is None:
        raise ValueError("cannot compute content crop for empty frame list")
    return left, top, right, bottom


def _content_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    width, height = image.size
    bg = image.getpixel((0, 0))
    pixels = image.load()
    left, top, right, bottom = width, height, 0, 0
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            pixel = pixels[x, y]
            if sum(abs(int(pixel[channel]) - int(bg[channel])) for channel in range(3)) > 24:
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)
    if right <= left or bottom <= top:
        return 0, 0, width, height
    pad_x = max(40, (right - left) // 5)
    pad_y = max(40, (bottom - top) // 5)
    return max(0, left - pad_x), max(0, top - pad_y), min(width, right + pad_x), min(height, bottom + pad_y)


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    lines: Sequence[str],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_height: int,
) -> int:
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, fill=fill, font=font)
        current_y += line_height
    return current_y


def _load_json_mapping(path: Path) -> dict[str, object]:
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(key): value for key, value in loaded.items()}


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


def _resolve_ffmpeg(path: Path | None) -> Path | None:
    if path is not None:
        return path if path.exists() else None
    discovered = shutil.which("ffmpeg")
    return Path(discovered) if discovered is not None else None


def _resampling_lanczos() -> int:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return int(resampling.LANCZOS)
    return int(Image.LANCZOS)


def _format_rate(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100.0:.1f}%"
    return "n/a"


def _float_value(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"record field {key} must be numeric")


def _value(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    return "n/a" if value is None else str(value)
