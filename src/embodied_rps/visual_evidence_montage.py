"""Create presentation-ready montage images from SCHUNK visual evidence PNGs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-untyped]


GESTURES = ("rock", "paper", "scissors")
STATIC_IMAGE_RE = "{gesture}_view_yaw{yaw}_pitch{pitch}.png"
SEQUENCE_RE = re.compile(
    r"^(?P<gesture>rock|paper|scissors)_sequence_frame(?P<frame>\d+)_progress(?P<progress>[0-9.]+)_view_yaw(?P<yaw>-?\d+)_pitch(?P<pitch>-?\d+)\.png$"
)


@dataclass(frozen=True)
class MontageArtifacts:
    """Paths written by the montage generator."""

    static_montage: Path
    sequence_montage: Path
    manifest: Path


def create_schunk_visual_evidence_montages(
    *,
    render_dir: Path,
    out_dir: Path,
    yaw: int = 45,
    pitch: int = 20,
    cell_width: int = 320,
    cell_height: int = 180,
) -> MontageArtifacts:
    """Create static RPS and fist-start sequence montage images."""

    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("cell_width and cell_height must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    static_paths = _static_paths(render_dir, yaw=yaw, pitch=pitch)
    sequence = _sequence_paths(render_dir, yaw=yaw, pitch=pitch)
    static_montage = out_dir / f"schunk_visual_rig_static_yaw{yaw}_pitch{pitch}.png"
    sequence_montage = out_dir / f"schunk_visual_rig_sequence_yaw{yaw}_pitch{pitch}.png"
    manifest_path = out_dir / "schunk_visual_evidence_montage_manifest.json"

    _compose_grid(
        rows=[[(gesture, static_paths[gesture]) for gesture in GESTURES]],
        out_path=static_montage,
        title="Static RPS final poses - procedural visual rig",
        cell_width=cell_width,
        cell_height=cell_height,
        font=font,
    )
    sequence_rows: list[list[tuple[str, Path]]] = []
    frame_labels: list[str] = []
    for gesture in GESTURES:
        frames = sequence[gesture]
        sequence_rows.append([(f"{gesture} f{frame_index}", path) for frame_index, path in frames])
        if not frame_labels:
            frame_labels = [f"f{frame_index}" for frame_index, _ in frames]
    _compose_grid(
        rows=sequence_rows,
        out_path=sequence_montage,
        title="Fist-start to RPS sequence - procedural visual rig",
        cell_width=cell_width,
        cell_height=cell_height,
        font=font,
        column_labels=frame_labels,
    )

    manifest = {
        "status": "passed",
        "render_dir": render_dir.as_posix(),
        "out_dir": out_dir.as_posix(),
        "yaw_degrees": yaw,
        "pitch_degrees": pitch,
        "visual_evidence": "procedural_visual_rig",
        "static_montage": static_montage.as_posix(),
        "sequence_montage": sequence_montage.as_posix(),
        "static_sources": {gesture: path.as_posix() for gesture, path in static_paths.items()},
        "sequence_sources": {
            gesture: [{"frame_index": frame_index, "path": path.as_posix()} for frame_index, path in frames]
            for gesture, frames in sequence.items()
        },
        "claim_scope": "readable visualization only; not imported SCHUNK mesh proof",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return MontageArtifacts(static_montage=static_montage, sequence_montage=sequence_montage, manifest=manifest_path)


def _static_paths(render_dir: Path, *, yaw: int, pitch: int) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for gesture in GESTURES:
        path = render_dir / STATIC_IMAGE_RE.format(gesture=gesture, yaw=yaw, pitch=pitch)
        if path.exists():
            paths[gesture] = path
        else:
            missing.append(path.as_posix())
    if missing:
        raise FileNotFoundError(f"missing static montage sources: {missing}")
    return paths


def _sequence_paths(render_dir: Path, *, yaw: int, pitch: int) -> dict[str, list[tuple[int, Path]]]:
    by_gesture: dict[str, list[tuple[int, Path]]] = {gesture: [] for gesture in GESTURES}
    for path in sorted(render_dir.glob("*_sequence_frame*_view_yaw*_pitch*.png")):
        match = SEQUENCE_RE.match(path.name)
        if match is None:
            continue
        if int(match.group("yaw")) != yaw or int(match.group("pitch")) != pitch:
            continue
        gesture = match.group("gesture")
        frame = int(match.group("frame"))
        by_gesture[gesture].append((frame, path))
    missing = [gesture for gesture, frames in by_gesture.items() if not frames]
    if missing:
        raise FileNotFoundError(f"missing sequence montage sources for gestures: {missing}")
    frame_counts = {len(frames) for frames in by_gesture.values()}
    if len(frame_counts) != 1:
        raise ValueError(f"sequence montage frame counts differ by gesture: {_counts_json(by_gesture)}")
    return {gesture: sorted(frames, key=lambda item: item[0]) for gesture, frames in by_gesture.items()}


def _compose_grid(
    *,
    rows: Sequence[Sequence[tuple[str, Path]]],
    out_path: Path,
    title: str,
    cell_width: int,
    cell_height: int,
    font: ImageFont.ImageFont,
    column_labels: Sequence[str] | None = None,
) -> None:
    if not rows or not rows[0]:
        raise ValueError("montage rows must not be empty")
    margin = 16
    title_h = 34
    label_h = 24
    col_label_h = 22 if column_labels else 0
    row_count = len(rows)
    col_count = max(len(row) for row in rows)
    width = margin * 2 + col_count * cell_width
    height = margin * 2 + title_h + col_label_h + row_count * (cell_height + label_h)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    top = margin + title_h
    if column_labels:
        for col_index, label in enumerate(column_labels):
            draw.text((margin + col_index * cell_width + 6, top), label, fill=(80, 80, 80), font=font)
        top += col_label_h
    for row_index, row in enumerate(rows):
        y = top + row_index * (cell_height + label_h)
        for col_index, (label, source) in enumerate(row):
            x = margin + col_index * cell_width
            image = _load_resized(source, cell_width, cell_height)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline=(210, 210, 210), width=1)
            draw.text((x + 6, y + cell_height + 5), label, fill=(30, 30, 30), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _load_resized(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((width, height), _resampling_lanczos())
        canvas = Image.new("RGB", (width, height), (238, 238, 238))
        x = (width - rgb.width) // 2
        y = (height - rgb.height) // 2
        canvas.paste(rgb, (x, y))
        return canvas


def _counts_json(by_gesture: Mapping[str, Sequence[tuple[int, Path]]]) -> dict[str, int]:
    return {gesture: len(frames) for gesture, frames in by_gesture.items()}


def _resampling_lanczos() -> int:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return int(resampling.LANCZOS)
    return int(Image.LANCZOS)
