"""Compose realtime skeleton overlay video with a SCHUNK response preview panel."""

from __future__ import annotations

import json
from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-untyped]


def create_realtime_schunk_demo_composite(
    *,
    overlay_video: Path,
    response_preview_image: Path,
    out_dir: Path,
    output_size: tuple[int, int] = (1920, 1080),
) -> dict[str, object]:
    """Create a side-by-side final-demo video from realtime overlay and SCHUNK preview."""

    width, height = output_size
    if width <= 0 or height <= 0:
        raise ValueError("output_size must be positive")
    if not overlay_video.exists():
        raise FileNotFoundError(f"overlay video does not exist: {overlay_video}")
    if not response_preview_image.exists():
        raise FileNotFoundError(f"response preview image does not exist: {response_preview_image}")

    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / "realtime_schunk_demo_composite.mp4"
    poster_path = out_dir / "realtime_schunk_demo_composite_poster.png"
    manifest_path = out_dir / "realtime_schunk_demo_composite_manifest.json"

    capture = cv2.VideoCapture(str(overlay_video))
    if not capture.isOpened():
        raise ValueError(f"could not open overlay video: {overlay_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        fps = 30.0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not open output video writer: {mp4_path}")

    response_preview = Image.open(response_preview_image).convert("RGB")
    font = ImageFont.load_default()
    frame_count = 0
    first_frame: Image.Image | None = None
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            overlay_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            composite = _compose_frame(
                overlay_rgb,
                response_preview,
                output_size=(width, height),
                font=font,
            )
            if first_frame is None:
                first_frame = composite.copy()
            writer.write(cv2.cvtColor(np.asarray(composite), cv2.COLOR_RGB2BGR))
            frame_count += 1
    finally:
        capture.release()
        writer.release()
        response_preview.close()

    if frame_count == 0 or first_frame is None:
        raise ValueError("overlay video did not contain frames")
    first_frame.save(poster_path)
    manifest: dict[str, object] = {
        "status": "passed",
        "overlay_video": overlay_video.as_posix(),
        "response_preview_image": response_preview_image.as_posix(),
        "out_dir": out_dir.as_posix(),
        "frame_count": frame_count,
        "fps": fps,
        "source_overlay_width": source_width,
        "source_overlay_height": source_height,
        "output_width": width,
        "output_height": height,
        "outputs": {
            "mp4": mp4_path.as_posix(),
            "poster_png": poster_path.as_posix(),
        },
        "claim_scope": "demo composition only; combines realtime overlay with metadata-driven SCHUNK response preview",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _compose_frame(
    overlay: Image.Image,
    response_preview: Image.Image,
    *,
    output_size: tuple[int, int],
    font: ImageFont.ImageFont,
) -> Image.Image:
    width, height = output_size
    canvas = Image.new("RGB", (width, height), (243, 245, 248))
    draw = ImageDraw.Draw(canvas)
    header_h = max(48, height // 18)
    draw.rectangle((0, 0, width, header_h), fill=(28, 33, 40))
    draw.text((24, header_h // 2 - 5), "Few-shot skeleton RPS demo", fill=(248, 250, 252), font=font)
    draw.text((width // 2, header_h // 2 - 5), "Camera prediction + SCHUNK response", fill=(219, 234, 254), font=font)

    margin = 20
    gutter = 16
    content_top = header_h + margin
    content_h = height - content_top - margin
    left_w = int((width - margin * 2 - gutter) * 0.62)
    right_w = width - margin * 2 - gutter - left_w
    left_box = (margin, content_top, margin + left_w, content_top + content_h)
    right_box = (margin + left_w + gutter, content_top, width - margin, content_top + content_h)
    _paste_fit(canvas, overlay, left_box, background=(226, 232, 240))
    _paste_fit(canvas, response_preview, right_box, background=(235, 238, 242))
    draw.rectangle(left_box, outline=(190, 198, 208), width=2)
    draw.rectangle(right_box, outline=(190, 198, 208), width=2)
    draw.text((left_box[0] + 12, left_box[1] + 12), "Realtime MediaPipe skeleton prediction", fill=(15, 23, 42), font=font)
    draw.text((right_box[0] + 12, right_box[1] + 12), "SCHUNK response preview", fill=(15, 23, 42), font=font)
    return canvas


def _paste_fit(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    background: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    tile = Image.new("RGB", (width, height), background)
    copied = image.copy()
    copied.thumbnail((width - 24, height - 40), _resampling_lanczos())
    tile.paste(copied, ((width - copied.width) // 2, (height - copied.height) // 2 + 12))
    canvas.paste(tile, (left, top))


def _resampling_lanczos() -> int:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return int(resampling.LANCZOS)
    return int(Image.LANCZOS)


__all__ = ["create_realtime_schunk_demo_composite"]
