"""SVG skeleton previews for five-joint RPS hand trajectories."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from embodied_rps.config import load_kinematic_config
from embodied_rps.dataset import load_synthetic_dataset


@dataclass(frozen=True)
class SkeletonPreviewArtifacts:
    """Paths written by one skeleton preview render."""

    montage_svg: Path
    animation_svg: Path


def render_skeleton_preview(
    *,
    dataset_path: Path,
    hand_config_path: Path,
    out_dir: Path,
    episode_index: int,
    frame_count: int,
    prefix: str,
) -> SkeletonPreviewArtifacts:
    """Render one dataset episode as a static montage SVG and animated SVG."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    dataset = load_synthetic_dataset(dataset_path)
    hand_config = load_kinematic_config(hand_config_path)
    if episode_index < 0 or episode_index >= int(dataset.positions.shape[0]):
        raise ValueError("episode_index is out of range")
    positions = cast(NDArray[np.float32], dataset.positions[episode_index])
    label = str(dataset.label_names[int(dataset.labels[episode_index])])
    frame_indices = _frame_indices(int(positions.shape[0]), frame_count)
    out_dir.mkdir(parents=True, exist_ok=True)
    montage_svg = out_dir / f"{prefix}_episode_{episode_index}_{label}_montage.svg"
    animation_svg = out_dir / f"{prefix}_episode_{episode_index}_{label}_animation.svg"
    joint_names = hand_config.joint_names
    montage_svg.write_text(
        _render_montage(positions, frame_indices, joint_names, label=label, episode_index=episode_index),
        encoding="utf-8",
    )
    animation_svg.write_text(
        _render_animation(positions, frame_indices, joint_names, label=label, episode_index=episode_index),
        encoding="utf-8",
    )
    return SkeletonPreviewArtifacts(montage_svg=montage_svg, animation_svg=animation_svg)


def render_preview_set(
    *,
    dataset_path: Path,
    hand_config_path: Path,
    out_dir: Path,
    split: str,
    frame_count: int,
    prefix: str,
) -> list[SkeletonPreviewArtifacts]:
    """Render one preview per class from the requested split."""

    dataset = load_synthetic_dataset(dataset_path)
    split_index = _split_index(dataset.split_names, split)
    artifacts: list[SkeletonPreviewArtifacts] = []
    for label_index, label_name in enumerate(dataset.label_names):
        matching = np.where((dataset.labels == label_index) & (dataset.splits == split_index))[0]
        if int(matching.shape[0]) == 0:
            raise ValueError(f"No episode found for label {label_name} in split {split}")
        episode_index = int(matching[0])
        artifacts.append(
            render_skeleton_preview(
                dataset_path=dataset_path,
                hand_config_path=hand_config_path,
                out_dir=out_dir,
                episode_index=episode_index,
                frame_count=frame_count,
                prefix=prefix,
            )
        )
    return artifacts


def _render_montage(
    positions: NDArray[np.float32],
    frame_indices: tuple[int, ...],
    joint_names: tuple[str, ...],
    *,
    label: str,
    episode_index: int,
) -> str:
    width = 260 * len(frame_indices)
    height = 260
    parts = [_svg_header(width, height)]
    parts.append(f'<text x="16" y="24" class="title">episode {episode_index} target {html.escape(label)}</text>')
    for column, frame_index in enumerate(frame_indices):
        offset_x = 130 + column * 260
        parts.append(f'<g transform="translate({offset_x},138)">')
        parts.append(_render_hand_frame(positions[frame_index], joint_names, scale=82.0))
        parts.append(f'<text x="-36" y="104" class="caption">frame {frame_index}</text>')
        parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)


def _render_animation(
    positions: NDArray[np.float32],
    frame_indices: tuple[int, ...],
    joint_names: tuple[str, ...],
    *,
    label: str,
    episode_index: int,
) -> str:
    width = 320
    height = 300
    frame_count = len(frame_indices)
    duration_s = max(1.2, frame_count * 0.28)
    duration_text = f"{duration_s:.2f}s"
    parts = [_svg_header(width, height)]
    parts.append(
        "<style>"
        ".anim-frame{opacity:0;animation-name:showframe;animation-duration:"
        + duration_text
        + ";animation-iteration-count:infinite;animation-timing-function:steps(1,end);}"
        "@keyframes showframe{0%{opacity:1}100%{opacity:0}}"
        "</style>"
    )
    parts.append(f'<text x="16" y="24" class="title">animation episode {episode_index} target {html.escape(label)}</text>')
    for frame_number, frame_index in enumerate(frame_indices):
        delay = -duration_s + (frame_number * duration_s / frame_count)
        parts.append(
            f'<g class="anim-frame" style="animation-delay:{delay:.3f}s" transform="translate(160,158)">'
        )
        parts.append(_render_hand_frame(positions[frame_index], joint_names, scale=96.0))
        parts.append(f'<text x="-36" y="120" class="caption">frame {frame_index}</text>')
        parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)


def _render_hand_frame(frame: NDArray[np.float32], joint_names: tuple[str, ...], *, scale: float) -> str:
    finger_bases = (-0.78, -0.38, 0.0, 0.38, 0.76)
    base_lengths = (0.58, 0.86, 0.92, 0.82, 0.68)
    neutral_angle = (-0.58, -0.22, 0.0, 0.20, 0.42)
    parts = ['<rect x="-48" y="18" width="96" height="72" rx="18" class="palm"/>']
    for index, joint_name in enumerate(joint_names):
        curl = float(np.clip(frame[index], 0.0, 1.05))
        base_x = finger_bases[index] * scale
        base_y = 20.0
        length = base_lengths[index] * scale
        points = _finger_points(base_x, base_y, length, neutral_angle[index], curl)
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline points="{point_text}" class="finger finger-{index}"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" class="joint"/>')
        parts.append(
            f'<text x="{base_x - 30.0:.1f}" y="108" class="joint-label">{html.escape(joint_name)}</text>'
        )
    return "\n".join(parts)


def _finger_points(base_x: float, base_y: float, length: float, neutral_angle: float, curl: float) -> tuple[tuple[float, float], ...]:
    segment = length / 3.0
    points: list[tuple[float, float]] = [(base_x, base_y)]
    angle = -math.pi / 2.0 + neutral_angle
    curl_delta = curl * 0.72
    x = base_x
    y = base_y
    for segment_index in range(3):
        local_angle = angle + curl_delta * segment_index
        x += math.cos(local_angle) * segment
        y += math.sin(local_angle) * segment
        points.append((x, y))
    return tuple(points)


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        "<style>"
        "svg{background:#f8fafc;font-family:Arial,sans-serif;}"
        ".title{font-size:14px;font-weight:700;fill:#111827;}"
        ".caption,.joint-label{font-size:9px;fill:#475569;}"
        ".palm{fill:#e0f2fe;stroke:#0f172a;stroke-width:2;}"
        ".finger{fill:none;stroke:#0f172a;stroke-width:5;stroke-linecap:round;stroke-linejoin:round;}"
        ".finger-0{stroke:#2563eb}.finger-1{stroke:#16a34a}.finger-2{stroke:#dc2626}"
        ".finger-3{stroke:#9333ea}.finger-4{stroke:#ea580c}"
        ".joint{fill:#ffffff;stroke:#0f172a;stroke-width:1.2;}"
        "</style>"
    )


def _frame_indices(sequence_length: int, frame_count: int) -> tuple[int, ...]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    raw = np.linspace(0, sequence_length - 1, num=frame_count)
    return tuple(int(round(value)) for value in raw.tolist())


def _split_index(split_names: tuple[str, ...], split: str) -> int:
    for index, split_name in enumerate(split_names):
        if split_name == split:
            return index
    raise ValueError(f"Unknown split: {split}")
