"""Temporal skeleton review strips for v7d prompt-pose candidates."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-untyped]

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_REVIEW_ROOT = Path("artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618")
DEFAULT_SHORTLIST_ROOT = Path("artifacts/real_skeleton_v7d_manual_review_shortlist_20260618")
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_temporal_review_20260618")
SHORTLIST_FILENAME = "seed_required_shortlist.csv"
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)
OUTPUT_FIELDS: tuple[str, ...] = (
    "rank",
    "segment_id",
    "target_name",
    "proposal_role",
    "frame_count",
    "detection_coverage",
    "temporal_strip",
    "preview_image",
    "skeleton_npz",
    "review_instruction",
)
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = ("preview_image", "skeleton_npz", "temporal_strip")


@dataclass(frozen=True)
class V7DTemporalReviewConfig:
    """Inputs for rendering v7d temporal review strips."""

    project_root: Path = field(default_factory=Path.cwd)
    review_root: Path = DEFAULT_REVIEW_ROOT
    shortlist_root: Path = DEFAULT_SHORTLIST_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    max_rows: int | None = None
    frames_per_strip: int = 6
    panel_size: tuple[int, int] = (170, 170)


def write_v7d_temporal_review(config: V7DTemporalReviewConfig) -> dict[str, object]:
    """Render temporal skeleton strips for seed-required v7d review candidates."""

    project_root = config.project_root.resolve()
    review_root = _resolve_path(project_root, config.review_root)
    shortlist_root = _resolve_path(project_root, config.shortlist_root)
    output_root = _resolve_path(project_root, config.output_root)
    strips_root = output_root / "temporal_strips"
    output_root.mkdir(parents=True, exist_ok=True)
    strips_root.mkdir(parents=True, exist_ok=True)

    shortlist_path = shortlist_root / SHORTLIST_FILENAME
    if not shortlist_path.exists():
        raise FileNotFoundError(f"Missing v7d seed-required shortlist: {shortlist_path}")
    rows = _read_csv(shortlist_path)
    if config.max_rows is not None:
        rows = rows[: max(0, int(config.max_rows))]
    output_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for row in rows:
        _reject_heldout_metadata(row, context=shortlist_path)
        segment_id = str(row.get("segment_id", "")).strip()
        skeleton_rel = str(row.get("skeleton_npz", "")).strip()
        skeleton_path = review_root / skeleton_rel
        if not skeleton_path.exists():
            failures.append({"code": "missing_segment_npz", "segment_id": segment_id, "path": skeleton_rel})
            continue
        strip_path = strips_root / f"{segment_id}_temporal_strip.png"
        _render_strip(
            skeleton_path=skeleton_path,
            output_path=strip_path,
            row=row,
            frames_per_strip=config.frames_per_strip,
            panel_size=config.panel_size,
        )
        output_row = {
            "rank": row.get("shortlist_rank", ""),
            "segment_id": segment_id,
            "target_name": row.get("target_name", ""),
            "proposal_role": row.get("proposal_role", ""),
            "frame_count": row.get("frame_count", ""),
            "detection_coverage": row.get("detection_coverage", ""),
            "temporal_strip": _display_path(strip_path, base=output_root),
            "preview_image": row.get("preview_image", ""),
            "skeleton_npz": skeleton_rel,
            "review_instruction": "inspect_temporal_strip_and_original_preview_before_editing_decision_template",
        }
        _reject_heldout_metadata(output_row, context=output_root / "temporal_review_manifest.csv")
        output_rows.append(output_row)

    role_counts = dict(Counter(str(row.get("proposal_role", "")) for row in output_rows))
    status = "failed" if failures else "awaiting_manual_review"
    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "review_root": _display_path(review_root, base=project_root),
        "shortlist_root": _display_path(shortlist_root, base=project_root),
        "shortlist_csv": _display_path(shortlist_path, base=project_root),
        "temporal_manifest_csv": _display_path(output_root / "temporal_review_manifest.csv", base=project_root),
        "temporal_review_html": _display_path(output_root / "temporal_review.html", base=project_root),
        "temporal_strip_count": len(output_rows),
        "role_counts": role_counts,
        "failures": failures,
        "decisions_applied": False,
        "review_manifest_modified": False,
        "seed_package_created": False,
        "training_started": False,
        "review_policy": (
            "temporal strips are review evidence only; approval still requires explicit decisions in "
            "seed_required_decision_template.csv"
        ),
        "heldout_policy": "heldout */test paths are rejected from temporal review metadata",
    }
    _write_csv(output_root / "temporal_review_manifest.csv", OUTPUT_FIELDS, output_rows)
    (output_root / "temporal_review.html").write_text(_html(summary=summary, rows=output_rows), encoding="utf-8")
    (output_root / "temporal_review_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "temporal_review_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _render_strip(
    *,
    skeleton_path: Path,
    output_path: Path,
    row: Mapping[str, object],
    frames_per_strip: int,
    panel_size: tuple[int, int],
) -> None:
    with np.load(skeleton_path, allow_pickle=False) as data:
        canonical = data["canonical_landmarks"].astype(np.float32)
        detected = data["detected"].astype(bool) if "detected" in data.files else np.ones((len(canonical),), dtype=bool)
        active_prompts = data["active_prompts"] if "active_prompts" in data.files else np.asarray([""] * len(canonical))
        times_s = data["times_s"].astype(np.float32) if "times_s" in data.files else np.arange(len(canonical), dtype=np.float32)
    if canonical.ndim != 3 or canonical.shape[1:] != (21, 3):
        raise ValueError(f"Expected canonical landmarks shape (T, 21, 3), got {canonical.shape} from {skeleton_path}")
    frame_indices = _sample_indices(len(canonical), max(1, int(frames_per_strip)))
    panel_w, panel_h = panel_size
    header_h = 44
    footer_h = 22
    image = Image.new("RGB", (panel_w * len(frame_indices), panel_h + header_h + footer_h), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for panel_index, frame_index in enumerate(frame_indices):
        x0 = panel_index * panel_w
        panel = (x0, header_h, x0 + panel_w, header_h + panel_h)
        draw.rectangle(panel, fill=(255, 255, 255), outline=(190, 200, 210))
        label = f"{frame_index + 1}/{len(canonical)} t={float(times_s[frame_index]):.2f}s"
        prompt = str(active_prompts[frame_index]) if frame_index < len(active_prompts) else ""
        draw.text((x0 + 8, 8), label, fill=(15, 23, 42), font=font)
        draw.text((x0 + 8, 23), f"prompt={prompt}", fill=(37, 99, 235), font=font)
        if bool(detected[frame_index]):
            _draw_landmarks(draw, canonical[frame_index], panel)
        else:
            draw.text((x0 + 24, header_h + panel_h // 2), "not detected", fill=(185, 28, 28), font=font)
    title = f"{row.get('target_name')} / {row.get('proposal_role')} / {row.get('segment_id')}"
    draw.text((8, panel_h + header_h + 4), title[:120], fill=(15, 23, 42), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _draw_landmarks(
    draw: ImageDraw.ImageDraw,
    landmarks: NDArray[np.float32],
    panel: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = panel
    coords = landmarks[:, :2].astype(np.float32)
    finite = np.isfinite(coords).all(axis=1)
    if not bool(finite.any()):
        return
    valid = coords[finite]
    min_xy = valid.min(axis=0)
    max_xy = valid.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-4)
    margin = 18

    def point(index: int) -> tuple[int, int]:
        xy = (coords[index] - min_xy) / span
        x = x0 + margin + float(xy[0]) * max(1, (x1 - x0 - margin * 2))
        y = y0 + margin + float(xy[1]) * max(1, (y1 - y0 - margin * 2))
        return int(round(x)), int(round(y))

    for start, end in HAND_CONNECTIONS:
        if finite[start] and finite[end]:
            draw.line([point(start), point(end)], fill=(30, 64, 175), width=2)
    for index in range(21):
        if finite[index]:
            px, py = point(index)
            radius = 3 if index in {4, 8, 12, 16, 20} else 2
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(245, 158, 11), outline=(15, 23, 42))


def _sample_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return [0]
    if count <= 1:
        return [max(0, length // 2)]
    return sorted({int(round(value)) for value in np.linspace(0, length - 1, count)})


def _html(*, summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>V7d Temporal Review</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#0f172a}"
        ".row{margin:18px 0;padding:12px;border:1px solid #cbd5e1;background:white}"
        "img{max-width:100%;height:auto;border:1px solid #cbd5e1}.meta{font-size:13px;color:#475569}</style>",
        "</head>",
        "<body>",
        "<h1>V7d Temporal Review</h1>",
        f"<p>Status: <code>{summary.get('status')}</code>. These strips are review evidence only; they do not approve training rows.</p>",
    ]
    for row in rows:
        lines.extend(
            [
                '<div class="row">',
                f"<h2>{_escape(str(row.get('rank', '')))}. {_escape(str(row.get('segment_id', '')))}</h2>",
                f"<p class=\"meta\">target={_escape(str(row.get('target_name', '')))} role={_escape(str(row.get('proposal_role', '')))} coverage={_escape(str(row.get('detection_coverage', '')))}</p>",
                f"<img src=\"{_escape(str(row.get('temporal_strip', '')))}\" alt=\"temporal strip\">",
                "</div>",
            ]
        )
    lines.extend(["</body>", "</html>", ""])
    return "\n".join(lines)


def _summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V7d Temporal Review Summary",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Temporal strips: `{summary.get('temporal_strip_count')}`",
        f"- HTML: `{summary.get('temporal_review_html')}`",
        "- Decisions applied: `False`",
        "- Seed package created: `False`",
        "- Training started: `False`",
        "",
        "## Role Counts",
        "",
    ]
    role_counts = summary.get("role_counts", {})
    if isinstance(role_counts, Mapping):
        for role, count in sorted(role_counts.items()):
            lines.append(f"- `{role}`: `{count}`")
    lines.append("")
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _reject_heldout_metadata(row: Mapping[str, object], *, context: Path) -> None:
    for field_name in PATH_FIELDS_TO_AUDIT:
        value = str(row.get(field_name, "")).strip()
        if value and _is_heldout_test_path(value):
            raise ValueError(f"{context} contains held-out test path in {field_name}: {value}")


def _is_heldout_test_path(value: str) -> bool:
    normalized = value.replace("\\", "/").lower().rstrip("/")
    return "/test/" in normalized or normalized.endswith("/test")


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["V7DTemporalReviewConfig", "write_v7d_temporal_review"]
