"""Validation helpers for SCHUNK Isaac render evidence."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Final

PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class DecodedPng:
    """Minimal decoded PNG data for same-format render comparisons."""

    width: int
    height: int
    channels: int
    pixels: bytes


@dataclass(frozen=True)
class ImageDifference:
    """Pixel and hash comparison between two PNG render outputs."""

    first_path: str
    second_path: str
    first_sha256: str
    second_sha256: str
    same_sha256: bool
    mean_abs_diff: float
    changed_pixel_ratio: float
    is_distinct: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "first_path": self.first_path,
            "second_path": self.second_path,
            "first_sha256": self.first_sha256,
            "second_sha256": self.second_sha256,
            "same_sha256": self.same_sha256,
            "mean_abs_diff": self.mean_abs_diff,
            "changed_pixel_ratio": self.changed_pixel_ratio,
            "is_distinct": self.is_distinct,
        }


def sha256_file(path: Path) -> str:
    """Return the SHA256 hash of a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_png_files(
    first: Path,
    second: Path,
    *,
    min_mean_abs_diff: float = 1.0,
    min_changed_pixel_ratio: float = 0.005,
) -> ImageDifference:
    """Compare two PNGs and decide whether they are visually distinct enough."""

    first_image = decode_png(first)
    second_image = decode_png(second)
    if (first_image.width, first_image.height, first_image.channels) != (second_image.width, second_image.height, second_image.channels):
        raise ValueError("PNG images must share width, height, and channel count for render comparison")
    if len(first_image.pixels) != len(second_image.pixels):
        raise ValueError("PNG pixel buffers have different lengths")

    total_abs_diff = 0
    changed_pixels = 0
    channels = first_image.channels
    for offset in range(0, len(first_image.pixels), channels):
        first_pixel = first_image.pixels[offset : offset + channels]
        second_pixel = second_image.pixels[offset : offset + channels]
        if first_pixel != second_pixel:
            changed_pixels += 1
        for first_value, second_value in zip(first_pixel, second_pixel):
            total_abs_diff += abs(first_value - second_value)

    byte_count = max(len(first_image.pixels), 1)
    pixel_count = max(first_image.width * first_image.height, 1)
    mean_abs_diff = float(total_abs_diff) / float(byte_count)
    changed_pixel_ratio = float(changed_pixels) / float(pixel_count)
    first_hash = sha256_file(first)
    second_hash = sha256_file(second)
    is_distinct = (
        first_hash != second_hash
        and mean_abs_diff >= min_mean_abs_diff
        and changed_pixel_ratio >= min_changed_pixel_ratio
    )
    return ImageDifference(
        first_path=first.as_posix(),
        second_path=second.as_posix(),
        first_sha256=first_hash,
        second_sha256=second_hash,
        same_sha256=first_hash == second_hash,
        mean_abs_diff=mean_abs_diff,
        changed_pixel_ratio=changed_pixel_ratio,
        is_distinct=is_distinct,
    )


def validate_same_view_rps_images(
    render_dir: Path,
    *,
    yaw_degrees: float,
    pitch_degrees: float,
    min_mean_abs_diff: float = 1.0,
    min_changed_pixel_ratio: float = 0.005,
) -> dict[str, Any]:
    """Validate that rock, paper, and scissors are distinct for one camera view."""

    paths = {
        gesture: render_dir / f"{gesture}_view_yaw{yaw_degrees:.0f}_pitch{pitch_degrees:.0f}.png"
        for gesture in ("rock", "paper", "scissors")
    }
    for gesture, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {gesture} render for validation: {path}")

    comparisons = {
        "rock_vs_paper": compare_png_files(
            paths["rock"],
            paths["paper"],
            min_mean_abs_diff=min_mean_abs_diff,
            min_changed_pixel_ratio=min_changed_pixel_ratio,
        ).to_json(),
        "scissors_vs_paper": compare_png_files(
            paths["scissors"],
            paths["paper"],
            min_mean_abs_diff=min_mean_abs_diff,
            min_changed_pixel_ratio=min_changed_pixel_ratio,
        ).to_json(),
        "scissors_vs_rock": compare_png_files(
            paths["scissors"],
            paths["rock"],
            min_mean_abs_diff=min_mean_abs_diff,
            min_changed_pixel_ratio=min_changed_pixel_ratio,
        ).to_json(),
    }
    status = "passed" if all(bool(item["is_distinct"]) for item in comparisons.values()) else "failed"
    return {
        "status": status,
        "yaw_degrees": yaw_degrees,
        "pitch_degrees": pitch_degrees,
        "comparisons": comparisons,
    }


def validate_render_run_outputs(
    render_dir: Path,
    *,
    run_start_epoch: float,
    required_visual_evidence: str = "schunk_mesh",
) -> dict[str, Any]:
    """Validate that the Isaac render wrapper produced fresh articulation evidence."""

    if required_visual_evidence not in {"schunk_mesh", "schunk_mesh_with_link_skeleton", "procedural_visual_rig", "articulation_only"}:
        raise ValueError(
            "required_visual_evidence must be one of "
            "'schunk_mesh', 'schunk_mesh_with_link_skeleton', 'procedural_visual_rig', or 'articulation_only'"
        )

    diagnostics_path = render_dir / "render_diagnostics.json"
    records_path = render_dir / "render_records.json"
    _require_fresh_file(diagnostics_path, run_start_epoch=run_start_epoch)
    _require_fresh_file(records_path, run_start_epoch=run_start_epoch)

    diagnostics = _load_json_mapping(diagnostics_path)
    articulation_status = diagnostics.get("articulation_status")
    if articulation_status != "initialized":
        raise ValueError(f"render_diagnostics.json articulation_status must be initialized, got {articulation_status!r}")
    dof_names = _parse_non_empty_string_list(diagnostics.get("dof_names"), "render_diagnostics.json must contain non-empty dof_names")
    dof_name_count = len(dof_names)
    render_validation_status = diagnostics.get("render_validation_status")
    if render_validation_status != "passed":
        raise ValueError(f"render_validation_status must be passed, got {render_validation_status!r}")

    records = _load_json_list(records_path)
    if not records:
        raise ValueError("render_records.json must contain at least one render record")
    image_count = 0
    visual_evidence_counts: dict[str, int] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"render_records.json item {index} must be a mapping")
        image_path_raw = record.get("image_path")
        if not isinstance(image_path_raw, str) or image_path_raw == "":
            raise ValueError(f"render record {index} must contain a non-empty image_path")
        image_path = _resolve_record_path(render_dir, image_path_raw)
        _require_fresh_file(image_path, run_start_epoch=run_start_epoch)
        if not isinstance(record.get("image_sha256"), str) or record.get("image_sha256") == "":
            raise ValueError(f"render record {index} must contain image_sha256")
        if not isinstance(record.get("requested_joint_positions"), Mapping):
            raise ValueError(f"render record {index} must contain requested_joint_positions")
        if not isinstance(record.get("observed_joint_positions"), Mapping):
            raise ValueError(f"render record {index} must contain observed_joint_positions")
        _validate_record_visual_evidence(
            record,
            index=index,
            required_visual_evidence=required_visual_evidence,
        )
        visual_key = _record_visual_evidence_key(record)
        visual_evidence_counts[visual_key] = visual_evidence_counts.get(visual_key, 0) + 1
        image_count += 1

    return {
        "status": "passed",
        "render_dir": render_dir.as_posix(),
        "run_start_epoch": run_start_epoch,
        "articulation_status": articulation_status,
        "render_validation_status": render_validation_status,
        "dof_count": dof_name_count,
        "record_count": len(records),
        "image_count": image_count,
        "required_visual_evidence": required_visual_evidence,
        "visual_evidence_counts": visual_evidence_counts,
        "diagnostics_mtime": diagnostics_path.stat().st_mtime,
        "records_mtime": records_path.stat().st_mtime,
    }


def _validate_record_visual_evidence(
    record: Mapping[str, Any],
    *,
    index: int,
    required_visual_evidence: str,
) -> None:
    overlay = _mapping_or_empty(record.get("debug_skeleton_overlay"))
    overlay_status = str(overlay.get("status", "missing"))
    if overlay_status == "applied":
        raise ValueError(
            f"render record {index} uses debug_skeleton_overlay; "
            "2D overlays cannot satisfy SCHUNK render evidence"
        )

    if required_visual_evidence == "articulation_only":
        return

    visual_sync = _mapping_or_empty(record.get("visual_sync"))
    visual_sync_status = str(visual_sync.get("status", "missing"))
    procedural = _mapping_or_empty(record.get("procedural_visual_rig"))
    procedural_status = str(procedural.get("status", "missing"))
    link_skeleton = _mapping_or_empty(record.get("schunk_link_skeleton"))
    link_skeleton_status = str(link_skeleton.get("status", "missing"))

    if required_visual_evidence == "schunk_mesh":
        if visual_sync_status != "synced":
            raise ValueError(
                f"render record {index} lacks synchronized SCHUNK mesh evidence; "
                f"visual_sync.status={visual_sync_status!r}"
            )
        if procedural_status == "applied":
            raise ValueError(
                f"render record {index} uses procedural_visual_rig; "
                "procedural visualization must not satisfy SCHUNK mesh evidence"
            )
        if link_skeleton_status == "applied":
            raise ValueError(
                f"render record {index} uses schunk_link_skeleton; "
                "link-skeleton visualization must use explicit schunk_mesh_with_link_skeleton evidence mode"
            )
        return

    if required_visual_evidence == "schunk_mesh_with_link_skeleton":
        if visual_sync_status != "synced":
            raise ValueError(
                f"render record {index} lacks synchronized SCHUNK mesh evidence; "
                f"visual_sync.status={visual_sync_status!r}"
            )
        if link_skeleton_status != "applied":
            raise ValueError(
                f"render record {index} lacks SCHUNK link-skeleton evidence; "
                f"schunk_link_skeleton.status={link_skeleton_status!r}"
            )
        if procedural_status == "applied":
            raise ValueError(
                f"render record {index} uses procedural_visual_rig; "
                "procedural visualization must not satisfy SCHUNK link-skeleton evidence"
            )
        link_skeleton_source = str(link_skeleton.get("source", "missing"))
        if link_skeleton_source != "schunk_link_skeleton":
            raise ValueError(
                f"render record {index} has invalid SCHUNK link-skeleton source; "
                f"schunk_link_skeleton.source={link_skeleton_source!r}"
            )
        palm_connection_count = int(link_skeleton.get("palm_connection_count", 0))
        if palm_connection_count < 5:
            raise ValueError(
                f"render record {index} has disconnected SCHUNK link-skeleton evidence; "
                f"palm_connection_count={palm_connection_count}"
            )
        curl_kinematics = _mapping_or_empty(link_skeleton.get("curl_kinematics"))
        curl_mode = str(curl_kinematics.get("mode", "missing"))
        if curl_mode != "articulated_constant_length":
            raise ValueError(
                f"render record {index} has invalid SCHUNK curl kinematics; "
                f"curl_kinematics.mode={curl_mode!r}"
            )
        max_error_raw = curl_kinematics.get("max_segment_length_error_m")
        if not isinstance(max_error_raw, (int, float)):
            raise ValueError(
                f"render record {index} must report numeric SCHUNK segment length error; "
                f"max_segment_length_error_m={max_error_raw!r}"
            )
        if float(max_error_raw) > 1.0e-9:
            raise ValueError(
                f"render record {index} does not preserve SCHUNK skeleton segment lengths; "
                f"max_segment_length_error_m={float(max_error_raw):.12g}"
            )
        return

    if required_visual_evidence == "procedural_visual_rig":
        if procedural_status != "applied":
            raise ValueError(
                f"render record {index} lacks procedural visual rig evidence; "
                f"procedural_visual_rig.status={procedural_status!r}"
            )
        return

    raise AssertionError(f"unhandled visual evidence mode: {required_visual_evidence}")


def _record_visual_evidence_key(record: Mapping[str, Any]) -> str:
    overlay_status = str(_mapping_or_empty(record.get("debug_skeleton_overlay")).get("status", "missing"))
    procedural_status = str(_mapping_or_empty(record.get("procedural_visual_rig")).get("status", "missing"))
    visual_sync_status = str(_mapping_or_empty(record.get("visual_sync")).get("status", "missing"))
    link_skeleton_status = str(_mapping_or_empty(record.get("schunk_link_skeleton")).get("status", "missing"))
    return (
        f"visual_sync={visual_sync_status};"
        f"link_skeleton={link_skeleton_status};"
        f"procedural={procedural_status};"
        f"overlay={overlay_status}"
    )


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _require_fresh_file(path: Path, *, run_start_epoch: float) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required render output: {path}")
    stat = path.stat()
    if stat.st_size <= 0:
        raise ValueError(f"{path.name} is empty")
    if stat.st_mtime < run_start_epoch:
        raise ValueError(f"{path.name} is stale; mtime={stat.st_mtime}, run_start_epoch={run_start_epoch}")


def _load_json_mapping(path: Path) -> dict[str, Any]:
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    parsed: dict[str, Any] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise ValueError(f"{path.name} must use string keys")
        parsed[key] = value
    return parsed


def _load_json_list(path: Path) -> list[Any]:
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError(f"{path.name} must contain a JSON list")
    return loaded


def _parse_non_empty_string_list(value: object, message: str) -> list[str]:
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(message)
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "":
            raise ValueError(message)
        parsed.append(item)
    return parsed


def _resolve_record_path(render_dir: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.exists():
        return path
    container_prefix = "/workspace/embodied-final/"
    if image_path.startswith(container_prefix):
        workspace = render_dir.parent.parent
        mapped = workspace / image_path[len(container_prefix) :]
        if mapped.exists():
            return mapped
    by_name = render_dir / path.name
    if by_name.exists():
        return by_name
    return path


def decode_png(path: Path) -> DecodedPng:
    """Decode non-interlaced 8-bit grayscale/RGB/RGBA PNGs using stdlib only."""

    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"not a PNG file: {path}")
    offset = len(PNG_SIGNATURE)
    width: int | None = None
    height: int | None = None
    bit_depth: int | None = None
    color_type: int | None = None
    interlace_method: int | None = None
    idat = bytearray()

    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError(f"truncated PNG chunk header: {path}")
        chunk_length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_length
        if chunk_end + 4 > len(data):
            raise ValueError(f"truncated PNG chunk body: {path}")
        chunk_data = data[chunk_start:chunk_end]
        offset = chunk_end + 4
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter, interlace_method = struct.unpack(">IIBBBBB", chunk_data)
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or bit_depth is None or color_type is None or interlace_method is None:
        raise ValueError(f"PNG missing IHDR: {path}")
    if bit_depth != 8:
        raise ValueError(f"unsupported PNG bit depth {bit_depth}: {path}")
    if interlace_method != 0:
        raise ValueError(f"unsupported interlaced PNG: {path}")
    channels_by_color_type = {0: 1, 2: 3, 6: 4}
    if color_type not in channels_by_color_type:
        raise ValueError(f"unsupported PNG color type {color_type}: {path}")
    channels = channels_by_color_type[color_type]
    decompressed = zlib.decompress(bytes(idat))
    stride = width * channels
    expected = (stride + 1) * height
    if len(decompressed) != expected:
        raise ValueError(f"unexpected PNG scanline size for {path}: expected {expected}, got {len(decompressed)}")

    pixels = bytearray()
    previous = bytearray(stride)
    cursor = 0
    for _row_index in range(height):
        filter_type = decompressed[cursor]
        cursor += 1
        scanline = bytearray(decompressed[cursor : cursor + stride])
        cursor += stride
        reconstructed = _unfilter_scanline(filter_type, scanline, previous, channels)
        pixels.extend(reconstructed)
        previous = reconstructed
    return DecodedPng(width=width, height=height, channels=channels, pixels=bytes(pixels))


def _unfilter_scanline(filter_type: int, scanline: bytearray, previous: bytearray, bytes_per_pixel: int) -> bytearray:
    reconstructed = bytearray(len(scanline))
    for index, value in enumerate(scanline):
        left = reconstructed[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        up = previous[index]
        up_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        if filter_type == 0:
            predictor = 0
        elif filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = up
        elif filter_type == 3:
            predictor = (left + up) // 2
        elif filter_type == 4:
            predictor = _paeth(left, up, up_left)
        else:
            raise ValueError(f"unsupported PNG filter type: {filter_type}")
        reconstructed[index] = (value + predictor) & 0xFF
    return reconstructed


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    distance_left = abs(estimate - left)
    distance_up = abs(estimate - up)
    distance_up_left = abs(estimate - up_left)
    if distance_left <= distance_up and distance_left <= distance_up_left:
        return left
    if distance_up <= distance_up_left:
        return up
    return up_left
