"""SCHUNK SVH asset audit, skeleton extraction, and preview rendering."""

from __future__ import annotations

import json
import math
import struct
import subprocess
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from xml.etree import ElementTree

import yaml

from embodied_rps.domain import Gesture, REQUIRED_GESTURES

Vector3 = tuple[float, float, float]
PoseMap = dict[str, float]
GesturePoseMap = dict[Gesture, PoseMap]
RenderBackend = Literal["kinematic_schunk_proxy"]


@dataclass(frozen=True)
class UrdfJointInfo:
    name: str
    joint_type: str
    parent: str | None
    child: str | None
    axis: Vector3 | None
    lower: float | None
    upper: float | None
    mimic_joint: str | None = None
    mimic_multiplier: float | None = None
    mimic_offset: float | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "type": self.joint_type,
            "parent": self.parent,
            "child": self.child,
            "axis": None if self.axis is None else list(self.axis),
            "lower": self.lower,
            "upper": self.upper,
            "mimic_joint": self.mimic_joint,
            "mimic_multiplier": self.mimic_multiplier,
            "mimic_offset": self.mimic_offset,
        }


@dataclass(frozen=True)
class UrdfJointSchema:
    robot_name: str
    link_names: tuple[str, ...]
    joints: tuple[UrdfJointInfo, ...]
    mesh_files: tuple[str, ...]

    @property
    def link_count(self) -> int:
        return len(self.link_names)

    @property
    def joint_count(self) -> int:
        return len(self.joints)

    @property
    def revolute_joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints if joint.joint_type == "revolute")

    @property
    def revolute_joint_count(self) -> int:
        return len(self.revolute_joint_names)

    @property
    def fixed_joint_count(self) -> int:
        return sum(1 for joint in self.joints if joint.joint_type == "fixed")

    def to_json(self) -> dict[str, object]:
        return {
            "robot_name": self.robot_name,
            "link_count": self.link_count,
            "joint_count": self.joint_count,
            "revolute_joint_count": self.revolute_joint_count,
            "fixed_joint_count": self.fixed_joint_count,
            "link_names": list(self.link_names),
            "revolute_joint_names": list(self.revolute_joint_names),
            "mesh_files": list(self.mesh_files),
            "joints": [joint.to_json() for joint in self.joints],
        }


@dataclass(frozen=True)
class SchunkAssetConfig:
    repository_url: str
    revision: str
    sparse_path: str
    local_root: Path
    urdf_candidates: tuple[str, ...]
    license_path: Path
    expected_revolute_joints: int
    expected_total_joints: int
    audit_output_path: Path
    joint_schema_output_path: Path
    skeleton_schema_output_path: Path
    isaac_import_output_path: Path
    usd_output_dir: Path
    docker_image: str
    isaac_cache_root: Path


@dataclass(frozen=True)
class SchunkPoseConfig:
    joint_names: tuple[str, ...]
    gestures: GesturePoseMap
    unused_passive_joints: tuple[str, ...]


@dataclass(frozen=True)
class MultiviewCaptureConfig:
    asset_config_path: Path
    pose_config_path: Path
    out_dir: Path
    metadata_path: Path
    yaw_degrees: tuple[float, ...]
    pitch_degrees: tuple[float, ...]
    distance_m: float
    focal_length_mm: float
    gestures: tuple[Gesture, ...]
    image_width: int
    image_height: int


@dataclass(frozen=True)
class SchunkPreviewOutput:
    preview_images: tuple[Path, ...]
    metadata_path: Path


def load_schunk_asset_config(path: Path) -> SchunkAssetConfig:
    root = _load_yaml_mapping(path, "schunk asset config")
    source = _as_mapping(_required(root, "source"), "source")
    asset = _as_mapping(_required(root, "asset"), "asset")
    expected = _as_mapping(_required(root, "expected"), "expected")
    output = _as_mapping(_required(root, "output"), "output")
    isaac = _as_mapping(_required(root, "isaac"), "isaac")
    local_root = Path(_required_string(source, "local_root"))
    return SchunkAssetConfig(
        repository_url=_required_string(source, "repository_url"),
        revision=_required_string(source, "revision"),
        sparse_path=_required_string(source, "sparse_path"),
        local_root=local_root,
        urdf_candidates=_string_tuple(_required(asset, "urdf_candidates"), "asset.urdf_candidates"),
        license_path=local_root / _required_string(asset, "license_path"),
        expected_revolute_joints=_required_int(expected, "revolute_joints"),
        expected_total_joints=_required_int(expected, "total_joints"),
        audit_output_path=Path(_required_string(output, "audit_json")),
        joint_schema_output_path=Path(_required_string(output, "joint_schema_json")),
        skeleton_schema_output_path=Path(_required_string(output, "skeleton_schema_json")),
        isaac_import_output_path=Path(_required_string(output, "isaac_import_json")),
        usd_output_dir=Path(_required_string(output, "usd_output_dir")),
        docker_image=_required_string(isaac, "docker_image"),
        isaac_cache_root=Path(_required_string(isaac, "cache_root")),
    )


def load_schunk_pose_config(path: Path) -> SchunkPoseConfig:
    root = _load_yaml_mapping(path, "schunk pose config")
    joint_names = _string_tuple(_required(root, "joint_names"), "joint_names")
    gestures_root = _as_mapping(_required(root, "gestures"), "gestures")
    gestures: GesturePoseMap = {}
    for gesture in REQUIRED_GESTURES:
        pose_mapping = _as_mapping(_required(gestures_root, gesture), f"gestures.{gesture}")
        gestures[gesture] = {joint_name: _as_float(_required(pose_mapping, joint_name), joint_name) for joint_name in joint_names}
    return SchunkPoseConfig(
        joint_names=joint_names,
        gestures=gestures,
        unused_passive_joints=_string_tuple(root.get("unused_passive_joints", ()), "unused_passive_joints"),
    )


def load_multiview_capture_config(path: Path) -> MultiviewCaptureConfig:
    root = _load_yaml_mapping(path, "multiview capture config")
    camera = _as_mapping(_required(root, "camera"), "camera")
    output = _as_mapping(_required(root, "output"), "output")
    image = _as_mapping(_required(root, "image"), "image")
    gestures = tuple(cast(Sequence[Gesture], _string_tuple(_required(root, "gestures"), "gestures")))
    return MultiviewCaptureConfig(
        asset_config_path=Path(_required_string(root, "asset_config")),
        pose_config_path=Path(_required_string(root, "pose_config")),
        out_dir=Path(_required_string(output, "preview_dir")),
        metadata_path=Path(_required_string(output, "metadata_jsonl")),
        yaw_degrees=_float_tuple(_required(camera, "yaw_degrees"), "camera.yaw_degrees"),
        pitch_degrees=_float_tuple(_required(camera, "pitch_degrees"), "camera.pitch_degrees"),
        distance_m=_as_positive_float(_required(camera, "distance_m"), "camera.distance_m"),
        focal_length_mm=_as_positive_float(_required(camera, "focal_length_mm"), "camera.focal_length_mm"),
        gestures=gestures,
        image_width=_as_positive_int(_required(image, "width"), "image.width"),
        image_height=_as_positive_int(_required(image, "height"), "image.height"),
    )


def ensure_sparse_dex_urdf(config: SchunkAssetConfig) -> None:
    if any((config.local_root / candidate).exists() for candidate in config.urdf_candidates):
        return
    config.local_root.parent.mkdir(parents=True, exist_ok=True)
    if not config.local_root.exists():
        _run_checked(("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "--branch", config.revision, config.repository_url, str(config.local_root)))
    _run_checked(("git", "-C", str(config.local_root), "sparse-checkout", "set", config.sparse_path))


def resolve_schunk_urdf(config: SchunkAssetConfig) -> Path:
    for candidate in config.urdf_candidates:
        path = config.local_root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"No configured SCHUNK URDF exists under {config.local_root}")


def parse_urdf_schema(urdf_path: Path) -> UrdfJointSchema:
    root = ElementTree.parse(urdf_path).getroot()
    links = tuple(link.attrib["name"] for link in root.findall("link") if "name" in link.attrib)
    joints: list[UrdfJointInfo] = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        joints.append(
            UrdfJointInfo(
                name=joint.attrib.get("name", ""),
                joint_type=joint.attrib.get("type", ""),
                parent=None if parent is None else parent.attrib.get("link"),
                child=None if child is None else child.attrib.get("link"),
                axis=None if axis is None else _parse_vector3(axis.attrib.get("xyz", "0 0 0")),
                lower=None if limit is None else _optional_float(limit.attrib.get("lower")),
                upper=None if limit is None else _optional_float(limit.attrib.get("upper")),
                mimic_joint=None if mimic is None else mimic.attrib.get("joint"),
                mimic_multiplier=None if mimic is None else _optional_float(mimic.attrib.get("multiplier", "1.0")),
                mimic_offset=None if mimic is None else _optional_float(mimic.attrib.get("offset", "0.0")),
            )
        )
    meshes = tuple(mesh.attrib["filename"] for mesh in root.findall(".//mesh") if "filename" in mesh.attrib)
    return UrdfJointSchema(root.attrib.get("name", urdf_path.stem), links, tuple(joints), meshes)


def audit_schunk_asset(*, urdf_path: Path, expected_revolute_joints: int, expected_total_joints: int, license_path: Path) -> dict[str, object]:
    schema = parse_urdf_schema(urdf_path)
    missing_meshes = [mesh for mesh in schema.mesh_files if not _mesh_exists(urdf_path.parent, mesh)]
    count_errors: list[str] = []
    if schema.revolute_joint_count != expected_revolute_joints:
        count_errors.append(f"expected {expected_revolute_joints} revolute joints, found {schema.revolute_joint_count}")
    if schema.joint_count != expected_total_joints:
        count_errors.append(f"expected {expected_total_joints} total joints, found {schema.joint_count}")
    license_id = _detect_license_id(license_path)
    failures = list(count_errors)
    if missing_meshes:
        failures.append("missing mesh files")
    if license_id == "missing":
        failures.append("missing license file")
    return {
        "status": "passed" if not failures else "failed",
        "urdf_path": urdf_path.as_posix(),
        "license_path": license_path.as_posix(),
        "license_id": license_id,
        "mesh_count": len(schema.mesh_files),
        "missing_meshes": missing_meshes,
        "count_errors": count_errors,
        "joint_schema": schema.to_json(),
        "notes": [
            "Use the URDF-derived simulated SCHUNK SVH approximation wording for paper claims.",
            "Visual mesh fidelity does not imply verified physical or dynamic fidelity.",
        ],
    }


def write_asset_audit_outputs(config: SchunkAssetConfig) -> dict[str, object]:
    ensure_sparse_dex_urdf(config)
    urdf = resolve_schunk_urdf(config)
    audit = audit_schunk_asset(urdf_path=urdf, expected_revolute_joints=config.expected_revolute_joints, expected_total_joints=config.expected_total_joints, license_path=config.license_path)
    config.audit_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.joint_schema_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.audit_output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    config.joint_schema_output_path.write_text(json.dumps(parse_urdf_schema(urdf).to_json(), indent=2), encoding="utf-8")
    return audit


def validate_pose_coverage(pose_config: SchunkPoseConfig, *, controllable_joints: Sequence[str]) -> None:
    configured = set(pose_config.joint_names)
    expected = set(controllable_joints)
    missing = sorted(expected - configured)
    extra = sorted(configured - expected)
    if missing or extra:
        raise ValueError(f"SCHUNK pose joint_names must match controllable joints; missing={missing}, extra={extra}")
    for gesture in REQUIRED_GESTURES:
        if set(pose_config.gestures[gesture]) != configured:
            raise ValueError(f"Gesture {gesture} does not cover the configured SCHUNK joints")


def write_skeleton_schema(config: SchunkAssetConfig) -> dict[str, object]:
    ensure_sparse_dex_urdf(config)
    schema = parse_urdf_schema(resolve_schunk_urdf(config))
    payload: dict[str, object] = {
        "source": "dexsuite/dex-urdf robots/hands/schunk_hand",
        "representation": "robot-native SCHUNK SVH link skeleton",
        "normalization": {"palm_link": "palm", "x_axis_link": "index_base", "y_axis_link": "middle_base", "scale_link": "index_base"},
        "finger_chains": {
            "thumb": ["thumb_base", "thumb_mid", "thumb_tip"],
            "index": ["index_base", "index_mid", "index_tip"],
            "middle": ["middle_base", "middle_mid", "middle_tip"],
            "ring": ["ring_base", "ring_mid", "ring_tip"],
            "pinky": ["pinky_base", "pinky_mid", "pinky_tip"],
        },
        "controllable_joints": list(schema.revolute_joint_names),
        "deferred": "Human 21-keypoint retargeting remains out of v1 unless raw perception becomes necessary.",
    }
    config.skeleton_schema_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.skeleton_schema_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def normalize_link_positions(link_positions: Mapping[str, Sequence[float]], *, palm_link: str, x_axis_link: str, y_axis_link: str, scale_link: str) -> dict[str, Vector3]:
    origin = _vector_from_mapping(link_positions, palm_link)
    x_axis = _normalize(_sub(_vector_from_mapping(link_positions, x_axis_link), origin))
    y_raw = _normalize(_sub(_vector_from_mapping(link_positions, y_axis_link), origin))
    z_axis = _normalize(_cross(x_axis, y_raw))
    y_axis = _normalize(_cross(z_axis, x_axis))
    scale = _norm(_sub(_vector_from_mapping(link_positions, scale_link), origin))
    if scale <= 1e-9:
        raise ValueError("scale_link must be separated from palm_link")
    return {name: (_dot(rel := _scale(_sub(_as_vector3(pos, name), origin), 1.0 / scale), x_axis), _dot(rel, y_axis), _dot(rel, z_axis)) for name, pos in link_positions.items()}


def generate_pose_skeleton(joint_state: Mapping[str, float]) -> dict[str, Vector3]:
    positions: dict[str, Vector3] = {"palm": (0.0, 0.0, 0.0)}
    specs = (
        ("thumb", (-0.52, -0.15, 0.0), -0.75, ("Thumb", "j5", "j3", "j4"), 0.22),
        ("index", (-0.28, 0.42, 0.0), -0.18, ("Index", "index"), 0.30),
        ("middle", (0.0, 0.48, 0.0), 0.0, ("Middle", "middle"), 0.32),
        ("ring", (0.28, 0.42, 0.0), 0.18, ("Ring", "ring", "j12", "j16"), 0.29),
        ("pinky", (0.52, 0.34, 0.0), 0.34, ("Pinky", "pinky", "j13", "j17"), 0.25),
    )
    for name, base, angle, tokens, length in specs:
        curl = _mean_matching_joint_value(joint_state, tokens)
        positions[f"{name}_base"] = base
        current = base
        theta = math.pi / 2.0 + angle
        for idx, suffix in enumerate(("mid", "tip"), start=1):
            theta += curl * (0.35 + 0.18 * idx)
            current = (current[0] + math.cos(theta) * length, current[1] + math.sin(theta) * length, current[2] + 0.04 * idx * curl)
            positions[f"{name}_{suffix}"] = current
    return positions


def build_multiview_records(*, gesture: Gesture, joint_state: Mapping[str, float], normalized_links: Mapping[str, Vector3], yaw_degrees: Sequence[float], pitch_degrees: Sequence[float], distance_m: float, focal_length_mm: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for yaw in yaw_degrees:
        for pitch in pitch_degrees:
            rows.append({
                "gesture": gesture,
                "camera": {"yaw_deg": float(yaw), "pitch_deg": float(pitch), "distance_m": float(distance_m), "focal_length_mm": float(focal_length_mm)},
                "joint_state": {name: float(value) for name, value in joint_state.items()},
                "normalized_skeleton": {name: [float(component) for component in value] for name, value in normalized_links.items()},
                "visibility_mask": {name: True for name in normalized_links},
            })
    return rows


def render_schunk_pose_previews(*, pose_config: SchunkPoseConfig, out_dir: Path, metadata_path: Path, yaw_degrees: Sequence[float], pitch_degrees: Sequence[float], gestures: Sequence[Gesture], distance_m: float = 0.75, focal_length_mm: float = 35.0, image_width: int = 720, image_height: int = 520, backend: RenderBackend = "kinematic_schunk_proxy") -> SchunkPreviewOutput:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []
    rows: list[dict[str, object]] = []
    for gesture in gestures:
        state = pose_config.gestures[gesture]
        skeleton = generate_pose_skeleton(state)
        normalized = normalize_link_positions(skeleton, palm_link="palm", x_axis_link="index_base", y_axis_link="middle_base", scale_link="index_base")
        records = build_multiview_records(gesture=gesture, joint_state=state, normalized_links=normalized, yaw_degrees=yaw_degrees, pitch_degrees=pitch_degrees, distance_m=distance_m, focal_length_mm=focal_length_mm)
        for record in records:
            camera = cast(Mapping[str, object], record["camera"])
            yaw = _as_float(camera["yaw_deg"], "camera.yaw_deg")
            pitch = _as_float(camera["pitch_deg"], "camera.pitch_deg")
            path = out_dir / f"schunk_{gesture}_yaw{yaw:.0f}_pitch{pitch:.0f}.png"
            _render_pose_png(path, skeleton=skeleton, gesture=gesture, yaw_degrees=yaw, pitch_degrees=pitch, width=image_width, height=image_height)
            row = dict(record)
            row["render_backend"] = backend
            row["image_path"] = path.as_posix()
            rows.append(row)
            images.append(path)
    metadata_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return SchunkPreviewOutput(tuple(images), metadata_path)


def write_isaac_import_smoke(config: SchunkAssetConfig) -> dict[str, object]:
    ensure_sparse_dex_urdf(config)
    urdf = resolve_schunk_urdf(config)
    docker = subprocess.run(("docker", "ps"), capture_output=True, text=True, check=False)
    sudo = subprocess.run(("sudo", "-n", "docker", "ps"), capture_output=True, text=True, check=False)
    command = "sudo docker run --rm --gpus all --network=host -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=N " + f"-v {Path.cwd().as_posix()}:/workspace/embodied-final -v {config.isaac_cache_root.as_posix()}:/root/.cache/ov {config.docker_image} bash -lc 'cd /workspace/embodied-final && ./python.sh standalone_examples/api/isaacsim.asset.importer.urdf/urdf_import.py --urdf {urdf.as_posix()} --usd-path {config.usd_output_dir.as_posix()} --merge-mesh'"
    status = "ready_to_run"
    blocker = None
    if docker.returncode != 0 and sudo.returncode != 0:
        status = "password_required"
        blocker = "Docker requires sudo/password in this session; user-assisted command is required."
    payload: dict[str, object] = {"status": status, "blocker": blocker, "urdf_path": urdf.as_posix(), "usd_output_dir": config.usd_output_dir.as_posix(), "docker_image": config.docker_image, "user_assisted_command": command, "docker_check_stderr": docker.stderr.strip(), "sudo_check_stderr": sudo.stderr.strip()}
    config.isaac_import_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.isaac_import_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _render_pose_png(path: Path, *, skeleton: Mapping[str, Vector3], gesture: Gesture, yaw_degrees: float, pitch_degrees: float, width: int, height: int) -> None:
    pixels = [[(248, 250, 252) for _ in range(width)] for _ in range(height)]
    projected = _project_points(skeleton, yaw_degrees=yaw_degrees, pitch_degrees=pitch_degrees, width=width, height=height)
    for y in range(min(42, height)):
        for x in range(width):
            pixels[y][x] = (226, 232, 240)
    del gesture
    for chain in _finger_chains():
        color = _chain_color(chain[0])
        for start, end in zip(chain, chain[1:]):
            _draw_line(pixels, projected[start], projected[end], color=color, radius=6)
        for link in chain:
            _draw_circle(pixels, projected[link], radius=8, color=(15, 23, 42))
            _draw_circle(pixels, projected[link], radius=5, color=(255, 255, 255))
    _draw_circle(pixels, projected["palm"], radius=28, color=(203, 213, 225))
    _draw_circle(pixels, projected["palm"], radius=22, color=(226, 232, 240))
    _write_png(path, pixels)


def _project_points(points: Mapping[str, Vector3], *, yaw_degrees: float, pitch_degrees: float, width: int, height: int) -> dict[str, tuple[int, int]]:
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    rotated: dict[str, tuple[float, float]] = {}
    for name, (x, y, z) in points.items():
        xr = math.cos(yaw) * x + math.sin(yaw) * z
        zr = -math.sin(yaw) * x + math.cos(yaw) * z
        yr = math.cos(pitch) * y - math.sin(pitch) * zr
        rotated[name] = (xr, yr)
    xs = [p[0] for p in rotated.values()]
    ys = [p[1] for p in rotated.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    scale = min(width * 0.68, height * 0.68) / span
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    return {name: (int(round(width / 2.0 + (x - cx) * scale)), int(round(height / 2.0 - (y - cy) * scale + 20.0))) for name, (x, y) in rotated.items()}


def _write_png(path: Path, pixels: Sequence[Sequence[tuple[int, int, int]]]) -> None:
    height = len(pixels)
    width = len(pixels[0]) if height else 0
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for red, green, blue in row:
            raw.extend((red, green, blue))
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    png.extend(_png_chunk(b"IDAT", zlib.compress(bytes(raw), level=6)))
    png.extend(_png_chunk(b"IEND", b""))
    path.write_bytes(bytes(png))


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _draw_line(pixels: list[list[tuple[int, int, int]]], start: tuple[int, int], end: tuple[int, int], *, color: tuple[int, int, int], radius: int) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for step in range(steps + 1):
        t = step / steps
        _draw_circle(pixels, (int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t))), radius=radius, color=color)


def _draw_circle(pixels: list[list[tuple[int, int, int]]], center: tuple[int, int], *, radius: int, color: tuple[int, int, int]) -> None:
    cx, cy = center
    height = len(pixels)
    width = len(pixels[0]) if height else 0
    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius:
                pixels[y][x] = color


def _finger_chains() -> tuple[tuple[str, ...], ...]:
    return (("thumb_base", "thumb_mid", "thumb_tip"), ("index_base", "index_mid", "index_tip"), ("middle_base", "middle_mid", "middle_tip"), ("ring_base", "ring_mid", "ring_tip"), ("pinky_base", "pinky_mid", "pinky_tip"))


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


def _mean_matching_joint_value(joint_state: Mapping[str, float], tokens: Sequence[str]) -> float:
    values = [float(value) for name, value in joint_state.items() if any(token.lower() in name.lower() for token in tokens)]
    return 0.15 if not values else max(0.0, min(1.4, sum(values) / len(values)))


def _mesh_exists(root: Path, mesh: str) -> bool:
    return False if mesh.startswith("package://") else (root / mesh).exists()


def _detect_license_id(path: Path) -> str:
    if not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8", errors="replace")[:4096].lower()
    if "gnu general public license" in text and "version 3" in text:
        return "GPL-3.0"
    if "mit license" in text:
        return "MIT"
    return "unknown"


def _parse_vector3(text: str) -> Vector3:
    parts = text.split()
    if len(parts) != 3:
        raise ValueError(f"Expected 3-vector, got {text!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _optional_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def _vector_from_mapping(mapping: Mapping[str, Sequence[float]], key: str) -> Vector3:
    if key not in mapping:
        raise ValueError(f"Missing skeleton link {key}")
    return _as_vector3(mapping[key], key)


def _as_vector3(value: Sequence[float], label: str) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"{label} must be a 3-vector")
    return (float(value[0]), float(value[1]), float(value[2]))


def _sub(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _scale(vector: Vector3, factor: float) -> Vector3:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _dot(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2], left[0] * right[1] - left[1] * right[0])


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vector3) -> Vector3:
    norm = _norm(vector)
    if norm <= 1e-9:
        raise ValueError("Cannot normalize a near-zero vector")
    return _scale(vector, 1.0 / norm)


def _run_checked(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def _load_yaml_mapping(path: Path, label: str) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle)
    return _as_mapping(loaded, label)


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    parsed: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} must use string keys")
        parsed[key] = item
    return parsed


def _required(mapping: Mapping[str, object], key: str) -> object:
    if key not in mapping:
        raise ValueError(f"Missing required key: {key}")
    return mapping[key]


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = _required(mapping, key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(mapping: Mapping[str, object], key: str) -> int:
    return _as_positive_int(_required(mapping, key), key)


def _as_positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _as_float(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _as_positive_float(value: object, label: str) -> float:
    parsed = _as_float(value, label)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "":
            raise ValueError(f"{label} must contain non-empty strings")
        out.append(item)
    return tuple(out)


def _float_tuple(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of numbers")
    parsed = tuple(_as_float(item, label) for item in value)
    if not parsed:
        raise ValueError(f"{label} must not be empty")
    return parsed
