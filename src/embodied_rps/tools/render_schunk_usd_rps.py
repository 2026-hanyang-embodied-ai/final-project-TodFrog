"""Prepare a true Isaac SCHUNK USD RPS render job."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import yaml

from embodied_rps.pose_family import PoseFamilyLabel
from embodied_rps.schunk import load_schunk_asset_config, parse_urdf_schema, resolve_schunk_urdf
from embodied_rps.schunk_retargeting import load_schunk_retargeting_config, retarget_semantic_pose_to_schunk


_GESTURE_CURLS: dict[PoseFamilyLabel, dict[str, float]] = {
    "rock": {"thumb": 0.75, "index": 0.90, "middle": 0.90, "ring": 0.90, "pinky": 0.90},
    "paper": {"thumb": 0.20, "index": 0.08, "middle": 0.08, "ring": 0.08, "pinky": 0.08},
    "scissors": {"thumb": 0.50, "index": 0.08, "middle": 0.08, "ring": 0.90, "pinky": 0.90},
}


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare retargeted SCHUNK render metadata and user-assisted Docker command."""

    parser = argparse.ArgumentParser(description="Prepare true Isaac SCHUNK USD render job for RPS classes.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/isaac_schunk_render.yaml")
    args = parser.parse_args(argv)

    config = _load_yaml(args.config)
    output = _mapping(_required(config, "output"), "output")
    asset_config = load_schunk_asset_config(Path(_string(config, "asset_config")))
    retargeting = load_schunk_retargeting_config(Path(_string(config, "retargeting_config")))
    schema = parse_urdf_schema(resolve_schunk_urdf(asset_config))
    usd_path = Path(_string(config, "usd_path"))
    out_dir = Path(_string(output, "render_dir"))
    render_plan_path = Path(_string(output, "render_plan_json"))
    gestures = tuple(cast(Sequence[PoseFamilyLabel], _string_sequence(_required(config, "gestures"), "gestures")))
    out_dir.mkdir(parents=True, exist_ok=True)
    render_plan_path.parent.mkdir(parents=True, exist_ok=True)

    targets = {
        gesture: retarget_semantic_pose_to_schunk(retargeting, _GESTURE_CURLS[gesture], schema=schema)
        for gesture in gestures
    }
    user_command = _user_assisted_command(config_path=args.config, docker_image=_string(config, "docker_image"), cache_root=Path(_string(config, "cache_root")))
    sudo_check = subprocess.run(("sudo", "-n", "docker", "ps"), capture_output=True, text=True, check=False)
    status = "ready_to_run" if sudo_check.returncode == 0 else "password_required"
    payload: dict[str, object] = {
        "status": status,
        "blocker": None if status == "ready_to_run" else "Docker requires sudo/password; run the user_assisted_command manually.",
        "usd_path": usd_path.as_posix(),
        "usd_exists": usd_path.exists(),
        "render_dir": out_dir.as_posix(),
        "gestures": list(gestures),
        "joint_targets": targets,
        "expected_png_patterns": [str(out_dir / f"{gesture}_view*.png") for gesture in gestures],
        "user_assisted_command": user_command,
    }
    render_plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _user_assisted_command(*, config_path: Path, docker_image: str, cache_root: Path) -> str:
    workspace = Path.cwd()
    return (
        "sudo docker run --rm --gpus all --network=host "
        "--entrypoint bash -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=N "
        "-e PYTHONPATH=/workspace/embodied-final/src "
        f"-v {workspace.as_posix()}:/workspace/embodied-final "
        f"-v {cache_root.as_posix()}:/root/.cache/ov "
        f"{docker_image} -lc 'cd /isaac-sim && ./python.sh "
        f"/workspace/embodied-final/scripts/isaac_render_schunk_usd.py "
        f"--config /workspace/embodied-final/{config_path.as_posix()}'"
    )


def _load_yaml(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle)
    return _mapping(loaded, str(path))


def _mapping(value: object, label: str) -> Mapping[str, object]:
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


def _string(mapping: Mapping[str, object], key: str) -> str:
    value = _required(mapping, key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "":
            raise ValueError(f"{label} must contain strings")
        parsed.append(item)
    return tuple(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
