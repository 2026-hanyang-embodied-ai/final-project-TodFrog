"""Configuration loading and validation for the kinematic fallback setup."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from embodied_rps.domain import REQUIRED_GESTURES, ActuatorLimits, Gesture, HandPose


class ConfigError(ValueError):
    """Raised when a project configuration file is structurally invalid."""


@dataclass(frozen=True)
class OutputPaths:
    """Output directory configuration for pre-simulation artifacts."""

    logs_dir: str
    results_dir: str


@dataclass(frozen=True)
class KinematicConfig:
    """Validated kinematic fallback configuration."""

    joint_names: tuple[str, ...]
    gestures: dict[Gesture, HandPose]
    velocity_limits_rad_s: dict[str, float]
    response_delay_s: float
    deadline_s: float
    output: OutputPaths

    def actuator_limits(self) -> ActuatorLimits:
        """Build actuator limits from the validated config."""

        return ActuatorLimits(
            velocity_limits_rad_s=self.velocity_limits_rad_s,
            response_delay_s=self.response_delay_s,
        )


def load_kinematic_config(path: Path) -> KinematicConfig:
    """Load and validate the kinematic RPS YAML configuration."""

    with path.open("r", encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle)

    root = _as_mapping(loaded, "root")
    joint_names = _load_joint_names(_required(root, "joint_names"))
    gestures = _load_gestures(_required(root, "gestures"), joint_names)
    actuator = _as_mapping(_required(root, "actuator"), "actuator")
    output = _load_output(_required(root, "output"))

    velocity_limits = _load_float_mapping(
        _required(actuator, "velocity_limits_rad_s"),
        "actuator.velocity_limits_rad_s",
    )
    _require_exact_keys(
        velocity_limits.keys(),
        joint_names,
        "actuator.velocity_limits_rad_s",
    )

    response_delay_s = _as_non_negative_float(
        _required(actuator, "response_delay_s"),
        "actuator.response_delay_s",
    )
    deadline_s = _as_positive_float(_required(actuator, "deadline_s"), "actuator.deadline_s")

    return KinematicConfig(
        joint_names=joint_names,
        gestures=gestures,
        velocity_limits_rad_s=velocity_limits,
        response_delay_s=response_delay_s,
        deadline_s=deadline_s,
        output=output,
    )


def _load_joint_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError("joint_names must be a sequence of strings")

    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "":
            raise ConfigError("joint_names must contain non-empty strings")
        names.append(item)

    if len(names) == 0:
        raise ConfigError("joint_names must not be empty")
    if len(set(names)) != len(names):
        raise ConfigError("joint_names must not contain duplicates")

    return tuple(names)


def _load_gestures(value: object, joint_names: tuple[str, ...]) -> dict[Gesture, HandPose]:
    raw_gestures = _as_mapping(value, "gestures")
    gestures: dict[Gesture, HandPose] = {}
    for gesture in REQUIRED_GESTURES:
        pose_mapping = _load_float_mapping(_required(raw_gestures, gesture), f"gestures.{gesture}")
        _require_exact_keys(pose_mapping.keys(), joint_names, f"gesture {gesture} joint set")
        ordered_pose = {joint_name: pose_mapping[joint_name] for joint_name in joint_names}
        gestures[gesture] = HandPose(gesture=gesture, joints=ordered_pose)
    return gestures


def _load_output(value: object) -> OutputPaths:
    output = _as_mapping(value, "output")
    return OutputPaths(
        logs_dir=_as_non_empty_string(_required(output, "logs_dir"), "output.logs_dir"),
        results_dir=_as_non_empty_string(_required(output, "results_dir"), "output.results_dir"),
    )


def _load_float_mapping(value: object, label: str) -> dict[str, float]:
    raw_mapping = _as_mapping(value, label)
    parsed: dict[str, float] = {}
    for key, item in raw_mapping.items():
        parsed[key] = _as_float(item, f"{label}.{key}")
    return parsed


def _require_exact_keys(keys: Iterable[str], expected: tuple[str, ...], label: str) -> None:
    actual = set(keys)
    expected_set = set(expected)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        extra = sorted(actual - expected_set)
        raise ConfigError(f"{label} must match the configured joint set; missing={missing}, extra={extra}")


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")

    parsed: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError(f"{label} must use string keys")
        parsed[key] = item
    return parsed


def _required(mapping: Mapping[str, object], key: str) -> object:
    if key not in mapping:
        raise ConfigError(f"Missing required config key: {key}")
    return mapping[key]


def _as_float(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{label} must be a number")
    return float(value)


def _as_positive_float(value: object, label: str) -> float:
    parsed = _as_float(value, label)
    if parsed <= 0.0:
        raise ConfigError(f"{label} must be positive")
    return parsed


def _as_non_negative_float(value: object, label: str) -> float:
    parsed = _as_float(value, label)
    if parsed < 0.0:
        raise ConfigError(f"{label} must be non-negative")
    return parsed


def _as_non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ConfigError(f"{label} must be a non-empty string")
    return value
