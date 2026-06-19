"""Retarget semantic RPS pose families onto SCHUNK SVH joint targets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import yaml

from embodied_rps.pose_family import FingerName, PoseFamilyLabel, label_from_finger_curls
from embodied_rps.schunk import UrdfJointSchema

AmbiguousPolicy: TypeAlias = Literal["reject", "neutral"]


@dataclass(frozen=True)
class SchunkRetargetingConfig:
    """Semantic-to-SCHUNK joint target mapping."""

    joint_groups: dict[FingerName, tuple[str, ...]]
    spread_joints: dict[str, dict[str, float]]
    extended_fraction: float
    flexed_fraction: float
    thumb_relaxed_fraction: float
    thumb_wrapped_fraction: float
    ambiguous_policy: AmbiguousPolicy


def load_schunk_retargeting_config(path: Path) -> SchunkRetargetingConfig:
    """Load a SCHUNK retargeting YAML config."""

    with path.open("r", encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle)
    root = _as_mapping(loaded, "schunk retargeting config")
    joint_groups_root = _as_mapping(_required(root, "joint_groups"), "joint_groups")
    spread_root = _as_mapping(root.get("spread_joints", {}), "spread_joints")
    target_root = _as_mapping(_required(root, "joint_targets"), "joint_targets")

    joint_groups: dict[FingerName, tuple[str, ...]] = {}
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        joint_groups[finger] = _string_tuple(_required(joint_groups_root, finger), f"joint_groups.{finger}")
    spread_joints: dict[str, dict[str, float]] = {}
    for joint_name, values in spread_root.items():
        value_map = _as_mapping(values, f"spread_joints.{joint_name}")
        spread_joints[joint_name] = {label: _as_float(item, f"spread_joints.{joint_name}.{label}") for label, item in value_map.items()}
    policy_raw = _required_string(root, "ambiguous_policy")
    if policy_raw not in ("reject", "neutral"):
        raise ValueError("ambiguous_policy must be reject or neutral")
    return SchunkRetargetingConfig(
        joint_groups=joint_groups,
        spread_joints=spread_joints,
        extended_fraction=_fraction(_required_float(target_root, "extended_fraction"), "joint_targets.extended_fraction"),
        flexed_fraction=_fraction(_required_float(target_root, "flexed_fraction"), "joint_targets.flexed_fraction"),
        thumb_relaxed_fraction=_fraction(_required_float(target_root, "thumb_relaxed_fraction"), "joint_targets.thumb_relaxed_fraction"),
        thumb_wrapped_fraction=_fraction(_required_float(target_root, "thumb_wrapped_fraction"), "joint_targets.thumb_wrapped_fraction"),
        ambiguous_policy=cast(AmbiguousPolicy, policy_raw),
    )


def validate_retargeting_with_schema(config: SchunkRetargetingConfig, schema: UrdfJointSchema) -> None:
    """Validate that every configured joint exists and every revolute joint is covered."""

    schema_revolute = set(schema.revolute_joint_names)
    configured = set(config.spread_joints)
    for group in config.joint_groups.values():
        configured.update(group)
    missing_from_schema = sorted(configured - schema_revolute)
    missing_from_config = sorted(schema_revolute - configured)
    if missing_from_schema or missing_from_config:
        raise ValueError(f"SCHUNK retargeting joint coverage mismatch; missing_from_schema={missing_from_schema}, missing_from_config={missing_from_config}")


def retarget_semantic_pose_to_schunk(
    config: SchunkRetargetingConfig,
    finger_curls: Mapping[str, float],
    *,
    schema: UrdfJointSchema,
) -> dict[str, float]:
    """Map semantic finger curls into SCHUNK joint targets within URDF limits."""

    label = label_from_finger_curls(finger_curls)
    if label == "ambiguous":
        if config.ambiguous_policy == "reject":
            raise ValueError("ambiguous pose cannot be retargeted when ambiguous_policy=reject")
        label = "paper"
    limits = _joint_limits(schema)
    targets: dict[str, float] = {}
    for finger, joint_names in config.joint_groups.items():
        fraction = _target_fraction(config, label=label, finger=finger, curl=float(finger_curls.get(finger, 0.0)))
        for joint_name in joint_names:
            lower, upper = limits[joint_name]
            targets[joint_name] = lower + (upper - lower) * fraction
    for joint_name, by_label in config.spread_joints.items():
        lower, upper = limits[joint_name]
        raw_target = float(by_label.get(label, by_label.get("paper", lower)))
        targets[joint_name] = min(upper, max(lower, raw_target))
    validate_retargeting_with_schema(config, schema)
    return targets


def mimic_joint_names(schema: UrdfJointSchema) -> tuple[str, ...]:
    """Return revolute joints whose motion is defined by a URDF mimic relation."""

    return tuple(joint.name for joint in schema.joints if joint.joint_type == "revolute" and joint.mimic_joint is not None)


def command_joint_targets_for_schunk(targets: Mapping[str, float], schema: UrdfJointSchema) -> dict[str, float]:
    """Filter static pose targets down to primary command joints.

    SCHUNK's imported USD contains mimic joints for coupled phalanges and spread
    joints. Controller commands should address the primary joints; static render
    state can still include every revolute joint when a proof image needs exact
    visual placement.
    """

    schema_revolute = set(schema.revolute_joint_names)
    unknown_targets = sorted(set(targets) - schema_revolute)
    if unknown_targets:
        raise ValueError(f"targets contain joints not present in the SCHUNK schema: {unknown_targets}")

    command_targets: dict[str, float] = {}
    for joint in schema.joints:
        if joint.joint_type != "revolute" or joint.mimic_joint is not None:
            continue
        if joint.name not in targets:
            raise ValueError(f"missing target for primary command joint: {joint.name}")
        command_targets[joint.name] = float(targets[joint.name])
    return command_targets


def schunk_targets_radians_to_isaac_degrees(targets: Mapping[str, float]) -> dict[str, float]:
    """Convert URDF-radian retarget values for Isaac's imported USD articulation.

    The SCHUNK URDF parser keeps revolute limits in radians, matching the URDF
    source. Isaac's URDF-imported USD stores revolute joint limits in degrees,
    and the Articulation API for this imported asset reports/accepts those
    degree values. Rendering must therefore convert the semantic retarget values
    at the Isaac boundary instead of changing the source retargeting semantics.
    """

    return {joint_name: math.degrees(float(value)) for joint_name, value in targets.items()}


def _target_fraction(config: SchunkRetargetingConfig, *, label: PoseFamilyLabel, finger: FingerName, curl: float) -> float:
    if finger == "thumb":
        if label == "paper":
            return config.thumb_relaxed_fraction
        if label == "rock":
            return config.thumb_wrapped_fraction
        return min(config.thumb_wrapped_fraction, max(config.thumb_relaxed_fraction, curl))
    if label == "paper":
        return config.extended_fraction
    if label == "rock":
        return config.flexed_fraction
    if label == "scissors":
        return config.extended_fraction if finger in ("index", "middle") else config.flexed_fraction
    return min(config.flexed_fraction, max(config.extended_fraction, curl))


def _joint_limits(schema: UrdfJointSchema) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for joint in schema.joints:
        if joint.joint_type != "revolute":
            continue
        if joint.lower is None or joint.upper is None:
            raise ValueError(f"joint {joint.name} is missing URDF limits")
        limits[joint.name] = (float(joint.lower), float(joint.upper))
    return limits


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


def _required_float(mapping: Mapping[str, object], key: str) -> float:
    value = _required(mapping, key)
    return _as_float(value, key)


def _as_float(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _fraction(value: float, label: str) -> float:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of strings")
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "":
            raise ValueError(f"{label} must contain non-empty strings")
        parsed.append(item)
    if len(parsed) == 0:
        raise ValueError(f"{label} must not be empty")
    return tuple(parsed)
