"""Typed domain objects for pre-simulation RPS hand control checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

Gesture: TypeAlias = Literal["rock", "paper", "scissors", "neutral"]
REQUIRED_GESTURES: tuple[Gesture, ...] = ("neutral", "rock", "paper", "scissors")


@dataclass(frozen=True)
class HandPose:
    """A named hand pose represented as a joint-value vector."""

    gesture: Gesture
    joints: dict[str, float]

    def __post_init__(self) -> None:
        if len(self.joints) == 0:
            raise ValueError("HandPose.joints must not be empty")
        for joint_name, joint_value in self.joints.items():
            if joint_name == "":
                raise ValueError("HandPose joint names must not be empty")
            if not isinstance(joint_value, (int, float)) or isinstance(joint_value, bool):
                raise ValueError(f"HandPose joint {joint_name} must be numeric")
        object.__setattr__(
            self,
            "joints",
            {joint_name: float(joint_value) for joint_name, joint_value in self.joints.items()},
        )


@dataclass(frozen=True)
class ActuatorLimits:
    """Velocity and response-delay limits for the simplified hand controller."""

    velocity_limits_rad_s: dict[str, float]
    response_delay_s: float = 0.0
    acceleration_limits_rad_s2: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if len(self.velocity_limits_rad_s) == 0:
            raise ValueError("At least one actuator velocity limit is required")
        if self.response_delay_s < 0.0:
            raise ValueError("response_delay_s must be non-negative")
        for joint_name, velocity_limit in self.velocity_limits_rad_s.items():
            if velocity_limit <= 0.0:
                raise ValueError(f"Velocity limit for {joint_name} must be positive")
        if self.acceleration_limits_rad_s2 is not None:
            for joint_name, acceleration_limit in self.acceleration_limits_rad_s2.items():
                if acceleration_limit <= 0.0:
                    raise ValueError(f"Acceleration limit for {joint_name} must be positive")


@dataclass(frozen=True)
class FeasibilityResult:
    """Result of checking whether a target pose can be reached before a deadline."""

    feasible: bool
    required_time_s: float
    completion_time_s: float
    limiting_joint: str | None
    failure_reason: str | None
