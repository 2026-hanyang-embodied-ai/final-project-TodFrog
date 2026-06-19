"""Actuator feasibility checks for the kinematic fallback controller."""

from __future__ import annotations

from embodied_rps.domain import ActuatorLimits, FeasibilityResult, HandPose


def check_actuator_feasibility(
    *,
    current_pose: HandPose,
    target_pose: HandPose,
    limits: ActuatorLimits,
    remaining_time_s: float,
) -> FeasibilityResult:
    """Check whether the hand can reach the target pose within the remaining time."""

    if remaining_time_s < 0.0:
        return FeasibilityResult(
            feasible=False,
            required_time_s=0.0,
            completion_time_s=0.0,
            limiting_joint=None,
            failure_reason="negative_time_budget",
        )

    _validate_pose_joint_sets(current_pose, target_pose)

    required_motion_time_s = 0.0
    limiting_joint: str | None = None
    for joint_name, current_value in current_pose.joints.items():
        if joint_name not in limits.velocity_limits_rad_s:
            raise ValueError(f"Missing velocity limit for joint {joint_name}")
        velocity_limit = limits.velocity_limits_rad_s[joint_name]
        target_value = target_pose.joints[joint_name]
        joint_time_s = abs(target_value - current_value) / velocity_limit
        if joint_time_s > required_motion_time_s or limiting_joint is None:
            required_motion_time_s = joint_time_s
            limiting_joint = joint_name

    required_time_s = required_motion_time_s + limits.response_delay_s
    feasible = required_time_s <= remaining_time_s
    return FeasibilityResult(
        feasible=feasible,
        required_time_s=required_time_s,
        completion_time_s=required_time_s,
        limiting_joint=limiting_joint,
        failure_reason=None if feasible else "deadline_missed",
    )


def _validate_pose_joint_sets(current_pose: HandPose, target_pose: HandPose) -> None:
    current_joints = set(current_pose.joints)
    target_joints = set(target_pose.joints)
    if current_joints != target_joints:
        raise ValueError("current_pose and target_pose must define the same joints")
