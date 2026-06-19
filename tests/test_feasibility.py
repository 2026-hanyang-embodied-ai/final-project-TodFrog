from __future__ import annotations

import pytest

from embodied_rps.domain import ActuatorLimits, Gesture, HandPose
from embodied_rps.feasibility import check_actuator_feasibility


def _pose(name: Gesture, value: float) -> HandPose:
    return HandPose(
        gesture=name,
        joints={
            "thumb_curl": value,
            "index_curl": value,
            "middle_curl": value,
        },
    )


def _limits(speed: float, delay: float = 0.0) -> ActuatorLimits:
    return ActuatorLimits(
        velocity_limits_rad_s={
            "thumb_curl": speed,
            "index_curl": speed,
            "middle_curl": speed,
        },
        response_delay_s=delay,
    )


def test_feasible_response_before_deadline() -> None:
    result = check_actuator_feasibility(
        current_pose=_pose("neutral", 0.0),
        target_pose=_pose("rock", 0.6),
        limits=_limits(speed=2.0, delay=0.05),
        remaining_time_s=0.40,
    )

    assert result.feasible is True
    assert result.required_time_s == pytest.approx(0.35)
    assert result.completion_time_s == pytest.approx(0.35)
    assert result.limiting_joint == "thumb_curl"
    assert result.failure_reason is None


def test_deadline_miss_reports_limiting_joint() -> None:
    target = HandPose(
        gesture="paper",
        joints={
            "thumb_curl": 0.2,
            "index_curl": 0.9,
            "middle_curl": 0.3,
        },
    )

    result = check_actuator_feasibility(
        current_pose=_pose("neutral", 0.0),
        target_pose=target,
        limits=_limits(speed=1.0, delay=0.05),
        remaining_time_s=0.50,
    )

    assert result.feasible is False
    assert result.required_time_s == pytest.approx(0.95)
    assert result.limiting_joint == "index_curl"
    assert result.failure_reason == "deadline_missed"


def test_negative_remaining_time_is_infeasible() -> None:
    result = check_actuator_feasibility(
        current_pose=_pose("neutral", 0.0),
        target_pose=_pose("rock", 0.1),
        limits=_limits(speed=1.0),
        remaining_time_s=-0.01,
    )

    assert result.feasible is False
    assert result.failure_reason == "negative_time_budget"


def test_missing_velocity_limit_raises() -> None:
    limits = ActuatorLimits(
        velocity_limits_rad_s={"thumb_curl": 1.0, "index_curl": 1.0},
        response_delay_s=0.0,
    )

    with pytest.raises(ValueError, match="velocity limit"):
        check_actuator_feasibility(
            current_pose=_pose("neutral", 0.0),
            target_pose=_pose("rock", 0.1),
            limits=limits,
            remaining_time_s=1.0,
        )


def test_mismatched_pose_joints_raise() -> None:
    current = HandPose(gesture="neutral", joints={"thumb_curl": 0.0})
    target = HandPose(gesture="rock", joints={"index_curl": 1.0})

    with pytest.raises(ValueError, match="same joints"):
        check_actuator_feasibility(
            current_pose=current,
            target_pose=target,
            limits=ActuatorLimits(velocity_limits_rad_s={"thumb_curl": 1.0}),
            remaining_time_s=1.0,
        )
