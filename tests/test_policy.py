from __future__ import annotations

from embodied_rps.policy import CounterMovePolicy


def test_counter_move_policy_maps_rps_winning_responses() -> None:
    policy = CounterMovePolicy()

    assert policy.counter("rock") == "paper"
    assert policy.counter("paper") == "scissors"
    assert policy.counter("scissors") == "rock"
