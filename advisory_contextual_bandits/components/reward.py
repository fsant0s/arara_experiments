from __future__ import annotations

from .policy import ActionType

DEFAULT_WEIGHTS = {
    "decision": 1.0,
    "follow": 0.4,
    "overload": 0.3,
    "exploration": 0.2,
}


def compute_turn_reward(
    user_decided: bool,
    user_followed_suggestion: bool,
    overload_before: float,
    overload_after: float,
    exploration_balance: float,
    action: ActionType,
    turn: int,
    max_turns: int,
    weights: dict | None = None,
) -> float:
    w = weights or DEFAULT_WEIGHTS

    r_decision = 1.0 if user_decided else 0.0

    r_follow = 1.0 if user_followed_suggestion else -0.1

    r_overload = -(overload_after - overload_before)

    r_exploration = exploration_balance

    return (
        w.get("decision", 1.0) * r_decision
        + w.get("follow", 0.4) * r_follow
        + w.get("overload", 0.3) * r_overload
        + w.get("exploration", 0.2) * r_exploration
    )
