from __future__ import annotations

from .policy import ActionType

DEFAULT_WEIGHTS = {"decision": 1.0, "elicitation": 0.3, "overload": 0.2, "diversity": 0.2}


def compute_turn_reward(
    user_decided: bool,
    sigma_before: float,
    sigma_after: float,
    omega_before: float,
    omega_after: float,
    items_shown: list,
    has_divergent_items: bool,
    all_single_source: bool,
    turn: int,
    max_turns: int,
    action: ActionType,
    weights: dict | None = None,
) -> float:
    w = weights or DEFAULT_WEIGHTS

    r_decision = 1.0 if user_decided else 0.0

    r_elicitation = sigma_after - sigma_before
    if action == ActionType.CLARIFY_PREFERENCE:
        r_elicitation *= max(0.0, 1.0 - turn / max(max_turns, 1))

    r_overload = (
        -0.1 * max(0, len(items_shown) - 3)
        - (omega_after - omega_before)
    )

    if has_divergent_items:
        r_diversity = 0.3
    elif all_single_source and items_shown:
        r_diversity = -0.3
    else:
        r_diversity = 0.0

    return (
        w.get("decision", 1.0) * r_decision
        + w.get("elicitation", 0.3) * r_elicitation
        + w.get("overload", 0.2) * r_overload
        + w.get("diversity", 0.2) * r_diversity
    )
