from __future__ import annotations

from .policy import ActionType

DEFAULT_WEIGHTS = {
    "decision": 1.0,
    "elicitation": 0.3,
    "overload": 0.2,
    "diversity": 0.2,
    "pool_expansion": 0.3,
}


def compute_turn_reward(
    user_decided: bool,
    sigma_before: float,
    sigma_after: float,
    omega_before: float,
    omega_after: float,
    items_considered_for_internal_turn_state: list,
    has_divergent_items: bool,
    all_single_source: bool,
    turn: int,
    max_turns: int,
    action: ActionType,
    weights: dict | None = None,
    key_term_coverage_before: float = 0.0,
    key_term_coverage_after: float = 0.0,
) -> float:
    w = weights or DEFAULT_WEIGHTS

    r_decision = 1.0 if user_decided else 0.0

    r_elicitation = sigma_after - sigma_before
    if action == ActionType.CLARIFY_PREFERENCE:
        r_elicitation *= max(0.0, 1.0 - turn / max(max_turns, 1))

    r_overload = (
        -0.1 * max(0, len(items_considered_for_internal_turn_state) - 3)
        - (omega_after - omega_before)
    )

    if has_divergent_items:
        r_diversity = 0.3
    elif all_single_source and items_considered_for_internal_turn_state:
        r_diversity = -0.3
    else:
        r_diversity = 0.0

    r_pool = 0.0
    if action == ActionType.EXPAND_POOL:
        coverage_delta = key_term_coverage_after - key_term_coverage_before
        r_pool = coverage_delta if coverage_delta > 0 else -0.1

    return (
        w.get("decision", 1.0) * r_decision
        + w.get("elicitation", 0.3) * r_elicitation
        + w.get("overload", 0.2) * r_overload
        + w.get("diversity", 0.2) * r_diversity
        + w.get("pool_expansion", 0.3) * r_pool
    )
