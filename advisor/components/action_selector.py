from __future__ import annotations

from enum import Enum
from typing import Dict, List


class ActionType(Enum):
    CLARIFY_PREFERENCE = "a1_clarify_preference"
    SHOW_COMPARISON = "a2_show_comparison"
    PRESENT_CONSENSUS = "a3_present_consensus"
    HIGHLIGHT_DIVERGENCE = "a4_highlight_divergence"
    SYNTHESIZE_DECISION = "a5_synthesize_decision"
    END_SESSION = "a6_end_session"


class HeuristicPolicy:
    """
    Heuristic policy for action selection with anti-repetition guard.

    The triangulation result is static across the session, which means
    delta/gamma are constant and naive rules would pick the same action
    every single turn. The anti-repetition guard forces a progression:
      - same action repeated MAX_SAME_ACTION times → escalate to SYNTHESIZE
      - still repeating after that → force END_SESSION
    """

    MAX_SAME_ACTION = 2  # max consecutive times the same action may repeat

    def __init__(self) -> None:
        self._history: List[ActionType] = []

    def select(self, context: Dict[str, float], user_decided: bool = False) -> ActionType:
        if user_decided:
            action = ActionType.END_SESSION
            self._history.append(action)
            return action

        sigma = context.get("sigma", 0.0)
        gamma = context.get("gamma", 0.0)
        delta = context.get("delta", 0.0)
        omega = context.get("omega", 0.0)
        turn = context.get("turn", 0)

        # --- base rule ---
        # After enough turns, stop clarifying and push toward a decision.
        # Also force synthesis when user explicitly asks for a recommendation
        # (omega elevated by update_belief_state in that case).
        if turn >= 3 and (gamma > 0.3 or sigma >= 0.3 or omega >= 0.5):
            raw = ActionType.SYNTHESIZE_DECISION
        elif turn >= 3:
            raw = ActionType.SHOW_COMPARISON
        elif sigma < 0.3:
            raw = ActionType.CLARIFY_PREFERENCE
        elif delta >= 1.0:
            raw = ActionType.HIGHLIGHT_DIVERGENCE
        elif gamma > 0.5:
            raw = ActionType.PRESENT_CONSENSUS
        elif sigma >= 0.5 and turn >= 2:
            raw = ActionType.SYNTHESIZE_DECISION
        elif turn >= 2:
            raw = ActionType.SHOW_COMPARISON
        else:
            raw = ActionType.CLARIFY_PREFERENCE

        # --- anti-repetition: escalate if we've been stuck on the same action ---
        recent = self._history[-self.MAX_SAME_ACTION:]
        if len(recent) >= self.MAX_SAME_ACTION and all(a == raw for a in recent):
            if raw == ActionType.SYNTHESIZE_DECISION:
                raw = ActionType.END_SESSION
            else:
                raw = ActionType.SYNTHESIZE_DECISION

        self._history.append(raw)
        return raw
