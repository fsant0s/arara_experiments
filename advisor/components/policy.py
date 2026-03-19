from __future__ import annotations

from enum import Enum
from typing import Dict, List

import numpy as np


# ─── Action space ─────────────────────────────────────────────

class ActionType(Enum):
    CLARIFY_PREFERENCE = "a1_clarify_preference"
    SHOW_COMPARISON = "a2_show_comparison"
    PRESENT_CONSENSUS = "a3_present_consensus"
    HIGHLIGHT_DIVERGENCE = "a4_highlight_divergence"
    SYNTHESIZE_DECISION = "a5_synthesize_decision"
    END_SESSION = "a6_end_session"


ACTIONS = list(ActionType)
CONTEXT_KEYS = ["sigma", "gamma", "delta", "alpha", "omega", "turn", "items_seen_count"]


def _context_to_vec(context: Dict[str, float]) -> np.ndarray:
    return np.array([context.get(k, 0.0) for k in CONTEXT_KEYS], dtype=np.float64)


# ─── LinUCB policy (contextual bandit) ───────────────────────

class LinUCBPolicy:
    """
    LinUCB contextual bandit for action selection.

    Maintains per-action matrices M_a and vectors b_a.
    Selects actions via Upper Confidence Bound.
    """

    def __init__(
        self,
        n_actions: int = len(ACTIONS),
        d: int = len(CONTEXT_KEYS),
        alpha: float = 1.0,
    ) -> None:
        self.n_actions = n_actions
        self.d = d
        self.alpha = alpha
        self.M = [np.eye(d) for _ in range(n_actions)]
        self.b = [np.zeros(d) for _ in range(n_actions)]
        self._history: List[ActionType] = []

    def select(self, context: Dict[str, float], user_decided: bool = False) -> ActionType:
        if user_decided:
            action = ActionType.END_SESSION
            self._history.append(action)
            return action

        x = _context_to_vec(context)
        ucbs = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            M_inv = np.linalg.inv(self.M[a])
            theta = M_inv @ self.b[a]
            ucbs[a] = x @ theta + self.alpha * np.sqrt(x @ M_inv @ x)

        best = int(np.argmax(ucbs))
        action = ACTIONS[best]
        self._history.append(action)
        return action

    def update(self, action: ActionType, context: Dict[str, float], reward: float) -> None:
        idx = ACTIONS.index(action)
        x = _context_to_vec(context)
        self.M[idx] += np.outer(x, x)
        self.b[idx] += reward * x

    def reset(self) -> None:
        self._history.clear()

    def save(self, path: str) -> None:
        data = {}
        for i in range(self.n_actions):
            data[f"M_{i}"] = self.M[i]
            data[f"b_{i}"] = self.b[i]
        data["meta"] = np.array([self.n_actions, self.d, self.alpha])
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> LinUCBPolicy:
        data = np.load(path)
        meta = data["meta"]
        n_actions, d, alpha = int(meta[0]), int(meta[1]), float(meta[2])
        policy = cls(n_actions=n_actions, d=d, alpha=alpha)
        for i in range(n_actions):
            policy.M[i] = data[f"M_{i}"]
            policy.b[i] = data[f"b_{i}"]
        return policy
