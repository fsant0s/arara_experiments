from __future__ import annotations

import random as _random
from enum import Enum
from typing import Dict, List

import numpy as np


# ─── Action space ─────────────────────────────────────────────

class ActionType(Enum):
    SUGGEST_CROSS_RS_QUERY = "a0_suggest_cross_rs_query"
    SUMMARIZE_RESPONSES = "a1_summarize_responses"
    IDENTIFY_DIVERGENCES = "a2_identify_divergences"
    SUGGEST_COMPARISON = "a3_suggest_comparison"
    SIMPLIFY_AND_FOCUS = "a4_simplify_and_focus"
    SUGGEST_CRITERION = "a5_suggest_criterion"
    SYNTHESIZE_DECISION = "a6_synthesize_decision"
    END_SESSION = "a7_end_session"


ACTIONS = list(ActionType)
_EXPLORABLE_ACTIONS = [a for a in ACTIONS if a != ActionType.END_SESSION]

CONTEXT_KEYS = [
    "sigma",
    "turn",
    "overload_risk",
    "anchoring_risk",
    "n_items_discussed",
    "rs_agreement",
    "user_follow_rate",
    "exploration_balance",
]


def _context_to_vec(context: Dict[str, float]) -> np.ndarray:
    return np.array([context.get(k, 0.0) for k in CONTEXT_KEYS], dtype=np.float64)


# ─── LinUCB policy ────────────────────────────────────────────

class LinUCBPolicy:
    """LinUCB contextual bandit with ε-greedy warm-up."""

    def __init__(
        self,
        n_actions: int = len(ACTIONS),
        d: int = len(CONTEXT_KEYS),
        alpha: float = 3.0,
        epsilon: float = 0.4,
        warmup_steps: int = 100,
    ) -> None:
        self.n_actions = n_actions
        self.d = d
        self.alpha = alpha
        self.epsilon = epsilon
        self.warmup_steps = warmup_steps
        self._global_update_count: int = 0
        self.M = [np.eye(d) for _ in range(n_actions)]
        self.b = [np.zeros(d) for _ in range(n_actions)]
        self._history: List[ActionType] = []

    _MIN_TURN_FOR_ACTION: Dict[ActionType, int] = {
        ActionType.SYNTHESIZE_DECISION: 2,
    }

    _REPEAT_PENALTY = 0.15

    def select(
        self,
        context: Dict[str, float],
        user_decided: bool = False,
    ) -> ActionType:
        if user_decided:
            action = ActionType.END_SESSION
            self._history.append(action)
            return action

        turn = int(context.get("turn", 0))
        explorable = list(_EXPLORABLE_ACTIONS)

        if (
            self._global_update_count < self.warmup_steps
            and _random.random() < self.epsilon
        ):
            eligible = [
                a for a in explorable
                if turn >= self._MIN_TURN_FOR_ACTION.get(a, 0)
            ] or explorable
            action = _random.choice(eligible)
            self._history.append(action)
            return action

        x = _context_to_vec(context)
        ucbs = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            M_inv = np.linalg.inv(self.M[a])
            theta = M_inv @ self.b[a]
            ucbs[a] = x @ theta + self.alpha * np.sqrt(x @ M_inv @ x)

        ucbs[ACTIONS.index(ActionType.END_SESSION)] = -np.inf

        for act, min_turn in self._MIN_TURN_FOR_ACTION.items():
            if turn < min_turn:
                ucbs[ACTIONS.index(act)] = -np.inf

        if self._history:
            last = self._history[-1]
            consec = 0
            for past in reversed(self._history):
                if past == last:
                    consec += 1
                else:
                    break
            if consec >= 1:
                idx = ACTIONS.index(last)
                ucbs[idx] -= self._REPEAT_PENALTY * consec

        best = int(np.argmax(ucbs))
        action = ACTIONS[best]
        self._history.append(action)
        return action

    def update(self, action: ActionType, context: Dict[str, float], reward: float) -> None:
        idx = ACTIONS.index(action)
        x = _context_to_vec(context)
        self.M[idx] += np.outer(x, x)
        self.b[idx] += reward * x
        self._global_update_count += 1

    def reset(self) -> None:
        self._history.clear()

    def save(self, path: str) -> None:
        data = {}
        for i in range(self.n_actions):
            data[f"M_{i}"] = self.M[i]
            data[f"b_{i}"] = self.b[i]
        data["meta"] = np.array([
            self.n_actions, self.d, self.alpha,
            self.epsilon, self.warmup_steps, self._global_update_count,
        ])
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> LinUCBPolicy:
        data = np.load(path)
        meta = data["meta"]
        saved_n, d, alpha = int(meta[0]), int(meta[1]), float(meta[2])
        epsilon = float(meta[3]) if len(meta) > 3 else 0.4
        warmup = int(meta[4]) if len(meta) > 4 else 100
        gc = int(meta[5]) if len(meta) > 5 else 0
        n_actions = len(ACTIONS)
        policy = cls(n_actions=n_actions, d=d, alpha=alpha, epsilon=epsilon, warmup_steps=warmup)
        policy._global_update_count = gc
        for i in range(min(saved_n, n_actions)):
            policy.M[i] = data[f"M_{i}"]
            policy.b[i] = data[f"b_{i}"]
        return policy
