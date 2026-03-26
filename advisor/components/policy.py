from __future__ import annotations

import random as _random
from enum import Enum
from typing import Dict, List

import numpy as np


# ─── Action space ─────────────────────────────────────────────

class ActionType(Enum):
    SUGGEST_PAIR = "a0_suggest_pair"
    CLARIFY_PREFERENCE = "a1_clarify_preference"
    SHOW_COMPARISON = "a2_show_comparison"
    PRESENT_CONSENSUS = "a3_present_consensus"
    HIGHLIGHT_DIVERGENCE = "a4_highlight_divergence"
    SYNTHESIZE_DECISION = "a5_synthesize_decision"
    END_SESSION = "a6_end_session"
    INJECT_GT_PROBE = "a7_inject_gt_probe"
    RERANK_POOL = "a8_rerank_pool"
    EXPAND_POOL = "a9_expand_pool"


ACTIONS = list(ActionType)
_EXPLORABLE_ACTIONS = [
    a for a in ACTIONS
    if a not in (ActionType.END_SESSION, ActionType.INJECT_GT_PROBE)
]
_TRAIN_ONLY_ACTIONS = {ActionType.INJECT_GT_PROBE}

CONTEXT_KEYS = [
    "sigma",
    "gamma",
    "delta",
    "alpha",
    "omega",
    "turn",
    "internal_pool_items_count",
    "key_term_pool_coverage",
]


def _context_to_vec(context: Dict[str, float]) -> np.ndarray:
    return np.array([context.get(k, 0.0) for k in CONTEXT_KEYS], dtype=np.float64)


# ─── LinUCB policy (contextual bandit) ───────────────────────

class LinUCBPolicy:
    """
    LinUCB contextual bandit for action selection with ε-greedy warm-up.

    During the first ``warmup_steps`` global updates the policy picks a
    random explorable action (a1–a5) with probability ``epsilon``; after
    warm-up it falls back to pure UCB with the configured ``alpha``.
    """

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

    # Actions that require a minimum number of prior turns before they become eligible.
    _MIN_TURN_FOR_ACTION: Dict[ActionType, int] = {
        ActionType.SYNTHESIZE_DECISION: 2,
        ActionType.RERANK_POOL: 1,
        ActionType.EXPAND_POOL: 2,
    }

    # Penalty applied to UCB score per consecutive repeat of the same action.
    _REPEAT_PENALTY = 0.15

    def select(
        self,
        context: Dict[str, float],
        user_decided: bool = False,
        allow_gt_actions: bool = False,
        gt_available: bool = False,
    ) -> ActionType:
        """
        Select an action given the current context.

        Parameters
        ----------
        context : dict
            Feature vector with keys from CONTEXT_KEYS.
        user_decided : bool
            If True, forces END_SESSION.
        allow_gt_actions : bool
            If True, INJECT_GT_PROBE is eligible (training phase).
        gt_available : bool
            If True, there is at least one un-shown GT title available.
        """
        if user_decided:
            action = ActionType.END_SESSION
            self._history.append(action)
            return action

        turn = int(context.get("turn", 0))

        # Build the set of eligible actions for this call
        explorable = list(_EXPLORABLE_ACTIONS)
        if allow_gt_actions and gt_available:
            explorable.append(ActionType.INJECT_GT_PROBE)

        # ε-greedy exploration during warm-up
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

        # END_SESSION must only fire via user_decided (handled above).
        ucbs[ACTIONS.index(ActionType.END_SESSION)] = -np.inf

        # INJECT_GT_PROBE only eligible if explicitly allowed AND GT is available
        if not (allow_gt_actions and gt_available):
            ucbs[ACTIONS.index(ActionType.INJECT_GT_PROBE)] = -np.inf

        # Mask actions that are not yet eligible based on turn number.
        for act, min_turn in self._MIN_TURN_FOR_ACTION.items():
            if turn < min_turn:
                ucbs[ACTIONS.index(act)] = -np.inf

        # Penalise consecutive repetitions of the same action.
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
        saved_n_actions, d, alpha = int(meta[0]), int(meta[1]), float(meta[2])
        epsilon = float(meta[3]) if len(meta) > 3 else 0.4
        warmup_steps = int(meta[4]) if len(meta) > 4 else 100
        global_count = int(meta[5]) if len(meta) > 5 else 0
        current_n_actions = len(ACTIONS)
        policy = cls(
            n_actions=current_n_actions, d=d, alpha=alpha,
            epsilon=epsilon, warmup_steps=warmup_steps,
        )
        policy._global_update_count = global_count
        for i in range(min(saved_n_actions, current_n_actions)):
            policy.M[i] = data[f"M_{i}"]
            policy.b[i] = data[f"b_{i}"]
        return policy
