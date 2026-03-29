"""Advisor Mediator — helps the user navigate responses from two RSs.

During training the advisor knows the ground-truth items and can guide
the user strategically toward them. During test it operates without GT.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from components.policy import ActionType, LinUCBPolicy
from components.belief_state import (
    UserBeliefState,
    create_initial_state,
    update_belief_state,
    compute_context_vector,
)
from components.reward import compute_turn_reward
from rs_agent import ConversationalRS


# ─── Prompt templates per action ──────────────────────────────

_ACTION_INSTRUCTIONS: Dict[ActionType, str] = {
    ActionType.SUGGEST_CROSS_RS_QUERY: (
        "Suggest the user ask one RS about an item recommended by the other RS. "
        "Explain briefly why comparing across RSs could help their decision."
    ),
    ActionType.SUMMARIZE_RESPONSES: (
        "Provide a concise summary of what both RSs have recommended so far. "
        "Highlight key themes and the main differences."
    ),
    ActionType.IDENTIFY_DIVERGENCES: (
        "Point out where RS1 and RS2 disagree or offer different perspectives. "
        "Explain the divergence clearly so the user can decide which angle matters more."
    ),
    ActionType.SUGGEST_COMPARISON: (
        "Suggest the user ask one RS to compare two specific items "
        "(e.g. one from RS1 and one from RS2, or two from the same RS)."
    ),
    ActionType.SIMPLIFY_AND_FOCUS: (
        "The user may be overwhelmed. Help narrow down to 2-3 key options "
        "or a single criterion. Keep it simple and actionable."
    ),
    ActionType.SUGGEST_CRITERION: (
        "Suggest a concrete criterion the user could ask about "
        "(e.g. reviews, accessibility, length, recency, reading level). "
        "Frame it as a question they can copy-paste to an RS."
    ),
    ActionType.SYNTHESIZE_DECISION: (
        "Summarize the user's exploration journey and list their apparent favorites. "
        "Ask if they are ready to make a final decision."
    ),
    ActionType.END_SESSION: (
        "Confirm the user's final choices. Thank them and close the session."
    ),
}

_ADVISOR_SYSTEM = """\
You are a recommendation advisor mediating between a user and two \
recommendation systems ({rs1_name} and {rs2_name}).

Your role:
- Help the user navigate complex recommendations from both RSs.
- Suggest questions the user can ask the RSs.
- Summarize, compare, and simplify RS responses.
- Never recommend items directly — guide the user to discover them via the RSs.
- Be concise (3-5 sentences). Address the user directly.
- When suggesting a question, make it concrete enough that the user can send it.
- When mentioning item titles use **bold**.
{gt_block}
The user needs to choose {n_choices} {item_word}(s) by the end of the session."""

_GT_TRAINING_BLOCK = """
PRIVILEGED INFORMATION (training only):
The ideal items for this user are: {gt_items}.

Your strategy:
- When summarizing RS responses, EMPHASIZE any of these items that were mentioned.
- When suggesting cross-RS queries, steer toward these specific titles.
- When identifying divergences, highlight these items if one RS mentioned them.
- When simplifying, keep these items in the shortlist and drop others.
- You may reference these titles by name when suggesting comparisons or criteria
  (e.g. "You might want to ask RS2 about **Title X** — it seems to match your taste").
- Make the user curious about these titles so they explore and ultimately choose them.
"""

_ADVISOR_TURN_PROMPT = """\
{rs1_name} conversation so far:
{rs1_history}

---

{rs2_name} conversation so far:
{rs2_history}

---

User's current state:
- Preferences: {preferences}
- Positive items: {positives}
- Negative items: {negatives}
- Overload risk: {overload:.1f}
- Turn: {turn}

YOUR TASK ({action_label}):
{action_instruction}

Generate your advisor message now."""


class AdvisorMediator:
    """Mediator that selects actions via a bandit and generates guidance."""

    def __init__(
        self,
        chat_fn: Callable[[List[dict]], str],
        belief_llm_fn: Callable[[str], str],
        policy: LinUCBPolicy,
        rs1_name: str = "RS1",
        rs2_name: str = "RS2",
        domain: str = "book",
        n_choices: int = 1,
        ground_truth: Optional[List[str]] = None,
        training_phase: bool = False,
        max_turns: int = 6,
        bandit_log_path: Optional[str] = None,
        session_id: str = "",
    ) -> None:
        self.chat_fn = chat_fn
        self.belief_llm_fn = belief_llm_fn
        self.policy = policy
        self.rs1_name = rs1_name
        self.rs2_name = rs2_name
        self.domain = domain
        self.n_choices = n_choices
        self.ground_truth = list(ground_truth or [])
        self.training_phase = training_phase
        self.max_turns = max_turns
        self.session_id = session_id

        self.state: UserBeliefState = create_initial_state()
        self.user_decided: bool = False
        self.chosen_items: List[str] = []
        self.session_log: List[Dict[str, Any]] = []

        self._last_action: Optional[ActionType] = None
        self._last_context: Optional[dict] = None
        self._last_advisor_msg: str = ""
        self._bandit_log_path = bandit_log_path

        item_word = "book" if domain == "book" else "movie"
        gt_block = ""
        if training_phase and self.ground_truth:
            gt_block = _GT_TRAINING_BLOCK.format(
                gt_items=", ".join(f'"{t}"' for t in self.ground_truth)
            )
        self._system_prompt = _ADVISOR_SYSTEM.format(
            rs1_name=rs1_name,
            rs2_name=rs2_name,
            gt_block=gt_block,
            n_choices=n_choices,
            item_word=item_word,
        )

    def initial_intervention(
        self, rs1: ConversationalRS, rs2: ConversationalRS,
    ) -> tuple[ActionType, str]:
        """First advisor message after both RSs have responded to the initial query."""
        self.policy.reset()
        context = compute_context_vector(self.state)
        action = self.policy.select(context)
        self._last_action = action
        self._last_context = context

        msg = self._generate_message(action, rs1, rs2)
        self._last_advisor_msg = msg

        self.session_log.append({
            "turn": 0,
            "action": action.value,
            "advisor_message": msg,
            "user_response": "",
            "user_target": "",
            "belief_state": self.state.to_dict(),
        })
        return action, msg

    def process_turn(
        self,
        user_response: str,
        user_target: str,
        rs1: ConversationalRS,
        rs2: ConversationalRS,
        items_in_rs_response: List[str],
    ) -> tuple[ActionType, str]:
        """Process a user turn and generate the next advisor message."""
        overload_before = self.state.overload_risk

        self.state, decided, chosen = update_belief_state(
            state=self.state,
            user_response=user_response,
            user_target=user_target,
            advisor_message=self._last_advisor_msg,
            items_mentioned=items_in_rs_response,
            llm_client=self.belief_llm_fn,
        )

        if decided and chosen:
            self.user_decided = True
            self.chosen_items = chosen

        # Bandit update for previous action
        if self._last_action is not None and self._last_context is not None:
            followed = self.state.advisor_suggestions_followed > (
                self.state.advisor_suggestions_total - 1
            )
            reward = compute_turn_reward(
                user_decided=self.user_decided,
                user_followed_suggestion=followed,
                overload_before=overload_before,
                overload_after=self.state.overload_risk,
                exploration_balance=compute_context_vector(self.state).get(
                    "exploration_balance", 0.5
                ),
                action=self._last_action,
                turn=self.state.turn,
                max_turns=self.max_turns,
            )
            self.policy.update(self._last_action, self._last_context, reward)
            self._log_bandit_step(self._last_action, self._last_context, reward)

        context = compute_context_vector(self.state)
        context["rs_agreement"] = self._compute_rs_agreement(rs1, rs2)

        action = self.policy.select(context, user_decided=self.user_decided)
        self._last_action = action
        self._last_context = context

        msg = self._generate_message(action, rs1, rs2)
        self._last_advisor_msg = msg

        self.session_log.append({
            "turn": self.state.turn,
            "action": action.value,
            "advisor_message": msg,
            "user_response": user_response,
            "user_target": user_target,
            "belief_state": self.state.to_dict(),
        })

        return action, msg

    def _generate_message(
        self, action: ActionType,
        rs1: ConversationalRS, rs2: ConversationalRS,
    ) -> str:
        instruction = _ACTION_INSTRUCTIONS.get(action, "Help the user.")
        preferences = json.dumps(self.state.preference_dimensions) or "{}"
        positives = ", ".join(sorted(self.state.items_positive)) or "none yet"
        negatives = ", ".join(sorted(self.state.items_negative)) or "none yet"

        user_prompt = _ADVISOR_TURN_PROMPT.format(
            rs1_name=self.rs1_name,
            rs1_history=rs1.get_history_text() or "(no conversation yet)",
            rs2_name=self.rs2_name,
            rs2_history=rs2.get_history_text() or "(no conversation yet)",
            preferences=preferences,
            positives=positives,
            negatives=negatives,
            overload=self.state.overload_risk,
            turn=self.state.turn,
            action_label=action.value,
            action_instruction=instruction,
        )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat_fn(messages)

    def _compute_rs_agreement(
        self, rs1: ConversationalRS, rs2: ConversationalRS,
    ) -> float:
        t1 = {t.lower() for t in rs1.get_mentioned_titles()}
        t2 = {t.lower() for t in rs2.get_mentioned_titles()}
        if not t1 and not t2:
            return 0.0
        union = t1 | t2
        return len(t1 & t2) / len(union) if union else 0.0

    def _log_bandit_step(
        self, action: ActionType, context: dict, reward: float,
    ) -> None:
        if not self._bandit_log_path:
            return
        entry = {
            "session_id": self.session_id,
            "turn": self.state.turn,
            "action": action.value,
            "context": context,
            "reward": reward,
        }
        try:
            import os
            os.makedirs(os.path.dirname(self._bandit_log_path) or ".", exist_ok=True)
            with open(self._bandit_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass
