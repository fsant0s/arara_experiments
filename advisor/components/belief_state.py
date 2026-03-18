from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Set

from .triangulation import TriangulationResult


@dataclass
class UserBeliefState:
    preference_specificity: float = 0.0
    preference_dimensions: Dict[str, str] = field(default_factory=dict)
    items_seen: Set[str] = field(default_factory=set)
    items_positive: Set[str] = field(default_factory=set)
    items_negative: Set[str] = field(default_factory=set)
    anchoring_risk: float = 0.0
    overload_risk: float = 0.0
    turn: int = 0

    def to_dict(self) -> dict:
        return {
            "preference_specificity": self.preference_specificity,
            "preference_dimensions": dict(self.preference_dimensions),
            "items_seen": sorted(self.items_seen),
            "items_positive": sorted(self.items_positive),
            "items_negative": sorted(self.items_negative),
            "anchoring_risk": self.anchoring_risk,
            "overload_risk": self.overload_risk,
            "turn": self.turn,
        }


DIMENSION_KEYS = [
    # Stylistic
    "tone", "pace", "complexity", "emotional_register", "formality", "scope",
    # Thematic / content
    "theme", "genre", "audience", "purpose",
    # Domain-specific (books, movies, music)
    "setting", "protagonist_type", "mood", "length",
]


def create_initial_state() -> UserBeliefState:
    return UserBeliefState()


_UPDATE_PROMPT = """You are analyzing a user's response in a recommendation conversation.

CURRENT USER STATE (JSON):
{state_json}

ADVISOR'S LAST MESSAGE:
{advisor_message}

USER'S RESPONSE:
{user_response}

ITEMS SHOWN IN THIS TURN:
{items_shown}

Extract structured updates from the user's response. Return valid JSON with exactly these fields:
{{
  "new_dimensions": {{"key": "value"}},
  "positive_items": ["title1"],
  "negative_items": ["title2"],
  "anchoring_risk": 0.0,
  "overload_risk": 0.0,
  "user_decided": false,
  "chosen_item": null
}}

Rules:
- new_dimensions: only dimensions the user explicitly expressed. Keys from: {dimension_keys}
- positive_items / negative_items: item titles the user expressed interest in or rejected
- anchoring_risk: 0.0-1.0, high if user keeps referencing only the first item shown
- overload_risk: 0.0-1.0, high if user seems confused or asks to simplify
- user_decided: true only if user explicitly chose an item
- chosen_item: the title if user_decided is true, else null
- Do NOT invent dimensions the user did not mention
- Return ONLY valid JSON, no extra text"""


def update_belief_state(
    state: UserBeliefState,
    user_response: str,
    advisor_message: str,
    items_shown: List[str],
    llm_client: Callable[[str], str],
) -> UserBeliefState:
    """
    Update belief state by parsing the user's response with a deterministic LLM.

    Parameters
    ----------
    state : UserBeliefState
    user_response : str
    advisor_message : str
    items_shown : list of item titles shown in this turn
    llm_client : callable that takes a prompt string and returns a string
    """
    prompt = _UPDATE_PROMPT.format(
        state_json=json.dumps(state.to_dict(), indent=2),
        advisor_message=advisor_message,
        user_response=user_response,
        items_shown=json.dumps(items_shown),
        dimension_keys=", ".join(DIMENSION_KEYS),
    )

    raw = llm_client(prompt)

    try:
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            lines = raw_clean.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw_clean = "\n".join(lines)
        update = json.loads(raw_clean)
    except (json.JSONDecodeError, ValueError):
        update = {}

    new_dims = update.get("new_dimensions", {})
    if isinstance(new_dims, dict):
        for k, v in new_dims.items():
            if k in DIMENSION_KEYS and v:
                state.preference_dimensions[k] = str(v)

    for title in update.get("positive_items", []):
        if isinstance(title, str) and title:
            state.items_positive.add(title.strip().lower())

    for title in update.get("negative_items", []):
        if isinstance(title, str) and title:
            state.items_negative.add(title.strip().lower())

    for title in items_shown:
        state.items_seen.add(title.strip().lower())

    state.anchoring_risk = float(update.get("anchoring_risk", state.anchoring_risk))
    state.overload_risk = float(update.get("overload_risk", state.overload_risk))
    state.turn += 1

    filled = sum(1 for k in DIMENSION_KEYS if k in state.preference_dimensions)
    state.preference_specificity = filled / len(DIMENSION_KEYS)

    # If the user explicitly asks the advisor to pick/recommend something,
    # treat it as a strong signal: boost specificity and overload_risk so the
    # policy stops asking clarifying questions and synthesizes a recommendation.
    lower_resp = user_response.lower()
    if any(phrase in lower_resp for phrase in [
        "what do you think",
        "which would be a good start",
        "what would you recommend",
        "which one should i",
        "which do you think",
        "you think is best",
        "would you suggest",
        "help me decide",
        "what should i",
    ]):
        state.preference_specificity = max(state.preference_specificity, 0.5)
        state.overload_risk = max(state.overload_risk, 0.5)

    return state


def compute_context_vector(state: UserBeliefState, tri: TriangulationResult) -> dict:
    """Build the 7-dimensional context vector for the bandit."""
    return {
        "sigma": state.preference_specificity,
        "gamma": tri.item_convergence,
        "delta": 1.0 if tri.framing_divergence else 0.0,
        "alpha": state.anchoring_risk,
        "omega": state.overload_risk,
        "turn": state.turn,
        "items_seen_count": len(state.items_seen),
    }
