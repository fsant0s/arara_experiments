from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set


@dataclass
class UserBeliefState:
    preference_specificity: float = 0.0
    preference_dimensions: Dict[str, str] = field(default_factory=dict)
    items_discussed: Set[str] = field(default_factory=set)
    items_positive: Set[str] = field(default_factory=set)
    items_negative: Set[str] = field(default_factory=set)
    anchoring_risk: float = 0.0
    overload_risk: float = 0.0
    turn: int = 0
    rs_talk_counts: Dict[str, int] = field(default_factory=dict)
    advisor_suggestions_followed: int = 0
    advisor_suggestions_total: int = 0

    def to_dict(self) -> dict:
        return {
            "preference_specificity": self.preference_specificity,
            "preference_dimensions": dict(self.preference_dimensions),
            "items_discussed": sorted(self.items_discussed),
            "items_positive": sorted(self.items_positive),
            "items_negative": sorted(self.items_negative),
            "anchoring_risk": self.anchoring_risk,
            "overload_risk": self.overload_risk,
            "turn": self.turn,
            "rs_talk_counts": dict(self.rs_talk_counts),
            "advisor_suggestions_followed": self.advisor_suggestions_followed,
            "advisor_suggestions_total": self.advisor_suggestions_total,
        }


DIMENSION_KEYS = [
    "tone", "pace", "complexity", "emotional_register", "formality", "scope",
    "theme", "genre", "audience", "purpose",
    "setting", "protagonist_type", "mood", "length",
    "author", "sub_genre", "time_period", "specific_topic",
]


def create_initial_state() -> UserBeliefState:
    return UserBeliefState()


_UPDATE_PROMPT = """You are analyzing a user's response in a multi-party recommendation conversation.
The user is talking to two recommendation systems (RS) and an advisor who mediates.

CURRENT USER STATE (JSON):
{state_json}

LAST ADVISOR SUGGESTION:
{advisor_message}

USER'S RESPONSE (the user chose to talk to {user_target}):
{user_response}

Items mentioned in this turn:
{items_mentioned}

Extract structured updates. Return valid JSON with exactly these fields:
{{
  "new_dimensions": {{"key": "value"}},
  "positive_items": ["title1"],
  "negative_items": ["title2"],
  "anchoring_risk": 0.0,
  "overload_risk": 0.0,
  "user_decided": false,
  "chosen_items": [],
  "followed_advisor": null
}}

Rules:
- new_dimensions: only dimensions the user explicitly expressed. Keys from: {dimension_keys}
- positive_items: titles the user explicitly praised or asked about
- negative_items: ONLY titles the user explicitly rejected BY NAME
- anchoring_risk: 0.0-1.0, high if user keeps referencing only the first items shown
- overload_risk: 0.0-1.0, high if user seems confused or overwhelmed
- user_decided: true only if user explicitly chose final items
- chosen_items: list of chosen titles if user_decided is true, else []
- followed_advisor: true if the user followed the advisor's suggestion, false if they \
ignored it, null if unclear
- Return ONLY valid JSON, no extra text"""


def update_belief_state(
    state: UserBeliefState,
    user_response: str,
    user_target: str,
    advisor_message: str,
    items_mentioned: List[str],
    llm_client: Callable[[str], str],
) -> tuple[UserBeliefState, bool, List[str]]:
    """Update belief state and return (state, user_decided, chosen_items)."""
    prompt = _UPDATE_PROMPT.format(
        state_json=json.dumps(state.to_dict(), indent=2),
        advisor_message=advisor_message,
        user_target=user_target,
        user_response=user_response,
        items_mentioned=json.dumps(items_mentioned),
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

    for title in items_mentioned:
        state.items_discussed.add(title.strip().lower())

    state.anchoring_risk = float(update.get("anchoring_risk", state.anchoring_risk))
    state.overload_risk = float(update.get("overload_risk", state.overload_risk))
    state.turn += 1

    if user_target:
        state.rs_talk_counts[user_target] = state.rs_talk_counts.get(user_target, 0) + 1

    followed = update.get("followed_advisor")
    if followed is not None:
        state.advisor_suggestions_total += 1
        if followed:
            state.advisor_suggestions_followed += 1

    filled = sum(1 for k in DIMENSION_KEYS if k in state.preference_dimensions)
    state.preference_specificity = filled / len(DIMENSION_KEYS)

    lower_resp = user_response.lower()
    _DELEGATION_PHRASES = [
        "what do you think", "what would you recommend", "help me decide",
        "just pick one", "i can't decide", "too many", "you choose",
    ]
    if any(phrase in lower_resp for phrase in _DELEGATION_PHRASES):
        state.overload_risk = max(state.overload_risk, 0.6)

    _CONFUSION_PHRASES = [
        "not sure", "confused", "overwhelmed", "hard to choose",
        "hard to decide", "all of them", "they all seem",
    ]
    if any(phrase in lower_resp for phrase in _CONFUSION_PHRASES):
        state.overload_risk = max(state.overload_risk, 0.4)

    user_decided = bool(update.get("user_decided", False))
    chosen_items = update.get("chosen_items", []) or []
    if not isinstance(chosen_items, list):
        chosen_items = [chosen_items] if chosen_items else []

    return state, user_decided, chosen_items


def compute_context_vector(state: UserBeliefState) -> dict:
    """Build the context vector for the bandit (no triangulation needed)."""
    total_talks = sum(state.rs_talk_counts.values()) or 1
    counts = list(state.rs_talk_counts.values()) or [0]
    max_count = max(counts)
    min_count = min(counts) if len(counts) > 1 else 0
    balance = 1.0 - (max_count - min_count) / max(total_talks, 1)

    follow_rate = (
        state.advisor_suggestions_followed / max(state.advisor_suggestions_total, 1)
    )

    return {
        "sigma": state.preference_specificity,
        "turn": state.turn,
        "overload_risk": state.overload_risk,
        "anchoring_risk": state.anchoring_risk,
        "n_items_discussed": float(len(state.items_discussed)),
        "rs_agreement": 0.0,
        "user_follow_rate": follow_rate,
        "exploration_balance": balance,
    }
