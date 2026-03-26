from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Set

from .triangulation import TriangulationResult


@dataclass
class UserBeliefState:
    preference_specificity: float = 0.0
    preference_dimensions: Dict[str, str] = field(default_factory=dict)
    # Cumulative titles that entered the triangulation/debiasing pool in any prior turn (lowercase).
    # Not necessarily visible to the user in advisor text.
    items_in_session_internal_pool: Set[str] = field(default_factory=set)
    items_positive: Set[str] = field(default_factory=set)
    items_negative: Set[str] = field(default_factory=set)
    anchoring_risk: float = 0.0
    overload_risk: float = 0.0
    turn: int = 0

    def to_dict(self) -> dict:
        return {
            "preference_specificity": self.preference_specificity,
            "preference_dimensions": dict(self.preference_dimensions),
            "items_in_session_internal_pool": sorted(self.items_in_session_internal_pool),
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
    # Content-level (critical for GT matching: narrows the search space)
    "author", "sub_genre", "time_period", "specific_topic",
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

ITEMS CONSIDERED FOR INTERNAL TURN STATE (triangulation/debiasing pool; may not appear in advisor text):
{items_considered_for_internal_turn_state}

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
- positive_items: titles the user explicitly praised, asked about, or expressed interest in
- negative_items: ONLY titles the user explicitly rejected BY NAME (e.g. "I don't want X",
  "X doesn't appeal to me"). Do NOT put a title in negative_items just because the user
  said something vague like "the others don't interest me" or "not those" — unless the
  user named the specific title. When in doubt, leave it OUT of negative_items.
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
    items_considered_for_internal_turn_state: List[str],
    llm_client: Callable[[str], str],
) -> UserBeliefState:
    """
    Update belief state by parsing the user's response with a deterministic LLM.

    Parameters
    ----------
    state : UserBeliefState
    user_response : str
    advisor_message : str
    items_considered_for_internal_turn_state : list of item titles in the turn's internal pool
    llm_client : callable that takes a prompt string and returns a string
    """
    prompt = _UPDATE_PROMPT.format(
        state_json=json.dumps(state.to_dict(), indent=2),
        advisor_message=advisor_message,
        user_response=user_response,
        items_considered_for_internal_turn_state=json.dumps(items_considered_for_internal_turn_state),
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

    for title in items_considered_for_internal_turn_state:
        state.items_in_session_internal_pool.add(title.strip().lower())

    state.anchoring_risk = float(update.get("anchoring_risk", state.anchoring_risk))
    state.overload_risk = float(update.get("overload_risk", state.overload_risk))
    state.turn += 1

    filled = sum(1 for k in DIMENSION_KEYS if k in state.preference_dimensions)
    state.preference_specificity = filled / len(DIMENSION_KEYS)

    # ── Heuristic overload / delegation signals ──────────────
    lower_resp = user_response.lower()

    # User explicitly asks the advisor to pick/recommend → strong overload signal
    _DELEGATION_PHRASES = [
        "what do you think",
        "which would be a good start",
        "what would you recommend",
        "which one should i",
        "which do you think",
        "you think is best",
        "would you suggest",
        "help me decide",
        "what should i",
        "just pick one",
        "you choose",
        "i can't decide",
        "too many",
    ]
    if any(phrase in lower_resp for phrase in _DELEGATION_PHRASES):
        state.preference_specificity = max(state.preference_specificity, 0.5)
        state.overload_risk = max(state.overload_risk, 0.6)

    # Confusion / indecision signals → mild overload bump
    _CONFUSION_PHRASES = [
        "i'm not sure",
        "not sure",
        "hard to choose",
        "hard to decide",
        "confused",
        "overwhelmed",
        "all of them",
        "they all",
        "any of them",
        "either would",
        "both seem",
        "all seem",
    ]
    if any(phrase in lower_resp for phrase in _CONFUSION_PHRASES):
        state.overload_risk = max(state.overload_risk, 0.4)

    # Pool size heuristic: many items in pool + later turns → rising overload
    pool_size = len(state.items_in_session_internal_pool)
    if pool_size > 12 and state.turn >= 3:
        state.overload_risk = max(state.overload_risk, 0.3)
    if pool_size > 20:
        state.overload_risk = max(state.overload_risk, 0.5)

    return state


def _compute_key_term_pool_coverage(
    state: UserBeliefState,
    tri: TriangulationResult,
    shared_relationships: list | None = None,
) -> float:
    """Fraction of key-terms (author/category) that appear in at least one pool item.

    0.0 → none of the user's key attributes are covered by the current pool
    1.0 → all key-terms have at least one matching item
    """
    if not shared_relationships:
        return 0.0
    n_terms = 0
    n_covered = 0
    pool_text = " ".join(state.items_in_session_internal_pool).lower()
    for item in tri.all_items:
        pool_text += " " + (item.get("title", "") + " " + item.get("explanation", "")).lower()

    for rel in shared_relationships:
        if not isinstance(rel, (list, tuple)) or len(rel) < 2:
            continue
        n_terms += 1
        val_lower = str(rel[1]).lower()
        if val_lower in pool_text:
            n_covered += 1
        else:
            val_words = val_lower.split()
            if len(val_words) > 1 and all(w in pool_text for w in val_words):
                n_covered += 1
    return n_covered / n_terms if n_terms else 0.0


def compute_context_vector(
    state: UserBeliefState,
    tri: TriangulationResult,
    shared_relationships: list | None = None,
) -> dict:
    """Build the 8-dimensional context vector for the bandit."""
    return {
        "sigma": state.preference_specificity,
        "gamma": tri.item_convergence,
        "delta": 1.0 if tri.framing_divergence else 0.0,
        "alpha": state.anchoring_risk,
        "omega": state.overload_risk,
        "turn": state.turn,
        "internal_pool_items_count": len(state.items_in_session_internal_pool),
        "key_term_pool_coverage": _compute_key_term_pool_coverage(
            state, tri, shared_relationships
        ),
    }
