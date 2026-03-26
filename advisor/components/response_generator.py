from __future__ import annotations

import json
import random
import re
from typing import Callable, Dict, List

from .policy import ActionType
from .belief_state import UserBeliefState
from .debiasing import reorder_by_consensus, limit_items, ensure_divergent_item
from .triangulation import TriangulationResult


def _items_summary(items: List[dict], max_items: int = 3) -> str:
    """Format a short item list for inclusion in prompts."""
    lines = []
    for it in items[:max_items]:
        sources = it.get("sources", [])
        src_str = f" (recommended by {', '.join(sources)})" if sources else ""
        lines.append(f"- {it['title']}{src_str}: {it.get('explanation', 'No explanation.')}")
    return "\n".join(lines) if lines else "(no items)"


def _extract_query_keywords(query: str) -> set:
    """Extract significant words from the user query for relevance matching."""
    if not query:
        return set()
    stop = {
        "can", "you", "recommend", "some", "books", "book", "movies", "movie",
        "written", "by", "the", "a", "an", "in", "of", "and", "or", "for",
        "me", "suggest", "any", "about", "with", "like", "category", "genre",
    }
    words = set()
    for w in re.sub(r"[^\w\s]", " ", query.lower()).split():
        if len(w) > 2 and w not in stop:
            words.add(w)
    return words


def _extract_key_terms(shared_relationships: list | None) -> list[tuple[str, str]]:
    """Extract (rel_type, value) pairs from sharedRelationships.

    Returns e.g. [("WRITTEN_BY", "Graham Greene"), ("IN_CATEGORY", "Historical Fiction")].
    """
    if not shared_relationships:
        return []
    terms = []
    for rel in shared_relationships:
        if isinstance(rel, (list, tuple)) and len(rel) >= 2:
            terms.append((str(rel[0]), str(rel[1])))
    return terms


def _key_term_label(key_terms: list[tuple[str, str]]) -> str:
    """Human-readable summary of key-terms for prompt injection."""
    if not key_terms:
        return ""
    parts = []
    for rtype, rval in key_terms:
        if rtype == "WRITTEN_BY":
            parts.append(f"author: {rval}")
        elif rtype == "IN_CATEGORY":
            parts.append(f"category: {rval}")
        elif rtype == "HAS_TOPIC":
            parts.append(f"topic: {rval}")
        else:
            parts.append(rval)
    return ", ".join(parts)


def _query_relevance_score(
    item: dict,
    keywords: set,
    key_terms: list[tuple[str, str]] | None = None,
) -> float:
    """Score how relevant an item is to the user's query and shared relationships.

    Keyword matches give +1 each. Key-term matches (author/category/topic from
    sharedRelationships) give a +3 bonus per match — these are the strongest
    signal of alignment with what the user actually wants.
    """
    if not keywords and not key_terms:
        return 0
    text = (item.get("title", "") + " " + item.get("explanation", "")).lower()
    score = sum(1 for kw in keywords if kw in text) if keywords else 0

    for rtype, rval in (key_terms or []):
        rval_lower = rval.lower()
        if rval_lower in text:
            score += 3
        else:
            rval_words = rval_lower.split()
            if len(rval_words) > 1 and all(w in text for w in rval_words):
                score += 2
    return score


def _prepare_items(
    tri: TriangulationResult,
    state: UserBeliefState,
    max_per_turn: int = 3,
    user_query: str = "",
    shared_relationships: list | None = None,
) -> List[dict]:
    """Apply debiasing constraints and return items ready for presentation.

    Items are ranked by (consensus sources + query/key-term relevance),
    breaking ties randomly.  Key-term matches (author, category from
    sharedRelationships) receive a strong boost so GT-aligned items surface
    higher.  Source diversity is enforced: at least one item from each
    recsys agent is included before filling the remaining slots by rank.
    """
    keywords = _extract_query_keywords(user_query)
    key_terms = _extract_key_terms(shared_relationships)

    items = list(tri.all_items)
    random.shuffle(items)

    def _rank_score(it):
        sources = it.get("sources", [])
        rel = _query_relevance_score(it, keywords, key_terms)
        base = len(sources) + rel
        # Niche single-source items with strong key-term alignment get a boost
        # so they aren't buried behind popular consensus items
        if len(sources) == 1 and rel >= 3:
            base += 2
        return -base

    items.sort(key=_rank_score)
    items = [
        it
        for it in items
        if it["title"].strip().lower() not in state.items_in_session_internal_pool
    ]

    all_divergent = []
    for lst in tri.divergent_items.values():
        all_divergent.extend(lst)
    items = ensure_divergent_item(items, all_divergent, state.preference_dimensions)

    # Source diversity: guarantee at least 1 item from each recsys agent
    selected: List[dict] = []
    selected_titles: set = set()
    agents_covered: set = set()
    all_agents = set()
    for it in items:
        for src in it.get("sources", []):
            all_agents.add(src)

    for it in items:
        sources = it.get("sources", [])
        uncovered = [s for s in sources if s not in agents_covered]
        if uncovered:
            selected.append(it)
            selected_titles.add(it["title"].strip().lower())
            for s in sources:
                agents_covered.add(s)
        if agents_covered >= all_agents:
            break

    for it in items:
        if len(selected) >= max_per_turn:
            break
        if it["title"].strip().lower() not in selected_titles:
            selected.append(it)
            selected_titles.add(it["title"].strip().lower())

    return selected


_TEMPLATES: Dict[ActionType, str] = {
    ActionType.SUGGEST_PAIR: """You are a friendly advisor exploring options with a user.

Pick 2-3 items from the pool below and suggest them for the user to consider.
Give a brief, natural description of each and ask what the user thinks.

ITEMS AVAILABLE (pick 2-3):
{items_to_show}

USER'S PREFERENCES SO FAR:
{state_summary}

ITEMS ALREADY DISCUSSED: {positive_items}, {negative_items}

Rules:
- Suggest 2-3 items. Use their EXACT titles.
- Keep it conversational: "How about these two…", "What about…"
- After describing them, ask: "What do you think? Does either sound interesting, \
or should we look at others?"
- Do NOT list all options.""",

    ActionType.CLARIFY_PREFERENCE: """You are a friendly advisor helping a user figure out what they want.

Multiple systems suggested options. Before showing them, you want to understand the user better.

CROSS-SYSTEM ANALYSIS:
- Item convergence (Jaccard): {gamma:.2f}
- Framing divergence detected: {divergence}
- Consensus items: {consensus_summary}
- Total unique items: {total_items}

USER'S CURRENT PREFERENCE STATE:
{state_summary}

Ask ONE focused question to learn more about their taste.
Focus on concrete things: a specific author, sub-genre, time period, or topic.
For example: "Do you lean more toward [X] or [Y]?" or "Are you in the mood for \
something like [A] or more like [B]?"
Be brief, conversational, and natural. Do NOT list all items yet.
Present at most 2 directions the user could choose from.""",

    ActionType.SHOW_COMPARISON: """You are a friendly advisor exploring options with a user.

CROSS-SYSTEM ANALYSIS:
- {n_llms} systems were consulted
- Items where systems agree: {consensus_summary}
- Items where systems diverge: {divergent_summary}

USER'S PREFERENCES SO FAR:
{state_summary}

ITEMS TO PRESENT:
{items_to_show}

Show the user a brief comparison of 2-3 items that represent different directions.
Explain how they differ and ask which direction sounds more appealing.
Be concise. Max 3 items.""",

    ActionType.PRESENT_CONSENSUS: """You are a friendly advisor exploring options with a user.

The systems show strong agreement on these items:

CONSENSUS ITEMS:
{items_to_show}

USER'S PREFERENCES:
{state_summary}

Suggest the top 2-3 consensus items with brief explanations.
Frame them by how they relate to what the user has expressed so far.
Ask: "Do any of these sound interesting? Or shall we keep exploring?"
Do NOT present more than 3 items.""",

    ActionType.HIGHLIGHT_DIVERGENCE: """You are a friendly advisor exploring options with a user.

The systems interpret the user's request through different lenses.

SYSTEM FRAMINGS:
{divergent_summary}

USER'S PREFERENCES:
{state_summary}

ITEMS TO PRESENT:
{items_to_show}

Explain the divergence: one perspective focused on X, another on Y.
Suggest 2-3 items (one per perspective) and ask which direction sounds better.""",

    ActionType.SYNTHESIZE_DECISION: """You are a friendly advisor wrapping up an exploration session.

ITEMS THE USER REACTED POSITIVELY TO: {positive_items}
ITEMS THE USER REACTED NEGATIVELY TO: {negative_items}
USER'S PREFERENCE DIMENSIONS: {state_summary}

ALL ITEMS DISCUSSED IN THIS SESSION (these are the ONLY items that exist):
{all_session_items}

Briefly recap what you explored together and what caught the user's attention.
Then present a shortlist and ask the user to pick {n_choices} title(s).

CRITICAL RULES:
- You may ONLY mention titles that appear VERBATIM in the list above.
- Do NOT invent, add, or suggest ANY title not listed above.
- Do NOT paraphrase or abbreviate titles. Copy them exactly.
- Ask: "Which {n_choices} would you like to go with?" """,

    ActionType.END_SESSION: """You are a friendly advisor closing a session.

USER'S CHOSEN ITEM(S): {chosen_item}
ALL ITEMS DISCUSSED IN THIS SESSION:
{all_session_items}
SESSION SUMMARY:
- Turns: {turn}
- Preferences discovered: {state_summary}
- Systems consulted: {n_llms}

Confirm the user's choice(s) using their EXACT titles as listed above.
Briefly explain why they match what the user was looking for.
Keep it short and warm.

At the very end of your message, append the token EXACTLY as:
TERMINATE""",

    ActionType.INJECT_GT_PROBE: """You are a friendly advisor exploring options with a user.

Based on your analysis, you found a title that aligns well with what the user has been describing.

ITEM TO SUGGEST:
{gt_item_to_inject}

USER'S PREFERENCES SO FAR:
{state_summary}

OTHER ITEMS DISCUSSED:
{items_to_show}

Suggest this item naturally as something that might fit what the user is looking for.
Briefly explain why it relates to their interests. Ask if they'd like to know more
or continue exploring other options.
Be conversational and not pushy. Say "What about this one?" not "I recommend".
Do NOT reveal that this is a "special" suggestion.""",

    ActionType.RERANK_POOL: """You are a friendly advisor re-evaluating options with the user.

USER'S ARTICULATED PREFERENCES:
{state_summary}

ITEMS IN THE CURRENT POOL:
{pool_items}

ITEMS USER REACTED POSITIVELY TO: {positive_items}
ITEMS USER REACTED NEGATIVELY TO: {negative_items}

Based on what the user has told you, suggest the 2-3 items that seem like the
best fit. Explain why briefly. Set aside items that don't match.
Ask: "How about these? Do they sound closer to what you're looking for?" """,

    ActionType.EXPAND_POOL: """You are a friendly advisor who just found additional options.

Based on your conversation, you searched for more options that match the user's interests.

USER'S PREFERENCES:
{state_summary}

NEWLY DISCOVERED ITEMS (added to the pool):
{items_to_show}

ITEMS USER PREVIOUSLY LIKED: {positive_items}
ITEMS USER PREVIOUSLY DISLIKED: {negative_items}

Suggest 2-3 of these newly found items that best fit what the user described.
Explain why each one might appeal to them.
Ask: "What do you think of these? Should we keep exploring?" """,
}


_JUNK_BOLD_PATTERNS = re.compile(
    r"^("
    r"cross-system|analysis|summary|agreements?|divergences?|"
    r"comparison|overview|recommendation|suggested|next steps?|"
    r"system framings?|how it matches|preferences?|items?|"
    r"relevant alternatives?|top recommendation|"
    r"emotional register|audience|purpose|formality|"
    r"mistral|deepseek|llama|gpt|gemini|qwen|"
    r"final options?|highlighted item|systems? agreement|systems? divergence|"
    r"interpretation highlights?|interpretation differences?|"
    r"direction of interest|theme\b|tone\b|pace\b|setting\b|"
    r"protagonist type|complexity|scope|mood|genre|length|"
    r"which interpretation|which direction|question|"
    r"final recommendations?|purpose\s*&|theme\s*&|tone\s*&|"
    r"protagonist\s*type\s*&|why it matches"
    r")",
    re.IGNORECASE,
)


def _is_valid_bold_title(text: str, known_titles_lower: set, real_titles_lower: set | None = None) -> bool:
    """Return True only if the bold text looks like a real item title."""
    t = text.strip().strip(":").strip('"').strip("'")
    if not t or len(t) < 6:
        return False
    if t.lower() in known_titles_lower:
        return False
    if _JUNK_BOLD_PATTERNS.match(t):
        return False
    if len(t.split()) < 2:
        return False
    # Reject if the bold text contains '&' (markdown sub-header pattern like "Theme & Setting")
    if "&" in t:
        return False
    # If we know the real catalogue, only accept titles that partially match a real one
    if real_titles_lower:
        tl = t.lower()
        if not any(tl in rt or rt in tl for rt in real_titles_lower):
            return False
    return True


def generate_response(
    action: ActionType,
    belief_state: UserBeliefState,
    triangulation: TriangulationResult,
    llm_client: Callable[[str], str],
    chosen_item: str = "",
    gt_item_to_inject: str = "",
    user_query: str = "",
    shared_relationships: list | None = None,
    n_choices: int = 1,
) -> tuple[str, List[str]]:
    """
    Generate the Advisor's natural-language response for the given action.

    Returns
    -------
    (response_text, items_considered_for_internal_turn_state)
    """
    items = _prepare_items(
        triangulation, belief_state,
        user_query=user_query,
        shared_relationships=shared_relationships,
    )
    items_considered_for_internal_turn_state = [it["title"] for it in items]

    # Build a set of all known real titles from the triangulation for validation
    all_known_titles_lower = {
        it.get("title", "").strip().lower()
        for it in triangulation.all_items
        if it.get("title")
    }

    state_summary = json.dumps(belief_state.preference_dimensions) if belief_state.preference_dimensions else "(no preferences articulated yet)"

    consensus_summary = ", ".join(it["title"] for it in triangulation.consensus_items[:3]) or "(none)"
    divergent_lines = []
    for llm_name, ditems in triangulation.divergent_items.items():
        titles = ", ".join(it["title"] for it in ditems[:2])
        divergent_lines.append(f"  {llm_name}: {titles}")
    divergent_summary = "\n".join(divergent_lines) if divergent_lines else "(none)"

    template = _TEMPLATES.get(action, _TEMPLATES[ActionType.CLARIFY_PREFERENCE])

    pool_items_summary = ", ".join(list(belief_state.items_in_session_internal_pool)[:10]) or "(none)"

    all_session_titles = list(belief_state.items_in_session_internal_pool)
    all_session_items_block = "\n".join(
        f"  {i+1}. {t}" for i, t in enumerate(all_session_titles)
    ) if all_session_titles else "(none)"

    key_terms = _extract_key_terms(shared_relationships)
    kt_label = _key_term_label(key_terms)
    key_term_block = ""
    if kt_label:
        key_term_block = (
            f"\n\nKEY-TERM SIGNAL (from user's query context):\n"
            f"The user's request is strongly tied to: {kt_label}.\n"
            f"Prioritize items that match these key-terms when presenting options.\n"
            f"If clarifying preferences, ask about these specific dimensions first "
            f"(e.g. \"Are you looking specifically for books by [author]?\")."
        )

    prompt = template.format(
        n_llms=len(triangulation.per_llm_items),
        gamma=triangulation.item_convergence,
        divergence="Yes" if triangulation.framing_divergence else "No",
        consensus_summary=consensus_summary,
        divergent_summary=divergent_summary,
        total_items=len(triangulation.all_items),
        state_summary=state_summary,
        items_to_show=_items_summary(items),
        positive_items=", ".join(belief_state.items_positive) or "(none)",
        negative_items=", ".join(belief_state.items_negative) or "(none)",
        turn=belief_state.turn,
        chosen_item=chosen_item or "(none)",
        gt_item_to_inject=gt_item_to_inject or "(none)",
        pool_items=pool_items_summary,
        n_choices=n_choices,
        all_session_items=all_session_items_block,
    )
    prompt += key_term_block

    response = llm_client(prompt)

    # For INJECT_GT_PROBE, ensure the injected GT item is in the pool
    if action == ActionType.INJECT_GT_PROBE and gt_item_to_inject:
        gt_lower = gt_item_to_inject.strip().lower()
        if gt_lower not in {t.strip().lower() for t in items_considered_for_internal_turn_state}:
            items_considered_for_internal_turn_state.append(gt_item_to_inject)

    # Extract titles mentioned in bold (**Title**) in the LLM response.
    # Only add if it passes validation (real item title, not a markdown header).
    bold_titles = re.findall(r'\*\*(.+?)\*\*', response)
    known_lower = {t.strip().lower() for t in items_considered_for_internal_turn_state}
    for bt in bold_titles:
        bt_clean = bt.strip().strip(":")
        if _is_valid_bold_title(bt_clean, known_lower, all_known_titles_lower):
            items_considered_for_internal_turn_state.append(bt_clean)
            known_lower.add(bt_clean.lower())

    return response.strip(), items_considered_for_internal_turn_state
