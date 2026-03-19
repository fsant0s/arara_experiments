from __future__ import annotations

import json
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


def _prepare_items(
    tri: TriangulationResult,
    state: UserBeliefState,
    max_per_turn: int = 3,
) -> List[dict]:
    """Apply debiasing constraints and return items ready for presentation."""
    items = reorder_by_consensus(tri.all_items)
    items = [it for it in items if it["title"].strip().lower() not in state.items_seen]

    all_divergent = []
    for lst in tri.divergent_items.values():
        all_divergent.extend(lst)
    items = ensure_divergent_item(items, all_divergent, state.preference_dimensions)
    items = limit_items(items, max_per_turn)
    return items


_TEMPLATES: Dict[ActionType, str] = {
    ActionType.CLARIFY_PREFERENCE: """You are a recommendation advisor helping a user find what they want.
The user has asked for recommendations and received results from {n_llms} independent systems.

CROSS-SYSTEM ANALYSIS:
- Item convergence (Jaccard): {gamma:.2f}
- Framing divergence detected: {divergence}
- Consensus items: {consensus_summary}
- Total unique items: {total_items}

USER'S CURRENT PREFERENCE STATE:
{state_summary}

Your task: Ask ONE focused preference-clarifying question to reduce uncertainty.
Focus on the dimension where LLM outputs diverge most.
Be conversational, brief, and natural. Do not list all items yet.
Present at most 2-3 key directions the user could choose from.""",

    ActionType.SHOW_COMPARISON: """You are a recommendation advisor.

CROSS-SYSTEM ANALYSIS:
- {n_llms} systems were consulted
- Items where systems agree: {consensus_summary}
- Items where systems diverge: {divergent_summary}

USER'S PREFERENCES SO FAR:
{state_summary}

ITEMS TO PRESENT (already debiased):
{items_to_show}

Show the user a brief comparison of what the systems agree on vs. where they differ.
Highlight how different systems interpreted the query differently.
Be concise. Max 2-3 items. Ask which direction interests them.""",

    ActionType.PRESENT_CONSENSUS: """You are a recommendation advisor.

The systems show strong agreement. Present the top recommendations by consensus.

CONSENSUS ITEMS:
{items_to_show}

USER'S PREFERENCES:
{state_summary}

Present the consensus items with brief, synthesized explanations.
Frame them by how they match the user's stated preferences.
Mention any relevant alternatives briefly. Ask if they want to explore further or decide.""",

    ActionType.HIGHLIGHT_DIVERGENCE: """You are a recommendation advisor.
The recommendation systems interpret the user's query through different lenses.

SYSTEM FRAMINGS:
{divergent_summary}

USER'S PREFERENCES:
{state_summary}

ITEMS TO PRESENT:
{items_to_show}

Explain the divergence clearly: one system focused on X, another on Y.
Ask the user which interpretation matches their intent.
Present 2-3 items that represent the different perspectives.""",

    ActionType.SYNTHESIZE_DECISION: """You are a recommendation advisor.
The user has explored enough options. Time to help them decide.

ITEMS THE USER REACTED POSITIVELY TO: {positive_items}
ITEMS THE USER REACTED NEGATIVELY TO: {negative_items}
USER'S PREFERENCE DIMENSIONS: {state_summary}

TOP REMAINING ITEMS:
{items_to_show}

Synthesize the session: summarize what was explored, what the user liked,
and present a clear choice of 2-3 final options with brief rationale.
Ask the user to pick one.""",

    ActionType.END_SESSION: """You are a recommendation advisor closing a session.

USER'S CHOSEN ITEM: {chosen_item}
SESSION SUMMARY:
- Turns: {turn}
- Preferences discovered: {state_summary}
- Systems consulted: {n_llms}

Confirm the user's choice, briefly explain why it matches their preferences,
and suggest one follow-up read-alike if appropriate. Keep it short.

At the very end of your message, append the token EXACTLY as:
TERMINATE""",
}


def generate_response(
    action: ActionType,
    belief_state: UserBeliefState,
    triangulation: TriangulationResult,
    llm_client: Callable[[str], str],
    chosen_item: str = "",
) -> tuple[str, List[str]]:
    """
    Generate the Advisor's natural-language response for the given action.

    Returns
    -------
    (response_text, items_shown_titles)
    """
    items = _prepare_items(triangulation, belief_state)
    items_shown_titles = [it["title"] for it in items]

    state_summary = json.dumps(belief_state.preference_dimensions) if belief_state.preference_dimensions else "(no preferences articulated yet)"

    consensus_summary = ", ".join(it["title"] for it in triangulation.consensus_items[:3]) or "(none)"
    divergent_lines = []
    for llm_name, ditems in triangulation.divergent_items.items():
        titles = ", ".join(it["title"] for it in ditems[:2])
        divergent_lines.append(f"  {llm_name}: {titles}")
    divergent_summary = "\n".join(divergent_lines) if divergent_lines else "(none)"

    template = _TEMPLATES.get(action, _TEMPLATES[ActionType.CLARIFY_PREFERENCE])

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
    )

    response = llm_client(prompt)

    # Extract titles mentioned in bold (**Title**) in the LLM response that
    # aren't already in items_shown_titles — the LLM sometimes suggests items
    # beyond the triangulation set.
    bold_titles = re.findall(r'\*\*(.+?)\*\*', response)
    known_lower = {t.strip().lower() for t in items_shown_titles}
    for bt in bold_titles:
        bt_clean = bt.strip().strip(":")
        if bt_clean and bt_clean.lower() not in known_lower and len(bt_clean) > 5:
            items_shown_titles.append(bt_clean)
            known_lower.add(bt_clean.lower())

    return response.strip(), items_shown_titles
