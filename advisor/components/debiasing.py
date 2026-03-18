from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional


def reorder_by_consensus(
    items: List[dict],
    consensus_scores: Optional[Dict[str, int]] = None,
) -> List[dict]:
    """
    Re-order items by how many LLMs recommended each one.
    Ties are broken randomly to avoid anchoring on any single LLM's order.
    """
    if consensus_scores is None:
        consensus_scores = {}
        for it in items:
            title = it.get("title", "").strip().lower()
            consensus_scores[title] = len(it.get("sources", []))

    shuffled = list(items)
    random.shuffle(shuffled)
    shuffled.sort(
        key=lambda it: -consensus_scores.get(it.get("title", "").strip().lower(), 0)
    )
    return shuffled


def limit_items(items: List[dict], max_items: int = 3) -> List[dict]:
    """Enforce the choice-overload constraint: at most max_items per turn."""
    return items[:max_items]


def ensure_divergent_item(
    items: List[dict],
    divergent_pool: List[dict],
    user_prefs: Dict[str, str],
) -> List[dict]:
    """
    If all presented items are from consensus, insert one divergent item
    to counter confirmation bias. The divergent item is chosen from the
    pool of items unique to a single LLM.
    """
    has_divergent = any(len(it.get("sources", [])) < 2 for it in items)
    if has_divergent or not divergent_pool:
        return items

    candidate = divergent_pool[0]
    result = list(items)
    if len(result) >= 3:
        result[-1] = candidate
    else:
        result.append(candidate)
    return result


def synthesize_explanation(
    explanations: List[str],
    llm_client: Callable[[str], str],
) -> str:
    """Produce a single concise sentence from multiple LLM explanations."""
    if not explanations:
        return ""
    if len(explanations) == 1:
        return explanations[0]

    prompt = (
        "Synthesize the following recommendation explanations into ONE concise sentence "
        "(max 30 words). Do not add new information.\n\n"
    )
    for i, exp in enumerate(explanations, 1):
        prompt += f"Explanation {i}: {exp}\n"
    prompt += "\nSynthesized explanation:"

    return llm_client(prompt).strip()
