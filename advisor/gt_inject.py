"""
Inject every ground-truth title into the recsys pool: for each GT item, pick a
random recsys agent and a random top-k slot (no slot reused when possible).

Generates a plausible, unbiased explanation via the OpenAI ChatGPT API so that
injected items are indistinguishable from LLM-recommended ones.

Reproducible via ``session_rng(seed, user_id)``.
"""

from __future__ import annotations

import copy
import hashlib
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

# Load .env from project root if OPENAI_API_KEY not already set
if not os.getenv("OPENAI_API_KEY"):
    _env_path = Path(__file__).resolve().parents[1] / ".env"
    if _env_path.is_file():
        for _line in _env_path.read_text().splitlines():
            _line = _line.strip()
            if _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            _k = _k.strip()
            _v = _v.strip().strip('"').strip("'")
            if _k and _v:
                os.environ.setdefault(_k, _v)

_EXPLANATION_CACHE: Dict[str, str] = {}


def session_rng(seed: int, user_id: str) -> random.Random:
    """Reproducible RNG per (seed, user_id) across Python runs."""
    digest = hashlib.sha256(f"{seed}:{user_id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def distribute_gt_for_rs(
    gt_titles: List[str],
    rng: random.Random,
    n_rs: int = 2,
) -> List[List[str]]:
    """Distribute GT titles across N recommender systems.

    Returns a list of N lists, each containing the GT titles assigned to that RS.
    Uses the provided RNG for reproducibility.
    """
    shuffled = list(gt_titles)
    rng.shuffle(shuffled)
    buckets: List[List[str]] = [[] for _ in range(n_rs)]
    for i, title in enumerate(shuffled):
        buckets[i % n_rs].append(title)
    return buckets


def _generate_explanation_openai(
    gt_title: str,
    domain: str = "book",
    shared_relationships: Optional[List] = None,
) -> str:
    """Call OpenAI ChatGPT to produce a 1–2 sentence explanation matching the
    recsys output style (short, factual, no spoilers, no special emphasis).

    If *shared_relationships* are provided, the prompt instructs the model to
    naturally mention relevant author/category/topic information so the
    explanation is comparable in keyword density to LLM-generated items.
    """
    cache_key = f"{domain}:{gt_title}"
    if cache_key in _EXPLANATION_CACHE:
        return _EXPLANATION_CACHE[cache_key]

    context_hint = ""
    if shared_relationships:
        parts = []
        for rel in shared_relationships:
            if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                rtype, rval = rel[0], rel[1]
                if rtype == "WRITTEN_BY":
                    parts.append(f"by {rval}")
                elif rtype in ("IN_CATEGORY", "BELONGS_TO"):
                    parts.append(f"in the {rval} genre")
                elif rtype == "HAS_TOPIC":
                    parts.append(f"about {rval}")
        if parts:
            context_hint = (
                f" The {domain} is {', '.join(parts)}. "
                "Naturally mention this context in the description."
            )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        item_word = "book" if domain == "book" else "movie"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            max_tokens=80,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You generate short {item_word} descriptions for a recommendation system. "
                        "Output ONLY 1-2 sentences. No markdown, no quotes, no title repetition. "
                        "Be factual and concise, like: 'A heartwarming tale of friendship and courage "
                        "by [Author], exploring themes of [topic].'"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Write a short recommendation explanation for the {item_word}: "
                        f"{gt_title}.{context_hint}"
                    ),
                },
            ],
        )
        explanation = resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[gt_inject] OpenAI explanation failed for '{gt_title}': {exc}")
        explanation = _generate_explanation_fallback(
            gt_title, shared_relationships=shared_relationships
        )

    _EXPLANATION_CACHE[cache_key] = explanation
    return explanation


def _generate_explanation_fallback(
    gt_title: str,
    query: str = "",
    shared_relationships: Optional[List] = None,
) -> str:
    """Offline fallback when OpenAI is unavailable."""
    author = ""
    categories: List[str] = []
    if shared_relationships:
        for rel in shared_relationships:
            if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                if rel[0] == "WRITTEN_BY":
                    author = rel[1]
                elif rel[0] in ("IN_CATEGORY", "BELONGS_TO"):
                    categories.append(rel[1])

    parts = []
    if author:
        parts.append(f"A notable work by {author}")
    else:
        parts.append("A compelling title that matches your request")
    if categories:
        parts.append(f"in the {' / '.join(categories)} space")
    return " ".join(parts) + "."


def inject_all_gt_into_tri_data(
    tri_data: Dict[str, List[dict]],
    ground_truth: List[str],
    rng: random.Random,
    query: str = "",
    shared_relationships: Optional[List] = None,
    domain: str = "book",
) -> Dict[str, List[dict]]:
    """
    For **each** non-empty string in ``ground_truth``, replace one distinct
    slot in some agent's list with that title (random agent + random position).

    Explanations are generated via OpenAI ChatGPT API so they look identical
    to the LLM-generated ones (no bias / no special treatment).
    """
    if not ground_truth or not tri_data:
        return tri_data

    gt_list = [g.strip() for g in ground_truth if g and str(g).strip()]
    if not gt_list:
        return tri_data

    out = copy.deepcopy(tri_data)
    agents = [a for a, items in out.items() if items]
    if not agents:
        return out

    free_slots: List[tuple] = []
    for a in agents:
        for i in range(len(out[a])):
            free_slots.append((a, i))

    rng.shuffle(free_slots)
    rng.shuffle(gt_list)

    n = min(len(gt_list), len(free_slots))
    for k in range(n):
        gt = gt_list[k]
        agent, slot = free_slots[k]
        items = out[agent]
        prev = items[slot]
        rank = prev.get("rank", slot + 1)
        explanation = _generate_explanation_openai(
            gt, domain=domain, shared_relationships=shared_relationships,
        )
        items[slot] = {
            "title": gt,
            "explanation": explanation,
            "rank": rank,
        }
    return out


# Backward-compatible alias
inject_random_gt_into_tri_data = inject_all_gt_into_tri_data
