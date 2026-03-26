from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Union

from agents import Agent, Module, Orchestrator

from .aggregator import Aggregator

try:
    from ioflow import IOStream
    from formatting_utils import colored
    iostream = IOStream.get_default()
except ImportError:
    iostream = None

    def colored(text: str, _: str) -> str:
        return text


_SYSTEM_MESSAGE_BOOK = """You are a book recommendation system. Your output MUST be valid JSON only.

CRITICAL OUTPUT RULES (MANDATORY):
- Output ONLY raw JSON. Nothing else.
- NO thinking, NO reasoning, NO chain-of-thought, NO <think> blocks. Skip all internal reasoning.
- NO markdown, NO code blocks (no ```json or ```), NO backticks.
- NO preamble, NO explanation, NO "I'm sorry", NO refusal, NO commentary before or after.
- Your entire response must be a single JSON object that parses with json.loads().
- NEVER output an empty top_k_recommendations list. You MUST always return exactly 3 items (like a production recommender).
- NEVER refuse the task in prose — only valid JSON. If the query is broad, still suggest 3 plausible NEW books that fit it.

INPUT:
- User profile: persona, previously read books (title, description, review).
- User request: natural language.

TASK:
1. Recommend TOP-3 NEW books (best first) that satisfy the request. NEVER recommend books already in the user's history.
2. For each: rank (1-3), book_title (exact published title of the NEW book), explanation (1–2 sentences).

OUTPUT FORMAT (exactly this structure). Example:

{"top_k_recommendations":[{"rank":1,"book_title":"The Rosie Project","explanation":"Light-hearted novel with uplifting tone."},{"rank":2,"book_title":"Eleanor Oliphant Is Completely Fine","explanation":"Charming story with quirky protagonist."},{"rank":3,"book_title":"A Man Called Ove","explanation":"Uplifting tale of connection and community."}]}

- book_title = exact published title of a NEW book (not from user history).
- The array top_k_recommendations must contain exactly 3 objects.

Remember: Your response must be parseable JSON only. No other text.
"""

_SYSTEM_MESSAGE_MOVIE = """You are a movie recommendation system. Your output MUST be valid JSON only.

CRITICAL OUTPUT RULES (MANDATORY):
- Output ONLY raw JSON. Nothing else.
- NO thinking, NO reasoning, NO chain-of-thought, NO <think> blocks. Skip all internal reasoning.
- NO markdown, NO code blocks (no ```json or ```), NO backticks.
- NO preamble, NO explanation, NO "I'm sorry", NO refusal, NO commentary before or after.
- Your entire response must be a single JSON object that parses with json.loads().
- NEVER output an empty top_k_recommendations list. You MUST always return exactly 3 items (like a production recommender).
- NEVER refuse the task in prose — only valid JSON. If the query is broad, still suggest 3 plausible NEW movies that fit it.

INPUT:
- User profile (if available): persona, previously watched movies.
- User request: natural language.

TASK:
1. Recommend TOP-3 NEW movies (best first) that satisfy the request. NEVER recommend movies already in the user's history.
2. For each: rank (1-3), movie_title (exact title of the NEW movie, include release year in parentheses), explanation (1–2 sentences).

OUTPUT FORMAT (exactly this structure). Example:

{"top_k_recommendations":[{"rank":1,"movie_title":"Inception (2010)","explanation":"A mind-bending thriller with complex narrative layers."},{"rank":2,"movie_title":"Interstellar (2014)","explanation":"An epic space adventure with emotional depth."},{"rank":3,"movie_title":"The Prestige (2006)","explanation":"A twisting mystery about obsession and rivalry."}]}

- movie_title = exact title of a NEW movie (not from user history), with year in parentheses.
- The array top_k_recommendations must contain exactly 3 objects.

Remember: Your response must be parseable JSON only. No other text.
"""

DOMAIN_SYSTEM_MESSAGES = {
    "book": _SYSTEM_MESSAGE_BOOK,
    "movie": _SYSTEM_MESSAGE_MOVIE,
}

# Default for backward compatibility
system_message = _SYSTEM_MESSAGE_BOOK


def _is_rec_list(value) -> bool:
    """Return True if value looks like a list of recommendation dicts (has at least one item with a title)."""
    if not isinstance(value, list) or not value:
        return False
    return any(
        isinstance(item, dict) and (
            item.get("book_title") or item.get("movie_title") or item.get("title") or
            item.get("book__title")  # typo variant seen in the wild
        )
        for item in value
    )


def _find_recs_in_dict(parsed: dict):
    """
    Try known key names first, then fall back to scanning every key whose
    value looks like a list of recommendation dicts.  This handles arbitrary
    key names like 'top_3_new_books_to_read', 'top_3_recommendations', etc.
    """
    # 1. Known canonical keys (fast path)
    for key in ("top_k_recommendations", "top_3_books", "recommendations", "top_3_new_books"):
        val = parsed.get(key)
        if _is_rec_list(val):
            return val

    # 2. Scan all keys — pick the first list that contains rec-shaped dicts
    for val in parsed.values():
        if _is_rec_list(val):
            return val

    return []


def _normalize_to_canonical(parsed: dict) -> dict:
    """
    Convert model output to canonical format:
    {"top_k_recommendations": [{"rank":1,"book_title":"...","explanation":"..."}, ...]}

    Handles any key name that wraps a list of recommendation dicts, including
    creative names like 'top_3_new_books_to_read_by_author', etc.
    """
    recs = _find_recs_in_dict(parsed)
    if not isinstance(recs, list):
        return {"top_k_recommendations": []}

    normalized = []
    for i, rec in enumerate(recs):
        if not isinstance(rec, dict):
            continue
        # Filter out echoed history items (some models repeat user's past items)
        if rec.get("user_review") is not None:
            continue
        title = rec.get("book_title") or rec.get("movie_title") or rec.get("title") or rec.get("book__title") or ""
        explanation = rec.get("explanation") or rec.get("description") or ""
        rank = rec.get("rank", i + 1)
        normalized.append({
            "rank": rank,
            "book_title": title,
            "explanation": explanation,
        })
    return {"top_k_recommendations": normalized}


def parse_recommendations(raw_content):
    """Parse raw LLM output into structured recommendations. Output must be JSON."""
    if isinstance(raw_content, (list, dict)):
        return raw_content

    if not isinstance(raw_content, str):
        raise TypeError(f"Unsupported content type: {type(raw_content)}")

    text = raw_content.strip()

    # 0. Strip <think>...</think> blocks (deepseek-r1, qwen3, etc.)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # 1. Try markdown fence (```json ... ```)
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Try direct parse
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # 3. Fallback: extract JSON object (first { to last })
        obj_match = re.search(r"\{.*\}", text, re.DOTALL)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                raise ValueError(f"Output is not valid JSON. Raw: {text[:200]}...")
        else:
            raise ValueError(f"Output is not valid JSON. Raw: {text[:200]}...")

    if isinstance(parsed, dict):
        parsed = [_normalize_to_canonical(parsed)]
    elif isinstance(parsed, list):
        parsed = [_normalize_to_canonical(p) for p in parsed if isinstance(p, dict)]
    else:
        raise ValueError("Parsed content is not a list or dict")

    return parsed


_MAX_RETRIES = 3


def _has_valid_recs(recs) -> bool:
    """Check if parsed recommendations contain at least one item with a title."""
    if not isinstance(recs, list):
        return False
    for entry in recs:
        if not isinstance(entry, dict):
            continue
        items = entry.get("top_k_recommendations", [])
        if isinstance(items, list) and any(
            isinstance(it, dict) and (it.get("book_title") or it.get("movie_title") or it.get("title"))
            for it in items
        ):
            return True
    return False


def parallel(
    last_speaker: Agent, module: Module, selector: Agent = None
) -> Union[Agent, str, None]:
    """Run all recommender agents in parallel and aggregate results.

    Each agent is retried up to ``_MAX_RETRIES`` times if its output is
    empty, unparseable, or contains no actual recommendations.
    """
    if iostream:
        iostream.print(colored("\nStarted executing the recommendation systems...", "yellow"), flush=True)

    aggregator = Aggregator(name="aggregator")

    message = {}
    for agent in module.agents:
        recs = None
        last_error = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                for reply in agent.generate_reply(sender=selector):
                    agent.send(reply, aggregator, silent=False, request_reply=False)
                    raw = reply.chat_message.content
                    parsed = parse_recommendations(raw)
                    if _has_valid_recs(parsed):
                        recs = parsed
                        break
                    else:
                        last_error = f"Empty recommendations (attempt {attempt}/{_MAX_RETRIES})"
                        if iostream:
                            iostream.print(
                                colored(f"  [{agent.name}] {last_error}, retrying...", "yellow"),
                                flush=True,
                            )
            except Exception as e:
                last_error = f"Parse error: {e} (attempt {attempt}/{_MAX_RETRIES})"
                if iostream:
                    iostream.print(
                        colored(f"  [{agent.name}] {last_error}, retrying...", "yellow"),
                        flush=True,
                    )
            if recs is not None:
                break

        if recs is not None:
            message[agent.name] = recs
        else:
            if iostream:
                iostream.print(
                    colored(f"  [{agent.name}] Failed after {_MAX_RETRIES} attempts: {last_error}", "red"),
                    flush=True,
                )
            message[agent.name] = {
                "error": f"Failed after {_MAX_RETRIES} attempts: {last_error}",
            }

    aggregator.send(json.dumps(message, ensure_ascii=False), selector, silent=True, request_reply=False)

    if aggregator not in module.agents:
        module.agents.append(aggregator)

    if iostream:
        iostream.print(colored("Finished executing the recommendation systems.\n", "yellow"), flush=True)
    return aggregator


def _default_recsys_configs() -> Dict[str, Dict[str, Any]]:
    """Built-in Ollama configs when no config is passed (from central registry)."""
    try:
        from config.llm_clients import get_recsys_configs
    except ModuleNotFoundError:
        from advisor.config.llm_clients import get_recsys_configs
    return get_recsys_configs()


def create_recsys(
    recsys_llm_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    domain: str = "book",
) -> Orchestrator:
    """
    Create the recsys orchestrator with multiple LLM agents.

    Parameters
    ----------
    recsys_llm_configs : dict, optional
        Per-agent LLM configs (from config.llm_clients). Keys: agent names.
        When None, uses built-in Ollama defaults from central registry.
    domain : str
        "book" or "movie". Selects the system message for the recommender LLMs.
    """
    configs = recsys_llm_configs or _default_recsys_configs()
    sys_msg = DOMAIN_SYSTEM_MESSAGES.get(domain, _SYSTEM_MESSAGE_BOOK)

    agents = [
        Agent(name=name, llm_config=cfg, system_message=sys_msg)
        for name, cfg in configs.items()
    ]

    recsys_module = Module(
        name="recsys_module",
        agents=agents,
        speaker_selection_method=parallel,
        max_round=1,
    )

    recsys_orchestrator = Orchestrator(
        name="recsys_orchestrator",
        module=recsys_module,
        description="Orchestrator for recommender systems using multiple LLMs.",
    )

    return recsys_orchestrator
