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


system_message = """You are acting as an independent book recommendation system.

You will receive:
- A structured user profile containing previously interacted books.
  Each history book includes: ASIN, title, description, and review text.
- A natural language request from the user.

Your task is to recommend books that best satisfy the user's request.

You must:

1. Produce a ranked list of the TOP-3 recommended books (best first).
2. For EACH recommended book, provide:
   - A concise explanation (1–2 sentences) grounded strictly in the user's preference signals.
   - The list of ASINs from the user's history that most influenced this recommendation.

IMPORTANT RULES:
- Use only the information contained in the provided user profile to infer preferences.
- Do NOT recommend any book that appears in the user's history.
- For each recommended book, return between 1 and 3 ASINs in "used_history_asins".
- The ASINs must refer exclusively to books present in the user profile.
- Do NOT invent ASINs.
- Do NOT output history titles as identifiers.
- Each explanation must be specific to its corresponding recommended book.
- Return ONLY valid JSON. No extra text.

Output JSON format (single object):

{
  "top_k_recommendations": [
    {
      "rank": 1,
      "book_title": "<title>",
      "explanation": "<1–2 sentences grounded in the user's preferences>",
      "used_history_asins": ["<asin1>", "<asin2>"]
    },
    {
      "rank": 2,
      "book_title": "<title>",
      "explanation": "<1–2 sentences grounded in the user's preferences>",
      "used_history_asins": ["<asin1>"]
    },
    {
      "rank": 3,
      "book_title": "<title>",
      "explanation": "<1–2 sentences grounded in the user's preferences>",
      "used_history_asins": ["<asin1>", "<asin2>", "<asin3>"]
    }
  ]
}

"""


def parse_recommendations(raw_content):
    """Parse raw LLM output into structured recommendations."""
    if isinstance(raw_content, (list, dict)):
        return raw_content

    if not isinstance(raw_content, str):
        raise TypeError(f"Unsupported content type: {type(raw_content)}")

    text = raw_content.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    parsed = json.loads(text)

    if isinstance(parsed, dict):
        parsed = [parsed]
    elif not isinstance(parsed, list):
        raise ValueError("Parsed content is not a list or dict")

    return parsed


def parallel(
    last_speaker: Agent, module: Module, selector: Agent = None
) -> Union[Agent, str, None]:
    """Run all recommender agents in parallel and aggregate results."""
    if iostream:
        iostream.print(colored("\nStarted executing the recommendation systems...", "yellow"), flush=True)

    aggregator = Aggregator(name="aggregator")

    message = {}
    for agent in module.agents:
        for reply in agent.generate_reply(sender=selector):
            agent.send(reply, aggregator, silent=False, request_reply=False)
            raw = reply.chat_message.content
            try:
                recs = parse_recommendations(raw)
                message[agent.name] = recs
            except Exception as e:
                message[agent.name] = {
                    "error": f"Failed to parse output: {str(e)}",
                    "raw_output": raw,
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
) -> Orchestrator:
    """
    Create the recsys orchestrator with multiple LLM agents.

    Parameters
    ----------
    recsys_llm_configs : dict, optional
        Per-agent LLM configs (from config.llm_clients). Keys: agent names.
        When None, uses built-in Ollama defaults from central registry.
    """
    configs = recsys_llm_configs or _default_recsys_configs()

    agents = [
        Agent(name=name, llm_config=cfg, system_message=system_message)
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
