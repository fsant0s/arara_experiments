"""Thin wrapper around OpenAI-compatible chat completions.

Works with both OpenAI and Ollama (via /v1 compat endpoint).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional


def make_chat_fn(
    config: Dict[str, Any],
    *,
    temperature_override: Optional[float] = None,
) -> Callable[[List[dict]], str]:
    """Build a chat-completion callable from the project config format.

    Config format: ``{"config_list": [{"client": ..., "model": ..., ...}]}``

    Returns a callable ``chat(messages) -> str``.
    """
    from openai import OpenAI

    cfg = config["config_list"][0]
    client_type = cfg.get("client", "openai")
    api_key = cfg.get("api_key") or os.getenv("OPENAI_API_KEY") or "ollama"
    base_url = cfg.get("base_url")

    if client_type == "ollama" and base_url and not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    model = cfg["model"]
    temperature = (
        temperature_override
        if temperature_override is not None
        else cfg.get("temperature", 0.7)
    )

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    oai = OpenAI(**client_kwargs)

    def chat(messages: List[dict]) -> str:
        resp = oai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    return chat


def make_prompt_fn(config: Dict[str, Any]) -> Callable[[str], str]:
    """Single-prompt convenience wrapper (for belief state updates)."""
    chat_fn = make_chat_fn(config, temperature_override=0.0)

    def prompt(text: str) -> str:
        return chat_fn([{"role": "user", "content": text}])

    return prompt
