"""
Central registry of LLM client configs for the Advisor experiment.

All LLM configs (user, advisor, recsys) are defined here. ExperimentConfig
selects from this registry and passes configs to the appropriate actors.

Swapped roles (user ↔ recsys):
  - Simulated user: Ollama mixtral:8x7b (former recsys strongest model)
    → weaker / noisier role-play vs OpenAI.
  - Recsys: mixed — 1 agent on OpenAI gpt-4o-mini (former user model),
    the other 2 on Ollama (gemma2:9b, llama3.2) as before.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _build_config(
    client: str,
    model: str,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    response_format: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Build LLM config in the format expected by ARARA agents."""
    config = {
        "client": client,
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
        "base_url": base_url,
        **kwargs,
    }
    if response_format == "json_object":
        config["response_format"] = {"type": "json_object"}
    return {"config_list": [config]}


# ─── User (Ollama — former strongest recsys) ─────────────────

OLLAMA_BASE = dict(
    client="ollama",
    base_url="http://localhost:11434",
)

# Strongest model among the old recsys trio (local).
USER_MODEL = "gpt-4o-mini"
USER_TEMPERATURE = 0.7
def get_user_config() -> Dict[str, Any]:
    """Simulated user on local Ollama (weaker role-play than OpenAI mini)."""
    return _build_config(
        client="openai",
        model=USER_MODEL,
        temperature=USER_TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

# ─── Advisor (OpenAI) ────────────────────────────────────────

# Model and temperature for Advisor. Set here only.
ADVISOR_MODEL = "gpt-4o"
ADVISOR_TEMPERATURE = 0.7
def get_advisor_config() -> Dict[str, Any]:
    """Config for the Advisor (OpenAI). Model and temperature set in this file."""
    return {
        "config_list": [
            {
                "client": "openai",
                "temperature": ADVISOR_TEMPERATURE,
                "model": ADVISOR_MODEL,
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        ]
    }


# ─── Recsys (mixed: 1 OpenAI + 2 Ollama) ────────────────────

# Lower (e.g. 0.0) for more repeatable lists; paired advisor/baseline still need
# ``no_advisor_recsys_cache_dir`` (train_test_split) for identical pools.
RECSYS_TEMPERATURE = 0.7
OLLAMA_RECSYS_BASE = dict(
    **OLLAMA_BASE,
    response_format="json_object",
    format="json",
    think=False,
    temperature=RECSYS_TEMPERATURE,
)
def get_recsys_configs() -> Dict[str, Dict[str, Any]]:
    """Three recsys slots: gpt-4o-mini (OpenAI), gemma2:9b and llama3.2 (Ollama)."""
    return {
        "mixtral:8x7b": _build_config(**{**OLLAMA_RECSYS_BASE, "model": "mixtral:8x7b"}),
        "gemma2:9b": _build_config(**{**OLLAMA_RECSYS_BASE, "model": "gemma2:9b"}),
       # "llama3.2": _build_config(**{**OLLAMA_RECSYS_BASE, "model": "llama3.2"}),
    }


