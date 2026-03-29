"""
Central registry of LLM client configs for the Advisor experiment.

Roles:
  - Simulated user:  gpt-4o-mini  (OpenAI — needs OPENAI_API_KEY in environment)
  - Advisor:         gpt-4o       (OpenAI)
  - RS1:             llama3.1:latest  (Ollama)
  - RS2:             mixtral:8x7b    (Ollama)

Ollama runs on gpusic; local Mac often has Ollama on 11434, so the SSH tunnel uses a
free local port:

  ssh -f -N -L 11435:localhost:11434 gpusic

All Ollama settings below are constants — change them in this file if needed.
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
    **kwargs,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "client": client,
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
        "base_url": base_url,
        **kwargs,
    }
    return {"config_list": [config]}


# ─── User ──────────────────────────────────────────────────────
USER_MODEL = "gpt-4o-mini"
USER_TEMPERATURE = 0.7


def get_user_config() -> Dict[str, Any]:
    return _build_config(
        client="openai",
        model=USER_MODEL,
        temperature=USER_TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ─── Advisor ───────────────────────────────────────────────────
ADVISOR_MODEL = "gpt-4o"
ADVISOR_TEMPERATURE = 0.7


def get_advisor_config() -> Dict[str, Any]:
    return _build_config(
        client="openai",
        model=ADVISOR_MODEL,
        temperature=ADVISOR_TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ─── Recommender Systems (RS1 & RS2) via Ollama ───────────────
OLLAMA_BASE_URL = "http://127.0.0.1:11435"
OLLAMA_API_KEY_PLACEHOLDER = "ollama"

RS1_OLLAMA_MODEL = "llama3.1:latest"
RS2_OLLAMA_MODEL = "mixtral:8x7b"

RS1_TEMPERATURE = 0.7
RS2_TEMPERATURE = 0.7


def get_rs1_config() -> Dict[str, Any]:
    return _build_config(
        client="ollama",
        model=RS1_OLLAMA_MODEL,
        temperature=RS1_TEMPERATURE,
        api_key=OLLAMA_API_KEY_PLACEHOLDER,
        base_url=OLLAMA_BASE_URL,
    )


def get_rs2_config() -> Dict[str, Any]:
    return _build_config(
        client="ollama",
        model=RS2_OLLAMA_MODEL,
        temperature=RS2_TEMPERATURE,
        api_key=OLLAMA_API_KEY_PLACEHOLDER,
        base_url=OLLAMA_BASE_URL,
    )


def rs_models_display() -> tuple[str, str]:
    """Human-readable RS model names for logs / CLI banner."""
    return f"ollama/{RS1_OLLAMA_MODEL}", f"ollama/{RS2_OLLAMA_MODEL}"


# ─── Legacy helpers ────────────────────────────────────────────
def get_recsys_configs() -> Dict[str, Dict[str, Any]]:
    return {"rs1": get_rs1_config(), "rs2": get_rs2_config()}
