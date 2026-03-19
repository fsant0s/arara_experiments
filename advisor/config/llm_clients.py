"""
Central registry of LLM client configs for the Advisor experiment.

All LLM configs (user, advisor, recsys) are defined here. ExperimentConfig
selects from this registry and passes configs to the appropriate actors.
"""

from __future__ import annotations

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


# ─── User & Advisor (Ollama local) ────────────────────────────

OLLAMA_BASE = dict(
    client="ollama",
    base_url="http://localhost:11434",
)


def get_user_config(model: str = "mistral:7b", temperature: float = 0.7) -> Dict[str, Any]:
    """Config for the simulated user (Ollama)."""
    return _build_config(**{**OLLAMA_BASE, "model": model, "temperature": temperature})


def get_advisor_config(model: str = "qwen2.5:7b") -> Dict[str, Any]:
    """Config for the Advisor (Ollama, temperature=0)."""
    return _build_config(**{**OLLAMA_BASE, "model": model, "temperature": 0.0})


# ─── Recsys (Ollama local) ───────────────────────────────────

OLLAMA_RECSYS_BASE = dict(
    **OLLAMA_BASE,
    response_format="json_object",
    temperature=0.7,
)


def get_recsys_configs() -> Dict[str, Dict[str, Any]]:
    """Default recsys agent configs (Ollama)."""
    return {
        "qwen4b": _build_config(**{**OLLAMA_RECSYS_BASE, "model": "qwen:4b"}),
        "deepseekr17b": _build_config(**{**OLLAMA_RECSYS_BASE, "model": "deepseek-r1:7b"}),
        "llama38b": _build_config(**{**OLLAMA_RECSYS_BASE, "model": "llama3:8b"}),
    }


