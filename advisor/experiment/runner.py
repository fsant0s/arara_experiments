"""
Experiment runner using the ARARA framework.

Flow for Advisor condition:
    sim_user → recsys_orchestrator → advisor → sim_user → advisor → sim_user → ...
"""

from __future__ import annotations

import json
import os
import random
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from agents import Agent, Module, Orchestrator
from utils import get_llm_config, InstructRecDataset
from recsys.llms import create_recsys
from recsys.llms.user_message import user_message as user_message_template
from advisor import Advisor, create_advisor
from users.simulated_user import SimulatedUser
from users.confused_user import ConfusedUser

from experiment.evaluation import SessionLog, TurnLog


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    dataset_path: str
    n_users: int = 30
    seed: int = 42
    max_turns: int = 4
    advisor_model: str = "openai/gpt-4o"
    user_model: str = "openai/gpt-4o"
    user_temperature: float = 0.7
    recsys_models: List[str] = field(default_factory=lambda: [
        "openai/gpt-4o",
        "google/gemini-2.5-flash-lite",
        "anthropic/claude-3.5-sonnet",
    ])
    output_dir: str = "experiment_results"
    # user_type controls which simulated user class is used:
    # "simulated" (default) or "confused" for ConfusedUser.
    user_type: str = "simulated"


# ──────────────────────────────────────────────────────────────
# Custom speaker selection for multi-turn conversation
# ──────────────────────────────────────────────────────────────

def conversational_speaker_selection(
    last_speaker: Agent,
    module: Module,
    selector: Agent = None,
) -> Union[Agent, str, None]:
    """
    Deterministic speaker selection for the experiment:

        1st user message  →  recsys_orchestrator
        recsys done       →  advisor
        advisor responds  →  simulated_user
        user responds     →  advisor  (loop)

    The advisor's `triangulation` attribute indicates whether the
    recsys has already been queried in this session.
    """
    agents_by_name = {a.name: a for a in module.agents}
    advisor_agent = agents_by_name.get("advisor")
    # support both the standard simulated user and the confused variant
    sim_user = agents_by_name.get("simulated_user") or agents_by_name.get("confused_user")
    recsys = agents_by_name.get("recsys_orchestrator")

    if last_speaker == sim_user:
        if advisor_agent and advisor_agent.triangulation is None:
            return recsys
        return advisor_agent

    if last_speaker == recsys or (
        hasattr(last_speaker, "name") and "recsys" in last_speaker.name
    ):
        return advisor_agent

    if last_speaker == advisor_agent:
        if advisor_agent.user_decided:
            return None
        return sim_user

    if hasattr(last_speaker, "name") and last_speaker.name == "aggregator":
        return advisor_agent

    return None


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _build_user_message(entry: dict, max_history: int = 3) -> str:
    """Build the full user message from an InstructRec dataset entry."""
    persona = entry.get("persona", "")
    instruction = entry.get("instruction", "")
    titles = entry.get("title", []) or []
    descriptions = entry.get("description", []) or []
    reviews = entry.get("reviewText", []) or []
    asins = entry.get("asin", []) or []

    read_books = ""
    for i in range(min(max_history, len(titles))):
        asin = asins[i] if i < len(asins) else "N/A"
        title = titles[i] if i < len(titles) else "Unknown"
        desc = descriptions[i] if i < len(descriptions) else ""
        review = reviews[i] if i < len(reviews) else ""
        read_books += (
            f"- ID: {asin}\n"
            f"  Name: {title}\n"
            f"  Description: {desc}\n"
            f"  User Review: {review}\n\n"
        )

    return user_message_template.format(
        persona=persona,
        read_books=read_books or "(no history)",
        instruction=instruction,
    )


def _make_llm_config(model: str, temperature: float = 0.0) -> dict:
    return get_llm_config(
        client="openrouter",
        model=model,
        temperature=temperature,
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )


def _create_simulated_user(entry: dict, config: ExperimentConfig) -> SimulatedUser:
    titles = entry.get("title", []) or []
    reviews = entry.get("reviewText", []) or []
    history = [
        {"title": titles[j], "review": reviews[j] if j < len(reviews) else ""}
        for j in range(min(3, len(titles)))
    ]

    persona = entry.get("persona", "")
    instruction = entry.get("instruction", "")

    user_cls = SimulatedUser if config.user_type != "confused" else ConfusedUser

    return user_cls(
        persona=persona,
        instruction=instruction,
        history=history,
        max_turns=config.max_turns,
        llm_config=_make_llm_config(config.user_model, config.user_temperature),
    )


def _extract_advisor_session_log(
    advisor: Advisor,
    user_id: str,
) -> SessionLog:
    """Build a SessionLog from the Advisor's internal _session_log."""
    turns = []
    for entry in advisor._session_log:
        turns.append(TurnLog(
            turn=entry["turn"],
            action=entry["action"],
            advisor_message=entry["advisor_message"],
            user_response=entry["user_response"],
            belief_state=entry["belief_state"],
            items_shown=entry["items_shown"],
        ))

    tri_data = {}
    if advisor.triangulation and advisor.triangulation.per_llm_items:
        tri_data = {
            k: [{"title": it.get("title", ""), "explanation": it.get("explanation", "")}
                 for it in v]
            for k, v in advisor.triangulation.per_llm_items.items()
        }

    return SessionLog(
        user_id=user_id,
        condition="advisor",
        turns=turns,
        chosen_item=advisor.chosen_item or "none",
        final_state=advisor.state.to_dict() if advisor.state else None,
        recsys_outputs=tri_data,
    )


# ──────────────────────────────────────────────────────────────
# Advisor condition
# ──────────────────────────────────────────────────────────────

def _run_advisor_condition(
    user_id: str,
    entry: dict,
    user_msg: str,
    config: ExperimentConfig,
) -> SessionLog:
    """
    Run the full advisor condition via ARARA's orchestration:
        sim_user.talk_to(orchestrator, message=user_msg)
    """
    recsys_orchestrator = create_recsys()

    advisor = create_advisor(
        llm_config=_make_llm_config(config.advisor_model, temperature=0.0),
    )

    sim_user = _create_simulated_user(entry, config)

    # No transitions dict — conversational_speaker_selection is the sole
    # controller of the flow, including returning None to stop the session.
    max_round = 2 + (config.max_turns * 2) + 2
    main_module = Module(
        name="experiment_module",
        agents=[sim_user, recsys_orchestrator, advisor],
        speaker_selection_method=conversational_speaker_selection,
        max_round=max_round,
    )

    orchestrator = Orchestrator(
        name="experiment_orchestrator",
        module=main_module,
    )

    print(f"[Runner] Starting advisor session for {user_id}")
    sim_user.talk_to(orchestrator, message=user_msg)

    return _extract_advisor_session_log(advisor, user_id)


# ──────────────────────────────────────────────────────────────
# Main experiment runner
# ──────────────────────────────────────────────────────────────

def run_experiment(config: ExperimentConfig) -> Dict:
    random.seed(config.seed)

    print(f"[Experiment] Loading dataset from {config.dataset_path}")
    dataset = InstructRecDataset(config.dataset_path)
    n_total = len(dataset)
    n_sample = min(config.n_users, n_total)
    indices = random.sample(range(n_total), n_sample)

    print(f"[Experiment] Sampled {n_sample} users from {n_total}")

    advisor_logs: List[SessionLog] = []

    os.makedirs(config.output_dir, exist_ok=True)

    for i, idx in enumerate(indices):
        entry = dataset[idx]
        user_id = f"user_{idx}"
        user_msg = _build_user_message(entry)

        print(f"\n{'='*60}")
        print(f"[Experiment] User {i+1}/{n_sample} (idx={idx})")
        print(f"{'='*60}")

        # --- Advisor condition ---
        adv_log = None
        try:
            adv_log = _run_advisor_condition(user_id, entry, user_msg, config)
            advisor_logs.append(adv_log)
            print(f"[Experiment] Advisor session done — {len(adv_log.turns)} turns, chose: {adv_log.chosen_item}")
            print("#"*60)
        except Exception as e:
            print(f"[Experiment] Advisor session failed: {e}")
            traceback.print_exc()
        _save_log(adv_log, config.output_dir, user_id, "advisor")

    summary = {
        "n_advisor": len(advisor_logs),
    }
    summary_path = os.path.join(config.output_dir, "advisor_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Experiment] Advisor summary saved to {summary_path}")

    return summary


def _save_log(log: Optional[SessionLog], output_dir: str, user_id: str, condition: str):
    if log is None:
        return
    path = os.path.join(output_dir, f"{user_id}_{condition}.json")
    with open(path, "w") as f:
        json.dump(log.to_dict(), f, indent=2)
