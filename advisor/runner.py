"""
Experiment runner using the ARARA framework.

Flow for Advisor condition:
    sim_user → recsys_orchestrator → advisor → sim_user → advisor → sim_user → ...
"""

from __future__ import annotations

import json
import os
import random
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from agents import Agent, Module, Orchestrator
from load_dataset import InstructRecDataset
from recsys import create_recsys
from advisor import Advisor, create_advisor
from simulated_user import SimulatedUser

from evaluation import SessionLog, TurnLog
from gt_inject import inject_random_gt_into_tri_data, session_rng


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    dataset_path: str
    n_users: int = 30
    seed: int = 42
    max_turns: int = 4
    output_dir: str = "experiment_results"
    user_type: str = "simulated"  # kept for backward-compat; only SimulatedUser is used
    bandit_model_path: Optional[str] = None
    bandit_log_path: str = "bandit_logs.jsonl"
    # LLM configs from config.llm_clients. Built by run_experiment.py from CLI args.
    user_llm_config: Optional[Dict[str, Any]] = None
    advisor_llm_config: Optional[Dict[str, Any]] = None
    recsys_llm_configs: Optional[Dict[str, Dict[str, Any]]] = None
    # Dataset type: "instructrec" (default, .pkl) or "recassistbench" (.json)
    dataset_type: str = "instructrec"
    # Domain: "book" (default) or "movie". Controls recsys/user prompts.
    domain: str = "book"
    # RecAssistBench query field to use as the user instruction.
    query_field: str = "direct_description_query"
    # If True, inject each GT title into a random recsys agent at a random top-k slot
    # (distinct slots when possible; reproducible per session). Skipped when GT empty.
    inject_random_gt_in_pool: bool = False
    # If set, no_advisor loads ``{user_id}_advisor.json`` from this directory and uses
    # its ``recsys_outputs`` instead of calling the LLM recsys again — same pool as the
    # advisor run for paired evaluation (train_test_split enables this automatically).
    no_advisor_recsys_cache_dir: Optional[str] = None


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
        recsys done       →  simulated_user  (reacts confused to raw output)
        user reacts       →  advisor          (processes recsys + user reaction)
        advisor responds  →  simulated_user
        user responds     →  advisor  (loop)

    After recsys runs, ``module._advisor_recsys_finished`` is set so the next
    user turn routes to the advisor (``triangulation`` is still None until the
    advisor's first ``process`` call).
    """
    agents_by_name = {a.name: a for a in module.agents}
    advisor_agent = agents_by_name.get("advisor")
    sim_user = agents_by_name.get("simulated_user") or agents_by_name.get("confused_user")
    recsys = agents_by_name.get("recsys_orchestrator")

    recsys_finished = getattr(module, "_advisor_recsys_finished", False)

    if last_speaker == sim_user:
        if advisor_agent and advisor_agent.triangulation is None:
            if recsys_finished:
                return advisor_agent
            return recsys
        return advisor_agent

    if last_speaker == recsys or (
        hasattr(last_speaker, "name") and "recsys" in last_speaker.name
    ):
        module._advisor_recsys_finished = True
        return sim_user

    if hasattr(last_speaker, "name") and last_speaker.name == "aggregator":
        return sim_user

    if last_speaker == advisor_agent:
        if advisor_agent.user_decided:
            return None
        return sim_user

    return None


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

_USER_MESSAGE_TEMPLATE = """
USER PERSONA
{persona}

---

{items_label} PREVIOUSLY {verb} BY THE USER
Each entry contains the {item_word} title, a factual description, and the user's own review.

{read_items}

---

USER REQUEST
{instruction}
"""

_RECASSIST_USER_MESSAGE_TEMPLATE = """
USER REQUEST
{instruction}
"""

DOMAIN_LABELS = {
    "book": {"items_label": "BOOKS", "verb": "READ", "item_word": "book"},
    "movie": {"items_label": "MOVIES", "verb": "WATCHED", "item_word": "movie"},
}


def _load_recassistbench(path: str) -> List[dict]:
    """Load a RecAssistBench ExplicitQuery-style JSON file."""
    with open(path) as f:
        return json.load(f)


def _build_user_message(entry: dict, max_history: int = 3, domain: str = "book") -> str:
    """Build the full user message from an InstructRec dataset entry."""
    persona = entry.get("persona", "")
    instruction = entry.get("instruction", "")
    titles = entry.get("title", []) or []
    descriptions = entry.get("description", []) or []
    reviews = entry.get("reviewText", []) or []
    labels = DOMAIN_LABELS.get(domain, DOMAIN_LABELS["book"])
    read_items = ""
    for i in range(min(max_history, len(titles))):
        title = titles[i] if i < len(titles) else "Unknown"
        desc = descriptions[i] if i < len(descriptions) else ""
        review = reviews[i] if i < len(reviews) else ""
        read_items += (
            f"- Name: {title}\n"
            f"  Description: {desc}\n"
            f"  User Review: {review}\n\n"
        )

    return _USER_MESSAGE_TEMPLATE.format(
        persona=persona,
        items_label=labels["items_label"],
        verb=labels["verb"],
        item_word=labels["item_word"],
        read_items=read_items or "(no history)",
        instruction=instruction,
    )


def _build_user_message_recassist(entry: dict, query_field: str) -> str:
    """Build user message from a RecAssistBench entry (no persona/history)."""
    instruction = entry.get(query_field, "")
    return _RECASSIST_USER_MESSAGE_TEMPLATE.format(instruction=instruction).strip()


def _get_n_choices(entry: dict, config: ExperimentConfig) -> int:
    """How many items should the simulated user pick at the end of a session.

    For RecAssistBench: ``bookCount`` / ``movieCount`` from the entry (= len(bookSubset)).
    Falls back to 1 for InstructRec or missing field.
    """
    if config.dataset_type == "recassistbench":
        count_key = "bookCount" if config.domain == "book" else "movieCount"
        val = entry.get(count_key, 1)
        return max(1, int(val))
    return 1


def _build_persona_recassist(entry: dict) -> str:
    """Build a vague persona string from sharedRelationships."""
    shared_rels = entry.get("sharedRelationships", [])
    parts: List[str] = []

    for rel in (shared_rels or []):
        if not (isinstance(rel, (list, tuple)) and len(rel) >= 2):
            continue
        rel_type, rel_value = rel[0], rel[1]

        if rel_type == "WRITTEN_BY":
            parts.append(f"You heard the name {rel_value} somewhere, maybe an author?")
        elif rel_type in ("IN_CATEGORY", "BELONGS_TO"):
            parts.append(f"Something about {rel_value}? You're not really sure.")
        elif rel_type == "HAS_TOPIC":
            parts.append(f"You vaguely remember someone mentioning {rel_value}.")
        else:
            parts.append(f"Something about {rel_value}, you think.")

    sit_query = entry.get("situational_description_query", "")
    if sit_query:
        parts.append(f"Some context: {sit_query}")

    return " ".join(parts) if parts else "You have no idea what you want, just 'something good'."


def _create_simulated_user(entry: dict, config: ExperimentConfig) -> SimulatedUser:
    is_recassist = config.dataset_type == "recassistbench"

    if is_recassist:
        persona = _build_persona_recassist(entry)
        instruction = entry.get(config.query_field, "")
        history: List[dict] = []
    else:
        titles = entry.get("title", []) or []
        reviews = entry.get("reviewText", []) or []
        history = [
            {"title": titles[j], "review": reviews[j] if j < len(reviews) else ""}
            for j in range(min(3, len(titles)))
        ]
        persona = entry.get("persona", "")
        instruction = entry.get("instruction", "")

    llm_config = config.user_llm_config
    if llm_config is None:
        from config.llm_clients import get_user_config
        llm_config = get_user_config()

    n_choices = _get_n_choices(entry, config)

    return SimulatedUser(
        persona=persona,
        instruction=instruction,
        history=history,
        max_turns=config.max_turns,
        domain=config.domain,
        n_choices=n_choices,
        llm_config=llm_config,
    )


def _extract_ground_truth(entry: dict, config: ExperimentConfig) -> List[str]:
    """Extract ground-truth titles from the dataset entry."""
    if config.dataset_type == "recassistbench":
        subset_key = "bookSubset" if config.domain == "book" else "movieSubset"
        return list(entry.get(subset_key, []) or [])
    # InstructRec: raw ranked_lists ids — resolution requires ItemIndex
    # which is not available in the runner; return empty and let evaluate_experiment.py handle it.
    return []


def _extract_advisor_session_log(
    advisor: Advisor,
    user_id: str,
    initial_user_message: str = "",
    ground_truth: Optional[List[str]] = None,
    shared_relationships: Optional[List] = None,
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
            items_considered_for_internal_turn_state=entry.get(
                "items_considered_for_internal_turn_state",
                entry.get("items_shown", []),
            ),
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
        chosen_items=advisor.chosen_items,
        final_state=advisor.state.to_dict() if advisor.state else None,
        recsys_outputs=tri_data,
        initial_user_message=initial_user_message,
        ground_truth=ground_truth or [],
        shared_relationships=shared_relationships or [],
    )


# ──────────────────────────────────────────────────────────────
# No Advisor condition (baseline: user receives raw recsys output)
# ──────────────────────────────────────────────────────────────

def _no_advisor_speaker_selection(
    last_speaker: Agent,
    module: Module,
    selector: Agent = None,
) -> Union[Agent, str, None]:
    """
    Flow: user → recsys → user (choice) → stop.

    After parallel recsys, ``last_speaker`` may be a leaf LLM (e.g. mistral:7b), not
    ``aggregator`` or ``recsys_orchestrator``. The old ``_no_advisor_recsys_done`` flag
    was then never set, so the user was routed to recsys again → loop / echo.
    """
    agents_by_name = {a.name: a for a in module.agents}
    sim_user = agents_by_name.get("simulated_user") or agents_by_name.get("confused_user")
    recsys = agents_by_name.get("recsys_orchestrator")
    if sim_user is None or recsys is None:
        return None

    sim_name = sim_user.name
    last_name = getattr(last_speaker, "name", "") or ""
    is_user = last_speaker == sim_user or last_name == sim_name

    # 0 = user's first message (request)
    # 1 = recsys invoked; waiting for recsys-side completion
    # 2 = recommendations delivered; user's next message ends the session
    st = getattr(sim_user, "_no_advisor_stage", 0)

    if is_user:
        if st == 0:
            sim_user._no_advisor_stage = 1
            return recsys
        if st == 1:
            return recsys
        if st == 2:
            return None  # user posted choice after seeing JSON
        return None

    # Echo / extra turns after choice: stop without re-entering recsys
    if st >= 2:
        return None

    if st == 1 and not is_user:
        sim_user._no_advisor_stage = 2
        return sim_user

    return None


def _parse_recsys_output_for_no_advisor(payload_str: str) -> Dict[str, List[dict]]:
    """Parse recsys JSON output into {llm_name: [{"title", "explanation"}, ...]}."""
    try:
        payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
    except json.JSONDecodeError:
        return {}
    result = {}
    for agent_name, agent_data in (payload or {}).items():
        if isinstance(agent_data, str):
            try:
                agent_data = json.loads(agent_data)
            except json.JSONDecodeError:
                continue
        if isinstance(agent_data, list) and agent_data:
            agent_data = agent_data[0]
        if not isinstance(agent_data, dict) or "error" in agent_data:
            continue
        recs = agent_data.get("top_k_recommendations", [])
        items = []
        for rec in recs:
            if isinstance(rec, dict):
                title = rec.get("book_title") or rec.get("movie_title") or rec.get("title") or ""
                explanation = rec.get("explanation", "")
                if title:
                    items.append({"title": title, "explanation": explanation})
        if items:
            result[agent_name] = items
    return result


def _extract_chosen_item(user_response: str, all_titles: List[str]) -> str:
    """Extract a single chosen item from user response via fuzzy match."""
    items = _extract_chosen_items(user_response, all_titles, n=1)
    return items[0] if items else "none"


def _extract_chosen_items(
    user_response: str,
    all_titles: List[str],
    n: int = 1,
) -> List[str]:
    """Extract up to *n* chosen items from user response via fuzzy match."""
    if not user_response or not all_titles:
        return []
    resp_lower = user_response.strip().lower()
    found: List[str] = []
    found_lower: set = set()
    for title in all_titles:
        tl = title.lower()
        if tl in resp_lower or resp_lower in tl:
            if tl not in found_lower:
                found.append(title)
                found_lower.add(tl)
    quoted = re.findall(r'"([^"]+)"', user_response)
    for q in quoted:
        for title in all_titles:
            if q.lower() == title.lower() and title.lower() not in found_lower:
                found.append(title)
                found_lower.add(title.lower())
    return found[:n] if n else found


def _run_no_advisor_from_advisor_json_cache(
    user_id: str,
    entry: dict,
    user_msg: str,
    config: ExperimentConfig,
    cache_path: str,
    ground_truth: Optional[List[str]] = None,
) -> SessionLog:
    """
    Baseline using the exact ``recsys_outputs`` saved in a prior advisor session JSON.
    One simulated-user LLM call to pick a title; no second recsys draw.
    """
    with open(cache_path, encoding="utf-8") as f:
        cached = json.load(f)
    recsys_outputs: Dict[str, List[dict]] = cached.get("recsys_outputs") or {}
    if not recsys_outputs:
        raise ValueError(f"Cache file has no recsys_outputs: {cache_path}")

    n_choices = _get_n_choices(entry, config)
    sim_user = _create_simulated_user(entry, config)
    item_word = "book" if config.domain == "book" else "movie"
    _NO_ADVISOR_SYSTEM_EXTRA = f"""

NO-ADVISOR MODE: You will receive a JSON with {item_word} recommendations from multiple systems.
Extract all unique {item_word} titles and choose your TOP {n_choices} favorite(s) that best fit your preferences.
Reply with ONLY the exact {item_word} title(s), one per line. No explanation."""
    try:
        if getattr(sim_user, "_oai_system_message", None) and len(sim_user._oai_system_message) > 0:
            orig = sim_user._oai_system_message[0].get("content", "") or ""
            sim_user._oai_system_message[0]["content"] = orig + _NO_ADVISOR_SYSTEM_EXTRA
    except (IndexError, TypeError, KeyError):
        pass
    sim_user._max_turns = 1
    sim_user._turn_count = 0

    visible = json.dumps(recsys_outputs, ensure_ascii=False, indent=2)
    payload_text = (
        f"Here is the aggregated output from multiple recommendation systems (JSON):\n{visible}\n\n"
        f"Choose your TOP {n_choices} favorite {item_word} title(s) from the lists above."
    )
    sim_user._oai_messages = {
        sim_user.name: [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": payload_text},
        ]
    }

    user_choice_response = sim_user.get_human_input("")

    all_titles: List[str] = []
    for items in recsys_outputs.values():
        for it in items or []:
            if not isinstance(it, dict):
                continue
            t = (it.get("title") or "").strip()
            if t and t not in all_titles:
                all_titles.append(t)

    chosen_items = _extract_chosen_items(user_choice_response, all_titles, n=n_choices)
    chosen = chosen_items[0] if chosen_items else "none"

    items_considered_for_internal_turn_state = all_titles[:20]
    turn_log = TurnLog(
        turn=0,
        action="no_advisor_raw",
        advisor_message="[Raw recsys output - no mediation; recsys from advisor session cache]",
        user_response=user_choice_response,
        belief_state={},
        items_considered_for_internal_turn_state=items_considered_for_internal_turn_state,
    )

    tri_data = {
        k: [{"title": it.get("title", ""), "explanation": it.get("explanation", "")} for it in v]
        for k, v in recsys_outputs.items()
    }
    rels = cached.get("shared_relationships")
    if not rels:
        rels = entry.get("sharedRelationships", [])

    return SessionLog(
        user_id=user_id,
        condition="no_advisor",
        turns=[turn_log],
        chosen_item=chosen,
        chosen_items=chosen_items,
        final_state=None,
        recsys_outputs=tri_data,
        initial_user_message=user_msg,
        ground_truth=ground_truth or [],
        shared_relationships=rels or [],
    )


def _run_no_advisor_condition(
    user_id: str,
    entry: dict,
    user_msg: str,
    config: ExperimentConfig,
    ground_truth: Optional[List[str]] = None,
) -> SessionLog:
    """
    No Advisor baseline: user receives raw recsys output and chooses directly.
    No mediation, no debiasing, no preference elicitation.
    """
    cache_dir = getattr(config, "no_advisor_recsys_cache_dir", None)
    if cache_dir:
        cache_path = os.path.join(cache_dir, f"{user_id}_advisor.json")
        if os.path.isfile(cache_path):
            print(f"[Runner] no_advisor: using advisor recsys cache → {cache_path}")
            return _run_no_advisor_from_advisor_json_cache(
                user_id, entry, user_msg, config, cache_path, ground_truth=ground_truth
            )
        print(f"[Runner] WARNING: advisor cache not found ({cache_path}); running live recsys")

    recsys_orchestrator = create_recsys(config.recsys_llm_configs, domain=config.domain)
    n_choices = _get_n_choices(entry, config)
    sim_user = _create_simulated_user(entry, config)

    item_word = "book" if config.domain == "book" else "movie"
    _NO_ADVISOR_SYSTEM_EXTRA = f"""

NO-ADVISOR MODE: You will receive a JSON with {item_word} recommendations from multiple systems.
Extract all unique {item_word} titles and choose your TOP {n_choices} favorite(s) that best fit your preferences.
Reply with ONLY the exact {item_word} title(s), one per line. No explanation."""
    try:
        if getattr(sim_user, "_oai_system_message", None) and len(sim_user._oai_system_message) > 0:
            orig = sim_user._oai_system_message[0].get("content", "") or ""
            sim_user._oai_system_message[0]["content"] = orig + _NO_ADVISOR_SYSTEM_EXTRA
    except (IndexError, TypeError, KeyError):
        pass
    sim_user._no_advisor_stage = 0
    sim_user._max_turns = 1

    no_advisor_module = Module(
        name="no_advisor_module",
        agents=[sim_user, recsys_orchestrator],
        speaker_selection_method=_no_advisor_speaker_selection,
        max_round=4,
    )
    no_advisor_orchestrator = Orchestrator(
        name="no_advisor_orchestrator",
        module=no_advisor_module,
    )

    print(f"[Runner] Starting no_advisor session for {user_id}")
    sim_user.talk_to(no_advisor_orchestrator, message=user_msg)

    # Extract recsys output and user choice from conversation
    recsys_output_raw = ""
    user_choice_response = ""
    for sender, msgs in (sim_user._oai_messages or {}).items():
        if not msgs or len(msgs) < 2:
            continue
        # Standard: [0]=user request, [1]=aggregated JSON from recsys
        recsys_output_raw = msgs[1].get("content", "") or ""
        # After recsys, take the last non-empty message (handles echo loops / extra turns)
        for m in reversed(msgs[2:]):
            c = (m.get("content", "") or "").strip()
            if c:
                user_choice_response = c
                break
        if not user_choice_response and len(msgs) >= 3:
            user_choice_response = msgs[2].get("content", "") or ""
        break

    recsys_outputs = _parse_recsys_output_for_no_advisor(recsys_output_raw)
    if config.inject_random_gt_in_pool and ground_truth:
        rng = session_rng(config.seed, user_id)
        recsys_outputs = inject_random_gt_into_tri_data(
            recsys_outputs, ground_truth, rng,
            query=entry.get(config.query_field, ""),
            shared_relationships=entry.get("sharedRelationships"),
            domain=config.domain,
        )

    all_items = []
    all_titles = []
    for llm_name, items in recsys_outputs.items():
        for it in items:
            t = it.get("title", "")
            if t and t not in all_titles:
                all_titles.append(t)
                all_items.append(it)

    chosen_items = _extract_chosen_items(user_choice_response, all_titles, n=n_choices)
    chosen = chosen_items[0] if chosen_items else "none"

    items_considered_for_internal_turn_state = all_titles[:20]
    turn_log = TurnLog(
        turn=0,
        action="no_advisor_raw",
        advisor_message="[Raw recsys output - no mediation]",
        user_response=user_choice_response,
        belief_state={},
        items_considered_for_internal_turn_state=items_considered_for_internal_turn_state,
    )

    tri_data = {
        k: [{"title": it.get("title", ""), "explanation": it.get("explanation", "")} for it in v]
        for k, v in recsys_outputs.items()
    }

    return SessionLog(
        user_id=user_id,
        condition="no_advisor",
        turns=[turn_log],
        chosen_item=chosen,
        chosen_items=chosen_items,
        final_state=None,
        recsys_outputs=tri_data,
        initial_user_message=user_msg,
        ground_truth=ground_truth or [],
        shared_relationships=entry.get("sharedRelationships", []),
    )


# ──────────────────────────────────────────────────────────────
# Advisor condition
# ──────────────────────────────────────────────────────────────

def _run_advisor_condition(
    user_id: str,
    entry: dict,
    user_msg: str,
    config: ExperimentConfig,
    ground_truth: Optional[List[str]] = None,
    training_phase: bool = False,
) -> SessionLog:
    """
    Run the full advisor condition via ARARA's orchestration:
        sim_user.talk_to(orchestrator, message=user_msg)

    Parameters
    ----------
    training_phase : bool
        If True, enables training-only actions like INJECT_GT_PROBE.
    """
    recsys_orchestrator = create_recsys(config.recsys_llm_configs, domain=config.domain)

    log_path = os.path.join(config.output_dir, config.bandit_log_path)
    inj_rng = (
        session_rng(config.seed, user_id)
        if config.inject_random_gt_in_pool
        else None
    )
    n_choices = _get_n_choices(entry, config)
    advisor = create_advisor(
        llm_config=config.advisor_llm_config,
        bandit_model_path=config.bandit_model_path,
        bandit_log_path=log_path,
        max_turns=config.max_turns,
        session_id=user_id,
        ground_truth=ground_truth,
        training_phase=training_phase,
        inject_random_gt_in_pool=config.inject_random_gt_in_pool,
        inject_gt_rng=inj_rng,
        inject_gt_query=entry.get(config.query_field, ""),
        inject_gt_shared_relationships=entry.get("sharedRelationships"),
        n_choices=n_choices,
        domain=config.domain,
    )

    sim_user = _create_simulated_user(entry, config)

    # +2 for initial user msg + recsys response, +2 for user reaction + advisor first turn,
    # then max_turns advisor↔user pairs, +2 buffer for END_SESSION.
    max_round = 4 + (config.max_turns * 2) + 2
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

    return _extract_advisor_session_log(
        advisor, user_id,
        initial_user_message=user_msg,
        ground_truth=ground_truth or [],
        shared_relationships=entry.get("sharedRelationships", []),
    )


# ──────────────────────────────────────────────────────────────
# Main experiment runner
# ──────────────────────────────────────────────────────────────

def load_dataset_entries(config: ExperimentConfig):
    """Load dataset and return (entries_or_dataset, n_total)."""
    if config.dataset_type == "recassistbench":
        entries = _load_recassistbench(config.dataset_path)
        return entries, len(entries)
    ds = InstructRecDataset(config.dataset_path)
    return ds, len(ds)


def _get_completed_user_ids(output_dir: str, condition: str) -> set:
    """Return set of user_ids already completed (have a saved JSON file)."""
    completed = set()
    if not os.path.isdir(output_dir):
        return completed
    suffix = f"_{condition}.json"
    for fname in os.listdir(output_dir):
        if fname.endswith(suffix):
            user_id = fname[: -len(suffix)]
            completed.add(user_id)
    return completed


def run_experiment(
    config: ExperimentConfig,
    condition: str = "advisor",
    indices: Optional[List[int]] = None,
    training_phase: bool = False,
    resume: bool = True,
) -> Dict:
    """
    Run experiment for the given condition.

    Parameters
    ----------
    condition : "advisor" | "no_advisor"
    indices : list of int, optional
        Explicit dataset row indices to run. When provided, ``config.n_users``
        and ``config.seed`` are ignored for sampling (caller controls the split).
    training_phase : bool
        If True (and condition="advisor"), enables training-only actions like
        INJECT_GT_PROBE that use ground truth supervision.
    resume : bool
        If True (default), skip users that already have a saved session file.
    """
    print(f"[Experiment] Loading dataset from {config.dataset_path}")
    print(f"[Experiment] Dataset type: {config.dataset_type}, domain: {config.domain}")

    if config.dataset_type == "recassistbench":
        all_entries = _load_recassistbench(config.dataset_path)
        n_total = len(all_entries)
    else:
        dataset = InstructRecDataset(config.dataset_path)
        n_total = len(dataset)

    if indices is None:
        random.seed(config.seed)
        n_sample = min(config.n_users, n_total)
        indices = random.sample(range(n_total), n_sample)
    else:
        n_sample = len(indices)

    print(f"[Experiment] Running {n_sample} sessions (from {n_total} total)")
    print(f"[Experiment] Condition: {condition}")

    os.makedirs(config.output_dir, exist_ok=True)

    completed_ids = set()
    if resume:
        completed_ids = _get_completed_user_ids(config.output_dir, condition)
        if completed_ids:
            print(f"[Resume] Found {len(completed_ids)} completed sessions — will skip them.")

    logs: List[SessionLog] = []
    sessions_run = 0

    for i, idx in enumerate(indices):
        if config.dataset_type == "recassistbench":
            entry = all_entries[idx]
            user_id = f"user_{entry.get('data_idx', idx)}"
            user_msg = _build_user_message_recassist(entry, config.query_field)
        else:
            entry = dataset[idx]
            user_id = f"user_{idx}"
            user_msg = _build_user_message(entry, domain=config.domain)

        if resume and user_id in completed_ids:
            print(f"[Resume] Skipping {user_id} (already completed)")
            continue

        print(f"\n{'='*60}")
        print(f"[Experiment] User {i+1}/{n_sample} (idx={idx})")
        print(f"{'='*60}")

        gt = _extract_ground_truth(entry, config)

        log = None
        try:
            if condition == "no_advisor":
                log = _run_no_advisor_condition(user_id, entry, user_msg, config, ground_truth=gt)
            else:
                log = _run_advisor_condition(
                    user_id, entry, user_msg, config,
                    ground_truth=gt,
                    training_phase=training_phase,
                )
            logs.append(log)
            sessions_run += 1
            items_str = ", ".join(log.chosen_items) if log.chosen_items else log.chosen_item
            print(f"[Experiment] {condition} session done — {len(log.turns)} turns, chose: {items_str}")
            print("#"*60)
        except Exception as e:
            print(f"[Experiment] {condition} session failed: {e}")
            traceback.print_exc()
        _save_log(log, config.output_dir, user_id, condition)

    summary = {f"n_{condition}": len(logs), "sessions_run_this_call": sessions_run}
    summary_path = os.path.join(config.output_dir, f"{condition}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Experiment] Summary saved to {summary_path}")

    return summary


def _save_log(log: Optional[SessionLog], output_dir: str, user_id: str, condition: str):
    if log is None:
        return
    path = os.path.join(output_dir, f"{user_id}_{condition}.json")
    with open(path, "w") as f:
        json.dump(log.to_dict(), f, indent=2)
