"""
Session runner for the multi-party advisor experiment.

Flow (advisor condition):
    Round 0:  user query → RS1 + RS2 (parallel)  → Advisor intervenes
    Round N:  user → chosen RS → Advisor intervenes
    End:      user decides → extract chosen items

Flow (baseline / no_advisor):
    Round 0:  user query → RS1 + RS2  → user sees raw responses
    Round N:  user → chosen RS (no mediation)
    End:      user decides
"""
from __future__ import annotations

import json
import os
import random
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_utils import make_chat_fn, make_prompt_fn
from rs_agent import ConversationalRS
from advisor import AdvisorMediator
from simulated_user import SimulatedUser
from components.policy import LinUCBPolicy
from components.belief_state import compute_context_vector
from evaluation import SessionLog, TurnLog
from gt_inject import distribute_gt_for_rs, session_rng


# ─── Configuration ────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    dataset_path: str
    n_users: int = 30
    seed: int = 42
    max_turns: int = 8
    output_dir: str = "experiment_results"
    user_type: str = "simulated"
    bandit_model_path: Optional[str] = None
    bandit_log_path: str = "bandit_logs.jsonl"
    user_llm_config: Optional[Dict[str, Any]] = None
    advisor_llm_config: Optional[Dict[str, Any]] = None
    rs1_llm_config: Optional[Dict[str, Any]] = None
    rs2_llm_config: Optional[Dict[str, Any]] = None
    dataset_type: str = "recassistbench"
    domain: str = "book"
    query_field: str = "direct_description_query"
    inject_gt: bool = True


# ─── Dataset helpers ──────────────────────────────────────────

def _load_recassistbench(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)


def load_dataset_entries(config: ExperimentConfig):
    entries = _load_recassistbench(config.dataset_path)
    return entries, len(entries)


def _extract_ground_truth(entry: dict, config: ExperimentConfig) -> List[str]:
    subset_key = "bookSubset" if config.domain == "book" else "movieSubset"
    return list(entry.get(subset_key, []) or [])


def _get_n_choices(entry: dict, config: ExperimentConfig) -> int:
    count_key = "bookCount" if config.domain == "book" else "movieCount"
    return max(1, int(entry.get(count_key, 1)))


def _build_user_message_recassist(entry: dict, query_field: str) -> str:
    return (entry.get(query_field, "") or "").strip()


def _build_persona_recassist(entry: dict) -> str:
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
        else:
            parts.append(f"Something about {rel_value}, you think.")
    return " ".join(parts) if parts else "You have no idea what you want, just 'something good'."


# ─── Conversation summary builder ─────────────────────────────

def _normalize_rs_responses(
    rs1_name: str, rs2_name: str, partial: Dict[str, str],
) -> Dict[str, str]:
    """Always emit both RS keys so each JSON turn is self-contained."""
    return {
        rs1_name: (partial.get(rs1_name) or "").strip() or "",
        rs2_name: (partial.get(rs2_name) or "").strip() or "",
    }


def _build_conversation_summary(
    rs1: ConversationalRS,
    rs2: ConversationalRS,
    advisor_messages: List[str],
    user_messages: List[tuple[str, str]],
) -> str:
    """Build a readable conversation summary for the simulated user."""
    parts: List[str] = []

    if rs1.history:
        parts.append(f"--- {rs1.name} conversation ---")
        parts.append(rs1.get_history_text(max_messages=10))

    if rs2.history:
        parts.append(f"\n--- {rs2.name} conversation ---")
        parts.append(rs2.get_history_text(max_messages=10))

    return "\n\n".join(parts)


# ─── Session runners ──────────────────────────────────────────

def _run_advisor_session(
    user_id: str,
    entry: dict,
    user_query: str,
    config: ExperimentConfig,
    ground_truth: List[str],
    training_phase: bool = False,
) -> SessionLog:
    """Run one advisor session with the multi-party flow."""
    n_choices = _get_n_choices(entry, config)
    persona = _build_persona_recassist(entry)

    rs1_chat = make_chat_fn(config.rs1_llm_config or {})
    rs2_chat = make_chat_fn(config.rs2_llm_config or {})
    advisor_chat = make_chat_fn(config.advisor_llm_config or {})
    advisor_belief_fn = make_prompt_fn(config.advisor_llm_config or {})
    user_chat = make_chat_fn(config.user_llm_config or {})

    # Inject ALL GT into both RSs so both can surface every item
    rs1_gt: List[str] = list(ground_truth) if config.inject_gt and ground_truth else []
    rs2_gt: List[str] = list(ground_truth) if config.inject_gt and ground_truth else []

    rs1 = ConversationalRS("RS1", rs1_chat, domain=config.domain, gt_titles=rs1_gt, persona_key="rs1")
    rs2 = ConversationalRS("RS2", rs2_chat, domain=config.domain, gt_titles=rs2_gt, persona_key="rs2")

    # Bandit
    if config.bandit_model_path and os.path.isfile(config.bandit_model_path):
        policy = LinUCBPolicy.load(config.bandit_model_path)
    else:
        policy = LinUCBPolicy()

    log_path = os.path.join(config.output_dir, config.bandit_log_path) if training_phase else None

    advisor = AdvisorMediator(
        chat_fn=advisor_chat,
        belief_llm_fn=advisor_belief_fn,
        policy=policy,
        rs1_name=rs1.name,
        rs2_name=rs2.name,
        domain=config.domain,
        n_choices=n_choices,
        ground_truth=ground_truth if training_phase else None,
        training_phase=training_phase,
        max_turns=config.max_turns,
        bandit_log_path=log_path,
        session_id=user_id,
    )

    sim_user = SimulatedUser(
        chat_fn=user_chat,
        instruction=user_query,
        persona=persona,
        n_choices=n_choices,
        domain=config.domain,
        max_turns=config.max_turns,
        rs1_name=rs1.name,
        rs2_name=rs2.name,
    )

    # ── Round 0: Initial query to both RSs ──
    print(f"  [Round 0] Querying {rs1.name} and {rs2.name}...")
    rs1_resp = rs1.respond(user_query)
    rs2_resp = rs2.respond(user_query)
    print(f"  [Round 0] {rs1.name}: {len(rs1_resp)} chars, {rs2.name}: {len(rs2_resp)} chars")

    # GT repair: if the RS ignored the injection prompt, integrate missing titles
    if config.inject_gt and ground_truth:
        repair_fn = make_prompt_fn(config.advisor_llm_config or {})
        rs1_resp = rs1.repair_with_gt(rs1_resp, repair_fn)
        rs2_resp = rs2.repair_with_gt(rs2_resp, repair_fn)

    action, advisor_msg = advisor.initial_intervention(rs1, rs2)
    print(f"  [Round 0] Advisor action: {action.value}")

    turns: List[TurnLog] = []
    turns.append(TurnLog(
        turn=0,
        action=action.value,
        advisor_message=advisor_msg,
        user_message=user_query,
        user_target="",
        rs_responses=_normalize_rs_responses(
            rs1.name, rs2.name, {rs1.name: rs1_resp, rs2.name: rs2_resp},
        ),
        belief_state=advisor.state.to_dict(),
        items_mentioned=rs1.get_mentioned_titles() + rs2.get_mentioned_titles(),
    ))

    # ── Conversation loop ──
    for t in range(1, config.max_turns + 1):
        summary = _build_conversation_summary(rs1, rs2, [], [])
        target, user_msg = sim_user.respond(advisor_msg, summary)
        print(f"  [Round {t}] User → {target}: {user_msg[:80]}...")

        if target.lower() == "decision":
            advisor.user_decided = True
            chosen = _extract_chosen_from_message(
                user_msg, rs1.get_mentioned_titles() + rs2.get_mentioned_titles(), n_choices,
            )
            advisor.chosen_items = chosen
            action, advisor_msg = advisor.process_turn(
                user_msg, "decision", rs1, rs2, [],
            )
            turns.append(TurnLog(
                turn=t,
                action=action.value,
                advisor_message=advisor_msg,
                user_message=user_msg,
                user_target="decision",
                rs_responses=_normalize_rs_responses(rs1.name, rs2.name, {}),
                belief_state=advisor.state.to_dict(),
                items_mentioned=chosen,
            ))
            print(f"  [Round {t}] User decided: {chosen}")
            break

        # Route to RS
        if target == rs2.name:
            rs_resp = rs2.respond(user_msg)
            rs_items = rs2.get_mentioned_titles()
        else:
            rs_resp = rs1.respond(user_msg)
            rs_items = rs1.get_mentioned_titles()

        print(f"  [Round {t}] {target} responded: {len(rs_resp)} chars")

        action, advisor_msg = advisor.process_turn(
            user_msg, target, rs1, rs2, rs_items,
        )
        print(f"  [Round {t}] Advisor action: {action.value}")

        turns.append(TurnLog(
            turn=t,
            action=action.value,
            advisor_message=advisor_msg,
            user_message=user_msg,
            user_target=target,
            rs_responses=_normalize_rs_responses(
                rs1.name, rs2.name, {target: rs_resp},
            ),
            belief_state=advisor.state.to_dict(),
            items_mentioned=rs_items,
        ))

        if advisor.user_decided:
            print(f"  [Round {t}] Session ended (user decided)")
            break
    else:
        if not advisor.chosen_items:
            all_titles = rs1.get_mentioned_titles() + rs2.get_mentioned_titles()
            advisor.chosen_items = all_titles[:n_choices] if all_titles else []
            print(f"  [Max turns] Force-picking: {advisor.chosen_items}")

    # Save bandit model after training sessions
    if training_phase and config.bandit_model_path:
        model_dir = os.path.dirname(config.bandit_model_path) or "."
        os.makedirs(model_dir, exist_ok=True)

    return SessionLog(
        user_id=user_id,
        condition="advisor",
        turns=turns,
        chosen_items=advisor.chosen_items,
        ground_truth=ground_truth,
        rs1_gt_items=rs1_gt,
        rs2_gt_items=rs2_gt,
        rs1_history=rs1.history,
        rs2_history=rs2.history,
        final_state=advisor.state.to_dict(),
        initial_user_message=user_query,
        shared_relationships=entry.get("sharedRelationships", []),
        n_choices=n_choices,
    )


def _run_no_advisor_session(
    user_id: str,
    entry: dict,
    user_query: str,
    config: ExperimentConfig,
    ground_truth: List[str],
) -> SessionLog:
    """Baseline: user talks to RSs without advisor mediation."""
    n_choices = _get_n_choices(entry, config)
    persona = _build_persona_recassist(entry)

    rs1_chat = make_chat_fn(config.rs1_llm_config or {})
    rs2_chat = make_chat_fn(config.rs2_llm_config or {})
    user_chat = make_chat_fn(config.user_llm_config or {})

    # Inject ALL GT into both RSs (same as advisor condition for fair comparison)
    rs1_gt: List[str] = list(ground_truth) if config.inject_gt and ground_truth else []
    rs2_gt: List[str] = list(ground_truth) if config.inject_gt and ground_truth else []

    rs1 = ConversationalRS("RS1", rs1_chat, domain=config.domain, gt_titles=rs1_gt, persona_key="rs1")
    rs2 = ConversationalRS("RS2", rs2_chat, domain=config.domain, gt_titles=rs2_gt, persona_key="rs2")

    # Baseline simulated user (no advisor messages)
    sim_user = SimulatedUser(
        chat_fn=user_chat,
        instruction=user_query,
        persona=persona,
        n_choices=n_choices,
        domain=config.domain,
        max_turns=config.max_turns,
        rs1_name=rs1.name,
        rs2_name=rs2.name,
    )

    # Round 0
    print(f"  [Baseline Round 0] Querying {rs1.name} and {rs2.name}...")
    rs1_resp = rs1.respond(user_query)
    rs2_resp = rs2.respond(user_query)

    # GT repair: same post-processing as advisor condition for fair comparison
    if config.inject_gt and ground_truth:
        repair_fn = make_prompt_fn(config.advisor_llm_config or {})
        rs1_resp = rs1.repair_with_gt(rs1_resp, repair_fn)
        rs2_resp = rs2.repair_with_gt(rs2_resp, repair_fn)

    no_advisor_msg = (
        "You've received recommendations from both systems above. "
        "Feel free to ask either system follow-up questions, or make your decision."
    )

    turns: List[TurnLog] = []
    turns.append(TurnLog(
        turn=0,
        action="no_advisor",
        advisor_message=no_advisor_msg,
        user_message=user_query,
        user_target="",
        rs_responses=_normalize_rs_responses(
            rs1.name, rs2.name, {rs1.name: rs1_resp, rs2.name: rs2_resp},
        ),
        belief_state={},
        items_mentioned=[],
    ))

    chosen_items: List[str] = []
    for t in range(1, config.max_turns + 1):
        summary = _build_conversation_summary(rs1, rs2, [], [])
        target, user_msg = sim_user.respond(no_advisor_msg, summary)
        print(f"  [Baseline Round {t}] User → {target}: {user_msg[:80]}...")

        if target.lower() == "decision":
            chosen_items = _extract_chosen_from_message(
                user_msg, rs1.get_mentioned_titles() + rs2.get_mentioned_titles(), n_choices,
            )
            turns.append(TurnLog(
                turn=t,
                action="no_advisor_decision",
                advisor_message="",
                user_message=user_msg,
                user_target="decision",
                rs_responses=_normalize_rs_responses(rs1.name, rs2.name, {}),
                belief_state={},
                items_mentioned=chosen_items,
            ))
            break

        if target == rs2.name:
            rs_resp = rs2.respond(user_msg)
        else:
            rs_resp = rs1.respond(user_msg)

        turns.append(TurnLog(
            turn=t,
            action="no_advisor",
            advisor_message=no_advisor_msg,
            user_message=user_msg,
            user_target=target,
            rs_responses=_normalize_rs_responses(
                rs1.name, rs2.name, {target: rs_resp},
            ),
            belief_state={},
            items_mentioned=[],
        ))
    else:
        if not chosen_items:
            all_titles = rs1.get_mentioned_titles() + rs2.get_mentioned_titles()
            chosen_items = all_titles[:n_choices]

    return SessionLog(
        user_id=user_id,
        condition="no_advisor",
        turns=turns,
        chosen_items=chosen_items,
        ground_truth=ground_truth,
        rs1_gt_items=rs1_gt,
        rs2_gt_items=rs2_gt,
        rs1_history=rs1.history,
        rs2_history=rs2.history,
        final_state=None,
        initial_user_message=user_query,
        shared_relationships=entry.get("sharedRelationships", []),
        n_choices=n_choices,
    )


def _extract_chosen_from_message(
    message: str, all_titles: List[str], n: int,
) -> List[str]:
    """Extract up to n chosen titles from user decision message.

    Uses three passes (strict → substring → fuzzy) so small rephrasing
    or case differences don't cause a miss.
    """
    from difflib import SequenceMatcher

    msg_lower = message.lower()
    msg_stripped = re.sub(r"[^\w\s]", "", msg_lower)
    found: List[str] = []
    seen: set = set()

    def _add(title: str):
        tl = title.lower()
        if tl not in seen:
            found.append(title)
            seen.add(tl)

    for t in all_titles:
        if t.lower() in msg_lower:
            _add(t)

    if len(found) < (n or len(all_titles)):
        for t in all_titles:
            tl = t.lower()
            if tl in seen:
                continue
            ts = re.sub(r"[^\w\s]", "", tl)
            if ts and ts in msg_stripped:
                _add(t)

    if len(found) < (n or len(all_titles)):
        for t in all_titles:
            tl = t.lower()
            if tl in seen:
                continue
            ts = re.sub(r"[^\w\s]", "", tl)
            if ts and SequenceMatcher(None, ts, msg_stripped).ratio() >= 0.75:
                _add(t)

    return found[:n] if n else found


# ─── Main runner ──────────────────────────────────────────────

def _get_completed_user_ids(output_dir: str, condition: str) -> set:
    completed = set()
    if not os.path.isdir(output_dir):
        return completed
    suffix = f"_{condition}.json"
    for fname in os.listdir(output_dir):
        if fname.endswith(suffix):
            completed.add(fname[: -len(suffix)])
    return completed


def _save_log(log: Optional[SessionLog], output_dir: str, user_id: str, condition: str):
    if log is None:
        return
    path = os.path.join(output_dir, f"{user_id}_{condition}.json")
    with open(path, "w") as f:
        json.dump(log.to_dict(), f, indent=2, ensure_ascii=False)


def run_experiment(
    config: ExperimentConfig,
    condition: str = "advisor",
    indices: Optional[List[int]] = None,
    training_phase: bool = False,
    resume: bool = True,
) -> Dict:
    print(f"[Experiment] Dataset: {config.dataset_path} ({config.dataset_type}, {config.domain})")

    all_entries = _load_recassistbench(config.dataset_path)
    n_total = len(all_entries)

    if indices is None:
        random.seed(config.seed)
        n_sample = min(config.n_users, n_total)
        indices = random.sample(range(n_total), n_sample)
    else:
        n_sample = len(indices)

    print(f"[Experiment] {n_sample} sessions, condition={condition}")
    os.makedirs(config.output_dir, exist_ok=True)

    completed_ids = set()
    if resume:
        completed_ids = _get_completed_user_ids(config.output_dir, condition)
        if completed_ids:
            print(f"[Resume] {len(completed_ids)} already done — skipping")

    logs: List[SessionLog] = []
    sessions_run = 0

    for i, idx in enumerate(indices):
        entry = all_entries[idx]
        user_id = f"user_{entry.get('data_idx', idx)}"
        user_msg = _build_user_message_recassist(entry, config.query_field)

        if resume and user_id in completed_ids:
            print(f"[Resume] Skipping {user_id}")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{n_sample}] {user_id} — {condition}")
        print(f"{'='*60}")

        gt = _extract_ground_truth(entry, config)
        log = None
        try:
            if condition == "no_advisor":
                log = _run_no_advisor_session(user_id, entry, user_msg, config, gt)
            else:
                log = _run_advisor_session(
                    user_id, entry, user_msg, config, gt, training_phase=training_phase,
                )
            logs.append(log)
            sessions_run += 1
            items = ", ".join(log.chosen_items) or "none"
            print(f"[Done] {len(log.turns)} turns, chose: {items}")
        except Exception as e:
            print(f"[FAILED] {e}")
            traceback.print_exc()

        _save_log(log, config.output_dir, user_id, condition)

    summary = {f"n_{condition}": len(logs), "sessions_run": sessions_run}
    summary_path = os.path.join(config.output_dir, f"{condition}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary
