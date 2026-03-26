import os
import json
import random

from agent_messages import TextMessage
from agents import Agent, Module, Orchestrator, User
from agents.types import Response

from typing import Optional, Union, List, Dict, Generator, Tuple, Any


def _advisor_arara_response(text: str, advisor: Agent, orchestrator: Optional[Agent]) -> Response:
    """Wrap advisor text as ARARA Response so Orchestrator.run_chat does not double-send.

    If we yield a plain str, run_chat calls send() inside the generate_reply loop and
    again after the loop (arara orchestrator.py), duplicating console output.
    """
    recv = orchestrator if orchestrator is not None else advisor
    return Response(
        chat_message=TextMessage(content=text, sender=advisor, receiver=recv),
        to_reply=False,
    )

from components.triangulation import TriangulationResult, triangulate
from components.belief_state import (
    UserBeliefState,
    create_initial_state,
    update_belief_state,
    compute_context_vector,
)
from components.policy import ActionType, LinUCBPolicy, CONTEXT_KEYS
from components.response_generator import generate_response
from components.reward import compute_turn_reward
from components.logger import TurnLogger
from gt_inject import inject_random_gt_into_tri_data


class Advisor(Agent):

    DEFAULT_SYSTEM_MESSAGE = """
        You are a friendly advisor helping someone find what they're looking for.
        You have access to suggestions from multiple recommendation systems.
        Your role is to explore these options together with the user through
        natural conversation — asking about their tastes, showing a couple of
        options at a time, and helping them figure out what appeals to them.
        You suggest, you don't decide for them.
    """

    def __init__(
        self,
        name: Optional[str] = "advisor",
        system_message: Optional[Union[str, List]] = DEFAULT_SYSTEM_MESSAGE,
        bandit_model_path: Optional[str] = None,
        bandit_log_path: Optional[str] = None,
        max_turns: int = 6,
        session_id: str = "",
        ground_truth: Optional[List[str]] = None,
        training_phase: bool = False,
        inject_random_gt_in_pool: bool = False,
        inject_gt_rng: Optional[Any] = None,
        inject_gt_query: str = "",
        inject_gt_shared_relationships: Optional[List] = None,
        n_choices: int = 1,
        domain: str = "book",
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            **kwargs,
        )

        self._domain = domain

        if bandit_model_path:
            self.policy = LinUCBPolicy.load(bandit_model_path)
        else:
            self.policy = LinUCBPolicy()

        self.max_turns = max_turns
        self.triangulation: Optional[TriangulationResult] = None
        self.state: Optional[UserBeliefState] = None
        self._last_advisor_message: str = ""
        self._last_items_considered_for_internal_turn_state: List[str] = []
        self._user_decided: bool = False
        self._chosen_item: str = ""
        self._session_log: List[Dict[str, Any]] = []
        self._session_id: str = session_id
        self._logger: Optional[TurnLogger] = (
            TurnLogger(bandit_log_path) if bandit_log_path else None
        )

        self._ground_truth: List[str] = list(ground_truth or [])
        self._training_phase: bool = training_phase
        self._gt_injected: set = set()
        self._inject_random_gt_in_pool: bool = inject_random_gt_in_pool
        self._inject_gt_rng: Optional[Any] = inject_gt_rng
        self._inject_gt_query: str = inject_gt_query
        self._inject_gt_shared_relationships: Optional[List] = inject_gt_shared_relationships
        self._n_choices: int = max(1, n_choices)
        self._chosen_items: List[str] = []

        self.unregister_reply_func(Agent._generate_oai_reply)
        self.register_reply(Agent, Advisor.process)

    # ─── LLM wrapper ──────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """
        Adapter: ARARA's client.create() → plain string.
        Used by the components (belief_state, response_generator, debiasing)
        which expect Callable[[str], str].
        """
        messages = [{"role": "user", "content": prompt}]
        result = self.client.create(
            messages=messages,
            cache=self.client_cache,
            Agent=self,
        )
        if isinstance(result.content, str):
            return result.content
        return str(result.content)

    # ─── Payload parsing ──────────────────────────────────────

    def _parse_payload(self, content: Union[str, Dict]) -> Dict:
        """Parse a JSON payload that may arrive as string or dict."""
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            return {}
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
            return parsed
        except json.JSONDecodeError:
            return {}

    def _parse_recsys_for_triangulation(self, payload: Dict) -> Dict[str, List[dict]]:
        """
        Convert the aggregated parallel_execution output to the format
        expected by triangulate():
            { "llm_name": [{"title": str, "explanation": str, "rank": int}, ...] }
        """
        result = {}
        for agent_name, agent_data in payload.items():
            if isinstance(agent_data, str):
                agent_data = self._parse_payload(agent_data)
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
                    rank = rec.get("rank", 0)
                    if title:
                        items.append({"title": title, "explanation": explanation, "rank": rank})
            if items:
                result[agent_name] = items
        return result

    # ─── Session lifecycle ────────────────────────────────────

    def initialize_session(
        self,
        recsys_outputs: Dict[str, List[dict]],
        session_id: str = "",
    ) -> None:
        """Run triangulation on the N LLM outputs and create initial belief state."""
        self.triangulation = triangulate(recsys_outputs)
        self.state = create_initial_state()
        self._user_decided = False
        self._chosen_item = ""
        self._chosen_items = []
        self._last_advisor_message = ""
        self._last_items_considered_for_internal_turn_state = []
        self._session_log = []
        self._already_recapped = False
        if session_id:
            self._session_id = session_id
        self.policy.reset()

    def _get_unshown_gt(self) -> Optional[str]:
        """Return a GT title that hasn't been shown/injected yet, if any."""
        pool_lower = set()
        if self.state:
            pool_lower = {t.lower() for t in self.state.items_in_session_internal_pool}
        for gt in self._ground_truth:
            gt_lower = gt.strip().lower()
            if gt_lower not in pool_lower and gt_lower not in self._gt_injected:
                return gt
        return None

    def first_turn(self) -> Tuple[str, ActionType, UserBeliefState, List[str]]:
        """Generate the first advisor message (before any user response)."""
        context = compute_context_vector(
            self.state, self.triangulation,
            shared_relationships=self._inject_gt_shared_relationships,
        )

        unshown_gt = self._get_unshown_gt()
        gt_available = unshown_gt is not None

        action = self.policy.select(
            context,
            user_decided=False,
            allow_gt_actions=self._training_phase,
            gt_available=gt_available,
        )

        gt_item_to_inject = ""
        if action == ActionType.INJECT_GT_PROBE and unshown_gt:
            gt_item_to_inject = unshown_gt
            self._gt_injected.add(unshown_gt.strip().lower())

        response, items_considered = generate_response(
            action=action,
            belief_state=self.state,
            triangulation=self.triangulation,
            llm_client=self._call_llm,
            gt_item_to_inject=gt_item_to_inject,
            user_query=self._inject_gt_query,
            shared_relationships=self._inject_gt_shared_relationships,
            n_choices=self._n_choices,
        )

        self._last_advisor_message = response
        self._last_items_considered_for_internal_turn_state = items_considered
        return response, action, self.state, items_considered

    def process_conversation_turn(
        self, user_response: str
    ) -> Tuple[str, ActionType, UserBeliefState, List[str]]:
        """
        One conversational turn:
          1. Snapshot sigma/omega before update
          2. Update belief state
          3. Detect decision
          4. Select action via policy
          5. Generate advisor response
          6. Compute reward → update policy → log tuple
        """
        sigma_before = self.state.preference_specificity
        omega_before = self.state.overload_risk
        kt_coverage_before = compute_context_vector(
            self.state, self.triangulation,
            shared_relationships=self._inject_gt_shared_relationships,
        ).get("key_term_pool_coverage", 0.0)

        self.state = update_belief_state(
            state=self.state,
            user_response=user_response,
            advisor_message=self._last_advisor_message,
            items_considered_for_internal_turn_state=self._last_items_considered_for_internal_turn_state,
            llm_client=self._call_llm,
        )

        self._detect_final_user_decision(user_response)

        # If user decided but chose fewer items than expected, redirect to
        # SYNTHESIZE_DECISION so the advisor recapitulates all items and
        # asks for the full shortlist.  This prevents premature END_SESSION
        # when the level-0 user only names 1-2 titles out of n_choices.
        insufficient_choices = (
            self._user_decided
            and len(self._chosen_items) < self._n_choices
            and not getattr(self, "_already_recapped", False)
        )
        if insufficient_choices:
            self._user_decided = False
            self._already_recapped = True

        context = compute_context_vector(
            self.state, self.triangulation,
            shared_relationships=self._inject_gt_shared_relationships,
        )

        unshown_gt = self._get_unshown_gt()
        gt_available = unshown_gt is not None

        if insufficient_choices:
            action = ActionType.SYNTHESIZE_DECISION
        else:
            action = self.policy.select(
                context,
                user_decided=self._user_decided,
                allow_gt_actions=self._training_phase,
                gt_available=gt_available,
            )

        gt_item_to_inject = ""
        if action == ActionType.INJECT_GT_PROBE and unshown_gt:
            gt_item_to_inject = unshown_gt
            self._gt_injected.add(unshown_gt.strip().lower())

        if action == ActionType.EXPAND_POOL:
            self._expand_pool()

        response, items_considered = generate_response(
            action=action,
            belief_state=self.state,
            triangulation=self.triangulation,
            llm_client=self._call_llm,
            chosen_item=", ".join(self._chosen_items) if self._chosen_items else self._chosen_item,
            gt_item_to_inject=gt_item_to_inject,
            user_query=self._inject_gt_query,
            shared_relationships=self._inject_gt_shared_relationships,
            n_choices=self._n_choices,
        )

        has_divergent = self._items_have_divergent(items_considered)
        all_single = self._items_all_single_source(items_considered)

        kt_coverage_after = context.get("key_term_pool_coverage", 0.0)

        reward = compute_turn_reward(
            user_decided=self._user_decided,
            sigma_before=sigma_before,
            sigma_after=self.state.preference_specificity,
            omega_before=omega_before,
            omega_after=self.state.overload_risk,
            items_considered_for_internal_turn_state=items_considered,
            has_divergent_items=has_divergent,
            all_single_source=all_single,
            turn=self.state.turn,
            max_turns=self.max_turns,
            action=action,
            key_term_coverage_before=kt_coverage_before,
            key_term_coverage_after=kt_coverage_after,
        )

        self.policy.update(action, context, reward)

        if self._logger:
            ctx_list = [context.get(k, 0.0) for k in CONTEXT_KEYS]
            self._logger.log(
                session_id=self._session_id,
                turn=self.state.turn,
                context=ctx_list,
                action=action.value,
                reward=reward,
            )

        self._last_advisor_message = response
        self._last_items_considered_for_internal_turn_state = items_considered
        return response, action, self.state, items_considered

    _EXPAND_POOL_PROMPT = """You are helping find additional {item_word} recommendations.

USER'S ORIGINAL QUERY: {query}
KEY ATTRIBUTES: {key_terms}
USER'S PREFERENCES SO FAR: {prefs}
TITLES ALREADY IN THE POOL (do NOT repeat these): {existing}

Generate exactly 5 additional {item_word} titles that match the query and key attributes.
IMPORTANT:
- Include lesser-known and niche works, not just bestsellers.
- Each title must be a REAL published {item_word} — do NOT invent titles.
- If the query mentions a specific author, ALL 5 must be by that author.
- If the query mentions a category/topic, ALL 5 must fit that category.

Return ONLY a JSON list of objects: [{{"title": "...", "explanation": "..."}}]
No other text."""

    def _expand_pool(self) -> None:
        """Use the advisor LLM to generate additional titles and merge into triangulation."""
        if not self.triangulation:
            return

        existing = sorted({
            it.get("title", "").strip()
            for it in self.triangulation.all_items
            if it.get("title")
        })
        rels = self._inject_gt_shared_relationships or []
        kt_parts = []
        for rel in rels:
            if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                kt_parts.append(f"{rel[0]}: {rel[1]}")
        key_terms_str = ", ".join(kt_parts) or "(none)"

        prefs = json.dumps(self.state.preference_dimensions) if self.state.preference_dimensions else "(none)"
        item_word = "book"
        if self._inject_gt_query and ("movie" in self._inject_gt_query.lower()):
            item_word = "movie"

        prompt = self._EXPAND_POOL_PROMPT.format(
            item_word=item_word,
            query=self._inject_gt_query or "(unknown)",
            key_terms=key_terms_str,
            prefs=prefs,
            existing=", ".join(existing[:20]),
        )

        raw = self._call_llm(prompt)

        try:
            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                lines = raw_clean.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw_clean = "\n".join(lines)
            new_items = json.loads(raw_clean)
        except (json.JSONDecodeError, ValueError):
            return

        if not isinstance(new_items, list):
            return

        existing_lower = {t.strip().lower() for t in existing}
        added = 0
        for item in new_items:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            if not title or title.lower() in existing_lower:
                continue
            new_entry = {
                "title": title,
                "explanation": item.get("explanation", ""),
                "rank": 99,
                "sources": ["advisor_expand"],
            }
            self.triangulation.all_items.append(new_entry)
            self.triangulation.divergent_items.setdefault("advisor_expand", []).append(new_entry)
            existing_lower.add(title.lower())
            added += 1

        if added:
            print(f"[Advisor] EXPAND_POOL: added {added} new titles to pool")

    def _items_have_divergent(self, items: List[str]) -> bool:
        if not self.triangulation or not items:
            return False
        divergent_titles = set()
        for lst in self.triangulation.divergent_items.values():
            for it in lst:
                divergent_titles.add(it.get("title", "").strip().lower())
        return any(t.strip().lower() in divergent_titles for t in items)

    def _items_all_single_source(self, items: List[str]) -> bool:
        if not self.triangulation or not items:
            return False
        consensus_titles = {
            it.get("title", "").strip().lower()
            for it in self.triangulation.consensus_items
        }
        return not any(t.strip().lower() in consensus_titles for t in items)

    _DECISION_PROMPT = """You are analyzing a user's message in a recommendation conversation.

USER MESSAGE:
{user_response}

ITEMS THAT EXIST IN THIS CONVERSATION (ONLY these are valid choices):
{all_items_shown}

ADVISOR'S LAST MESSAGE TO THE USER:
{last_advisor_message}

The user was asked to pick {n_choices} item(s). Has the user made a final, explicit decision?

Return ONLY valid JSON, no extra text:
{{"decided": true, "chosen_items": ["title1", "title2"]}}
or:
{{"decided": false, "chosen_items": []}}

Rules:
- "decided" is true if the user explicitly commits to specific items (e.g. "I'll go with X and Y", "My picks are X, Y", "I choose X")
- The user should pick {n_choices} items, but any number >= 1 counts as decided
- CRITICAL: chosen_items MUST only contain titles from the "ITEMS THAT EXIST" list above.
  If the user mentions a title NOT in that list, ignore it.
- Match user mentions to the closest title in the list (fuzzy match is fine).
- If the user says something like "that one", "this one", or "that book", infer which item from the advisor's last message
- If the user is still asking questions, expressing preferences without choosing, or being vague → decided is false
- Phrases like "I'll go with X", "I'll stick with X", "I choose X" count as a final decision"""

    def _detect_final_user_decision(self, user_response: str) -> None:
        """
        Use a deterministic LLM call (temp=0) to detect whether the user
        has made a final decision and extract item title(s).
        Uses ALL items shown across the conversation so the LLM can
        resolve references like "that one" or items mentioned in earlier turns.

        Post-validation: each returned title must fuzzy-match a real
        item from the triangulation or the session pool.
        """
        all_items: List[str] = list(self._last_items_considered_for_internal_turn_state or [])
        if self.state:
            for title in self.state.items_in_session_internal_pool:
                normalized = title.strip().lower()
                if not any(existing.strip().lower() == normalized for existing in all_items):
                    all_items.append(title)

        real_titles_lower: set = set()
        if self.triangulation:
            for it in self.triangulation.all_items:
                t = it.get("title", "").strip().lower()
                if t:
                    real_titles_lower.add(t)
        for t in all_items:
            real_titles_lower.add(t.strip().lower())

        prompt = self._DECISION_PROMPT.format(
            user_response=user_response,
            all_items_shown=json.dumps(all_items) if all_items else "[]",
            last_advisor_message=self._last_advisor_message[:500] if self._last_advisor_message else "(none)",
            n_choices=self._n_choices,
        )
        try:
            raw = self._call_llm(prompt)
            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                lines = raw_clean.split("\n")
                raw_clean = "\n".join(l for l in lines if not l.strip().startswith("```"))
            result = json.loads(raw_clean)
            if result.get("decided") is True:
                titles = result.get("chosen_items") or []
                if isinstance(titles, str):
                    titles = [titles]
                # Legacy single-item field
                if not titles and result.get("chosen_item"):
                    titles = [result["chosen_item"]]
                validated: List[str] = []
                for title in titles:
                    if not title or title.lower() in ("none", "null", ""):
                        continue
                    chosen_lower = title.strip().lower()
                    is_real = any(
                        chosen_lower in rt or rt in chosen_lower
                        for rt in real_titles_lower
                    )
                    if is_real:
                        validated.append(title)
                if validated:
                    self._user_decided = True
                    self._chosen_items = validated
                    self._chosen_item = validated[0]
        except (json.JSONDecodeError, ValueError, Exception):
            pass

    @property
    def user_decided(self) -> bool:
        return self._user_decided

    @property
    def chosen_item(self) -> str:
        return self._chosen_item

    @property
    def chosen_items(self) -> List[str]:
        if self._chosen_items:
            return list(self._chosen_items)
        if self._chosen_item and self._chosen_item != "none":
            return [self._chosen_item]
        return []

    # ─── ARARA process() — entry point for the Orchestrator ──

    def process(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Module] = None,
    ) -> Generator[Tuple[bool, Optional[Union[str, Response]]], None, None]:
        """
        Dispatch based on session state.

        New flow: User → Recsys → User (reacts) → Advisor → User → Advisor …
        When the advisor is called for the first time the recsys output is
        somewhere earlier in the conversation history (not the last message).
        We find it, initialise triangulation, then handle the user's reaction
        as the first conversational turn.
        """
        if messages is None or not messages:
            err = json.dumps({"error": "No messages received."})
            yield [(True, _advisor_arara_response(err, self, sender))]
            return
        if sender is None:
            err = json.dumps({"error": "Sender not provided."})
            yield [(True, _advisor_arara_response(err, self, sender))]
            return

        if self.triangulation is None:
            yield from self._handle_first_advisor_turn(messages, sender=sender)
        else:
            yield from self._handle_conversational_turn(messages, sender=sender)

    # ─── Message handlers ─────────────────────────────────────

    def _find_recsys_payload(self, messages: List[Dict]) -> Optional[Dict]:
        """Scan conversation history backwards to find the recsys JSON payload."""
        for msg in reversed(messages):
            name = msg.get("name", "")
            if name in ("recsys_orchestrator", "aggregator"):
                payload = self._parse_payload(msg.get("content"))
                if payload and isinstance(payload, dict):
                    return payload
        return None

    def _handle_first_advisor_turn(
        self, messages: List[Dict], sender: Optional[Agent] = None,
    ) -> Generator[Tuple[bool, Optional[Union[str, Response]]], None, None]:
        """
        First time the advisor speaks.
        The recsys output is earlier in the conversation; the last message is
        the user's confused reaction.  Initialise triangulation, then respond
        to the user's reaction with a gentle exploration (not a dump of items).
        """
        payload = self._find_recsys_payload(messages)
        if not payload:
            err = json.dumps({"error": "Could not find recsys output in conversation history."})
            yield [(True, _advisor_arara_response(err, self, sender))]
            return

        tri_data = self._parse_recsys_for_triangulation(payload)
        if not tri_data:
            err = json.dumps({"error": "Could not extract recommendations for triangulation."})
            yield [(True, _advisor_arara_response(err, self, sender))]
            return

        if self._inject_random_gt_in_pool and self._ground_truth:
            rng = self._inject_gt_rng or random.Random(0)
            tri_data = inject_random_gt_into_tri_data(
                tri_data, self._ground_truth, rng,
                query=self._inject_gt_query,
                shared_relationships=self._inject_gt_shared_relationships,
                domain=self._domain,
            )

        self.initialize_session(tri_data)

        user_reaction = messages[-1].get("content", "")

        response_text, action, state, items_considered = self.first_turn()

        self._session_log.append({
            "turn": 0,
            "action": action.value,
            "advisor_message": response_text,
            "user_response": user_reaction,
            "belief_state": state.to_dict(),
            "items_considered_for_internal_turn_state": items_considered,
        })

        yield [(True, _advisor_arara_response(response_text, self, sender))]

    def _handle_conversational_turn(
        self, messages: List[Dict], sender: Optional[Agent] = None,
    ) -> Generator[Tuple[bool, Optional[Union[str, Response]]], None, None]:
        """Receive user reply → update belief → select action → respond."""
        user_response = messages[-1].get("content", "")
        response_text, action, state, items_considered = self.process_conversation_turn(user_response)

        self._session_log.append({
            "turn": state.turn,
            "action": action.value,
            "advisor_message": response_text,
            "user_response": user_response,
            "belief_state": state.to_dict(),
            "items_considered_for_internal_turn_state": items_considered,
        })

        yield [(True, _advisor_arara_response(response_text, self, sender))]


# ─── Factory ──────────────────────────────────────────────────

def create_advisor(
    llm_config: Optional[Dict] = None,
    bandit_model_path: Optional[str] = None,
    bandit_log_path: Optional[str] = None,
    max_turns: int = 6,
    session_id: str = "",
    ground_truth: Optional[List[str]] = None,
    training_phase: bool = False,
    inject_random_gt_in_pool: bool = False,
    inject_gt_rng: Optional[Any] = None,
    inject_gt_query: str = "",
    inject_gt_shared_relationships: Optional[List] = None,
    n_choices: int = 1,
    domain: str = "book",
    **kwargs,
) -> Advisor:
    """
    Create an Advisor agent for the conversational experiment.

    Parameters
    ----------
    llm_config : dict, optional
        Custom LLM configuration. If None, uses get_advisor_config() (model/temp set in config.llm_clients).
    bandit_model_path : str, optional
        Path to a pre-trained LinUCB model (.npz). Starts fresh if not provided.
    bandit_log_path : str, optional
        Path where (context, action, reward) tuples are logged as JSONL.
    max_turns : int
        Maximum turns per session (used in reward computation).
    session_id : str
        Identifier for this session (e.g. "user_0"). Used in bandit logs.
    ground_truth : list of str, optional
        Ground truth titles for this session (only used during training).
    training_phase : bool
        If True, enables training-only actions like INJECT_GT_PROBE.
    **kwargs
        Forwarded to the Advisor constructor.
    """
    if llm_config is None:
        from config.llm_clients import get_advisor_config
        llm_config = get_advisor_config()

    return Advisor(
        llm_config=llm_config,
        system_message=Advisor.DEFAULT_SYSTEM_MESSAGE,
        is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content", "") or ""),
        bandit_model_path=bandit_model_path,
        bandit_log_path=bandit_log_path,
        max_turns=max_turns,
        session_id=session_id,
        ground_truth=ground_truth,
        training_phase=training_phase,
        inject_random_gt_in_pool=inject_random_gt_in_pool,
        inject_gt_rng=inject_gt_rng,
        inject_gt_query=inject_gt_query,
        inject_gt_shared_relationships=inject_gt_shared_relationships,
        n_choices=n_choices,
        domain=domain,
        **kwargs,
    )
