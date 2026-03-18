import os
import json

from agents import Agent, Module, Orchestrator, User
from utils import get_llm_config

from typing import Optional, Union, List, Dict, Generator, Tuple, Any

from components.triangulation import TriangulationResult, triangulate
from components.belief_state import (
    UserBeliefState,
    create_initial_state,
    update_belief_state,
    compute_context_vector,
)
from components.action_selector import ActionType, HeuristicPolicy
from components.response_generator import generate_response


class Advisor(Agent):

    DEFAULT_SYSTEM_MESSAGE = """
        You are an Advisor that helps users make better decisions about recommendations.
        You analyze recommendations from multiple systems, compare their outputs,
        and help users articulate their preferences to make an informed choice.
    """

    def __init__(
        self,
        name: Optional[str] = "advisor",
        system_message: Optional[Union[str, List]] = DEFAULT_SYSTEM_MESSAGE,
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            **kwargs,
        )

        self.policy = HeuristicPolicy()
        self.triangulation: Optional[TriangulationResult] = None
        self.state: Optional[UserBeliefState] = None
        self._last_advisor_message: str = ""
        self._last_items_shown: List[str] = []
        self._user_decided: bool = False
        self._chosen_item: str = ""
        self._session_log: List[Dict[str, Any]] = []

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
                    title = rec.get("book_title") or rec.get("title") or ""
                    explanation = rec.get("explanation", "")
                    rank = rec.get("rank", 0)
                    if title:
                        items.append({"title": title, "explanation": explanation, "rank": rank})
            if items:
                result[agent_name] = items
        return result

    # ─── Session lifecycle ────────────────────────────────────

    def initialize_session(self, recsys_outputs: Dict[str, List[dict]]) -> None:
        """Run triangulation on the N LLM outputs and create initial belief state."""
        self.triangulation = triangulate(recsys_outputs)
        self.state = create_initial_state()
        self._user_decided = False
        self._chosen_item = ""
        self._last_advisor_message = ""
        self._last_items_shown = []
        self._session_log = []

    def first_turn(self) -> Tuple[str, ActionType, UserBeliefState, List[str]]:
        """Generate the first advisor message (before any user response)."""
        context = compute_context_vector(self.state, self.triangulation)
        action = self.policy.select(context, user_decided=False)

        response, items_shown = generate_response(
            action=action,
            belief_state=self.state,
            triangulation=self.triangulation,
            llm_client=self._call_llm,
        )

        self._last_advisor_message = response
        self._last_items_shown = items_shown
        return response, action, self.state, items_shown

    def process_conversation_turn(
        self, user_response: str
    ) -> Tuple[str, ActionType, UserBeliefState, List[str]]:
        """
        One conversational turn:
          1. Update belief state
          2. Detect decision
          3. Select action (heuristic policy)
          4. Generate advisor response
        """
        self.state = update_belief_state(
            state=self.state,
            user_response=user_response,
            advisor_message=self._last_advisor_message,
            items_shown=self._last_items_shown,
            llm_client=self._call_llm,
        )

        self._detect_decision(user_response)

        context = compute_context_vector(self.state, self.triangulation)
        action = self.policy.select(context, user_decided=self._user_decided)

        response, items_shown = generate_response(
            action=action,
            belief_state=self.state,
            triangulation=self.triangulation,
            llm_client=self._call_llm,
            chosen_item=self._chosen_item,
        )

        self._last_advisor_message = response
        self._last_items_shown = items_shown
        return response, action, self.state, items_shown

    _DECISION_PROMPT = """You are analyzing a user's message in a recommendation conversation.

USER MESSAGE:
{user_response}

ITEMS THAT HAVE BEEN DISCUSSED IN THIS CONVERSATION (may not be exhaustive):
{all_items_shown}

ADVISOR'S LAST MESSAGE TO THE USER:
{last_advisor_message}

Has the user made a final, explicit decision to choose one specific item?

Return ONLY valid JSON, no extra text:
{{"decided": true, "chosen_item": "the full title the user chose"}}
or:
{{"decided": false, "chosen_item": null}}

Rules:
- "decided" is true if the user explicitly commits to one specific item (e.g. "I'll go with X", "I choose X", "I'll take X", "I'll stick with X")
- The user may choose an item that was MENTIONED in the advisor's message but is NOT in the items list — that is still a valid decision
- Extract the full book/item title as accurately as possible from the user's message or the advisor's last message
- If the user says something like "that one", "this one", or "that book", infer which item from the advisor's last message
- If the user is still asking questions, expressing preferences without choosing, or being vague → decided is false
- Phrases like "I'll go with X", "I'll stick with X", "I choose X" count as a final decision"""

    def _detect_decision(self, user_response: str) -> None:
        """
        Use a deterministic LLM call (temp=0) to detect whether the user
        has made a final decision and extract the exact item title.
        Uses ALL items shown across the conversation so the LLM can
        resolve references like "that one" or items mentioned in earlier turns.
        """
        # Combine items from ALL turns (items_seen stores lowercase titles — use
        # _last_items_shown for the most recent turn to preserve original casing,
        # then fill in from items_seen for earlier turns)
        all_items: List[str] = list(self._last_items_shown or [])
        if self.state:
            for title in self.state.items_seen:
                normalized = title.strip().lower()
                if not any(existing.strip().lower() == normalized for existing in all_items):
                    all_items.append(title)

        prompt = self._DECISION_PROMPT.format(
            user_response=user_response,
            all_items_shown=json.dumps(all_items) if all_items else "[]",
            last_advisor_message=self._last_advisor_message[:500] if self._last_advisor_message else "(none)",
        )
        try:
            raw = self._call_llm(prompt)
            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                lines = raw_clean.split("\n")
                raw_clean = "\n".join(l for l in lines if not l.strip().startswith("```"))
            result = json.loads(raw_clean)
            if result.get("decided") is True:
                title = result.get("chosen_item") or ""
                if title and title.lower() not in ("none", "null", ""):
                    self._user_decided = True
                    self._chosen_item = title
        except (json.JSONDecodeError, ValueError, Exception):
            pass

    @property
    def user_decided(self) -> bool:
        return self._user_decided

    @property
    def chosen_item(self) -> str:
        return self._chosen_item

    # ─── ARARA process() — entry point for the Orchestrator ──

    def process(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Module] = None,
    ) -> Generator[Tuple[bool, Optional[str]], None, None]:
        """
        Dispatch by the `name` field of the last message
        (which identifies the real sender inside the Module).
        """
        if messages is None or not messages:
            yield [(True, json.dumps({"error": "No messages received."}))]
            return
        if sender is None:
            yield [(True, json.dumps({"error": "Sender not provided."}))]
            return

        last_sender_name = messages[-1].get("name", "")
        if last_sender_name in ("recsys_orchestrator", "aggregator"):
            yield from self._handle_recsys_response(messages)
        elif self.triangulation is not None:
            yield from self._handle_conversational_turn(messages)
        else:
            yield [(True, json.dumps({"error": f"Unexpected message from {last_sender_name}"}))]

    # ─── Message handlers ─────────────────────────────────────

    def _handle_recsys_response(
        self, messages: List[Dict],
    ) -> Generator[Tuple[bool, Optional[str]], None, None]:
        """Receive aggregated RecSys output → triangulate → first message."""
        payload = self._parse_payload(messages[-1].get("content"))

        if not payload or not isinstance(payload, dict):
            yield [(True, json.dumps({"error": "Payload must be a dict mapping agent names to outputs."}))]
            return

        tri_data = self._parse_recsys_for_triangulation(payload)
        if not tri_data:
            yield [(True, json.dumps({"error": "Could not extract recommendations for triangulation."}))]
            return

        self.initialize_session(tri_data)
        response_text, action, state, items_shown = self.first_turn()

        self._session_log.append({
            "turn": 0,
            "action": action.value,
            "advisor_message": response_text,
            "user_response": "",
            "belief_state": state.to_dict(),
            "items_shown": items_shown,
        })

        yield [(True, response_text)]

    def _handle_conversational_turn(
        self, messages: List[Dict],
    ) -> Generator[Tuple[bool, Optional[str]], None, None]:
        """Receive user reply → update belief → select action → respond."""
        user_response = messages[-1].get("content", "")
        response_text, action, state, items_shown = self.process_conversation_turn(user_response)

        self._session_log.append({
            "turn": state.turn,
            "action": action.value,
            "advisor_message": response_text,
            "user_response": user_response,
            "belief_state": state.to_dict(),
            "items_shown": items_shown,
        })

        yield [(True, response_text)]


# ─── Factory ──────────────────────────────────────────────────

def create_advisor(
    llm_config: Optional[Dict] = None,
    **kwargs,
) -> Advisor:
    """
    Create an Advisor agent for the conversational experiment.

    Parameters
    ----------
    llm_config : dict, optional
        Custom LLM configuration. Defaults to GPT-4o via OpenRouter.
    **kwargs
        Forwarded to the Advisor constructor (e.g. skills).
    """
    if llm_config is None:
        llm_config = get_llm_config(
            client="openrouter",
            model="openai/gpt-4o",
            api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

    return Advisor(
        llm_config=llm_config,
        system_message=Advisor.DEFAULT_SYSTEM_MESSAGE,
        is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content", "") or ""),
        **kwargs,
    )
