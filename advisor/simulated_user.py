import json
from typing import Optional, List

from agents import User


_SYSTEM_TEMPLATE = """You are a human user seeking recommendations. Respond naturally and concisely.

YOUR PERSONA:
{persona}

YOUR ORIGINAL REQUEST:
{instruction}

YOUR PAST READING/VIEWING HISTORY:
{history}

RULES:
- Respond as a regular person, not an AI or an expert.
- Express genuine preferences when asked. Be specific about what you like or dislike.
- If you are confused or overwhelmed, say so honestly.
- If you have made a decision, state it clearly (e.g., "I'll go with [title]").
- Keep responses to 2-4 sentences. Do not write essays.
- Do NOT invent preferences you do not have. If unsure, say "I'm not sure."
"""


class SimulatedUser(User):
    """
    A simulated user powered by an LLM for experiment automation.
    Extends ARARA's User and overrides get_human_input() so that,
    when the Orchestrator routes a message to this agent, the LLM
    generates a response instead of waiting for terminal input.

    Works within the standard ARARA flow:
        user.talk_to(orchestrator, message=...)
    The Orchestrator selects speakers via Module transitions; when it is
    this agent's turn, check_termination_and_human_reply calls
    get_human_input(), which calls the LLM with the conversation stored
    in self._oai_messages.
    """

    def __init__(
        self,
        persona: str,
        instruction: str,
        history: Optional[List[dict]] = None,
        max_turns: int = 6,
        name: str = "simulated_user",
        **kwargs,
    ):
        history_str = ""
        for item in (history or []):
            title = item.get("title", "Unknown")
            review = item.get("review", "")
            history_str += f"- {title}: {review}\n"

        system_message = _SYSTEM_TEMPLATE.format(
            persona=persona,
            instruction=instruction,
            history=history_str or "(no history)",
        )

        super().__init__(
            name=name,
            human_input_mode="ALWAYS",
            system_message=system_message,
            **kwargs,
        )

        self._max_turns = max_turns
        self._turn_count = 0
        self.persona = persona
        self.instruction = instruction

    # Keep only the last N messages from the conversation to prevent
    # context contamination / LLM echoing after many turns.
    _MAX_HISTORY_MESSAGES = 10

    def get_human_input(self, prompt: str) -> str:
        """
        Override ARARA's get_human_input to call the LLM.
        Reads the conversation history from self._oai_messages
        (populated automatically by the ARARA Orchestrator flow).
        Limits history to the last _MAX_HISTORY_MESSAGES entries to
        prevent the context from growing so large that the LLM starts
        echoing previous messages.
        """
        if self._turn_count >= self._max_turns:
            return "exit"

        messages = []
        for sender, msgs in self._oai_messages.items():
            if msgs:
                messages = msgs
                break

        if not messages:
            return "exit"

        last_content = (messages[-1].get("content", "") or "")
        if "TERMINATE" in last_content:
            return "exit"

        # Truncate to the most recent N messages to avoid context contamination
        if len(messages) > self._MAX_HISTORY_MESSAGES:
            messages = messages[-self._MAX_HISTORY_MESSAGES:]

        all_messages = list(self._oai_system_message or []) + list(messages)

        model_result = self.client.create(
            messages=all_messages,
            cache=self.client_cache,
            Agent=self,
        )

        self._turn_count += 1
        reply = model_result.content if isinstance(model_result.content, str) else str(model_result.content)
        return reply
