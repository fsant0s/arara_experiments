"""Simulated user for the multi-party recommendation scenario.

The user reads the advisor's suggestion, then decides:
  - which RS to talk to (rs1 or rs2), OR
  - to make a final decision.
Output is structured JSON so the runner can route the message.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional, Tuple


_SYSTEM_TEMPLATE = """\
You are role-playing a regular person looking for {item_word} suggestions.

WHAT YOU WANT:
{instruction}

WHO YOU ARE:
- You don't {consume_verb} very often, so you don't know many titles or authors.
- You tend to be vague: "something fun", "not too heavy".
- Long lists make you feel lost. You find it hard to compare things side by side.
- You trust confident explanations.
- You sometimes contradict yourself without noticing.
- You decide based on gut feeling, not analysis.

HOW THE CONVERSATION WORKS:
- Two recommendation systems ({rs1_name} and {rs2_name}) give you suggestions.
- An Advisor helps you navigate their responses (suggests questions, summaries, etc.).
- Each turn, you choose ONE system to talk to, OR announce your final decision.

HOW YOU RESPOND:
You MUST reply with valid JSON (no extra text) in this exact format:
{{
  "target": "<rs1_name>|<rs2_name>|decision",
  "message": "your message here"
}}

- Set "target" to the name of the RS you want to talk to.
- If you are ready to decide, set "target" to "decision" and list your choices \
in the message: "I'll go with [title1] and [title2]".
- Your message should be short and casual (1-3 sentences).
- Use fillers: "hmm", "I guess", "sounds cool".
- React honestly: if bored say so, if excited say so.
- You may follow or ignore the Advisor's suggestion — your choice.

WHEN YOU DECIDE:
- Pick EXACTLY {n_choices} {item_word_plural}.
- Use the EXACT title as it appeared in the conversation — do not rephrase.
- You must eventually decide (don't stall forever).
"""

_USER_TURN_PROMPT = """\
Here is what has happened so far:

{conversation_summary}

---

The Advisor says:
{advisor_message}

---

Reply with JSON: {{"target": "...", "message": "..."}}"""

_DOMAIN_STRINGS = {
    "book": {"item_word": "book", "item_word_plural": "books", "consume_verb": "read"},
    "movie": {"item_word": "movie", "item_word_plural": "movies", "consume_verb": "watch movies"},
}


class SimulatedUser:
    """LLM-based simulated user for the multi-party flow."""

    def __init__(
        self,
        chat_fn: Callable[[List[dict]], str],
        instruction: str,
        persona: str,
        n_choices: int = 1,
        domain: str = "book",
        max_turns: int = 6,
        rs1_name: str = "RS1",
        rs2_name: str = "RS2",
    ) -> None:
        self.chat_fn = chat_fn
        self.instruction = instruction
        self.persona = persona
        self.n_choices = max(1, n_choices)
        self.domain = domain
        self.max_turns = max_turns
        self.rs1_name = rs1_name
        self.rs2_name = rs2_name
        self._turn = 0

        ds = _DOMAIN_STRINGS.get(domain, _DOMAIN_STRINGS["book"])
        self.system_prompt = _SYSTEM_TEMPLATE.format(
            instruction=instruction,
            n_choices=self.n_choices,
            rs1_name=rs1_name,
            rs2_name=rs2_name,
            **ds,
        )

    def respond(
        self,
        advisor_message: str,
        conversation_summary: str,
    ) -> Tuple[str, str]:
        """Return (target, message).

        target is one of: rs1_name, rs2_name, "decision".
        """
        self._turn += 1

        if self._turn > self.max_turns:
            return "decision", f"I'll just go with whatever sounds best — please pick for me."

        user_prompt = _USER_TURN_PROMPT.format(
            conversation_summary=conversation_summary,
            advisor_message=advisor_message,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = self.chat_fn(messages)
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> Tuple[str, str]:
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return self.rs1_name, raw
            else:
                return self.rs1_name, raw

        target = str(data.get("target", self.rs1_name)).strip()
        message = str(data.get("message", "")).strip() or raw

        valid_targets = {
            self.rs1_name.lower(), self.rs2_name.lower(), "decision",
            "rs1", "rs2",
        }
        if target.lower() not in valid_targets:
            target = self.rs1_name

        if target.lower() == "rs1":
            target = self.rs1_name
        elif target.lower() == "rs2":
            target = self.rs2_name

        return target, message
