from __future__ import annotations

from typing import List, Optional

from agents import User


_SYSTEM_TEMPLATE = """
You are role-playing a regular person looking for {item_word} suggestions.

WHAT YOU WANT:
{instruction}

WHO YOU ARE (personality — stay in character throughout the conversation):
- You don't {consume_verb} very often, so you don't know many titles, authors, or genres.
- When you try to describe what you want, you tend to be vague: "something fun", \
"not too heavy", "I don't really know".
- Long lists of options make you feel lost. You find it hard to compare things \
side by side and tend to just go with whatever sounds familiar or catches your eye first.
- You trust confident explanations. If someone tells you why something fits you, \
that matters more to you than reading a list of features.
- You sometimes contradict yourself without noticing — you might say you want \
something short and then get excited about a long epic.
- You decide based on gut feeling, not analysis.

HOW YOU TALK:
- Short, casual responses (1-3 sentences).
- You sometimes use fillers: "I guess", "hmm", "maybe?", "sure, sounds ok".
- You ask simple questions when curious: "What's that about?", "Is it long?".
- You react honestly: if something sounds boring, say so; if it sounds cool, say so.

WHEN YOU DECIDE:
- You pick EXACTLY {n_choices} {item_word_plural} when you feel ready.
- Use the EXACT title as it appeared in the conversation — do not rephrase or shorten it.
- Format: "I'll go with [title1] and [title2]" or "My picks are: [title1], [title2]".
- You must always end up choosing {n_choices}, even if you're unsure.
"""

_DOMAIN_STRINGS = {
    "book": {
        "item_word": "book",
        "item_word_plural": "books",
        "consume_verb": "read",
    },
    "movie": {
        "item_word": "movie",
        "item_word_plural": "movies",
        "consume_verb": "watch movies",
    },
}


class SimulatedUser(User):
    """Simulated user: novice, easily confused, high cognitive load.

    This is the user profile where the advisor should provide the most benefit.
    """

    _MAX_HISTORY_MESSAGES = 10

    def __init__(
        self,
        persona: str,
        instruction: str,
        history: Optional[List[dict]] = None,
        max_turns: int = 6,
        name: str = "simulated_user",
        domain: str = "book",
        n_choices: int = 1,
        **kwargs,
    ):
        ds = _DOMAIN_STRINGS.get(domain, _DOMAIN_STRINGS["book"])
        system_message = _SYSTEM_TEMPLATE.format(
            instruction=instruction,
            n_choices=max(1, n_choices),
            **ds,
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

    def get_human_input(self, prompt: str) -> str:
        if self._turn_count >= self._max_turns:
            return "exit"

        messages = []
        for _sender, msgs in self._oai_messages.items():
            if msgs:
                messages = msgs
                break

        if not messages:
            return "exit"

        last_content = (messages[-1].get("content", "") or "")
        if "TERMINATE" in last_content:
            return "exit"

        if len(messages) > self._MAX_HISTORY_MESSAGES:
            messages = messages[-self._MAX_HISTORY_MESSAGES:]

        all_messages = list(self._oai_system_message or []) + list(messages)

        model_result = self.client.create(
            messages=all_messages,
            cache=self.client_cache,
            Agent=self,
        )

        self._turn_count += 1
        return model_result.content if isinstance(model_result.content, str) else str(model_result.content)
