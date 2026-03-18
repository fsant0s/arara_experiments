from typing import Optional, List

from agents import User

from .simulated_user import SimulatedUser


_CONFUSED_SYSTEM_TEMPLATE = """You are a human user seeking book recommendations. You tend to be INDECISIVE and VAGUE.

YOUR PERSONA:
{persona}

YOUR ORIGINAL REQUEST:
{instruction}

YOUR PAST READING/VIEWING HISTORY:
{history}

BEHAVIOR RULES (you are a "confused" user for testing):
- Often say things like "I'm not sure", "maybe", "it's hard to decide", "I'm torn between..."
- Give vague or contradictory preferences (e.g. "something light but also deep")
- Sometimes change your mind mid-conversation
- Ask the advisor for help instead of deciding quickly
- Express that you feel overwhelmed when many options are shown
- Take several turns before making a final choice
- Keep responses to 2-4 sentences. Do NOT write essays."""


class ConfusedUser(SimulatedUser):
    """
    A simulated user that behaves in a confused, indecisive way.
    Used to stress-test the Advisor's ability to guide users through overload.
    """

    def __init__(
        self,
        persona: str,
        instruction: str,
        history: Optional[List[dict]] = None,
        max_turns: int = 6,
        name: str = "confused_user",
        **kwargs,
    ):
        history_str = ""
        for item in (history or []):
            title = item.get("title", "Unknown")
            review = item.get("review", "")
            history_str += f"- {title}: {review}\n"

        system_message = _CONFUSED_SYSTEM_TEMPLATE.format(
            persona=persona,
            instruction=instruction,
            history=history_str or "(no history)",
        )

        User.__init__(
            self,
            name=name,
            human_input_mode="ALWAYS",
            system_message=system_message,
            **kwargs,
        )

        self._max_turns = max_turns
        self._turn_count = 0
        self.persona = persona
        self.instruction = instruction
