import os

from agents import Agent, Orchestrator, Module
from recsys.parallel_execution import parallel
from utils import get_llm_config

system_message="""You are acting as an independent book recommendation system.

You will receive:
- A structured user profile containing previously interacted books.
  Each history book includes: ASIN, title, description, and review text.
- A natural language request from the user.

Your task is to recommend books that best satisfy the user's request.

You must:

1. Produce a ranked list of the TOP-3 recommended books (best first).
2. For EACH recommended book, provide:
   - A concise explanation (1–2 sentences) grounded strictly in the user's preference signals.
   - The list of ASINs from the user's history that most influenced this recommendation.

IMPORTANT RULES:
- Use only the information contained in the provided user profile to infer preferences.
- Do NOT recommend any book that appears in the user's history.
- For each recommended book, return between 1 and 3 ASINs in "used_history_asins".
- The ASINs must refer exclusively to books present in the user profile.
- Do NOT invent ASINs.
- Do NOT output history titles as identifiers.
- Each explanation must be specific to its corresponding recommended book.
- Return ONLY valid JSON. No extra text.

Output JSON format (single object):

{
  "top_k_recommendations": [
    {
      "rank": 1,
      "book_title": "<title>",
      "explanation": "<1–2 sentences grounded in the user's preferences>",
      "used_history_asins": ["<asin1>", "<asin2>"]
    },
    {
      "rank": 2,
      "book_title": "<title>",
      "explanation": "<1–2 sentences grounded in the user's preferences>",
      "used_history_asins": ["<asin1>"]
    },
    {
      "rank": 3,
      "book_title": "<title>",
      "explanation": "<1–2 sentences grounded in the user's preferences>",
      "used_history_asins": ["<asin1>", "<asin2>", "<asin3>"]
    }
  ]
}

"""


def create_recsys() -> Orchestrator:

    chatgpt4o = Agent(
        name = "chatgpt4o",
        llm_config = get_llm_config(
            client="openrouter",
            model="openai/gpt-4o",
            api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            response_format = "json_object",
            temperature = 0.0,
        ),
        system_message=system_message,
    )

    gemini_2_5_flash_lite = Agent(
        name = "gemini_2_5_flash_lite",
        llm_config = get_llm_config(
            client="openrouter",
            model="google/gemini-2.5-flash-lite",
            api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            response_format = "json_object",
            temperature = 0.0,
        ),
        system_message=system_message,
    )

    claude_3_5_sonnet = Agent(
        name = "claude_3_5_sonnet",
        llm_config = get_llm_config(
            client="openrouter",
            model="anthropic/claude-3.5-sonnet",
            api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            response_format = "json_object",
            temperature = 0.0,
        ),
        system_message=system_message,
    )

    recsys_module = Module(
        name="recsys_module",
        agents=[chatgpt4o, gemini_2_5_flash_lite, claude_3_5_sonnet],
        speaker_selection_method=parallel,
        max_round=1,
    )

    recsys_orchestrator = Orchestrator(
        name="recsys_orchestrator",
        module=recsys_module,
        description="Orchestrator for recommender systems using multiple LLMs.",
    )

    return recsys_orchestrator

