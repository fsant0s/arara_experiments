# --- EXPLICIT MODULE (output: "Title (Year) [SEP] Title (Year) ...") --------
import os
from agents import Agent, Module, Orchestrator
from tools import movies
from clients import groq_llama3370b

llm_config = groq_llama3370b
# LLM config

# ------------------ Agents ------------------

# Planner (não usa tools)
planner = Agent(
    name="planner",
    llm_config=llm_config,
    description="Plans the EXPLICIT pipeline; does not call tools.",
    system_message="""
You are the PLANNER for this EXPLICIT module. You DO NOT call tools.

Flow (fixed):
1) Send the raw user text to `constraint_agent`.
2) Take the JSON from `constraint_agent` and forward it verbatim to `candidate_and_gate_agent`.
3) Return the final string from `candidate_and_gate_agent` as the module result.

Rules:
- Do NOT relax hard constraints yourself.
- Pass ONLY compact JSON between agents (no prose).
- The module must output a single-line string with movies separated by ' [SEP] '.
""",
)

# A1) Canoniza/valida atributos → hard/soft
constraint_agent = Agent(
    name="constraint_agent",
    llm_config=llm_config,
    description="Canonicalizes explicit attributes and emits hard_constraints + soft_hints.",
    system_message="""
You convert explicit user attributes into canonical constraints that Neo4j tools understand.

STRICT IO
- INPUT: raw user request text (explicit)
- OUTPUT (JSON only, no prose):
  {
    "hard_constraints": [{"rel": "Directed_by|Starring|Genre|Language|Produced_by|Year", "obj": "<string or int>"}],
    "soft_hints": [{"rel": "Directed_by|Starring|Genre|Language|Produced_by|Year", "obj": "<string or int>"}]
  }

Policies
- REQUIRED attribute → hard_constraints.
- PREFERENCE ("preferably", "if possible") → soft_hints.
- Validate/canonicalize with allowed tools before emitting.
- Year must be int; names are strings. Do not invent attributes.

Allowed tools:
- movies.list_nodes_by_type
- movies.get_available_genres
- movies.get_available_languages
- movies.get_existing_relations
- movies.get_existing_nodes
- movies.search_movies_by_title   # only if a title appears

Do NOT fetch movie candidates here; only produce constraints.
""",
    tools=[
        movies.list_nodes_by_type,
        movies.get_available_genres,
        movies.get_available_languages,
        movies.get_existing_relations,
        movies.get_existing_nodes,
        movies.search_movies_by_title,
    ],
    reflect_on_tool_use=True,
)

# A2) Candidatos por relação → CMR-Gate → Ranking → **STRING " [SEP] "**
candidate_and_gate_agent = Agent(
    name="candidate_and_gate_agent",
    llm_config=llm_config,
    description="Retrieves candidates per hard constraint, intersects (CMR-Gate), ranks with soft hints, and outputs a single '[SEP]'-joined line.",
    system_message="""
You take canonical constraints and produce the FINAL STRING.

STRICT IO
- INPUT (JSON only):
  {
    "hard_constraints": [{"rel":"Directed_by|Starring|Genre|Language|Produced_by|Year","obj":"str or int"}],
    "soft_hints": [{"rel":"Directed_by|Starring|Genre|Language|Produced_by|Year","obj":"str or int"}]
  }
- OUTPUT (STRING only, no JSON, no prose):
  "Title A (YYYY) [SEP] Title B (YYYY) [SEP] Title C (YYYY)"

Algorithm
1) For EACH hard constraint, call the SPECIFIC tool to get a list of movie titles:
   - Directed_by -> movies.get_movies_by_director(obj)
   - Starring    -> movies.get_movies_by_actor(obj)
   - Genre       -> movies.get_movies_by_genre(obj)
   - Language    -> movies.get_movies_by_language(obj)
   - Produced_by -> movies.get_movies_by_production_company(obj)
   - Year        -> movies.get_movies_by_year(int(obj) if needed)
   - Unknown relation -> movies.get_movies_by_relation(relation, obj)

2) Intersection (CMR-Gate):
   - Normalize for comparison only (trim spaces, lowercase for set ops, optionally strip trailing ' (YYYY)' if needed).
   - Keep a mapping {normalized_title -> original_display_title} using the strings returned by the tools.
   - Compute the intersection across ALL hard constraint lists.
   - The approved set = intersection (may be empty).

3) Ranking (only over approved set):
   - Equal base score for all (they already pass hard).
   - For each soft_hint a movie satisfies (e.g., Language/Genre), add a small bonus.
   - Optional tiny diversification; final tie-break is lexicographic by original_display_title.

4) Formatting:
   - Use the ORIGINAL display titles (with year if present in the tool output) for the final string.
   - Join in ranked order with ' [SEP] ' exactly.
   - No trailing '[SEP]'.
   - If there are ZERO approved items, output an EMPTY STRING.

Return ONLY the final string; no JSON, no extra text.
""",
    tools=[
        movies.get_movies_by_director,
        movies.get_movies_by_actor,
        movies.get_movies_by_genre,
        movies.get_movies_by_language,
        movies.get_movies_by_production_company,
        movies.get_movies_by_year,
        movies.get_movies_by_relation,
    ],
    reflect_on_tool_use=True,
)

# ------------------ Module wiring ------------------
allowed_transitions = {
    planner: [constraint_agent, candidate_and_gate_agent],
    constraint_agent: [planner],
    candidate_and_gate_agent: [],
}

module = Module(
    admin_name="explicit_module",
    agents=[planner, constraint_agent, candidate_and_gate_agent],
    speaker_selection_method="auto",
    allowed_or_disallowed_speaker_transitions=allowed_transitions,
    speaker_transitions_type="allowed",
)

orchestrator = Orchestrator(
    name="explicit_orchestrator",
    module=module,
    llm_config=llm_config,
    system_message="Só repasse a mensagem.",
    description="Explicit path: constraints -> CMR-gate -> ranked single-line '[SEP]' output.",
)
