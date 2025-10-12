# --- EXPLICIT MODULE (Constraint → Retrieve+Rank) ---
# Objetivo: gerar recomendações a partir de consultas explícitas, ex.:
#   "Can you suggest some movies directed by Mike Judge?"
#
# Saída (exatamente uma linha):
#   Title A (YYYY) [SEP] Title B (YYYY) [SEP] Title C (YYYY)
#
# Simplificado: sem Planner, sem Disambiguator. Só 2 agentes.
# Fluxo: Constraint Agent → Retrieve+Rank Agent

from agents import Agent, Module, Orchestrator
from tools import movies
from clients import groq_llama3370b, gpt_41

llm_config = gpt_41

# ===================== Constraint Agent (Step 1) =====================
constraint_agent = Agent(
    name="constraint_agent",
    llm_config=llm_config,
    description="Extracts and validates HARD (required) and SOFT (preference) constraints from the explicit query. No candidate retrieval.",
    system_message="""
You are the CONSTRAINT AGENT (Step 1) for explicit recommendations.

Input:
- User explicit query (e.g., "Can you suggest some movies directed by Mike Judge?")

Goal:
- Extract constraints as HARD (required) and SOFT (preferences).
- Validate that values exist and are compatible with allowed relations.
- Do NOT retrieve candidates.

Allowed relations: Directed_by, Starring, Genre, Language, Produced_by, Year

Rules:
- Required → HARD. Preferences → SOFT.
- Use tools only for validation (check if entities or values exist in the knowledge base).
- If a movie title appears, you may call movies.search_movies_by_title only to confirm existence.
- Never list catalogs or retrieve candidates here.

Output format (TEXT ONLY, exactly):

HARD:
- <Relation>: <Value>
- ...

SOFT:
- <Relation>: <Value>
- ...

No code, no JSON, no logs.

Implicit configuration (do not print):
- normalize_for_match: true
- intersection: AND
- limit_per_relation: 200
- top_k: 20
- ranking: use_soft_hints (small); tie_break: lexicographic
- empty_policy: do_not_relax_hard → return empty string
""",
    tools=[
        movies.list_nodes_by_type,
        movies.get_available_genres,
        movies.get_available_languages,
        movies.get_existing_relations,
        movies.get_existing_nodes,
        movies.search_movies_by_title,
        movies.get_movies_by_director,  # existence check only
        movies.get_movies_by_actor,     # existence check only
    ],
    reflect_on_tool_use=True,
)

# ===================== Retrieve + Rank Agent (Step 2) =====================
retrieve_rank_agent = Agent(
    name="retrieve_rank_agent",
    llm_config=llm_config,
    description=(
        "Retrieves movie candidates per HARD constraints, intersects results (AND/CMR-Gate), "
        "ranks with SOFT hints, and formats a single '[SEP]' line."
    ),
    system_message="""
You are the RETRIEVE+RANK AGENT (Step 2).

Input:
- HARD and SOFT lists from the Constraint Agent.

Goal:
Retrieve movies that satisfy HARD constraints,
optionally use SOFT hints to rank, and output a single formatted line.

Algorithm:
1) Retrieval (per HARD):
   Directed_by  → movies.get_movies_by_director
   Starring     → movies.get_movies_by_actor
   Genre        → movies.get_movies_by_genre
   Language     → movies.get_movies_by_language
   Produced_by  → movies.get_movies_by_production_company
   Year         → movies.get_movies_by_year
   Otherwise    → movies.get_movies_by_relation(Rel, Value)
   Keep ORIGINAL titles exactly as returned by tools.

2) Intersection (AND / CMR-Gate):
   - Normalize titles for comparison (lowercase, trim, strip quotes).
   - Maintain {normalized → original} mapping per list.
   - Intersect across all HARD result sets.
   - Recover ORIGINAL titles for the approved set.

3) Ranking:
   - Start all with equal score.
   - If a SOFT hint appears in title/metadata, apply a small bonus.
   - Break ties lexicographically.
   - Truncate to top_k=20.

4) Formatting:
   - If possible, include years as "Title (YYYY)".
   - Join ORIGINAL titles with ' [SEP] ' (no trailing separator).
   - If empty, return an empty string.

Output:
Exactly ONE LINE:
Title A (YYYY) [SEP] Title B (YYYY) [SEP] Title C (YYYY)

Constraints:
- No JSON, no code, no logs.
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

# ===================== Wiring =====================
allowed_transitions = {
    constraint_agent: [retrieve_rank_agent],
}

module = Module(
    admin_name="explicit_module",
    agents=[constraint_agent, retrieve_rank_agent],
    speaker_selection_method="auto",
    allowed_or_disallowed_speaker_transitions=allowed_transitions,
    speaker_transitions_type="allowed",
)

orchestrator = Orchestrator(
    name="explicit_orchestrator",
    module=module,
    llm_config=llm_config,
    system_message="Forward messages only. Do not interpret or modify content.",
    description=(
        "Explicit recommendation module: Constraint Agent extracts and validates HARD/SOFT constraints; "
        "Retrieve+Rank Agent fetches per HARD, intersects (AND), ranks with SOFT hints, "
        "and outputs a single '[SEP]' line."
    ),
)
