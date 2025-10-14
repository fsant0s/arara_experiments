from agents import Agent, Module, Orchestrator
from tools import movies
from clients import groq_llama3370b, gpt_41

from capabilities.memory import ListMemory, MemoryContent
from user_history import get_filtered_user_history


def create_explicit_orchestrator(
    data: dict,
    llm_config=gpt_41,
    use_memory: bool = True,
    memory_size: int = 10,
) -> Orchestrator:
    """
    Módulo explícito:
      - RetrieverAgent: recupera TODOS os itens relevantes (sem top-k).
      - RecommenderAgent: seleciona apenas top_k = movieCount com base nas preferências ANTIGAS do usuário
        fornecidas em `history_line` (sem adicionar itens; sem usar ferramentas).
      - Saída final: uma única linha com ' [SEP] ' entre os títulos.
    """

    movieCount = data.get("movieCount", None)
    top_k_value = movieCount if isinstance(movieCount, int) and movieCount > 0 else 3

    # ======== memória e history_line ========
    sequential_memory = None
    history_line = ""
    if use_memory:
        user_history = get_filtered_user_history(
            user_id=data["source_user"],
            groundtruth_movie_ids=data["movieSubsetId"],
            neo4j_conditions=data["sharedRelationships"],
        )
        limited_history = user_history[-memory_size:] if len(user_history) > memory_size else user_history
        if limited_history:
            history_line = " ".join(limited_history)
            sequential_memory = ListMemory(name="chat_history")
            sequential_memory.add(MemoryContent(content=history_line))

    _mem_kwargs = {"memory": [sequential_memory]} if sequential_memory else {}

    # ===================== Retriever Agent (recupera TUDO) =====================
    RetrieverAgent = Agent(
        name="RetrieverAgent",
        llm_config=llm_config,
        description=(
            "Retrieves a broad list of movies based on explicit user query parameters "
            "and outputs a single '[SEP]' line. Does NOT enforce top-k; returns all relevant items (deduped)."
        ),
        system_message="""
You are the RETRIEVER AGENT.

Input:
- A user request with explicit information such as directors, actors, genres, languages, producers, or years.

Goal:
- Retrieve a BROAD, COMPREHENSIVE list of relevant movies (no top-k limit).
- Output exactly ONE LINE containing all titles separated by ' [SEP] '.

Process:
1) Collect from all relevant movie tools:
   - movies.get_movies_by_director
   - movies.get_movies_by_actor
   - movies.get_movies_by_genre
   - movies.get_movies_by_language
   - movies.get_movies_by_production_company
   - movies.get_movies_by_year
   - movies.get_movies_by_relation
2) Normalize-for-dedup (lowercase, trim, strip quotes, '_'→' '), deduplicate, keep ORIGINAL surface forms.
3) Sort lexicographically by title (deterministic).
4) Do NOT truncate by top-k — return the full list.

Output (STRICT):
- Exactly ONE line: Title A (YYYY) [SEP] Title B (YYYY) [SEP] Title C (YYYY)
- Use ' [SEP] ' (single spaces). No leading/trailing [SEP], no commentary or extra lines.
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
        **_mem_kwargs,
    )

    # ===================== Recommender Agent (seleciona top_k pelo histórico) =====================
    RecommenderAgent = Agent(
        name="RecommenderAgent",
        llm_config=llm_config,
        description=(
            f"Selects exactly top_k={top_k_value} items FROM the Retriever's list, "
            "ranking solely by the user's PAST preferences provided via history_line. "
            "Does NOT add new items or call tools. Outputs a single '[SEP]' line."
        ),
        system_message=f"""
You are the RECOMMENDER AGENT.

Goal:
Select exactly top_k={top_k_value} movies from the Retriever's list,
using ONLY the user's PAST preferences below (history_line) to rank.

history_line (past user preferences; use ONLY this to decide relevance):
{history_line or "[No prior history available]"}

Inputs:
- One single line from the Retriever containing MANY titles (eligible set).
- You MUST NOT add new movies, fetch new data, or use tools.

Scoring (use only signals derivable from history_line):
- + Director overlap with names present in history_line.
- + Actor overlap with names present in history_line.
- + Genre keywords overlap with terms present in history_line.
- + Language keywords overlap with terms present in history_line.
- (Optional) Small boost for titles explicitly mentioned in history_line.
Tie-breakers (deterministic):
- 1) Alphabetical by title; 2) Year descending (if available).

Selection:
- Deduplicate by normalized title (lower/trim/strip quotes, '_'→' '), then keep ORIGINAL surface forms.
- Rank all eligible items using the rules above.
- Output EXACTLY {top_k_value} items; if the Retriever list has fewer than {top_k_value}, output all available.

STRICT Output:
- Exactly ONE LINE:
  Title A (YYYY) [SEP] Title B (YYYY) [SEP] Title C (YYYY)
- Use ' [SEP] ' (single spaces), no leading/trailing [SEP], no extra lines, no commentary.
- DO NOT call tools. DO NOT introduce items not present in the Retriever line.
""",
        tools=[],  # não chama ferramentas; só seleciona
        reflect_on_tool_use=True,
        **_mem_kwargs,
    )

    # ===================== Wiring =====================
    allowed_transitions = {RetrieverAgent: [RecommenderAgent]}

    module = Module(
        admin_name="explicit_module",
        agents=[RetrieverAgent, RecommenderAgent],
        speaker_selection_method="auto",
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
    )

    orchestrator = Orchestrator(
        name="explicit_orchestrator",
        module=module,
        llm_config=llm_config,
        description=(
            f"RetrieverAgent returns a comprehensive list; RecommenderAgent selects exactly top_k={top_k_value} "
            "purely based on past user preferences (history_line), outputting one '[SEP]' line."
        ),
    )

    return orchestrator
