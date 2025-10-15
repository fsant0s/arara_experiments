from agents import Agent, Module, Orchestrator
from tools import movies

from capabilities.memory import ListMemory, MemoryContent
from user_history import get_filtered_user_history


def create_implicit_orchestrator(
    data: dict,
    llm_config=None,
    use_memory: bool = True,
    memory_size: int = 10,
) -> Orchestrator:
    """
    Módulo implícito:
      - ProfileAgent: extrai sinais implícitos (gêneros/moods/era/idioma/duração/evitar) da consulta + histórico (history_line),
                      e canonicaliza onde possível (gêneros/idiomas) usando tools de catálogo.
      - ImplicitRetrieverAgent: gera um pool amplo com base no perfil + hábitos (diretores/atores frequentes etc.). NÃO corta por K.
      - ImplicitRecommenderAgent: escolhe exatamente top_k=movieCount, equilibrando preferência (history), novidade e diversidade.
      - Saída final: uma única linha com ' [SEP] ' entre os títulos.

    Requer em `data`: 'source_user', 'movieSubsetId', 'sharedRelationships'.
    Opcional: 'movieCount' (K alvo) — se ausente, usa K=3.
    """
    # ======== parâmetros e memória ========
    movieCount = data.get("movieCount", None)
    top_k_value = movieCount if isinstance(movieCount, int) and movieCount > 0 else 3

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


    # ===================== ProfileAgent (sinais implícitos + canonicalização) =====================
    ProfileAgent = Agent(
        name="ProfileAgent",
        llm_config=llm_config,
        description=(
            "Extracts implicit preference signals from the user's situational query and prior history. "
            "Canonicalizes genres/languages if possible using catalog tools. Outputs a compact PROFILE block."
        ),
        system_message=f"""
You are the PROFILE AGENT.

Goal:
From the user's implicit/situational description and the past history (below), derive a concise preference profile.

history_line (past user behavior; use to infer stable tastes):
{history_line or "[No prior history available]"}

Input:
- A situational/implicit description (no explicit entities guaranteed).

What to extract (when present or inferable):
- Genres (canonicalize to known catalog genres when possible)
- Moods/Tone (e.g., lighthearted, gritty, suspenseful)
- Era/Years (e.g., 90s, 2000s, or a year range)
- Language(s)
- Length preference (short / standard / long)
- Directors/Actors likely preferred (from history hints)
- Avoid (e.g., gore, slow-burn, very long)
- Novelty preference (if user hints “something new”)

Use tools ONLY for canonicalization mappings (no candidate retrieval):
- movies.get_available_genres
- movies.get_available_languages

OUTPUT (TEXT ONLY, STRICT):
PROFILE:
- Genres: <comma-separated or empty>
- Moods: <comma-separated or empty>
- Era: <free text or empty>
- Languages: <comma-separated or empty>
- Length: <short|standard|long|empty>
- Directors: <comma-separated or empty>
- Actors: <comma-separated or empty>
- Avoid: <comma-separated or empty>
- Novelty: <prefer_new|prefer_familiar|none>
""",
        tools=[
            movies.get_available_genres,
            movies.get_available_languages,
        ],
        reflect_on_tool_use=True,
    )

    # ===================== ImplicitRetrieverAgent (gera POOL amplo; não corta por K) =====================
    ImplicitRetrieverAgent = Agent(
        name="ImplicitRetrieverAgent",
        llm_config=llm_config,
        description=(
            "Builds a broad candidate pool using the PROFILE info and user habits; "
            "returns a single '[SEP]' line with many titles. Does NOT enforce top-k."
        ),
        system_message="""
You are the IMPLICIT RETRIEVER AGENT.

Input:
- PROFILE block from ProfileAgent (see fields: Genres, Moods, Era, Languages, Length, Directors, Actors, Avoid, Novelty).
- Optional user history implicitly available via context.

Goal:
- Generate a BROAD candidate pool from catalog tools based on PROFILE signals (genres, languages, era/years, directors, actors).
- Include items matching multiple signals first, but DO NOT cut by top-k here.

Tools to call for candidate generation:
- movies.get_movies_by_genre ← Allowed genres ONLY (see full list below).  
- movies.get_movies_by_language
- movies.get_movies_by_year
- movies.get_movies_by_director
- movies.get_movies_by_actor
- movies.get_movies_by_production_company (optional)
- movies.get_movies_by_relation ← Allowed relations ONLY:  
    `['Based_on', 'Cinematography', 'Color_process', 'Directed_by', 'Distributed_by', 'Edited_by', 'Genre', 'Language', 'Music_by', 'Narrated_by', 'Produced_by', 'Production_Country', 'Screenplay_by', 'Starring', 'Written_by']`

⚠️ **Important rule:**  
Always use the **canonical form** of the relation exactly as listed above.  
- If a user mentions a near-synonym (e.g., *Cinematographer*), map it to **Cinematography**.  
- If unsure, **do not invent** a new relation — use only the allowed ones.  

Heuristics:
- If "Era" resembles a decade (e.g., 90s) map to a year range (1990–1999) and call get_movies_by_year per year or via relation fallback.
- If "Length" is "short", prefer earlier years or known short runtimes when metadata strings hint at that (best-effort).
- If "Novelty" is "prefer_new", you MAY down-rank items explicitly present in history when composing the pool (but do not drop them entirely).
- Avoid obvious conflicts in "Avoid" (e.g., skip “gore” genres if present).

Process:
1) Aggregate results across applicable signals.
2) Normalize for dedup (lower/trim/strip quotes, replace '_' with space); keep ORIGINAL surface form.
3) Sort lexicographically (deterministic).
4) DO NOT truncate by K. Return the full pool line.

OUTPUT (STRICT):
- Exactly ONE LINE with titles: Title A (YYYY) [SEP] Title B (YYYY) [SEP] Title C (YYYY)
- Use ' [SEP] ' (single spaces). No leading/trailing [SEP], no commentary.

---

### Allowed movies relations (use EXACTLY these):
`['Based_on', 'Cinematography', 'Color_process', 'Directed_by', 'Distributed_by', 'Edited_by', 'Genre', 'Language', 'Music_by', 'Narrated_by', 'Produced_by', 'Production_Country', 'Screenplay_by', 'Starring', 'Written_by']`

### Allowed Genres:
`['10', '1970s', '20', '26', '29', '35', '42', '61', 'AOR', 'Action', 'Action-adventure', 'Action|Adventure', 'Action|Adventure|Animation', "Action|Adventure|Animation|Children's|Fantasy", 'Action|Adventure|Comedy', 'Action|Adventure|Comedy|Romance', 'Action|Adventure|Drama', 'Action|Adventure|Fantasy', 'Action|Adventure|Horror|Thriller', 'Action|Adventure|Sci-Fi', 'Action|Adventure|Sci-Fi|Thriller', 'Action|Adventure|Sci-Fi|Thriller|War', 'Action|Adventure|Thriller', "Action|Children's", 'Action|Comedy', 'Action|Comedy|Crime|Drama', 'Action|Crime', 'Action|Crime|Drama', 'Action|Crime|Drama|Thriller', 'Action|Drama', 'Action|Drama|Romance', 'Action|Drama|Thriller', 'Action|Drama|Thriller|War', 'Action|Drama|War', 'Action|Horror', 'Action|Horror|Sci-Fi', 'Action|Horror|Sci-Fi|Thriller', 'Action|Horror|Thriller', 'Action|Mystery|Romance|Thriller', 'Action|Sci-Fi', 'Action|Sci-Fi|Thriller', 'Action|Sci-Fi|War', 'Action|Thriller', 'Action|War', 'Action|Western', 'Adventure', "Adventure|Animation|Children's", "Adventure|Animation|Children's|Sci-Fi", "Adventure|Children's", "Adventure|Children's|Comedy|Fantasy", "Adventure|Children's|Fantasy", 'Adventure|Comedy', 'Adventure|Comedy|Musical', 'Adventure|Comedy|Sci-Fi', 'Adventure|Drama', 'Adventure|Drama|Thriller', 'Adventure|Fantasy', 'Adventure|Fantasy|Romance', 'Adventure|Fantasy|Sci-Fi', 'Adventure|Musical', 'Adventure|Musical|Romance', 'Adventure|War', 'Alternative metal', 'Alternative pop/rock', 'Alternative rock', 'Animated sitcom', 'Animation', "Animation|Children's", "Animation|Children's|Comedy", "Animation|Children's|Comedy|Musical", "Animation|Children's|Musical", 'Animation|Comedy', 'Animation|Musical', 'Animation|Sci-Fi', 'Anime', 'Avant-garde', 'Beach party', 'Blues', 'Britpop', 'Children', "Children's", "Children's music", "Children's|Comedy", "Children's|Comedy|Drama", "Children's|Comedy|Fantasy", "Children's|Comedy|Sci-Fi", "Children's|Comedy|Western", "Children's|Drama", 'Christian metal', 'Christian rock', 'Christmas', 'Classical', 'Comedy', 'Comedy-drama', 'Comedy|Crime', 'Comedy|Crime|Drama', 'Comedy|Documentary', 'Comedy|Drama', 'Comedy|Drama|Romance', 'Comedy|Drama|Thriller']`
""",
        tools=[
            movies.get_movies_by_genre,
            movies.get_movies_by_language,
            movies.get_movies_by_year,
            movies.get_movies_by_director,
            movies.get_movies_by_actor,
            movies.get_movies_by_production_company,
            movies.get_movies_by_relation,
        ],
        reflect_on_tool_use=True,
    )

    # ===================== ImplicitRecommenderAgent (seleciona exatamente top_k) =====================
    ImplicitRecommenderAgent = Agent(
        name="ImplicitRecommenderAgent",
        llm_config=llm_config,
        description=(
            f"Selects exactly top_k={top_k_value} items from the implicit pool optimizing preference × novelty × diversity, "
            "guided by history_line. Outputs a single '[SEP]' line."
        ),
        system_message=f"""
You are the IMPLICIT RECOMMENDER AGENT.

Goal:
Select exactly top_k={top_k_value} movies from the candidate pool produced by the ImplicitRetrieverAgent.

history_line (past behavior; use to weight preferences and novelty):
{history_line or "[No prior history available]"}

Inputs:
- One single line containing MANY titles (the candidate pool).
- The PROFILE textual block (preferences inferred) may be available upstream for context (do not reprint).

Scoring signals (combine, transparent trade-offs):
- Preference alignment (from history_line): + directors/actors/genres/languages the user tends to consume.
- Novelty: + if the user likes “something new”, penalize items already in history_line or very similar clusters.
- Diversity: small bonus for covering varied directors/actors/years among the final K.
- Conflict avoidance: demote items conflicting with "Avoid" terms if known.
Tie-breaking: alphabetical by title, then year desc (if available).

Selection:
- Deduplicate by normalized title (lower/trim/strip quotes, '_'→' ') and keep ORIGINAL surface form for output.
- Rank candidates and select exactly {top_k_value}. If the pool contains fewer than {top_k_value}, output all available.
- DO NOT add movies beyond the candidate pool; DO NOT call tools here.

STRICT Output:
- Exactly ONE LINE:
  Title A (YYYY) [SEP] Title B (YYYY) [SEP] Title C (YYYY)
- Use ' [SEP] ' (single spaces). No leading/trailing [SEP], no extra lines, no commentary.
""",
        tools=[],  # sem tools aqui; apenas seleção
        reflect_on_tool_use=True,
    )

    # ===================== Wiring =====================
    allowed_transitions = {
        ProfileAgent: [ImplicitRetrieverAgent],
        ImplicitRetrieverAgent: [ImplicitRecommenderAgent],
    }

    module = Module(
        admin_name="implicit_module",
        agents=[ProfileAgent, ImplicitRetrieverAgent, ImplicitRecommenderAgent],
        speaker_selection_method="round_robin",
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
    )

    orchestrator = Orchestrator(
        name="implicit_orchestrator",
        module=module,
        llm_config=llm_config,
        description="""
            Handles implicit recommendation queries in which the user expresses intent indirectly through examples, without explicitly naming the desired attributes or entities.
            It focuses on understanding the relational pattern implied by the user’s examples and generating recommendations that follow the same underlying connection or context.
            Examples of implicit queries:
                - Please recommend some movies starring the same actor as in The - - Return of the Musketeers (1989) and The Omega Code (1999).
                - Please recommend some movies featuring the same actor as seen in - Heartburn (1986) and Man Trouble (1992).
                - Please recommend some movies featuring the same actor who starred -in Bram Stoker's Dracula (1992) and Great Balls of Fire! (1989).
        """
    )

    return orchestrator
