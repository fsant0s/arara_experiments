from agents import Agent, Module, Orchestrator
from tools import movies

from capabilities.memory import ListMemory, MemoryContent
from user_history_movie import get_filtered_user_history


def create_explicit_orchestrator(
    data: dict,
    llm_config=None,
    use_memory: bool = True,
    memory_size: int = 10,
) -> Orchestrator:
    """
    Explicit module:
    * **RetrieverAgent:** retrieves **ALL** relevant items (no top-k).
    * **RecommenderExplicitgent:** selects only **top_k = movieCount** based on the user's **PAST** preferences provided in `history_line` (without adding items; without using tools).
    * **Final output:** a single line with `' [SEP] '` between titles.

    """

    movieCount = data.get("movieCount", None)
    top_k_value = movieCount if isinstance(movieCount, int) and movieCount > 0 else 3

    # ======== memória e history_line ========
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
    
    # ===================== Retriever Agent (recupera TUDO) =====================
    RetrieverAgent = Agent(
        name="RetrieverAgent",
        llm_config=llm_config,
        description=(
            "Retrieves a broad list of movies based on explicit user query parameters "
            "and outputs a single '[SEP]' line. Does NOT enforce top-k; returns all relevant items (deduped)."
        ),
        system_message="""
## 🧠 RETRIEVER AGENT

### Input:
A user request with explicit information such as directors, actors, genres, languages, producers, or years.

---

### Goal:
- Retrieve a **BROAD, COMPREHENSIVE** list of relevant movies (no top-k limit).  
- Output **exactly ONE LINE** containing all titles separated by `' [SEP] '`.

---

### Process:

1. **Collect** from all relevant movie tools:
   - `movies.get_movies_by_director`  
   - `movies.get_movies_by_actor`  
   - `movies.get_movies_by_genre` ← Allowed genres ONLY (see full list below).  
   - `movies.get_movies_by_language`  
   - `movies.get_movies_by_production_company`  
   - `movies.get_movies_by_year`  
   - `movies.get_movies_by_relation` ← Allowed relations ONLY:  
     `['Based_on', 'Cinematography', 'Color_process', 'Directed_by', 'Distributed_by', 'Edited_by', 'Genre', 'Language', 'Music_by', 'Narrated_by', 'Produced_by', 'Production_Country', 'Screenplay_by', 'Starring', 'Written_by']`

   ⚠️ **Important rule:**  
   Always use the **canonical form** of the relation exactly as listed above.  
   - If a user mentions a near-synonym (e.g., *Cinematographer*), map it to **Cinematography**.  
   - If unsure, **do not invent** a new relation — use only the allowed ones.  

2. **Normalize-for-dedup**  
   - Convert to lowercase  
   - Trim and strip quotes  
   - Convert `_` → ` `  
   - Deduplicate  
   - Keep **original surface forms**

3. **Sort** lexicographically by title (deterministic).

4. **Do NOT truncate** by top-k — return the **full list**.

---

### Output (STRICT):

- Exactly **ONE line**:  
  `Title A (YYYY) [SEP] Title B (YYYY) [SEP] Title C (YYYY)`  
- Use `' [SEP] '` (single spaces).  
- No leading/trailing `[SEP]`, **no commentary or extra lines**.

---

### Allowed movies relations (use EXACTLY these):
`['Based_on', 'Cinematography', 'Color_process', 'Directed_by', 'Distributed_by', 'Edited_by', 'Genre', 'Language', 'Music_by', 'Narrated_by', 'Produced_by', 'Production_Country', 'Screenplay_by', 'Starring', 'Written_by']`

### Allowed Genres:
`['10', '1970s', '20', '26', '29', '35', '42', '61', 'AOR', 'Action', 'Action-adventure', 'Action|Adventure', 'Action|Adventure|Animation', "Action|Adventure|Animation|Children's|Fantasy", 'Action|Adventure|Comedy', 'Action|Adventure|Comedy|Romance', 'Action|Adventure|Drama', 'Action|Adventure|Fantasy', 'Action|Adventure|Horror|Thriller', 'Action|Adventure|Sci-Fi', 'Action|Adventure|Sci-Fi|Thriller', 'Action|Adventure|Sci-Fi|Thriller|War', 'Action|Adventure|Thriller', "Action|Children's", 'Action|Comedy', 'Action|Comedy|Crime|Drama', 'Action|Crime', 'Action|Crime|Drama', 'Action|Crime|Drama|Thriller', 'Action|Drama', 'Action|Drama|Romance', 'Action|Drama|Thriller', 'Action|Drama|Thriller|War', 'Action|Drama|War', 'Action|Horror', 'Action|Horror|Sci-Fi', 'Action|Horror|Sci-Fi|Thriller', 'Action|Horror|Thriller', 'Action|Mystery|Romance|Thriller', 'Action|Sci-Fi', 'Action|Sci-Fi|Thriller', 'Action|Sci-Fi|War', 'Action|Thriller', 'Action|War', 'Action|Western', 'Adventure', "Adventure|Animation|Children's", "Adventure|Animation|Children's|Sci-Fi", "Adventure|Children's", "Adventure|Children's|Comedy|Fantasy", "Adventure|Children's|Fantasy", 'Adventure|Comedy', 'Adventure|Comedy|Musical', 'Adventure|Comedy|Sci-Fi', 'Adventure|Drama', 'Adventure|Drama|Thriller', 'Adventure|Fantasy', 'Adventure|Fantasy|Romance', 'Adventure|Fantasy|Sci-Fi', 'Adventure|Musical', 'Adventure|Musical|Romance', 'Adventure|War', 'Alternative metal', 'Alternative pop/rock', 'Alternative rock', 'Animated sitcom', 'Animation', "Animation|Children's", "Animation|Children's|Comedy", "Animation|Children's|Comedy|Musical", "Animation|Children's|Musical", 'Animation|Comedy', 'Animation|Musical', 'Animation|Sci-Fi', 'Anime', 'Avant-garde', 'Beach party', 'Blues', 'Britpop', 'Children', "Children's", "Children's music", "Children's|Comedy", "Children's|Comedy|Drama", "Children's|Comedy|Fantasy", "Children's|Comedy|Sci-Fi", "Children's|Comedy|Western", "Children's|Drama", 'Christian metal', 'Christian rock', 'Christmas', 'Classical', 'Comedy', 'Comedy-drama', 'Comedy|Crime', 'Comedy|Crime|Drama', 'Comedy|Documentary', 'Comedy|Drama', 'Comedy|Drama|Romance', 'Comedy|Drama|Thriller']`
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

    # ===================== Recommender Agent (seleciona top_k pelo histórico) =====================
    RecommenderExplicitgent = Agent(
        name="RecommenderExplicitgent",
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
- DO NOT call tools. DO NOT introduce items not present in the Retriever line. DO NOT add explanations, commentary, or apologies. Just output the titles separeted by [SEP].
""",
        tools=[],  # não chama ferramentas; só seleciona
        reflect_on_tool_use=True,
    )

    # ===================== Wiring =====================
    allowed_transitions = {RetrieverAgent: [RecommenderExplicitgent]}

    module = Module(
        name="explicit_module",
        agents=[RetrieverAgent, RecommenderExplicitgent],
        speaker_selection_method="round_robin",
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
    )

    orchestrator = Orchestrator(
        name="explicit_orchestrator",
        module=module,
        llm_config=llm_config,
        description="""
            Handles explicit recommendation queries in which the user clearly specifies the entities or attributes that define the recommendation scope — such as a director, actor, or genre.
            It focuses on generating recommendations that directly satisfy the explicit condition expressed in the query, maintaining a clear and factual connection to the mentioned entity.
            Examples of explicit queries:
            Can you suggest some movies directed by Spike Lee?
            Can you suggest some movies directed by Mike Judge?
            Can you recommend some films directed by Mark Mothersbaugh?
        """,
    )

    return orchestrator
