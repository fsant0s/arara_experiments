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
    Implicit orchestrator with 6 specialized agents.
    
    Expects `data` to contain:
    - 'source_user': User ID
    - 'direct_description_query': User query mentioning reference movies
    - 'movieCount': Target K for recommendations
    - 'multihop_info': Array of reference movies with relations (for validation)
    - 'sharedRelationships': Expected shared relations (for validation)
    - 'movieSubset': Ground truth expected recommendations (for validation)
    """
    
    # ======== Parameters & Memory Setup ========
    movieCount = data.get("movieCount", None)
    top_k_value = movieCount if isinstance(movieCount, int) and movieCount > 0 else 3
    
    history_line = ""
    if use_memory:
        user_history = get_filtered_user_history(
            user_id=data["source_user"],
            groundtruth_movie_ids=data.get("movieSubsetId", []),
            neo4j_conditions=data.get("sharedRelationships", []),
        )
        limited_history = user_history[-memory_size:] if len(user_history) > memory_size else user_history
        if limited_history:
            history_line = " ".join(limited_history)
    
    # ============ 1️⃣ TITLE NORMALIZER ============
    TitleNormalizer = Agent(
        name="TitleNormalizer",
        llm_config=llm_config,
        description="Normalize movie titles from query (handle underscores, quotes, article position).",
        system_message="""
You are the TITLE NORMALIZER.

Input: User's direct_description_query mentioning reference movies (e.g., "The Immigrant (1917)")

Task:
1. Extract ALL movie titles mentioned in the query (usually 2-3 reference movies)
2. Normalize each title:
   - Replace underscores "_" with spaces
   - Remove extra quotes (keep only inner quotes if any)
   - Handle article positioning: "The_Immigrant" → "The Immigrant" OR "Immigrant, The"?
   - Preserve year in format "(YYYY)"
3. Output exact titles in format: "Title (YYYY)"

Normalization examples:
- "The_Immigrant (1917)" → "The Immigrant (1917)"
- "A_King in New York (1957)" → "A King in New York (1957)"
- "Heartburn (1986)" → "Heartburn (1986)"

CRITICAL:
- ALWAYS include the year "(YYYY)" for each title
- If unsure about article position, try BOTH variants:
  "The Title (YYYY)" and "Title, The (YYYY)"
- Output one title per line, then final summary

OUTPUT (STRICT):
Normalized Titles:
- Reference 1: Title A (YYYY)
- Reference 2: Title B (YYYY)
- (Reference 3: Title C (YYYY) if present)
""",
        tools=[],
    )
    
    # ============ 2️⃣ COMMON ATTRIBUTE EXTRACTOR ============
    CommonAttributeExtractor = Agent(
        name="CommonAttributeExtractor",
        llm_config=llm_config,
        description="Extract shared actors/directors from the two reference movies.",
        system_message=f"""
You are the COMMON ATTRIBUTE EXTRACTOR.

Input: 
- Two normalized reference movie titles (from TitleNormalizer)
- Example: "The Immigrant (1917)" and "A King in New York (1957)"

history_line (user preferences context):
{history_line or "[No prior history available]"}

Task:
1. For EACH reference movie title:
   - Call: movies.get_movie_details_by_title("Title (YYYY)")
   - Extract ALL people from each relation:
     * actors (from "Starring" relation)
     * directors (from "Directed_by" relation)
     * composers (from "Music_by")
     * writers (from "Written_by")
     * producers (from "Produced_by")

2. Find INTERSECTION across both movies:
   - ACTORS common to BOTH? → YES: list them
   - DIRECTORS common to BOTH? → YES: list them
   - Other relations common? → YES: list them

3. Validate intersection is non-empty:
   - If empty → "No common attributes found" (should not happen in well-formed queries)
   - If found → proceed

CRITICAL:
- ALWAYS include the year "(YYYY)" in movie titles when calling get_movie_details_by_title
- Return ALL common people for each relation (not just first)
- If "Starring" is common relation, list ALL shared actors

OUTPUT (JSON, one line):
{{
 "reference_movies": ["Title A (YYYY)", "Title B (YYYY)"],
 "common_attributes": {{
   "actors": ["Actor1", "Actor2"],
   "directors": ["Director1"],
   "composers": [],
   "writers": [],
   "producers": []
 }},
 "primary_relation": "Starring",
 "primary_people": ["Actor1", "Actor2"],
 "is_valid": true
}}
""",
        tools=[movies.get_movie_details_by_title],
    )
    
    # ============ 3️⃣ RELATION TYPE DETECTOR ============
    RelationTypeDetector = Agent(
        name="RelationTypeDetector",
        llm_config=llm_config,
        description="Detect which relation type (Starring, Directed_by, etc.) the query is asking for.",
        system_message="""
You are the RELATION TYPE DETECTOR.

Input:
- Original user query (direct_description_query)
- Common attributes found (from CommonAttributeExtractor)

Task:
Determine the PRIMARY relation the user is asking for:

1. Analyze query language:
   - "same actor" / "same actor who appeared" → Starring (PRIMARY)
   - "same director" / "directed by" → Directed_by (PRIMARY)
   - "same composer" / "scored by" → Music_by (PRIMARY)
   - "same writer" / "written by" → Written_by (PRIMARY)
   - "same producer" → Produced_by (PRIMARY)

2. Cross-validate with common attributes found:
   - If common_attributes has matching people → confirm relation
   - If common_attributes is empty for detected relation → flag as error

3. Determine if secondary relations should be used:
   - Query mentions only 1 relation type? → Use only primary
   - Query mentions "also featured in" + multiple relations? → May need secondary

CRITICAL:
- DEFAULT RELATION: "Starring" (80% of queries)
- If query ambiguous, use "Starring"
- Output relation name EXACTLY as in DB: "Directed_by", "Starring", "Music_by", etc.

OUTPUT (JSON, one line):
{{
 "primary_relation": "Starring",
 "primary_people": ["Actor1", "Actor2"],
 "secondary_relations": [],
 "secondary_people": {{}},
 "query_clarity": "high|medium|low",
 "detected_language_hints": ["same actor", "starred in"]
}}
""",
        tools=[],
    )
    
    # ============ 4️⃣ MULTI-HOP RETRIEVER ============
    MultiHopRetriever = Agent(
        name="MultiHopRetriever",
        llm_config=llm_config,
        description="Find OTHER movies with the common actors/people.",
        system_message=f"""
You are the MULTI-HOP RETRIEVER.

Input:
- Primary relation type: "Starring" or "Directed_by" etc.
- Common people: ["Actor1", "Actor2"] or ["Director1"]

Task:
1. For EACH common person:
   - Call: movies.retrieve_titles_by_condition(relation, person, limit=400)
   - Relation must be EXACTLY: "Starring", "Directed_by", "Music_by", "Produced_by", etc.
   - person must be EXACT name from common attributes

2. Combine results:
   - All people are from SAME relation? → UNION all results (get all movies)
   - Multiple relations? → May INTERSECT (stricter)

3. Filter out reference movies:
   - Reference movies should NOT appear in final recommendations
   - Remove any title that was in the original query

4. Normalize and sort:
   - Dedup by normalized title (lowercase, trim, etc.)
   - Sort alphabetically (deterministic)
   - Keep ORIGINAL surface form for output

CRITICAL:
- Call retrieve_titles_by_condition with EXACT relation name
- ALL common people should be searched (union mode for same relation)
- Output deterministic (sorted)

OUTPUT (STRICT):
Title A (YYYY) [SEP] Title B (YYYY) [SEP] Title C (YYYY)
Or if no results: NO_CANDIDATES
""",
        tools=[movies.retrieve_titles_by_condition],
    )
    
    # ============ 5️⃣ RECOMMENDER AGENT ============
    RecommenderAgent = Agent(
        name="RecommenderAgent",
        llm_config=llm_config,
        description=f"Select exactly top_k={top_k_value} recommendations from the candidate pool.",
        system_message=f"""
You are the HISTORY-AWARE RECOMMENDER.

history_line (user preferences context):
{history_line or "[No prior history available]"}

Input:
- Candidate pool line: "Title A (YYYY) [SEP] Title B (YYYY) [SEP] ..."
- Target K: {top_k_value}

Task:
1. Parse candidate titles and deduplicate (normalized: lowercase, trim, replace '_' with ' ')
2. Rank by:
   - Preference alignment (from history_line): + if matches user's typical choices
   - Novelty: + if not in history, - if seen before
   - Diversity: small bonus for variety in years/directors/actors
   - Conflict avoidance: - if conflicting with history

3. Select exactly {top_k_value} items
   - If pool has fewer than {top_k_value} items, return all
   - Sort alphabetically then year desc (deterministic tie-breaking)

STRICT Output:
- Exactly ONE LINE:
  Title A (YYYY) [SEP] Title B (YYYY) [SEP] Title C (YYYY)
- Use ' [SEP] ' (single spaces)
- No leading/trailing [SEP]
- No extra lines or commentary
""",
        tools=[],
    )
    
    
    # ============ WIRING: Module + Orchestrator ============
    allowed_transitions = {
        TitleNormalizer: [CommonAttributeExtractor],
        CommonAttributeExtractor: [RelationTypeDetector],
        RelationTypeDetector: [MultiHopRetriever],
        MultiHopRetriever: [RecommenderAgent],
    }
    
    module = Module(
        name="implicit_module",
        agents=[
            TitleNormalizer,
            CommonAttributeExtractor,
            RelationTypeDetector,
            MultiHopRetriever,
            RecommenderAgent,
        ],
        speaker_selection_method="round_robin",
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
    )
    
    orchestrator = Orchestrator(
        name="implicit_orchestrator",
        module=module,
        llm_config=llm_config,
        description="""
Implicit Query Orchestrator.

Handles implicit/inferential recommendations where users reference movies and ask for 
similar recommendations based on shared attributes (actors, directors, etc.).

Multi-hop aware: Extracts shared people from reference movies → Finds OTHER movies with same people.

Examples:
- "Please recommend movies starring the same actor as in A King in New York (1957) 
   and The Immigrant (1917)." → Find common actor → Find OTHER movies
- "Please recommend movies directed by the same director as Pulp Fiction (1994) 
   and Reservoir Dogs (1992)." → Find common director → Find OTHER movies
""",
    )
    
    return orchestrator
