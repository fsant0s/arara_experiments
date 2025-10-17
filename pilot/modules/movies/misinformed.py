from agents import Agent, Module, Orchestrator
from tools import movies
from user_history import get_filtered_user_history


def create_misinformed_orchestrator(data, llm_config=None, use_memory=True, memory_size=10) -> Orchestrator:
    """
    Orchestrator for handling misinformed queries.
      
    Args:
        data: Dictionary with query context including:
            - query: User's query string
            - movieCount: Expected number of recommendations (top_k)
            - movieSubsetId: Ground truth movie IDs
            - sharedRelationships: Correct relations connecting ground truth movies
            - multihop_info: Complex relation info (for validation)
        llm_config: LLM configuration for agents
        use_memory: Whether to use user history as context
        memory_size: Number of historical movies to include
    
    Returns:
        Orchestrator instance for processing misinformed queries
    """
    movie_count = data.get("movieCount", 2)
    top_k = movie_count if movie_count > 0 else 2

    # ============== user history context ==============
    history_line = ""
    if use_memory:
        user_history = get_filtered_user_history(
            user_id=data.get("source_user", ""),
            groundtruth_movie_ids=data.get("movieSubsetId", []),
            neo4j_conditions=data.get("sharedRelationships", []),
        )
        if user_history:
            limited_history = user_history[-memory_size:] if len(user_history) > memory_size else user_history
            history_line = " ".join(limited_history)

    # 1️⃣ ENHANCED ENTITY RESOLVER
    # Detects MULTIPLE misinformations and extracts all (movie, relation, person) tuples
    EntityResolver = Agent(
        name="EntityResolver",
        llm_config=llm_config,
        description="Extract all mentioned movies, people, and relations (multiple per query).",
        system_message="""
You are the ENHANCED ENTITY RESOLVER.

Goal:
Extract ALL mentioned movie titles (WITH RELEASE YEAR), people, and intended relations from the query.
Handle MULTIPLE misinformations in a single query.

CRITICAL INSTRUCTIONS:
- ALWAYS include the release year when calling get_movie_details_by_title
- Format: "Title (YYYY)" - Example: "Bamboozled (2000)", "Girl 6 (1996)"
- If you don't know the year, make a reasonable guess based on context clues
- NEVER call get_movie_details_by_title with just the title without year

Process:
1️⃣ For EACH movie mentioned in the query:
   - Identify the movie title
   - Determine its release year (from query context, or make educated guess)
   - Call movies.get_movie_details_by_title("Title (YYYY)") with the year included
   - Example: If user says "Bamboozled", call with "Bamboozled (2000)"
   - This ensures the database returns the CORRECT movie

2️⃣ For EACH (movie, relation, person) combination mentioned:
   - Extract the movie title with year as "Title (YYYY)"
   - Identify the relation type (Directed_by, Starring, Music_by, Produced_by, Written_by)
   - Extract the mentioned person name exactly as stated
   - Note if multiple people/relations mentioned for same movie

3️⃣ Handle edge cases:
   - If title is very generic, try context to find year
   - If person name is incomplete/nickname, preserve as stated
   - Do NOT assume year - extract from query or context

STRICT OUTPUT (JSON, one line):
{
 "movies": [
  {
   "title": "<Title (YYYY)>",
   "found": true/false,
   "year": YYYY
  }
 ],
 "misinformations": [
  {
   "movie": "<Title (YYYY)>",
   "relation": "<Directed_by|Starring|Music_by|Produced_by|Written_by>",
   "mentioned_person": "<PersonName>"
  }
 ],
 "notes": "Any edge cases or ambiguities"
}
""",
        tools=[movies.get_movie_details_by_title],
    )

    # 2️⃣ ENHANCED FACT CHECKER
    # Validates each tuple and detects WRONG RELATIONS (not just wrong persons)
    FactChecker = Agent(
        name="FactChecker",
        llm_config=llm_config,
        description="Validate each (movie, relation, person) tuple and detect wrong relations.",
        system_message=f"""
You are the ENHANCED FACT CHECKER.

history_line (user preferences context):
{history_line or "[No prior history available]"}

Input: JSON from EntityResolver with array of misinformations.

CRITICAL INSTRUCTIONS:
- The movie title ALWAYS includes year: "Title (YYYY)"
- ALWAYS call get_movie_details_by_title with the EXACT title including year
- Example: "Bamboozled (2000)" - pass it exactly as received
- DO NOT strip the year from the movie title

For EACH misinformation tuple:
 1. Get the movie title WITH YEAR from input (format: "Title (YYYY)")
 2. Call movies.get_movie_details_by_title("Title (YYYY)") - include the year!
 3. From the result, extract ALL true people for claimed relation:
    - Directed_by → result.directors (ALL of them)
    - Starring → result.actors (ALL of them)
    - Music_by → result.composers (ALL of them)
    - Produced_by → result.producers (ALL of them)
    - Written_by → result.writers (ALL of them)
 4. Check if mentioned_person matches ANY true person (case-insensitive)
 5. If NO match:
    a. Search for mentioned_person in ALL other relations (directors, actors, etc)
    b. If found → person is real but WRONG RELATION
    c. If not found → person is FAKE/INEXISTENT
 6. Output ALL people in "true_people" array - for comprehensive retrieval

CRITICAL RULES:
- "true_people" MUST contain ALL people for the relation (not just the first)
- Include everyone from the relation list (all directors, all actors, etc)
- This allows Retriever to search for all contributors, improving recall
- Retriever will intelligently combine results to find best matches

Output JSON (one line):
{{
 "analysis": [
  {{
   "movie": "<Title (YYYY)>",
   "claimed_relation": "<Relation>",
   "mentioned_person": "<PersonName>",
   "is_misinformation": true/false,
   "reason": "<misinformation|correct_but_wrong_relation|person_not_found>",
   "actual_relation": "<Correct relation if wrong>",
   "correct_person": "<Actual primary person>",
   "true_people": ["<ONLY_FIRST_PRIMARY_PERSON>"]
  }}
 ]
}}
""",
        tools=[movies.get_movie_details_by_title],
    )

    # 3️⃣ SMART FACT CORRECTION AGENT
    # Handles multiple corrections and chooses correct person/relation intelligently
    FactCorrectionAgent = Agent(
        name="FactCorrectionAgent",
        llm_config=llm_config,
        description="Generate corrected conditions for complex/multi-misinformation queries.",
        system_message="""
You are the SMART FACT CORRECTION AGENT.

Input: JSON from FactChecker with analysis of misinformations, including "true_people" array.

CRITICAL: Valid Relation Names (EXACT SPELLING REQUIRED)
Your JSON output MUST use EXACTLY these relation names:
- "Directed_by" (NOT "director")
- "Starring" (NOT "actor")
- "Genre"
- "Language"
- "Produced_by" (NOT "producer")
- "Music_by" (NOT "composer")
- "Year"

Goal:
- For each misinformation, decide the CORRECTION strategy
- Create ONE primary_condition with first true person
- Create secondary_conditions for ALL other true_people (for comprehensive retrieval)
- Handle edge cases (multiple possible people, empty lists)

Strategy:
1️⃣ PRIMARY CONDITION: Use first person from true_people array
   - Relation: correct relation name from CRITICAL list
   - Value: first person from true_people
   
2️⃣ SECONDARY CONDITIONS: For each additional person in true_people
   - Create one condition per person
   - Same relation as primary
   - This enables Retriever to search for ALL contributors
   
3️⃣ For FAKE person → use correct_person from analysis
4️⃣ For WRONG RELATION → correct to proper relation name from CRITICAL list
5️⃣ If person is correct but relation wrong → extract TRUE relation

Output PRIMARY + SECONDARY CONDITIONS:
- Primary: first person from true_people (most canonical)
- Secondary: all other people (ensures comprehensive coverage)

STRICT OUTPUT (JSON, one line):
{
 "primary_condition": {"relation":"<ExactRelationName>","value":"<FirstPerson>"},
 "secondary_conditions": [{"relation":"<ExactRelationName>","value":"<SecondPerson>"},{"relation":"<ExactRelationName>","value":"<ThirdPerson>"}],
 "correction_summary": "<Human-readable explanation of what was corrected>",
 "confidence": "high|medium|low"
}

EXAMPLE WITH MULTIPLE DIRECTORS:
{"primary_condition":{"relation":"Directed_by","value":"Spike Lee"},"secondary_conditions":[{"relation":"Directed_by","value":"Kwame Jackson"},{"relation":"Directed_by","value":"Damon Dash"}],"correction_summary":"Movie has multiple directors/producers; searching for all","confidence":"high"}
""",
    )

    # 4️⃣ MULTI-HOP AWARE RETRIEVER
    # Processes multiple conditions and intelligently combines results
    Retriever = Agent(
        name="Retriever",
        llm_config=llm_config,
        description="Retrieve movies satisfying corrected conditions, handling multiple relations.",
        system_message="""
You are the MULTI-HOP AWARE RETRIEVER.

Input: JSON from FactCorrectionAgent with primary and secondary conditions.

CRITICAL: Relation Names
You MUST use EXACTLY these relation names when calling retrieve_titles_by_condition:
- "Directed_by" (NOT "director" or "directed by")
- "Starring" (NOT "actor" or "acts in")
- "Genre"
- "Language"
- "Produced_by" (NOT "producer")
- "Music_by" (NOT "composer")
- "Year"

Action:
1️⃣ For primary_condition:
   - Extract: (relation_name, value_name) from input JSON
   - Call: movies.retrieve_titles_by_condition(relation_name, value_name, limit=400)
   - VERIFY relation_name is from CRITICAL list above
   - These are the main candidates

2️⃣ For secondary_conditions (if any):
   - Call movies.retrieve_titles_by_condition for EACH secondary condition
   - For SAME relation (e.g., multiple directors): UNION results (combine all)
   - For DIFFERENT relations: INTERSECT with primary (keep only overlaps)
   - Example: If both directors and actors appear → find movies with BOTH relations

3️⃣ Smart Merging:
   - If secondary_conditions all have SAME relation → UNION (more inclusive)
   - If secondary_conditions have DIFFERENT relations → INTERSECT (strict multi-hop)
   - Deduplicate and sort alphabetically

4️⃣ Output result as [SEP]-joined line

Examples:
- Primary: ("Directed_by", "Spike Lee") → [A, B, C, D, ...]
- Secondary: ("Directed_by", "Other Director") → [C, D, E, F, ...] 
- Same relation → UNION: [A, B, C, D, E, F, ...]

- Primary: ("Directed_by", "Spike Lee") → [A, B, C, D, ...]
- Secondary: ("Starring", "Actor X") → [B, C, G, H, ...]
- Different relations → INTERSECT: [B, C, ...]

Output a single line:
Title A (YYYY) [SEP] Title B (YYYY) [SEP] ...
If no results, output exactly: NO_CANDIDATES.
""",
        tools=[movies.retrieve_titles_by_condition],
    )

    # 5️⃣ HISTORY-AWARE RECOMMENDER
    # Balances determinism with history preference
    Recommender = Agent(
        name="Recommender",
        llm_config=llm_config,
        description=f"Return top-{top_k} recommendations with history awareness.",
        system_message=f"""
You are the HISTORY-AWARE RECOMMENDER.

history_line (movies user previously watched):
{history_line or "[No prior history available]"}

Input: line of candidate titles separated by ' [SEP] '.

Steps:
1. Parse candidates and normalize (lowercase, trim, dedup)
2. Separate candidates into TWO groups:
   - [IN_HISTORY]: Titles appearing in history_line
   - [NEW]: Titles NOT in history_line
3. Sort each group alphabetically (A-Z for determinism)
4. Select top_k={top_k}:
   - Prioritize NEW titles first (to recommend diverse content)
   - If insufficient NEW titles, add from IN_HISTORY
   - Alternative: Balance 70% NEW, 30% IN_HISTORY
5. Output exactly one line of titles separated by ' [SEP] '
6. Do not add commentary

Example output:
New Movie (2000) [SEP] Another New (1999) [SEP] Previously Watched (1998)
""",
    )

    allowed_transitions = {
        EntityResolver: [FactChecker],
        FactChecker: [FactCorrectionAgent],
        FactCorrectionAgent: [Retriever],
        Retriever: [Recommender],
    }

    module = Module(
        name="misinformed_module",
        agents=[
            EntityResolver,
            FactChecker,
            FactCorrectionAgent,
            Retriever,
            Recommender,
        ],
        speaker_selection_method="round_robin",
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
    )

    return Orchestrator(
        name="misinformed_orchestrator",
        module=module,
        llm_config=llm_config,
        description="""
            MisInformed Query Orchestrator.
            Handles misinformed recommendation queries in which the user provides incorrect or hallucinated information about movies, such as misattributed directors, actors, or production details.
            It focuses on identifying and correcting these factual inconsistencies before generating recommendations, ensuring that the suggested movies are accurate and contextually aligned with the user’s stated interests.
            Examples of misinformed queries:
            I recently watched Spy Kids 2: The Island of Lost Dreams and Spy Kids 3-D: Game Over, both featuring Chris Savino's work. I'm curious to discover more films directed by Chris Savino from 2003 to 2005, as I appreciate his unique storytelling style.
            I recently watched Girl 6 and was fascinated by its direction. I know that Yoshihisa Kishimoto directed it, and I'm eager to find other films that he has directed. Additionally, I noticed that he starred in both Girl 6 and When We Were Kings, so if any of his directed films also feature him, that would be great!
            I just watched Electric Dreams and was fascinated by the direction of Alison Ball-Gabriel. I'm eager to find other movies directed by her to see her unique style again.
""",
    )
