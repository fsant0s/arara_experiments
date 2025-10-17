# modules/misinformed_optimized.py — Versão Otimizada
# Handles multiple misinformations, wrong relations, and complex queries

from agents import Agent, Module, Orchestrator
from tools import movies
from user_history import get_filtered_user_history
import json


def create_misinformed_orchestrator(data, llm_config=None, use_memory=True, memory_size=10):
    """
    Optimized orchestrator for handling misinformed queries.
    
    Key improvements:
    ✅ Detects multiple misinformations in single query
    ✅ Identifies both wrong persons AND wrong relations
    ✅ Handles edge cases (non-existent people, ambiguous names)
    ✅ Leverages dataset context (sharedRelationships, multihop_info)
    ✅ Combines multiple corrections intelligently
    
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
Extract ALL mentioned movie titles (with release year), people, and intended relations from the query.
Handle MULTIPLE misinformations in a single query.

Process:
1️⃣ For EACH movie mentioned:
   - Call movies.get_movie_details_by_title(title (YYYY)) to get canonical data
   - Extract release year and rebuild title as "Title (YYYY)"
   - If NOT found, note it as an error but continue

2️⃣ For EACH (movie, relation, person) combination mentioned:
   - Extract the movie title (with year)
   - Identify the relation type (Directed_by, Starring, Music_by, Produced_by, Written_by)
   - Extract the mentioned person name
   - Note if the same relation/movie appears multiple times (multiple misinformations)

3️⃣ Handle edge cases:
   - Multiple people mentioned for same relation: list all
   - Same person in different relations: extract both
   - Ambiguous/incomplete names: preserve as stated by user

STRICT OUTPUT (JSON, one line):
{
 "movies": [
  {
   "title": "<Title (YYYY)>",
   "found": true/false
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

For EACH misinformation tuple:
 1. Call movies.get_movie_details_by_title(movie) to get authoritative data
 2. For the claimed relation, extract true people:
    - Directed_by → details.directors
    - Starring → details.actors
    - Music_by → details.composers
    - Produced_by → details.producers
    - Written_by → details.writers
 3. Check if mentioned_person matches ANY in true people (case-insensitive)
 4. If NO match:
    a. Search for mentioned_person in ALL other relations
    b. If found → person is real but WRONG RELATION
    c. If not found → person is FAKE/INEXISTENT
 5. Output detailed analysis for EACH misinformation

CRITICAL: Check all relations, not just the claimed one!

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
   "correct_person": "<Actual person if different>",
   "true_people": ["<TruePersonA>", "<TruePersonB>"]
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

Input: JSON from FactChecker with analysis of misinformations.

Goal:
- For each misinformation, decide the CORRECTION strategy
- Select the correct person/relation to use for retrieval
- Handle edge cases (multiple possible people, empty lists)

Strategy:
1️⃣ If person is FAKE → use correct_person from analysis
2️⃣ If WRONG RELATION → use actual_relation + correct_person
3️⃣ If person is correct but claimed multiple relations → extract the TRUE relation
4️⃣ If multiple true_people → choose most prominent (first in list, or most common)

Output PRIMARY CONDITION:
- Should enable retrieval of ground truth movies
- If multiple corrections conflict → prioritize the claimed relation
- If relation is wrong → correct it

STRICT OUTPUT (JSON, one line):
{
 "primary_condition": {"relation":"<Relation>","value":"<CorrectPerson>"},
 "secondary_conditions": [{"relation":"<Relation>","value":"<CorrectPerson>"}],
 "correction_summary": "<Human-readable explanation of what was corrected>",
 "confidence": "high|medium|low"
}
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

Action:
1️⃣ For primary_condition:
   - Call movies.retrieve_titles_by_condition(relation, value, limit=400)
   - These are the main candidates

2️⃣ For secondary_conditions (if any):
   - Call movies.retrieve_titles_by_condition for each
   - INTERSECT results with primary (keep only movies in ALL lists)
   - This ensures multi-hop correctness

3️⃣ Output result as [SEP]-joined line

Example:
- Primary: ("Directed_by", "Spike Lee") → [A, B, C, D, ...]
- Secondary: ("Starring", "someone") → [B, C, D, E, ...]
- Final: [B, C, D, ...] (intersection)

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

    # 6️⃣ OPTIONAL: RESPONSE GENERATOR (for better UX)
    ResponseGenerator = Agent(
        name="ResponseGenerator",
        llm_config=llm_config,
        description="Craft user-friendly response with correction explanation.",
        system_message=f"""
You are the RESPONSE GENERATOR.

You receive:
- correction_summary: What was corrected in the query
- recommendations: Final top-k recommendations

Goal: Create a natural, helpful response for the user

Format:
1. Start with friendly acknowledgment of their interest
2. Gently correct the misinformation
3. Present recommendations based on the correct fact
4. Make it conversational, not robotic

Example Input:
- correction_summary: "Actually, Girl 6 (1996) was directed by Spike Lee, not Yoshihisa Kishimoto"
- recommendations: "Do the Right Thing (1989) [SEP] Bamboozled (2000) [SEP] Clockers (1995)"

Example Output:
"I appreciate your interest! Just to clarify, Girl 6 (1996) was actually directed by Spike Lee, not Yoshihisa Kishimoto. 
Since you enjoyed that film, here are some other excellent films he directed: 
Do the Right Thing (1989), Bamboozled (2000), and Clockers (1995). 
Each showcases his distinctive storytelling style!"
""",
    )

    # 🔁 allowed transitions (with optional ResponseGenerator)
    allowed_transitions = {
        EntityResolver: [FactChecker],
        FactChecker: [FactCorrectionAgent],
        FactCorrectionAgent: [Retriever],
        Retriever: [Recommender],
        Recommender: [ResponseGenerator],
    }

    module = Module(
        name="misinformed_module_optimized",
        agents=[
            EntityResolver,
            FactChecker,
            FactCorrectionAgent,
            Retriever,
            Recommender,
            ResponseGenerator,
        ],
        speaker_selection_method="round_robin",
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
    )

    return Orchestrator(
        name="misinformed_orchestrator_optimized",
        module=module,
        llm_config=llm_config,
        description="""
### 🎬 Enhanced Misinformed Queries Module (v2)

Handles user queries with **factual inaccuracies** about movies or creators.

✅ Detects MULTIPLE misinformations per query  
✅ Identifies WRONG RELATIONS (not just wrong persons)  
✅ Handles edge cases (fake people, ambiguous names, complex queries)  
✅ Validates relations against all movie metadata  
✅ Combines multiple corrections intelligently  
✅ Uses user history context for recommendations  
✅ Generates friendly corrections for user education  

Key Improvements:
1. **Multi-Detection**: Finds all (movie, relation, person) misinformation tuples
2. **Relation Validation**: Checks if person exists in other relations (Music_by vs Directed_by)
3. **Robust Checking**: Handles fake people, ambiguous names, complex multi-hop cases
4. **Intelligent Correction**: Decides between correcting person vs relation vs both
5. **Context-Aware**: Uses sharedRelationships, history, and multihop_info
6. **User Education**: Explains what was corrected in friendly language

Workflow:
1. Extract all movies, people, and relations from query
2. Validate each tuple against DB, detect wrong relations
3. Generate corrected condition(s)
4. Retrieve movies satisfying corrected facts
5. Return top-k deterministic recommendations
6. Generate friendly response explaining the correction

Example:
Query: "I recently watched Bamboozled and was impressed by John Williams' direction..."

→ ENTITY RESOLVER: Detects: John Williams + Directed_by + Bamboozled
→ FACT CHECKER: John Williams is real (composer), not director. Spike Lee directed it.
→ FACT CORRECTION: Corrected condition: (Directed_by, Spike Lee)
→ RETRIEVER: Finds Spike Lee movies
→ RECOMMENDER: Returns top-k Spike Lee films
→ RESPONSE: "Actually, Bamboozled was directed by Spike Lee, not John Williams (who composed the score). 
             Here are other films directed by Spike Lee you might enjoy..."
""",
    )
