from agents import Agent, Module, Orchestrator
from tools import books

from capabilities.memory import ListMemory, MemoryContent
from user_history_book import get_filtered_user_history


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
    - 'direct_description_query': User query mentioning reference books
    - 'bookCount': Target K for recommendations
    - 'multihop_info': Array of reference books with relations (for validation)
    - 'sharedRelationships': Expected shared relations (for validation)
    - 'bookSubset': Ground truth expected recommendations (for validation)
    """
    
    # ======== Parameters & Memory Setup ========
    bookCount = data.get("bookCount", None)
    top_k_value = bookCount if isinstance(bookCount, int) and bookCount > 0 else 3
    
    history_line = ""
    if use_memory:
        user_history = get_filtered_user_history(
            user_id=data["source_user"],
            groundtruth_book_ids=data.get("bookSubset", []),
            neo4j_conditions=data.get("sharedRelationships", []),
        )
        limited_history = user_history[-memory_size:] if len(user_history) > memory_size else user_history
        if limited_history:
            history_line = " ".join(limited_history)
    
    # ============ 1️⃣ TITLE NORMALIZER ============
    TitleNormalizer = Agent(
        name="TitleNormalizer",
        llm_config=llm_config,
        description="Normalize book titles from query (handle underscores, quotes, article position).",
        system_message="""
You are the TITLE NORMALIZER.

Input: User's direct_description_query mentioning reference books (e.g., "The Great Gatsby")

Task:
1. Extract ALL book titles mentioned in the query (usually 2-3 reference books)
2. Normalize each title:
   - Replace underscores "_" with spaces
   - Remove extra quotes (keep only inner quotes if any)
   - Handle article positioning: "The_Great_Gatsby" → "The Great Gatsby" OR "Great Gatsby, The"?
   - Preserve original title format
3. Output exact titles in format: "Title"

Normalization examples:
- "The_Great_Gatsby" → "The Great Gatsby"
- "A_Clockwork_Orange" → "A Clockwork Orange"
- "1984" → "1984"

CRITICAL:
- Do NOT add years to book titles (unlike movies)
- If unsure about article position, try BOTH variants:
  "The Title" and "Title, The"
- Output one title per line, then final summary

OUTPUT (STRICT):
Normalized Titles:
- Reference 1: Title A
- Reference 2: Title B
- (Reference 3: Title C if present)
""",
        tools=[],
    )
    
    # ============ 2️⃣ COMMON ATTRIBUTE EXTRACTOR ============
    CommonAttributeExtractor = Agent(
        name="CommonAttributeExtractor",
        llm_config=llm_config,
        description="Extract shared authors/categories from the two reference books.",
        system_message=f"""
You are the COMMON ATTRIBUTE EXTRACTOR.

Input: 
- Two normalized reference book titles (from TitleNormalizer)
- Example: "The Great Gatsby" and "To Kill a Mockingbird"

history_line (user preferences context):
{history_line or "[No prior history available]"}

Task:
1. For EACH reference book title:
   - Call: books.get_book_details_by_title("Title")
   - Extract ALL attributes from each relation:
     * authors (from "WRITTEN_BY" relation)
     * categories (from "BELONGS_TO" relation)

2. Find INTERSECTION across both books:
   - AUTHORS common to BOTH? → YES: list them
   - CATEGORIES common to BOTH? → YES: list them

3. Validate intersection is non-empty:
   - If empty → "No common attributes found" (should not happen in well-formed queries)
   - If found → proceed

CRITICAL:
- Do NOT include years in book titles when calling get_book_details_by_title
- Return ALL common people for each relation (not just first)
- If "WRITTEN_BY" is common relation, list ALL shared authors

OUTPUT (JSON, one line):
{{
 "reference_books": ["Title A", "Title B"],
 "common_attributes": {{
   "authors": ["Author1", "Author2"],
   "categories": ["Category1"]
 }},
 "primary_relation": "WRITTEN_BY",
 "primary_people": ["Author1", "Author2"],
 "is_valid": true
}}
""",
        tools=[books.get_book_details_by_title],
    )
    
    # ============ 3️⃣ RELATION TYPE DETECTOR ============
    RelationTypeDetector = Agent(
        name="RelationTypeDetector",
        llm_config=llm_config,
        description="Detect which relation type (WRITTEN_BY, BELONGS_TO, etc.) the query is asking for.",
        system_message="""
You are the RELATION TYPE DETECTOR.

Input:
- Original user query (direct_description_query)
- Common attributes found (from CommonAttributeExtractor)

Task:
Determine the PRIMARY relation the user is asking for:

1. Analyze query language:
   - "same author" / "written by" → WRITTEN_BY (PRIMARY)
   - "same category" / "genre" / "type" → BELONGS_TO (PRIMARY)

2. Cross-validate with common attributes found:
   - If common_attributes has matching people → confirm relation
   - If common_attributes is empty for detected relation → flag as error

3. Determine if secondary relations should be used:
   - Query mentions only 1 relation type? → Use only primary
   - Query mentions "also in" + multiple relations? → May need secondary

CRITICAL:
- DEFAULT RELATION: "WRITTEN_BY" (most common for books)
- If query ambiguous, use "WRITTEN_BY"
- Output relation name EXACTLY as in DB: "WRITTEN_BY", "BELONGS_TO"

OUTPUT (JSON, one line):
{{
 "primary_relation": "WRITTEN_BY",
 "primary_people": ["Author1", "Author2"],
 "secondary_relations": [],
 "secondary_people": {{}},
 "query_clarity": "high|medium|low",
 "detected_language_hints": ["same author", "written by"]
}}
""",
        tools=[],
    )
    
    # ============ 4️⃣ MULTI-HOP RETRIEVER ============
    MultiHopRetriever = Agent(
        name="MultiHopRetriever",
        llm_config=llm_config,
        description="Find OTHER books with the common authors/categories.",
        system_message=f"""
You are the MULTI-HOP RETRIEVER.

Input:
- Primary relation type: "WRITTEN_BY" or "BELONGS_TO" etc.
- Common people: ["Author1", "Author2"] or ["Category1"]

Task:
1. For EACH common person:
   - Call: books.retrieve_titles_by_condition(relation, person, limit=400)
   - Relation must be EXACTLY: "WRITTEN_BY", "BELONGS_TO"
   - person must be EXACT name from common attributes

2. Combine results:
   - All people are from SAME relation? → UNION all results (get all books)
   - Multiple relations? → May INTERSECT (stricter)

3. Filter out reference books:
   - Reference books should NOT appear in final recommendations
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
Title A [SEP] Title B [SEP] Title C
Or if no results: NO_CANDIDATES
""",
        tools=[books.retrieve_titles_by_condition],
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
- Candidate pool line: "Title A [SEP] Title B [SEP] ..."
- Target K: {top_k_value}

Task:
1. Parse candidate titles and deduplicate (normalized: lowercase, trim, replace '_' with ' ')
2. Rank by:
   - Preference alignment (from history_line): + if matches user's typical choices
   - Novelty: + if not in history, - if seen before
   - Diversity: small bonus for variety in authors/categories
   - Conflict avoidance: - if conflicting with history

3. Select exactly {top_k_value} items
   - If pool has fewer than {top_k_value} items, return all
   - Sort alphabetically (deterministic tie-breaking)

STRICT Output:
- Exactly ONE LINE:
  Title A [SEP] Title B [SEP] Title C
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

Handles implicit/inferential recommendations where users reference books and ask for 
similar recommendations based on shared attributes (authors, categories, etc.).

Multi-hop aware: Extracts shared people from reference books → Finds OTHER books with same people.

Examples:
- "Please recommend books written by the same author as The Great Gatsby 
   and To Kill a Mockingbird." → Find common author → Find OTHER books
- "Please recommend books in the same category as 1984 
   and Brave New World." → Find common category → Find OTHER books
""",
    )
    
    return orchestrator
