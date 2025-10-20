from agents import Agent, Module, Orchestrator
from tools import books
from user_history_book import get_filtered_user_history


def create_misinformed_orchestrator(data, llm_config=None, use_memory=True, memory_size=10) -> Orchestrator:
    """
    Orchestrator for handling misinformed queries.
      
    Args:
        data: Dictionary with query context including:
            - query: User's query string
            - bookCount: Expected number of recommendations (top_k)
            - bookSubset: Ground truth book IDs
            - sharedRelationships: Correct relations connecting ground truth books
            - multihop_info: Complex relation info (for validation)
        llm_config: LLM configuration for agents
        use_memory: Whether to use user history as context
        memory_size: Number of historical books to include
    
    Returns:
        Orchestrator instance for processing misinformed queries
    """
    book_count = data.get("bookCount", 2)
    top_k = book_count if book_count > 0 else 2

    # ============== user history context ==============
    history_line = ""
    if use_memory:
        user_history = get_filtered_user_history(
            user_id=data.get("source_user", ""),
            groundtruth_book_ids=data.get("bookSubset", []),
            neo4j_conditions=data.get("sharedRelationships", []),
        )
        if user_history:
            limited_history = user_history[-memory_size:] if len(user_history) > memory_size else user_history
            history_line = " ".join(limited_history)

    # 1️⃣ ENHANCED ENTITY RESOLVER
    # Detects MULTIPLE misinformations and extracts all (book, relation, person) tuples
    EntityResolver = Agent(
        name="EntityResolver",
        llm_config=llm_config,
        description="Extract all mentioned books, people, and relations (multiple per query).",
        system_message="""
You are the ENHANCED ENTITY RESOLVER.

Goal:
Extract ALL mentioned book titles, people, and intended relations from the query.
Handle MULTIPLE misinformations in a single query.

CRITICAL INSTRUCTIONS:
- Do NOT include years when calling get_book_details_by_title
- Format: "Title" - Example: "The Great Gatsby", "1984"
- NEVER call get_book_details_by_title with year included

Process:
1️⃣ For EACH book mentioned in the query:
   - Identify the book title
   - Call books.get_book_details_by_title("Title") without year
   - Example: If user says "The Great Gatsby", call with "The Great Gatsby"
   - This ensures the database returns the CORRECT book

2️⃣ For EACH (book, relation, person) combination mentioned:
   - Extract the book title as "Title"
   - Identify the relation type (WRITTEN_BY, BELONGS_TO)
   - Extract the mentioned person name exactly as stated
   - Note if multiple people/relations mentioned for same book

3️⃣ Handle edge cases:
   - If title is very generic, try context to find specific book
   - If person name is incomplete/nickname, preserve as stated
   - Do NOT add years to book titles

STRICT OUTPUT (JSON, one line):
{
 "books": [
  {
   "title": "<Title>",
   "found": true/false
  }
 ],
 "misinformations": [
  {
   "book": "<Title>",
   "relation": "<WRITTEN_BY|BELONGS_TO>",
   "mentioned_person": "<PersonName>"
  }
 ],
 "notes": "Any edge cases or ambiguities"
}
""",
        tools=[books.get_book_details_by_title],
    )

    # 2️⃣ ENHANCED FACT CHECKER
    # Validates each tuple and detects WRONG RELATIONS (not just wrong persons)
    FactChecker = Agent(
        name="FactChecker",
        llm_config=llm_config,
        description="Validate each (book, relation, person) tuple and detect wrong relations.",
        system_message=f"""
You are the ENHANCED FACT CHECKER.

history_line (user preferences context):
{history_line or "[No prior history available]"}

Input: JSON from EntityResolver with array of misinformations.

CRITICAL INSTRUCTIONS:
- The book title does NOT include year: "Title"
- ALWAYS call get_book_details_by_title with the EXACT title without year
- Example: "The Great Gatsby" - pass it exactly as received
- DO NOT add year to the book title

For EACH misinformation tuple:
 1. Get the book title from input (format: "Title")
 2. Call books.get_book_details_by_title("Title") - without year!
 3. From the result, extract ALL true people for claimed relation:
    - WRITTEN_BY → result.authors (ALL of them)
    - BELONGS_TO → result.categories (ALL of them)
 4. Check if mentioned_person matches ANY true person (case-insensitive)
 5. If NO match:
    a. Search for mentioned_person in ALL other relations (authors, categories, etc)
    b. If found → person is real but WRONG RELATION
    c. If not found → person is FAKE/INEXISTENT
 6. Output ALL people in "true_people" array - for comprehensive retrieval

CRITICAL RULES:
- "true_people" MUST contain ALL people for the relation (not just the first)
- Include everyone from the relation list (all authors, all categories, etc)
- This allows Retriever to search for all contributors, improving recall
- Retriever will intelligently combine results to find best matches

Output JSON (one line):
{{
 "analysis": [
  {{
   "book": "<Title>",
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
        tools=[books.get_book_details_by_title],
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
- "WRITTEN_BY" (NOT "author" or "written by")
- "BELONGS_TO" (NOT "category" or "genre")

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

EXAMPLE WITH MULTIPLE AUTHORS:
{"primary_condition":{"relation":"WRITTEN_BY","value":"F. Scott Fitzgerald"},"secondary_conditions":[{"relation":"WRITTEN_BY","value":"Zelda Fitzgerald"}],"correction_summary":"Book has multiple authors; searching for all","confidence":"high"}
""",
    )

    # 4️⃣ MULTI-HOP AWARE RETRIEVER
    # Processes multiple conditions and intelligently combines results
    Retriever = Agent(
        name="Retriever",
        llm_config=llm_config,
        description="Retrieve books satisfying corrected conditions, handling multiple relations.",
        system_message="""
You are the MULTI-HOP AWARE RETRIEVER.

Input: JSON from FactCorrectionAgent with primary and secondary conditions.

CRITICAL: Relation Names
You MUST use EXACTLY these relation names when calling retrieve_titles_by_condition:
- "WRITTEN_BY" (NOT "author" or "written by")
- "BELONGS_TO" (NOT "category" or "genre")

Action:
1️⃣ For primary_condition:
   - Extract: (relation_name, value_name) from input JSON
   - Call: books.retrieve_titles_by_condition(relation_name, value_name, limit=400)
   - VERIFY relation_name is from CRITICAL list above
   - These are the main candidates

2️⃣ For secondary_conditions (if any):
   - Call books.retrieve_titles_by_condition for EACH secondary condition
   - For SAME relation (e.g., multiple authors): UNION results (combine all)
   - For DIFFERENT relations: INTERSECT with primary (keep only overlaps)
   - Example: If both authors and categories appear → find books with BOTH relations

3️⃣ Smart Merging:
   - If secondary_conditions all have SAME relation → UNION (more inclusive)
   - If secondary_conditions have DIFFERENT relations → INTERSECT (strict multi-hop)
   - Deduplicate and sort alphabetically

4️⃣ Output result as [SEP]-joined line

Examples:
- Primary: ("WRITTEN_BY", "F. Scott Fitzgerald") → [A, B, C, D, ...]
- Secondary: ("WRITTEN_BY", "Other Author") → [C, D, E, F, ...] 
- Same relation → UNION: [A, B, C, D, E, F, ...]

- Primary: ("WRITTEN_BY", "F. Scott Fitzgerald") → [A, B, C, D, ...]
- Secondary: ("BELONGS_TO", "Fiction") → [B, C, G, H, ...]
- Different relations → INTERSECT: [B, C, ...]

Output a single line:
Title A [SEP] Title B [SEP] ...
If no results, output exactly: NO_CANDIDATES.
""",
        tools=[books.retrieve_titles_by_condition],
    )

    # 5️⃣ HISTORY-AWARE RECOMMENDER
    # Balances determinism with history preference
    Recommender = Agent(
        name="Recommender",
        llm_config=llm_config,
        description=f"Return top-{top_k} recommendations with history awareness.",
        system_message=f"""
You are the HISTORY-AWARE RECOMMENDER.

history_line (books user previously read):
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
New Book [SEP] Another New Book [SEP] Previously Read Book
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
            Handles misinformed recommendation queries in which the user provides incorrect or hallucinated information about books, such as misattributed authors, categories, or publication details.
            It focuses on identifying and correcting these factual inconsistencies before generating recommendations, ensuring that the suggested books are accurate and contextually aligned with the user's stated interests.
            Examples of misinformed queries:
            I recently read The Great Gatsby and To Kill a Mockingbird, both written by Ernest Hemingway. I'm curious to discover more books written by Ernest Hemingway, as I appreciate his unique writing style.
            I recently read 1984 and was fascinated by its author. I know that George Orwell wrote it, and I'm eager to find other books that he has written. Additionally, I noticed that he wrote both 1984 and Animal Farm, so if any of his books are in the same category, that would be great!
            I just read Brave New World and was fascinated by the writing of Aldous Huxley. I'm eager to find other books written by him to see his unique style again.
""",
    )
