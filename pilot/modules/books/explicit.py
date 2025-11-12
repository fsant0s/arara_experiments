from agents import Agent, Module, Orchestrator
from tools import books

from user_history_book import get_filtered_user_history


def create_explicit_orchestrator(
    data: dict,
    llm_config=None,
    use_memory: bool = True,
    memory_size: int = 10,
) -> Orchestrator:
    """
   Explicit module:
      - RetrieverAgent: retrieves ALL relevant items (no top-k).
      - RecommenderExplicitgent: selects only top_k = bookCount based on the user's OLD preferences
        provided in `history_line` (no adding items; no using tools).
      - Final output: a single line with ' [SEP] ' between titles.
    """

    bookCount = data.get("bookCount", None)
    top_k_value = bookCount if isinstance(bookCount, int) and bookCount > 0 else 3

    # ======== memória e history_line ========
    history_line = ""
    if False:
        user_history = get_filtered_user_history(
            user_id=data["source_user"],
            groundtruth_book_ids=data["bookSubset"],
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
            "Retrieves a broad list of books based on explicit user query parameters "
            "and outputs a single '[SEP]' line. Does NOT enforce top-k; returns all relevant items (deduped)."
        ),
        system_message="""
## 🧠 RETRIEVER AGENT

### Input:
A user request with explicit information such as authors, categories, or other book attributes.

---

### Goal:
- Retrieve a **BROAD, COMPREHENSIVE** list of relevant books (no top-k limit).  
- Output **exactly ONE LINE** containing all titles separated by `' [SEP] '`.

---

### Process:

1. **Collect** from all relevant book tools:
   - `books.get_books_by_author`  
   - `books.get_books_by_category`  
   - `books.get_books_by_relation` ← Allowed relations ONLY:  
     `['WRITTEN_BY', 'BELONGS_TO']`

   ⚠️ **Important rule:**  
   Always use the **canonical form** of the relation exactly as listed above.  
   - If a user mentions a near-synonym (e.g., *written by*), map it to **WRITTEN_BY**.  
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
  `Title A [SEP] Title B [SEP] Title C`  
- Use `' [SEP] '` (single spaces).  
- No leading/trailing `[SEP]`, **no commentary or extra lines**.

---

### Allowed book relations (use EXACTLY these):
`['WRITTEN_BY', 'BELONGS_TO']`
""",
        tools=[
            books.get_books_by_author,
            books.get_books_by_category,
            books.get_books_by_relation,
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
Select exactly top_k={top_k_value} books from the Retriever's list,
using ONLY the user's PAST preferences below (history_line) to rank.

history_line (past user preferences; use ONLY this to decide relevance):
{history_line or "[No prior history available]"}

Inputs:
- One single line from the Retriever containing MANY titles (eligible set).
- You MUST NOT add new books, fetch new data, or use tools.

Scoring (use only signals derivable from history_line):
- + Author overlap with names present in history_line.
- + Category keywords overlap with terms present in history_line.
- (Optional) Small boost for titles explicitly mentioned in history_line.
Tie-breakers (deterministic):
- 1) Alphabetical by title.

Selection:
- Deduplicate by normalized title (lower/trim/strip quotes, '_'→' '), then keep ORIGINAL surface forms.
- Rank all eligible items using the rules above.
- Output EXACTLY {top_k_value} items; if the Retriever list has fewer than {top_k_value}, output all available.

STRICT Output:
- Exactly ONE LINE:
  Title A [SEP] Title B [SEP] Title C
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
            Handles explicit recommendation queries in which the user clearly specifies the entities or attributes that define the recommendation scope — such as an author or category.
            It focuses on generating recommendations that directly satisfy the explicit condition expressed in the query, maintaining a clear and factual connection to the mentioned entity.
            Examples of explicit queries:
            Can you suggest some books written by Stephen King?
            Can you suggest some books in the Science Fiction category?
            Can you recommend some books by J.K. Rowling?
        """,
    )

    return orchestrator
