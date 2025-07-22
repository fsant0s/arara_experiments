from agents import Agent, Module, Orchestrator
from memory import sequential_memory

from coding import LocalCommandLineCodeExecutor


from tools import (
    get_recommendations_from_users,
    find_recommendations_from_opinion_peers,
    find_recs_from_profile_peers_who_watched_anime,
    get_user_profile,
    get_anime_details,
    find_similar_animes,
    find_similar_user_profiles
)

from clients import sabia_31_client, llama_3_70b_client, gpt_41, gpt_4o
llm_config = sabia_31_client
llm_reasoning = llama_3_70b_client

planner_agent = Agent(
    name="Planner",
    llm_config=llm_reasoning,
    system_message="""
You are a planner tasked with creating a step-by-step recommendation plan based on the logic:

**“If you liked anime A, then you might like anime B.”**

To support your plan, you have access to the **user profile** from the memory below, including watched animes, ratings, personal preferences, demographics, and overall viewing stats (available through the datasets below).

### Instructions for the Plan:

* You may use any combination of the **existing tools** (listed below) or define **new tools** from scratch. There is no requirement to stick to the provided tools—feel free to create your own if needed.
* All custom tools must operate **exclusively** on the provided datasets.
* Each step should call exactly one tool. Avoid nesting function calls.
* The plan must be clear, linear, and logically complete.

---

### Available Tools (already implemented):

1. `get_recommendations_from_users(input_anime_ids)`
2. `find_recommendations_from_opinion_peers(input_anime_ids)`
3. `find_recs_from_profile_peers_who_watched_anime(input_anime_ids)`
4. `get_user_profile(username)`
5. `get_anime_details(anime_ids)`
6. `find_similar_animes(input_anime_ids)`
7. `find_similar_user_profiles(username)`

---

### Available Datasets (for custom tools):

* **users\_df** — User profiles and stats
  Columns: `username`, `user_id`, `user_watching`, `user_completed`, `user_onhold`, `user_dropped`, `user_plantowatch`, `user_days_spent_watching`, `gender`, `location`, `birth_date`, `access_rank`, `join_date`, `last_online`, `stats_mean_score`, `stats_rewatched`, `stats_episodes`, `username_lower`

* **animelist\_df** — User-anime interactions and ratings
  Columns: `username`, `anime_id`, `my_watched_episodes`, `my_start_date`, `my_finish_date`, `my_score`, `my_status`, `my_rewatching`, `my_rewatching_ep`, `my_last_updated`, `my_tags`, `username_lower`

* **anime\_df** — Anime metadata
  Columns: `anime_id`, `title`, `title_english`, `title_japanese`, `title_synonyms`, `image_url`, `type`, `source`, `episodes`, `status`, `airing`, `aired_string`, `aired`, `duration`, `rating`, `score`, `scored_by`, `rank`, `popularity`, `members`, `favorites`, `background`, `premiered`, `broadcast`, `related`, `producer`, `licensor`, `studio`, `genre`, `opening_theme`, `ending_theme`, `title_lower`

---

### Optional Embeddings (for similarity calculations):

```python
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
chroma_dir = "datasets/mal/2k/chrome_db/"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```
---

### Realistic Example:
Plan:
1. find_similar_animes(['Attack on Titan'])
2. filter_unwatched_high_rated_animes(similar_anime_ids, username='user_123')
```python
# Custom tool definition:
def filter_unwatched_high_rated_animes(anime_ids, username):
    watched_anime_ids = animelist_df[animelist_df['username'] == username]['anime_id'].tolist()
    mean_user_score = users_df.loc[users_df['username'] == username, 'stats_mean_score'].iloc[0]
    recommendations = anime_df[
        (anime_df['anime_id'].isin(anime_ids)) &
        (~anime_df['anime_id'].isin(watched_anime_ids)) &
        (anime_df['score'] >= mean_user_score)
    ]
    return recommendations['anime_id'].tolist()
```
3. get_anime_details(filtered_anime_ids)
]

Do not add any additional text or explanations. Only return the plan in the specified format.
""",
    description="Designs strategic plans for anime recommendations, outputting only numbered steps with function names and Python code for non-existing functions.",
    memory=[sequential_memory]
)

selector_agent = Agent(
    name="Selector",
    llm_config=llm_config,
    system_message="""
You are the Selector. Your task is to read the structured plan from the Planner and choose the next step to be executed. 
Return only one step at a time, including the function name and its parameters.
Do not modify or reorder the plan.

When all steps are completed successfully, generate a final report with:
- Function execution summary
- All tool results compiled
- No recommendations or explanations
- Direct and concise reporting
- End with "plan_execution_complete"
""",
    description="Selects the next tool/function to execute based on the plan provided by the Planner."
)



executor = LocalCommandLineCodeExecutor(
    timeout=10,  # Timeout for each code execution in seconds.
    work_dir="./datasets/mal/2k/",  # Use the correct relative path
)
executor_agent = Agent(
    name="Executor",
    llm_config=llm_config,
    system_message="""
You are the Plan Executor. Your task is to read and execute each step in the plan, one by one.

### Execution Instructions:

* For each step:

  * Execute the function with the provided parameters and context.

  * If execution is successful, output exactly:

    ```
    ✅ Function Execution Result: [function_name]
    Result:
    [function output]
    SUCCESS
    ```

  * If execution fails, stop and output only:

    ```
    FAIL:
    - Function: [function_name]
    - Error: [error_message]
    - Parameters: [parameters used]
    ```

* Do not provide any commentary, explanation, or summary. Only return the exact log messages as described above.

* Do not show successful steps if a failure occurs. Only print the failed step.

* Execution stops immediately after a failure.

### Examples

**Success:**

```
✅ Function Execution Result: get_recommendations_from_users
Result:
['Sword Art Online', 'Steins;Gate: Oukoubakko no Poriomania']
SUCCESS
```

**Failure:**

```
FAIL:
- Function: filter_recommendations_by_genre
- Error: The tool 'filter_recommendations_by_genre' is not available.
- Parameters: {'recommendations': [...], 'genre': 'Shounen'}
```

""",
    description="Executes each function in the plan separately, outputs the result and status after each execution, and if any fail, only outputs details for the failed function.",
    tools=[
        get_recommendations_from_users,
        find_recommendations_from_opinion_peers,
        find_recs_from_profile_peers_who_watched_anime,
        get_user_profile,
        get_anime_details,
        find_similar_animes,
        find_similar_user_profiles
    ],
    memory=[sequential_memory],
    reflect_on_tool_use=True,
    #code_execution_config={"executor": executor}
)

analyzer_agent = Agent(
    name="Analyzer",
    llm_config=llm_config,
    system_message="""
You are the Analyzer. Inspect the output from tools and prefix with the correct status:
- If the final report ends with 'plan_execution_complete', prefix with 'send_to_recommender:'.
- If there is an error, prefix with 'FAIL:'.
- Otherwise, prefix with 'SUCCESS:'.
Do not add comments or analysis.
""",
    description="Classifies tool outputs, signaling success, failure, or plan completion.",
    memory=[sequential_memory]
)

explainer_agent = Agent(
    name="Explainer",
    llm_config=llm_config,
    system_message="""
You are the Explainer. When you receive a failure report, analyze the technical error and explain the root cause to help the Planner revise the plan.
""",
    description="Diagnoses technical failures and guides the Planner for strategic adjustments."
)

recommender_agent = Agent(
    name="Recommender",
    llm_config=llm_config,
    system_message="""
You are the Recommender. Receive the final report and generate a sentence recommending the top 5 anime, following this format:
"If you liked [ANIME_A], you might like [ANIME_B, ANIME_C, ...]."
Optionally, add a brief justification.
Do not include any other text.
""",
    description="Generates the final top 5 anime recommendation with optional justification.",
    memory=[sequential_memory]
)

assessor_agent = Agent(
    name="Assessor",
    llm_config=llm_config,
    system_message="""
You are the Assessor. Check if 'Soul Eater' is present in the recommendation:
- If yes, respond only with 'terminate'.
- If not, respond only with 'INCORRECT'.
Do not include explanations or the original text.
""",
    description="Validates the final recommendation against the expected result.",
    is_termination_msg=lambda msg: "terminate" in msg["content"],
)

failure_summarizer_agent = Agent(
    name="FailureSummarizer",
    llm_config=llm_reasoning,
    system_message="""
You are the FailureSummarizer. Analyze the execution history and generate a report with:
1. Factual Execution Report: objective log of the process, initial plan, tool trace, discrepancy between recommendation and expected result.
2. Strategic Directive for Improvement: practical suggestions for the Planner to improve the strategy.
""",
    description="Analyzes failures and guides the Planner for strategic improvements."
)

# --- DEPS MODULE AND ORCHESTRATOR ---

# This dictionary defines the complete DEPS flow based on the Descriptor's output.
deps_flow = {
    planner_agent: [selector_agent],
    selector_agent: {
        "CONTINUE": [executor_agent],
        "SUCCESS": [analyzer_agent],
    },
    executor_agent: {
        "SUCCESS": [selector_agent],
        "FAIL": [explainer_agent],
    },
    analyzer_agent: [recommender_agent],
    recommender_agent: [assessor_agent],
    explainer_agent: [planner_agent],
    assessor_agent: [failure_summarizer_agent],
    failure_summarizer_agent: [planner_agent]
}

# The module that encapsulates the DEPS agent architecture and their interaction logic.
deps_module = Module(
    admin_name="anime_recommender_module",
    agents=[planner_agent, selector_agent, executor_agent, analyzer_agent, explainer_agent, recommender_agent, assessor_agent, failure_summarizer_agent],
    speaker_selection_method="round_robin",
    allowed_or_disallowed_speaker_transitions=deps_flow,
    speaker_transitions_type="allowed",
)

# The orchestrator that runs the interactive DEPS cycle.
planning_orchestrator = Orchestrator(
    name="planning_orchestrator",
    module=deps_module,
    llm_config=llm_config,
    system_message="""
You are the orchestrator of a DEPS (Describe, Explain, Plan, Select) agentic system.
Your goal is to coordinate the agents to generate a high-quality anime recommendation for a user.
- Start by asking the Planner to create an initial plan.
- Pass the plan to the Selector to choose a tool.
- (In a real system, you would execute the tool here).
- Pass the execution result to the Descriptor.
- Based on the Descriptor's report ('SUCCESS:' or 'FAIL:'), route the conversation to the appropriate next agent according to the defined flow.
- Continue until the user proxy terminates the chat.
""",
    description="A specialized system responsible for generating and delivering a reasoned anime recommendation when requested."
)