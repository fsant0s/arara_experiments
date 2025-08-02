from agents import Agent, Module, Orchestrator
from memories import sequential_memory

from promtps import (
    planner_agent_prompt,
    selector_agent_prompt,
    explainer_agent_prompt,
    executor_agent_prompt,
    analyzer_agent_prompt,
    recommender_agent_prompt,
)

from tools import (
    get_recommendations_from_users,
    find_recommendations_from_opinion_peers,
    find_recs_from_profile_peers_who_watched_anime,
    get_user_profile,
    get_anime_details,
    find_similar_animes,
    find_similar_user_profiles
)

from clients import groq_llama3370b, sabia_31, gpt_41
llm_config = gpt_41
llm_reasoning = gpt_41

planner_agent = Agent(
    name="Planner",
    llm_config=llm_reasoning,
    system_message=planner_agent_prompt,
    description="Designs strategic plans for anime recommendations, outputting only numbered steps with function names and Python code for non-existing functions.",
    memory=[sequential_memory]
)

selector_agent = Agent(
    name="Selector",
    llm_config=llm_config,
    system_message=selector_agent_prompt,
    description="Selects the next tool/function to execute based on the plan provided by the Planner."
)

explainer_agent = Agent(
    name="Explainer",
    llm_config=llm_config,
    system_message=explainer_agent_prompt,
    description="Diagnoses technical failures and guides the Planner for strategic adjustments."
)

executor_agent = Agent(
    name="Executor",
    llm_config=llm_config,
    system_message=executor_agent_prompt,
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
)

analyzer_agent = Agent(
    name="Analyzer",
    llm_config=llm_config,
    system_message=analyzer_agent_prompt,
    description="Classifies tool outputs, signaling success, failure, or plan completion.",
    memory=[sequential_memory]
)

recommender_agent = Agent(
    name="Recommender",
    llm_config=llm_config,
    system_message=recommender_agent_prompt,
    description="Generates the final top 5 anime recommendation with optional justification.",
    memory=[sequential_memory]
)


flow = {
    planner_agent: [selector_agent],
    selector_agent: {
        "CONTINUE": [executor_agent],
        "plan_execution_complete": [analyzer_agent],
    },
     explainer_agent: [planner_agent],
    executor_agent: {
        "SUCCESS": [selector_agent],
        "FAIL": [explainer_agent],
    },
    analyzer_agent: [recommender_agent],
}

deps_module = Module(
    admin_name="anime_recommender_module",
    agents=[planner_agent, selector_agent, executor_agent, analyzer_agent, explainer_agent, recommender_agent],
    speaker_selection_method="auto",
    allowed_or_disallowed_speaker_transitions=flow,
    speaker_transitions_type="allowed",
)

planning_orchestrator = Orchestrator(
    name="planning_orchestrator",
    module=deps_module,
    description="A specialized system responsible for generating and delivering a reasoned anime recommendation when requested."
)