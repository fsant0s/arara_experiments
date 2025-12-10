import os

from pyparsing import Union
from ioflow import IOStream
from formatting_utils import colored

from agents import Agent, Orchestrator, Module
from builtin_agents import Aggregator
from .system_message import system_message


iostream = IOStream.get_default()

def get_llm_config(client: str = "maritaca", temperature: float = 0.0, model: str = "sabia-3.1", api_key: str = None, base_url: str = None):
    return {
        "config_list": [
            {
                "client": client,
                "temperature": temperature,
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
            }
        ]
    }

#chatgpt = Agent(
#    name = "gpt4",
#    llm_config = get_llm_config(
#        client="openai",
#        model="gpt-4",
#        api_key=os.getenv("OPENAI_API_KEY")
#    ),
#    system_message="You are a recommender. Respond to the user requests accordingly. Strictly #output: [item1, item2, ..., itemn]",
#)

chatgpt = Agent(
    name = "chatgpt",
    llm_config = get_llm_config(
        client="openrouter",
        model="openai/gpt-4o",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    ),
    system_message=system_message,
)

gemini25flashi25pro = Agent(
    name = "gemini-2.5-flash",
    llm_config = get_llm_config(
        client="openrouter",
        model="google/gemini-2.5-flash",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    ),
    system_message=system_message,
)

claudeopus4 = Agent(
    name = "claude-opus-4",
    llm_config = get_llm_config(
        client="openrouter",
        model="anthropic/claude-opus-4",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    ),
    system_message=system_message,
)

def parallel(
    last_speaker: Agent, module: Module, selector: Agent = None
) -> Union[Agent, str, None]:

    iostream.print(colored("\nStarted executing the recommendation systems...", "yellow"), flush=True)
    aggregator = Aggregator(
        name="aggregator",
    )

    message = {'content': '', 'name': 'all', 'role': 'user'}
    for agent in module.agents:
        for reply in agent.generate_reply(sender=selector):
            message['content'] += f"Recommendation from {agent.name}: {reply.chat_message.content}\n\n"
            agent.send(reply, aggregator, silent=False, request_reply=False)

    aggregator.send(message['content'], selector, silent=True, request_reply=False)

    if aggregator not in module.agents: module.agents.append(aggregator)
    
    iostream.print(colored("Finished executing the recommendation systems.\n", "yellow"), flush=True)
    return aggregator

recsy_llm_module = Module(
    name="recsys_llm_module",
    agents=[chatgpt, gemini25flashi25pro, claudeopus4],
    speaker_selection_method=parallel,
    max_round=1,
)

recsys_orchestrator = Orchestrator(
    name="recsys_orchestrator",
    module=recsy_llm_module,
    description="Orchestrator for recommender systems using multiple LLMs.",
)

