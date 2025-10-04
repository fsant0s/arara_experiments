
from agents import Agent
from clients import groq_llama3370b
from memories import sequential_memory

conversational = Agent(
    name="conversational",
    description=
        "Conversational agent for casual anime discussions and general chat. " \
        "Handles casual conversations about anime topics, but directs recommendation requests to the Planner."
    ,
    system_message=f"""
      system: |
        Você é um gerador de listas. Sempre retorne os itens separados por [SEP] e nada mais.
        Exemplo: Item1 [SEP] Item2 [SEP] Item3
    """,
    llm_config=groq_llama3370b,
    memory=[sequential_memory],
)
