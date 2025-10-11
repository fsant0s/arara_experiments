import os
import sys
import ast

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from users import ExplicitUser
from agents import Agent, Orchestrator, Module
from clients import groq_llama3370b, ollama_llama32
from agents.helpers.graph_utils import visualize_speaker_transitions_dict

from evaluation import report_metrics

from capabilities.memory import ListMemory, MemoryContent
from user_history import get_filtered_user_history
from tools import movies
from datasets.RecAssistBench import Dataloader
from neo4j_client import connect_to_neo4j

if not connect_to_neo4j():
    sys.exit(1)

model_name = "llama-3.1-70b-instruct"
dataloader = Dataloader("movie/ImplicitQuery.json")
dataset = dataloader.load()
data = dataset[1]
print("--------------------------------")
print("-------- DATA INICIAL ----------")
print("Data:", data)
print("--------------------------------\n")


prediction = dataloader.get_result(data_idx=data["data_idx"], model_name=model_name)

user = ExplicitUser()

user_history = get_filtered_user_history(
    user_id=data['source_user'],
    groundtruth_movie_ids=data['movieSubsetId'],
    neo4j_conditions=data['sharedRelationships']
)
# Limita o histórico para reduzir tokens (pega últimos 20 itens)
limited_history = user_history[-20:] if len(user_history) > 20 else user_history
sequential_memory = ListMemory(name="chat_history")
sequential_memory.add(MemoryContent(content=" ".join(limited_history)))
prediction = dataloader.get_result(data_idx=data["data_idx"], model_name=model_name)

conversational = Agent(
    name="conversational",
    description=
        "Assistente de recomendação de filmes que usa ferramentas para buscar informações " \
        "e faz recomendações personalizadas baseadas no contexto do usuário."
    ,
    system_message=f"""Assistente de recomendação de filmes.

PROCESSO:
1. Analise o pedido e histórico do usuário
2. Use tools para buscar informações
3. Combine resultados das tools com contexto do usuário
4. Selecione os melhores filmes

SAÍDA:
Retorne APENAS títulos separados por [SEP]. Sem JSON, explicações ou numeração.

Exemplo: City Lights [SEP] Modern Times [SEP] The Great Dictator [SEP]""",
    llm_config=groq_llama3370b,
    tools=movies.tools,
    reflect_on_tool_use=True,
    tool_call_summary_format="{result}",  # Apenas o resultado bruto
)

user.talk_to(conversational, message=data['direct_description_query'], silent=False)
arara_response = conversational.last_message()['content']

arara_eval = dataloader.evaluate_response(arara_response, data_idx=data['data_idx'])

report_metrics(
    dataloader=dataloader,
    data=data,
    arara_response=arara_response,
    model_name=model_name,
    k=0,
)
