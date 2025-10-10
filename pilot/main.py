import os
import sys
import ast

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from users import ExplicitUser
from agents import Agent
from clients import groq_llama3370b, ollama_llama32

from evaluation import report_metrics

from capabilities.memory import ListMemory, MemoryContent
from user_history import get_filtered_user_history
from tools import movies
from datasets.RecAssistBench import Dataloader
from neo4j_client import connect_to_neo4j

if not connect_to_neo4j():
    sys.exit(1)

model_name = "llama-3.1-70b-instruct"
dataloader = Dataloader("movie/ExplicitQuery.json")
dataset = dataloader.load()
data = dataset[0]
print("data", data)
prediction = dataloader.get_result(data_idx=data["data_idx"], model_name=model_name)

user = ExplicitUser()

user_history = get_filtered_user_history(
    user_id=data['source_user'],
    groundtruth_movie_ids=data['movieSubsetId'],
    neo4j_conditions=data['sharedRelationships']
)
sequential_memory = ListMemory(name="chat_history")
sequential_memory.add(MemoryContent(content=" ".join(user_history)))
prediction = dataloader.get_result(data_idx=data["data_idx"], model_name=model_name)
prediction_length = len(prediction['response'].split(' [SEP] ')) if prediction else 0

conversational = Agent(
    name="conversational",
    description=
        "Conversational agent for casual anime discussions and general chat. " \
        "Handles casual conversations about anime topics, but directs recommendation requests to the Planner."
    ,
    system_message=f"""
      system: |
        Você é um gerador de listas de filmes. 
        
        Quando o usuário pedir filmes, use as tools disponíveis para buscar.
        Depois de receber os resultados das tools, formate a resposta APENAS com os títulos dos filmes separados por [SEP].
        
        IMPORTANTE: Retorne APENAS os títulos separados por [SEP], sem JSON, sem explicações, sem numeração.
        
        ***TROQUE AS DATAS DOS FILMES POR [SEP]

        Retorne exatamente {prediction_length} itens.
        Exemplo de formato correto: Bamboozled (2000) [SEP] Do the Right Thing (1989) [SEP] Clockers (1995)
    """,
    llm_config=groq_llama3370b,
    # memory=[sequential_memory],
    tools=movies.tools,
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
