import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from users import ExplicitUser
from agents import Agent
from clients import groq_llama3370b

from evaluation import report_metrics
from datasets.recassistbench import Dataloader

from capabilities.memory import ListMemory, MemoryContent
from user_history import get_filtered_user_history

model_name = "llama-3.1-70b-instruct"
dataloader = Dataloader("movie/ExplicitQuery.json")
dataset = dataloader.load()
data = dataset[0]
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
        Você é um gerador de listas. Sempre retorne os itens separados por [SEP] e nada mais.
        Retorne exatamente {prediction_length} itens.
        Exemplo: Item1 [SEP] Item2 [SEP] Item3
    """,
    llm_config=groq_llama3370b,
    memory=[sequential_memory],
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
