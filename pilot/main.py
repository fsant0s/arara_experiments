import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from users import ExplicitUser
from agents import Orchestrator, Module
from modules import explicit_orchestrator
from clients import groq_llama3370b, ollama_llama32
from agents.helpers.graph_utils import visualize_speaker_transitions_dict

from evaluation import report_metrics

from user_history import get_filtered_user_history
from tools import movies
from datasets.recassistbench import Dataloader
from neo4j_client import connect_to_neo4j

if not connect_to_neo4j():
    sys.exit(1)

llm_config = groq_llama3370b
model_name = "llama-3.1-70b-instruct"
dataloader = Dataloader("movie/ExplicitQuery.json")
dataset = dataloader.load()
data = dataset[1]
print("--------------------------------")
print("-------- DATA INICIAL ----------")
print("Data:", data)
print("--------------------------------\n")


prediction = dataloader.get_result(data_idx=data["data_idx"], model_name=model_name)
user_history = get_filtered_user_history(
    user_id=data['source_user'],
    groundtruth_movie_ids=data['movieSubsetId'],
    neo4j_conditions=data['sharedRelationships']
)
# Limita o histórico para reduzir tokens (pega últimos 20 itens)
limited_history = user_history[-20:] if len(user_history) > 20 else user_history
prediction = dataloader.get_result(data_idx=data["data_idx"], model_name=model_name)

user = ExplicitUser() #TODO: User can be implicit or explicit

main_module = Module(
    admin_name="main_module",
    agents=[user, explicit_orchestrator],
    speaker_selection_method="round_robin",
)

# ------------------ Orchestrator principal ------------------
main_orchestrator = Orchestrator(
    name="main_orchestrator",
    module=main_module,
    llm_config=llm_config,
    system_message="Só repasse a mensagem.",
    description="Routes to the Explicit or Implicit module based on the user query.",
)

user.talk_to(main_orchestrator, message=data['direct_description_query'], silent=False)
arara_response = main_orchestrator.last_message(user)['content']

arara_eval = dataloader.evaluate_response(arara_response, data_idx=data['data_idx'])

report_metrics(
    dataloader=dataloader,
    data=data,
    arara_response=arara_response,
    model_name=model_name,
    k=0,
)
