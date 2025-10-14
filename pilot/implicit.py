import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents import Orchestrator, Module
from agents.helpers.graph_utils import visualize_speaker_transitions_dict


from modules import create_implicit_orchestrator
from users import ImplicitExplicitUser
from clients import groq_llama3370b, gpt_41

from evaluation import report_metrics


from datasets.recassistbench import Dataloader
from neo4j_client import connect_to_neo4j

if not connect_to_neo4j():
    sys.exit(1)

llm_config = groq_llama3370b
model_name = "llama-3.1-70b-instruct"
dataloader = Dataloader("movie/ImplicitQuery.json")
dataset = dataloader.load()
data = dataset[1]

print("--------------------------------")
print("-------- DATA INICIAL ----------")
print("Data:", data)
print("--------------------------------\n")

user = ImplicitExplicitUser()

# Toggle to enable/disable memory usage inside explicit module
USE_MEMORY = True
MEMORY_SIZE = 10  # adjust if needed

implicit_orchestrator = create_implicit_orchestrator(
    data,
    llm_config=llm_config,
    use_memory=USE_MEMORY,
    memory_size=MEMORY_SIZE,
)

main_module = Module(
    admin_name="main_module",
    agents=[user, implicit_orchestrator],
    speaker_selection_method="round_robin",
)

# ------------------ Orchestrator principal ------------------
main_orchestrator = Orchestrator(
    name="main_orchestrator",
    module=main_module,
    llm_config=llm_config,
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
