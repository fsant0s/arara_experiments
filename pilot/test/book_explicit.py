import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
PILOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if PILOT_DIR not in sys.path:
    sys.path.insert(0, PILOT_DIR)

from agents import Orchestrator, Module


from modules.books.explicit import create_explicit_orchestrator
from arara_user import AraraUser
from clients import gpt_4o

from evaluation import report_metrics


from datasets.recassistbench.book_dataloader import Dataloader
from neo4j_client import connect_to_neo4j

if not connect_to_neo4j():
    sys.exit(1)

llm_config = gpt_4o
model_name = "gpt-4o" #"llama-3.1-70b-instruct"
dataloader = Dataloader("book/ExplicitQuery.json")
dataset = dataloader.load(data_idx=3826) 
data = dataset[0]

print("--------------------------------")
print("-------- DATA INICIAL ----------")
print("Data:", data)
print("--------------------------------\n")


# Toggle to enable/disable memory usage inside explicit module
USE_MEMORY = True
MEMORY_SIZE = 10 # adjust if needed

user = AraraUser() 
explicit_orchestrator = create_explicit_orchestrator(
    data,
    llm_config=llm_config,
    use_memory=USE_MEMORY,
    memory_size=MEMORY_SIZE,
)

main_module = Module(
    name="main_module",
    agents=[user, explicit_orchestrator],
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
