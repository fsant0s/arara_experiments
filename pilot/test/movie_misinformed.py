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
from agents.helpers.graph_utils import visualize_speaker_transitions_dict

from modules.movies.misinformed import create_misinformed_orchestrator
from arara_user import AraraUser  # reuse a generic user wrapper
from clients import gpt_41

from evaluation import report_metrics
from datasets.recassistbench.movie_dataloader import Dataloader
from neo4j_client import connect_to_neo4j

# ------------------ Neo4j connection ------------------
if not connect_to_neo4j():
    sys.exit(1)

# ------------------ Config ------------------
llm_config = gpt_41
model_name = "gpt-4o"

# You can pass a specific index here if desired (e.g., data_idx=0)
dataloader = Dataloader("movie/MisinformedQuery.json")
dataset = dataloader.load()  # or dataloader.load(data_idx=<int>) to load a single sample
data = dataset[11]

print("--------------------------------")
print("-------- DATA INICIAL ----------")
print("Data:", data)
print("--------------------------------\n")

user = AraraUser()

# Toggle to enable/disable memory usage inside misinformed module
USE_MEMORY = True
MEMORY_SIZE = 10  # adjust if needed

misinformed_orchestrator = create_misinformed_orchestrator(
    data,
    llm_config=llm_config,
    use_memory=USE_MEMORY,
    memory_size=MEMORY_SIZE,
)

main_module = Module(
    name="main_module",
    agents=[user, misinformed_orchestrator],
    speaker_selection_method="round_robin",
)

# ------------------ Orchestrator principal ------------------
main_orchestrator = Orchestrator(
    name="main_orchestrator",
    module=main_module,
    llm_config=llm_config,
    description="Routes to the MisinformedCondition module for robustness testing against incorrect user claims.",
)

# Use the natural-language query that contains the misinformation
user.talk_to(main_orchestrator, message=data["query"], silent=False)
arara_response = main_orchestrator.last_message(user)["content"]

# Evaluate against ground truth for the corrected (true) condition
arara_eval = dataloader.evaluate_response(arara_response, data_idx=data["data_idx"])

report_metrics(
    dataloader=dataloader,
    data=data,
    arara_response=arara_response,
    model_name=model_name,
    k=0,
)
