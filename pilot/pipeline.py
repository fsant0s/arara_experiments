import os 
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from users import ExplicitUser
from agents import Agent
from clients import groq_llama3370b, ollama_llama32
from capabilities.memory import ListMemory, MemoryContent
from datasets.RecAssistBench import Dataloader
from user_history import get_filtered_user_history
from neo4j_client import connect_to_neo4j
from tools import movies

def load_memory(data):
  user_history = get_filtered_user_history(
    user_id=data["source_user"],
    groundtruth_movie_ids=data["movieSubsetId"],
    neo4j_conditions=data["sharedRelationships"]
  )

  memory = ListMemory(name="chat_history")
  memory.add(MemoryContent(content=" ".join(user_history)))
  return memory

def save_response(dataset_name, query_type, with_memory, results):
  """Salva todas as respostas em formato JSONL"""
  os.makedirs(dataset_name, exist_ok=True)
  filepath = f"{dataset_name}/{query_type}Query_arara-history{with_memory}_prediction.jsonl"
  
  with open(filepath, "w") as f:
    for result in results:
      f.write(json.dumps(result) + "\n")

def main(*args):
  # Conecta ao Neo4j no início
  if not connect_to_neo4j():
    print("❌ Erro: Não foi possível conectar ao Neo4j")
    sys.exit(1)
  
  dataset_name = args[0] if len(args) > 0 and args[0] else "movie"
  query_type = args[1] if len(args) > 1 and args[1] else "Explicit"
  with_memory = args[2] if len(args) > 2 and args[2] else False
  
  if isinstance(with_memory, str):
    with_memory = with_memory.lower() == "true"

  dataloader = Dataloader(f"{dataset_name}/{query_type}Query.json")
  dataset = dataloader.load()
  
  results = []

  for data in dataset:
      memory = load_memory(data) if with_memory else None
      user = ExplicitUser()

      conversational = Agent(
        name="conversational",
        description=
            "You are a recommendation assistant focused on recommending items based on the user's query. " \
            "Your goal is to recommend items that are relevant to the user's query."
        ,
        system_message="""
          system: |
            You are a movie recommendation assistant.
            
            When the user asks for movies, use the available tools to search.
            After receiving the tool results, format your response with ONLY the movie titles separated by [SEP].
            
            IMPORTANT: Return ONLY the titles separated by [SEP], without JSON, without explanations, without numbering.
            
            Correct format example: Bamboozled (2000) [SEP] Do the Right Thing (1989) [SEP] Clockers (1995)
        """,
        tools=movies.tools,
        llm_config=groq_llama3370b,
        # memory=[memory] if memory else None,
      )

      user.talk_to(conversational, message=data['direct_description_query'], silent=False)
      arara_response = conversational.last_message()['content']
      
      result = {
        "id": str(data["data_idx"]),
        "response": arara_response
      }
      results.append(result)
  
  # Salva todas as respostas de uma vez
  save_response(dataset_name, query_type, with_memory, results)

if __name__ == "__main__":
  # args: dataset_name="movie" query_type="Implicit"
  main(*sys.argv[1:])


# python pilot/pipeline.py movie Implicit True