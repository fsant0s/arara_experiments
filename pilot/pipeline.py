import os 
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
ARARA_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir, "arara", "src"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if ARARA_PATH not in sys.path:
    sys.path.insert(0, ARARA_PATH)

from users import ExplicitUser
from agents import Orchestrator, Module
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
  
  # Limita o histórico para reduzir tokens (pega últimos 20 itens)
  limited_history = user_history[-20:] if len(user_history) > 20 else user_history

  memory = ListMemory(name="chat_history")
  memory.add(MemoryContent(content=" ".join(limited_history)))
  return memory

def validate_response(response):
  """Valida se a resposta está no formato esperado"""
  if not response:
    return False, "Resposta vazia"
  
  # Verifica se tem [SEP]
  if '[SEP]' not in response:
    return False, "Sem separador [SEP]"
  
  # Verifica se tem pelo menos um título
  parts = [p.strip() for p in response.split('[SEP]') if p.strip()]
  if len(parts) == 0:
    return False, "Nenhum título encontrado"
  
  # Verifica se tem anos inválidos (só números)
  if all(p.replace('[', '').replace(']', '').strip().isdigit() for p in parts):
    return False, "Apenas números/anos (sem títulos)"
  
  return True, f"{len(parts)} filmes encontrados"

def save_response(dataset_name, query_type, with_memory, results):
  """Salva todas as respostas em formato JSONL"""
  # Usa caminho absoluto relativo ao script
  output_dir = os.path.join(CURRENT_DIR, "llm_results", "arara")
  os.makedirs(output_dir, exist_ok=True)
  filepath = f"{output_dir}/{dataset_name}-{query_type}Query_arara-history{with_memory}_prediction.jsonl"
  
  print(f"📁 Salvando em: {filepath}")
  
  with open(filepath, "w") as f:
    for result in results:
      f.write(json.dumps(result) + "\n")
  
  print(f"✅ Arquivo salvo com {len(results)} resultados")

def main(*args):
  # Conecta ao Neo4j no início
  if not connect_to_neo4j():
    sys.exit(1)
  
  dataset_name = args[0] if len(args) > 0 and args[0] else "movie"
  query_type = args[1] if len(args) > 1 and args[1] else "Explicit"
  with_memory = args[2] if len(args) > 2 and args[2] else False
  
  if isinstance(with_memory, str):
    with_memory = with_memory.lower() == "true"

  dataloader = Dataloader(f"{dataset_name}/{query_type}Query.json")
  dataset = dataloader.load()
  
  results = []
  total = len(dataset)  # Total de itens a processar
  
  print(f"🚀 Iniciando processamento: {total} itens")
  print("=" * 60)

  llm_config = ollama_llama32
  model_name = "llama-3.2:latest"
  
  try:
      for idx, data in enumerate(dataset, start=1):
          print(f"\n📊 [{idx}/{total}] Processando data_idx={data['data_idx']}")
          
          max_retries = 3
          arara_response = None
          
          for attempt in range(max_retries):
              # Importar módulo fresh para cada tentativa
              from modules import explicit as explicit_module
              import importlib
              importlib.reload(explicit_module)
              explicit_orchestrator = explicit_module.orchestrator
              
              user = ExplicitUser()
              
              main_module = Module(
                  admin_name="main_module",
                  agents=[user, explicit_orchestrator],
                  speaker_selection_method="round_robin",
              )

              main_orchestrator = Orchestrator(
                  name="main_orchestrator",
                  module=main_module,
                  llm_config=llm_config,
                  system_message="Só repasse a mensagem.",
                  description="Routes to the Explicit or Implicit module based on the user query.",
              )

              user.talk_to(main_orchestrator, message=data['direct_description_query'], silent=True)
              arara_response = main_orchestrator.last_message(user)['content']
              
              # Validar resposta
              is_valid, validation_msg = validate_response(arara_response)
              
              if is_valid:
                  print(f"✅ Válido: {validation_msg}")
                  break
              else:
                  print(f"⚠️  Tentativa {attempt+1}/{max_retries} - {validation_msg}")
                  if attempt < max_retries - 1:
                      print(f"   Resposta: {arara_response[:100]}...")
                      print(f"   Tentando novamente...")
                  else:
                      print(f"❌ Resposta inválida após {max_retries} tentativas")
          
          result = {
              "id": str(data["data_idx"]),
              "response": arara_response if arara_response else "ERROR: Invalid response"
          }
          results.append(result)
          print(f"✅ [{idx}/{total}] Concluído")
          
          save_response(dataset_name, query_type, with_memory, results)

  except KeyboardInterrupt:
      print("\n\nInterrompido pelo usuário")
      print(f"📊 Processados {len(results)}/{total} itens")
  except Exception as e:
      print(f"\n❌ Erro: {e}")
  
  # Salvamento final
  print("\n" + "=" * 60)
  print(f"💾 Salvamento final de {len(results)} resultados...")
  save_response(dataset_name, query_type, with_memory, results)
  print(f"✅ Pipeline concluído! {len(results)}/{total} itens")
  print("=" * 60)

if __name__ == "__main__":
  # args: dataset_name="movie" query_type="Implicit"
  main(*sys.argv[1:])


# python pilot/pipeline.py movie Implicit True