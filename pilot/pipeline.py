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
  
  # Limita o histórico para reduzir tokens (pega últimos 20 itens)
  limited_history = user_history[-20:] if len(user_history) > 20 else user_history

  memory = ListMemory(name="chat_history")
  memory.add(MemoryContent(content=" ".join(limited_history)))
  return memory

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
  
  try:
      for idx, data in enumerate(dataset, start=1):
          print(f"\n📊 [{idx}/{total}] Processando data_idx={data['data_idx']}")
          
          memory = load_memory(data) if with_memory else None
          user = ExplicitUser()

          conversational = Agent(
              name="conversational",
              description=
                  "Assistente de recomendação de filmes que usa ferramentas para buscar informações " \
                  "e faz recomendações personalizadas baseadas no contexto do usuário."
              ,
              system_message="""Movie recommendation assistant.

PROCESS:
1. Analyze user request
2. Use tools to search for information
3. Select the best movies

OUTPUT:
Return ONLY movie titles separated by [SEP]. No JSON, no explanations, no numbering.
IMPORTANT: Keep original movie titles in English.

Example: City Lights [SEP] Modern Times [SEP] The Great Dictator [SEP]""",
              llm_config=groq_llama3370b,
              tools=movies.tools,
              reflect_on_tool_use=True,
              tool_call_summary_format="{result}",  # Apenas o resultado bruto
              # memory=[memory] if memory else None,
          )

          try:
              user.talk_to(conversational, message=data['direct_description_query'], silent=True)
              arara_response = conversational.last_message()['content']
              
              result = {
                "id": str(data["data_idx"]),
                "response": arara_response
              }
              results.append(result)
              print(f"✅ [{idx}/{total}] Concluído com sucesso")
          except Exception as e:
              print(f"❌ [{idx}/{total}] Erro: {str(e)}")
              result = {
                "id": str(data["data_idx"]),
                "response": f"ERROR: {str(e)}"
              }
              results.append(result)
          
          # Salva a cada 10 itens para não perder progresso
          if idx % 10 == 0:
              print(f"\n💾 Salvamento intermediário ({idx} itens)...")
              save_response(dataset_name, query_type, with_memory, results)
  
  except KeyboardInterrupt:
      print("\n\n⚠️  Processo interrompido pelo usuário (Ctrl+C)")
      print(f"📊 Processados {len(results)}/{total} itens até agora")
  
  # Salva todas as respostas uma última vez
  print("\n" + "=" * 60)
  print(f"💾 Salvamento final de {len(results)} resultados...")
  save_response(dataset_name, query_type, with_memory, results)
  print(f"✅ Pipeline concluído! {len(results)}/{total} itens processados")
  print("=" * 60)

if __name__ == "__main__":
  # args: dataset_name="movie" query_type="Implicit"
  main(*sys.argv[1:])


# python pilot/pipeline.py movie Implicit True