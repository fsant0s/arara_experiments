import os 
import sys
import json
import random

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents import Orchestrator, Module
from clients import (
  groq_llama3370b, 
  ollama_llama32, 
  gpt_41, 
  openrouter_llama370b, 
  openrouter_claude35, 
  openrouter_gpt4o, 
  gpt_4o,
  openrouter_deepseek_chat,
)

from arara_user import AraraUser
from modules.movies import create_implicit_orchestrator, create_explicit_orchestrator, create_misinformed_orchestrator
from datasets.recassistbench import Dataloader

from neo4j_client import connect_to_neo4j

MEMORY_SIZE = 20 # default memory window size
COUNTER = 50

VALID_DATASETS = {"movie", "book"}
VALID_MODELS = {
    "groq_llama3370b": groq_llama3370b,
    "ollama_llama32": ollama_llama32,
    "gpt_41": gpt_41,
    "openrouter_llama370b": openrouter_llama370b,
    "openrouter_claude35": openrouter_claude35,
    "openrouter_gpt4o": openrouter_gpt4o,
    "gpt_4o": gpt_4o,
    "openrouter_deepseek_chat": openrouter_deepseek_chat,
}

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

def save_response(
      dataset_name, 
      llm_config_name, 
      query_type, 
      predict_type_name, 
      use_memory, 
      data,
      arara_response,
    ):
  """Salva todas as respostas em formato JSONL"""
  # Usa caminho absoluto relativo ao script

  result = {
        "type": query_type,
        "predicted_type": predict_type_name,
        "id": str(data["data_idx"]),
        "response": arara_response if arara_response else "ERROR: Invalid response"
      }

  output_dir = os.path.join(PROJECT_ROOT, "datasets", "recassistbench", "llm_results", f"arara_{llm_config_name}")

  os.makedirs(output_dir, exist_ok=True)

  filepath = f"{output_dir}/{dataset_name}-{query_type}Query_arara_{llm_config_name}_{use_memory}-prediction.jsonl"

  with open(filepath, "a") as f:
    f.write(json.dumps(result) + "\n")

  print(f"✅ Arquivo salvo com {len(result)} resultados")

def main(*args):
  # Conecta ao Neo4j no início
  if not connect_to_neo4j():
    sys.exit(1)
  
  # --- Dataset name ---
  if len(args) < 1 or args[0] not in VALID_DATASETS:
      raise ValueError(
          f"❌ Invalid dataset name. Please choose one of: {', '.join(sorted(VALID_DATASETS))}"
      )
  dataset_name = args[0]

  # --- Model name ---
  if len(args) < 2 or args[1] not in VALID_MODELS:
      raise ValueError(
          f"❌ Invalid or missing model name. "
          f"Valid options: {', '.join(sorted(VALID_MODELS.keys()))}"
      )
  llm_config_name = args[1]
  llm_config = VALID_MODELS[llm_config_name]

  memory_arg = args[2]
  if memory_arg not in {"True", "False", "true", "false"}:
      raise ValueError("❌ Invalid memory flag. Use True or False (case-insensitive).")
  
  # Convert to actual boolean
  use_memory = memory_arg.lower() == "true"

  impl_dataloader = Dataloader(f"{dataset_name}/ImplicitQuery.json")
  expl_dataloader = Dataloader(f"{dataset_name}/ExplicitQuery.json")
  mis_dataloader = Dataloader(f"{dataset_name}/MisinformedQuery.json")

  dataset_size = 200
  dataset = random.sample(mis_dataloader.load(), dataset_size) + random.sample(expl_dataloader.load(), dataset_size) + random.sample(impl_dataloader.load(), dataset_size)
  total = len(dataset)  # Total de itens a processar
  
  print(f"🚀 Iniciando processamento: {total} itens")
  print("=" * 60)

  try:
    for idx, data in enumerate(dataset, start=1):
      print(f"\n📊 [{idx}/{total}] Processando data_idx={data['data_idx']}")
      max_retries = 3
      arara_response = None

      for attempt in range(max_retries):
        
        user = AraraUser(
          name ="User",
          description="""
            Acts as the entry point of the conversational process, providing queries that express individual preferences, goals, or contextual needs.
            The user can formulate three main types of recommendation requests:

            - Explicit queries**, where the user directly specifies entities or attributes (e.g., “Can you suggest some movies directed by Spike Lee?”).
            - Implicit queries**, where the user conveys intent indirectly through examples or situational cues (e.g., “Please recommend some movies starring the same actor as in *The Return of the Musketeers* (1989) and *The Omega Code* (1999).”).
            - Misinformed queries**, where the user expresses interest based on an incorrect assumption or mistaken belief about a movie fact (e.g., “I recently watched *The Holiday* and really enjoyed Prof. T.’s direction. I’m looking for more movies directed by him.”).

            The user’s input determines which orchestration path is activated, guiding the system toward explicit reasoning, implicit inference, or misinformed clarification before generating the final recommendations.
          """
        )

        explicit_orchestrator = create_explicit_orchestrator(
          data,
          llm_config=llm_config,
          use_memory=use_memory,
          memory_size=MEMORY_SIZE,
        )
        implicit_orchestrator = create_implicit_orchestrator(
          data,
          llm_config=llm_config,
          use_memory=use_memory,
          memory_size=MEMORY_SIZE,
        )

        misinformed_orchestrator = create_misinformed_orchestrator(
          data,
          llm_config=llm_config,
          use_memory=use_memory,
          memory_size=MEMORY_SIZE,
        )

        main_module = Module(
          name="main_module",
          agents=[user, explicit_orchestrator, implicit_orchestrator, misinformed_orchestrator],
          speaker_selection_method="auto",
          max_round=2,
        )

        # ------------------ Orchestrator principal ------------------
        main_orchestrator = Orchestrator(
          name="main_orchestrator",
          module=main_module,
          llm_config=llm_config,
          description="Routes to the either Explicit, Implicit, or Misinformed module based on the user query.",
        )

        message = data.get('direct_description_query') or data.get('query')
        user.talk_to(main_orchestrator, message=message, silent=False)
        arara_response = main_orchestrator.last_message(user)['content']

        # Validar resposta
        is_valid, validation_msg = True, "It is valid." #validate_response(arara_response)
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


      predict_type = list(main_orchestrator._oai_messages.values())[1]
      predict_type_name = predict_type[1]['name']

      is_misinformed = bool(data.get("misinformed"))
      entry_type = "Explicit"

      if is_misinformed:
        entry_type = "Misinformed"
      elif data.get("multihop_info"):
        entry_type = "Implicit"
  
      print(f"✅ [{idx}/{total}] Concluído")
      save_response(dataset_name, 
                    llm_config_name, 
                    entry_type,  
                    predict_type_name,
                    "historyTrue" if use_memory else None, 
                    data,
                    arara_response
                    )
      
  except KeyboardInterrupt:
      print("\n\nInterrompido pelo usuário")
      print(f"📊 Processados {idx}/{total} itens (Implicit)")


  else:   
    # Salvamento final
    print("\n" + "=" * 60)
    print(f"✅ Pipeline concluído! {idx}/{total} itens")
    print("=" * 60)

if __name__ == "__main__":
# Example: python pilot/pipeline.py book Implicit True
  main(*sys.argv[1:])

