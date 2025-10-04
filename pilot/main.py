import os

from agents import Agent
from users import ExplicitUser
from capabilities.memory import ListMemory, MemoryContent
from clients import ollama_llama32
from dataloader import Dataloader
from user_history import get_filtered_user_history

user = ExplicitUser()
sequential_memory = ListMemory(name="chat_history")

dataloader = Dataloader("movie/ExplicitQuery.json")
dataset = dataloader.load()
data = dataset[1]


# ===== Avalia Arara =====
user_history = get_filtered_user_history(
    user_id=data['source_user'], 
    groundtruth_movie_ids=data['movieSubsetId'], 
    neo4j_conditions=data['sharedRelationships'],
    percentage=0.1
)
prompt_user_history = f"Este é o histórico do usuário: {', '.join(user_history)}"  

print(prompt_user_history)
# sequential_memory.add(MemoryContent(content=prompt_user_history))

conversational = Agent(
    name="conversational",
    description=
        "Conversational agent for casual anime discussions and general chat. " \
        "Handles casual conversations about anime topics, but directs recommendation requests to the Planner."
    ,
    system_message=f"""
      system: |
        Você é um gerador de listas. Sempre retorne os itens separados por [SEP] e nada mais.
        Exemplo: Item1 [SEP] Item2 [SEP] Item3
    """,
    llm_config=ollama_llama32,
    # memory=[sequential_memory],
)

user.talk_to(conversational, message=data['direct_description_query'])
arara_response = conversational.last_message()['content']
# Avalia em tempo real
arara_eval = dataloader.evaluate_response(arara_response, data_idx=data['data_idx'])

print(f"FTR:       {arara_eval['ftr']}")
print(f"Recall:    {arara_eval['recall']:.3f}")
print(f"Precision: {arara_eval['precision']:.3f}")
print(f"NDCG:      {arara_eval['ndcg']:.3f}")
print(f"\nFilmes preditos: {arara_eval['predicted_titles']}")
print(f"Filmes matched:  {arara_eval['matched_titles']}")
print(f"Ground truth:    {arara_eval['ground_truth']}")
print("=" * 80)


# ===== Avalia modelo existente =====
model_name = "llama-3.1-70b-instruct"
# model_name = "DeepSeek-R1"
prediction = dataloader.get_result(data_idx=data['data_idx'], model_name=model_name)

print(f"\n{'=' * 80}")
print(f"=== Predição do {model_name} ===")
if prediction:
    print(f"Response: {prediction['response']}")
    
    # Pega resultado da avaliação pré-calculada
    eval_result = dataloader.get_eval_result(data_idx=data['data_idx'], model_name=model_name)
    
    print(f"\n=== Métricas (pré-calculadas) ===")
    if eval_result:
        print(f"Recall:          {eval_result.get('recall', 'N/A'):.3f}")
        print(f"Precision:       {eval_result.get('precision', 'N/A'):.3f}")
        print(f"NDCG:            {eval_result.get('ndcg', 'N/A'):.3f}")
        print(f"Satisfied Ratio: {eval_result.get('satisfied_ratio', 'N/A'):.3f}")