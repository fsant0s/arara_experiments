import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from users import ExplicitUser
from mas import conversational

from datasets.recassistbench import Dataloader

dataloader = Dataloader("movie/ExplicitQuery.json")
dataset = dataloader.load()
data = dataset[1]

user = ExplicitUser()
user.talk_to(conversational, message=data['direct_description_query'], silent=False)
arara_response = conversational.last_message()['content']
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