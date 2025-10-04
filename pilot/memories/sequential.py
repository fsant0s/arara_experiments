
from capabilities.memory import ListMemory, MemoryContent
from user_history import get_filtered_user_history

from datasets.recassistbench import Dataloader
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

sequential_memory = ListMemory(name="chat_history")
sequential_memory.add(MemoryContent(content=prompt_user_history))