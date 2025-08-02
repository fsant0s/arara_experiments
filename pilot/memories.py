from capabilities.memory import ListMemory, MemoryContent

sequential_memory = ListMemory(name="chat_history")
sequential_memory.add(MemoryContent(content="User's name is Shocked."))