from langchain.memory import ConversationBufferMemory

# ==========================================
# CREATE MEMORY
# ==========================================

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ==========================================
# SAVE CONVERSATIONS
# ==========================================

memory.save_context(
    {"input": "Explain BFS"},
    {"output": "BFS means Breadth First Search"}
)

memory.save_context(
    {"input": "Explain DFS"},
    {"output": "DFS means Depth First Search"}
)

# ==========================================
# LOAD MEMORY
# ==========================================

chat_history = memory.load_memory_variables({})

print("\nChat History:\n")

print(chat_history)