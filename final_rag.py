from langchain_groq import ChatGroq
from dotenv import load_dotenv

from hybrid_retriever import hybrid_search

import os

# ==========================================
# LOAD ENV VARIABLES
# ==========================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# ==========================================
# LOAD GROQ LLM
# ==========================================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"

)

# ==========================================
# SIMPLE CHAT MEMORY
# ==========================================

chat_history = []

# ==========================================
# CHAT LOOP
# ==========================================

print("\n🚀 PlacementGPT Started!")
print("Type 'exit' to quit.\n")

while True:

    query = input("Ask Question: ")

    # ======================================
    # EXIT CONDITION
    # ======================================

    if query.lower() == "exit":

        print("\nGoodbye!\n")

        break

    # ======================================
    # HYBRID RETRIEVAL
    # ======================================

    vector_results, bm25_results = hybrid_search(query)

    # ======================================
    # BUILD CONTEXT
    # ======================================

    context = ""

    # VECTOR RESULTS
    for doc in vector_results:

        context += doc.page_content + "\n"

    # BM25 RESULTS
    for doc in bm25_results:

        context += doc + "\n"

    # ======================================
    # CHAT HISTORY
    # ======================================

    history_text = ""

    for chat in chat_history:

        history_text += f"""
        User: {chat['user']}
        AI: {chat['ai']}
        """

    # ======================================
    # FINAL PROMPT
    # ======================================

    prompt = f"""
    You are PlacementGPT,
    an AI assistant for placement preparation.

    Previous Conversation:
    {history_text}

    Context:
    {context}

    User Question:
    {query}

    Instructions:
    - Answer clearly
    - Use the given context
    - Keep answers concise
    - Help with interview preparation
    """

    # ======================================
    # GENERATE RESPONSE
    # ======================================

    response = llm.invoke(prompt)

    answer = response.content

    # ======================================
    # DISPLAY RESPONSE
    # ======================================

    print("\n🤖 AI Response:\n")

    print(answer)

    print("\n" + "=" * 70 + "\n")

    # ======================================
    # SAVE MEMORY
    # ======================================

    chat_history.append({
        "user": query,
        "ai": answer
    })