from langchain_groq import ChatGroq
from hybrid_retriever import hybrid_search

# ==========================================
# LOAD LLM
# ==========================================

llm = ChatGroq(
    groq_api_key="YOUR_GROQ_API_KEY",
    model_name="llama3-8b-8192"
)

# ==========================================
# SIMPLE CHAT MEMORY
# ==========================================

chat_history = []

# ==========================================
# CHAT LOOP
# ==========================================

while True:

    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    # ======================================
    # HYBRID SEARCH
    # ======================================

    vector_results, bm25_results = hybrid_search(query)

    # ======================================
    # CREATE CONTEXT
    # ======================================

    context = ""

    # VECTOR RESULTS
    for doc in vector_results:

        context += doc.page_content + "\n"

    # BM25 RESULTS
    for doc in bm25_results:

        context += doc + "\n"

    # ======================================
    # ADD CHAT HISTORY
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

    Give a clear and concise answer.
    """

    # ======================================
    # GENERATE RESPONSE
    # ======================================

    response = llm.invoke(prompt)

    answer = response.content

    print("\nAI Response:\n")

    print(answer)

    # ======================================
    # SAVE MEMORY
    # ======================================

    chat_history.append({
        "user": query,
        "ai": answer
    })