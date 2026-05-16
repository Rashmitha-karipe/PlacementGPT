import streamlit as st
from langchain_groq import ChatGroq
from hybrid_retriever import hybrid_search

# ==========================
# STREAMLIT CONFIG
# ==========================
st.set_page_config(
    page_title="PlacementGPT",
    page_icon="🚀",
    layout="wide"
)

# ==========================
# LOAD API KEY (STREAMLIT CLOUD SAFE)
# ==========================
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = None

if not groq_api_key:
    st.error("❌ GROQ API KEY not found. Add it in Streamlit Secrets.")
    st.stop()

# ==========================
# LOAD LLM
# ==========================
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0E1117;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

.main-title {
    text-align: center;
    font-size: 46px;
    font-weight: bold;
    color: #00FFAA;
}

.sub-title {
    text-align: center;
    color: #AAAAAA;
    font-size: 18px;
    margin-bottom: 20px;
}

.user-msg {
    background: #1E88E5;
    padding: 12px;
    border-radius: 10px;
    margin: 8px 0;
    color: white;
}

.ai-msg {
    background: #262730;
    padding: 12px;
    border-radius: 10px;
    margin: 8px 0;
    border: 1px solid #444;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# TITLE
# ==========================
st.markdown('<div class="main-title">🚀 PlacementGPT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hybrid RAG Assistant for Placement Prep</div>', unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.header("⚙️ Settings")

    company = st.selectbox(
        "Select Company",
        ["Amazon", "TCS", "Infosys", "Wipro", "Accenture"]
    )

    st.markdown("---")
    st.subheader("✨ Features")
    st.write("✔ Hybrid Retrieval (BM25 + Vector)")
    st.write("✔ Groq LLM")
    st.write("✔ Conversational AI")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ==========================
# SESSION STATE
# ==========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================
# DISPLAY CHAT HISTORY
# ==========================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

# ==========================
# USER INPUT
# ==========================
query = st.chat_input("Ask placement questions...")

# ==========================
# HANDLE QUERY
# ==========================
if query:

    st.session_state.messages.append({"role": "user", "content": query})

    # ==========================
    # HYBRID RETRIEVAL (SAFE)
    # ==========================
    try:
        vector_results, bm25_results = hybrid_search(query)
    except Exception as e:
        st.error(f"Retrieval error: {e}")
        vector_results, bm25_results = [], []

    # ==========================
    # BUILD CONTEXT
    # ==========================
    context = ""

    for doc in vector_results:
        if hasattr(doc, "page_content"):
            context += doc.page_content + "\n"

    for doc in bm25_results:
        context += str(doc) + "\n"

    # ==========================
    # CHAT HISTORY
    # ==========================
    history = ""
    for msg in st.session_state.messages:
        history += f"{msg['role']}: {msg['content']}\n"

    # ==========================
    # PROMPT
    # ==========================
    prompt = f"""
You are PlacementGPT, an AI assistant for placement preparation.

Company: {company}

Conversation:
{history}

Context:
{context}

User Question:
{query}

Instructions:
- Be concise
- Give clear answers
- Focus on placement prep
"""

    # ==========================
    # GENERATE RESPONSE
    # ==========================
    with st.spinner("🤖 Thinking..."):
        response = llm.invoke(prompt)
        answer = response.content

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()