import streamlit as st
from langchain-groq import ChatGroq
from hybrid_retriever import hybrid_search
from dotenv import load_dotenv
import os

# ==========================================
# LOAD ENV VARIABLES
# ==========================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="PlacementGPT",
    page_icon="🚀",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

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
    margin-top: 10px;
}

.sub-title {
    text-align: center;
    color: #AAAAAA;
    margin-bottom: 30px;
    font-size: 18px;
}

.user-msg {
    background: linear-gradient(135deg, #1E88E5, #1565C0);
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 12px;
    color: white;
    font-size: 16px;
}

.ai-msg {
    background-color: #262730;
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 18px;
    color: white;
    font-size: 16px;
    border: 1px solid #444;
}

.stChatInput input {
    background-color: #262730 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.markdown(
    '<div class="main-title">🚀 PlacementGPT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Conversational Hybrid RAG for Placement Preparation</div>',
    unsafe_allow_html=True
)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Settings")

    company = st.selectbox(
        "Select Company",
        [
            "Amazon",
            "TCS",
            "Infosys",
            "Wipro",
            "Accenture"
        ]
    )

    st.markdown("---")

    st.subheader("✨ Features")

    st.write("✅ Hybrid Retrieval")
    st.write("✅ BM25 Retrieval")
    st.write("✅ Semantic Search")
    st.write("✅ Conversational Memory")
    st.write("✅ Groq LLM")
    st.write("✅ Streamlit UI")

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []
        st.rerun()

# ==========================================
# LOAD GROQ MODEL
# ==========================================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for msg in st.session_state.messages:

    if msg["role"] == "user":

        st.markdown(
            f'<div class="user-msg">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f'<div class="ai-msg">🤖 {msg["content"]}</div>',
            unsafe_allow_html=True
        )

# ==========================================
# CHAT INPUT
# ==========================================

query = st.chat_input("Ask placement questions...")

# ==========================================
# HANDLE USER QUERY
# ==========================================

if query:

    # SAVE USER MESSAGE
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # ======================================
    # HYBRID SEARCH
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
    # CONVERSATION HISTORY
    # ======================================

    history = ""

    for msg in st.session_state.messages:

        history += f"{msg['role']}: {msg['content']}\n"

    # ======================================
    # FINAL PROMPT
    # ======================================

    prompt = f"""
    You are PlacementGPT,
    an AI assistant for placement preparation.

    Company Focus:
    {company}

    Previous Conversation:
    {history}

    Context:
    {context}

    User Question:
    {query}

    Instructions:
    - Give concise answers
    - Explain clearly
    - Help with placements
    - Use the provided context
    """

    # ======================================
    # GENERATE RESPONSE
    # ======================================

    with st.spinner("🤖 Thinking..."):

        response = llm.invoke(prompt)

        answer = response.content

    # SAVE AI RESPONSE
    st.session_state.messages.append({

        "role": "assistant",
        "content": answer
    })

    # RERUN
    st.rerun()