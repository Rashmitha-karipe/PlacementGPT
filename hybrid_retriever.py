from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================
# STEP 1: LOAD EMBEDDING MODEL
# ==========================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# STEP 2: LOAD CHROMA VECTOR DATABASE
# ==========================================

vectorstore = Chroma(
    persist_directory="vectordb",
    embedding_function=embedding_model
)

print("\nVector database loaded successfully!")

# ==========================================
# STEP 3: VECTOR RETRIEVER
# ==========================================

vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ==========================================
# STEP 4: LOAD DOCUMENTS FOR BM25
# ==========================================

all_docs = vectorstore.get()["documents"]

print(f"\nTotal documents loaded: {len(all_docs)}")

# ==========================================
# STEP 5: TOKENIZE DOCUMENTS
# ==========================================

tokenized_docs = [doc.split() for doc in all_docs]

# ==========================================
# STEP 6: CREATE BM25
# ==========================================

bm25 = BM25Okapi(tokenized_docs)

print("\nBM25 retriever initialized!")

# ==========================================
# STEP 7: HYBRID SEARCH FUNCTION
# ==========================================

def hybrid_search(query):

    print(f"\nSearching for: {query}\n")

    # ----------------------------
    # VECTOR SEARCH
    # ----------------------------

    vector_results = vector_retriever.invoke(query)

    # ----------------------------
    # BM25 SEARCH
    # ----------------------------

    tokenized_query = query.split()

    bm25_results = bm25.get_top_n(
        tokenized_query,
        all_docs,
        n=3
    )

    return vector_results, bm25_results

# ==========================================
# STEP 8: TEST HYBRID SEARCH
# ==========================================

query = "Amazon graph interview questions"

vector_results, bm25_results = hybrid_search(query)

# ==========================================
# DISPLAY VECTOR RESULTS
# ==========================================

print("\n" + "=" * 60)
print("VECTOR SEARCH RESULTS")
print("=" * 60)

for i, doc in enumerate(vector_results, start=1):

    print(f"\nVECTOR RESULT {i}")
    print("-" * 50)

    print(doc.page_content[:500])

# ==========================================
# DISPLAY BM25 RESULTS
# ==========================================

print("\n" + "=" * 60)
print("BM25 SEARCH RESULTS")
print("=" * 60)

for i, doc in enumerate(bm25_results, start=1):

    print(f"\nBM25 RESULT {i}")
    print("-" * 50)

    print(doc[:500])