from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os

# ==========================================
# PHASE 3 — VECTOR DATABASE CREATION
# ==========================================

# Folder containing TXT files
DATA_PATH = "data"

# Folder where vector DB will be stored
VECTOR_DB_PATH = "vectordb"

# ==========================================
# STEP 1: LOAD TEXT DOCUMENTS
# ==========================================

documents = []

print("\nLoading text files...\n")

for file in os.listdir(DATA_PATH):

    if file.endswith(".txt"):

        file_path = os.path.join(DATA_PATH, file)

        try:

            print(f"Loading: {file}")

            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

            loaded_docs = loader.load()

            documents.extend(loaded_docs)

            print(f"Successfully loaded: {file}\n")

        except Exception as e:

            print(f"Error loading {file}")
            print(e)
            print("\nSkipping this file...\n")

# ==========================================
# STEP 2: CHECK DOCUMENTS
# ==========================================

if len(documents) == 0:

    print("No valid text documents found.")
    exit()

print(f"Total documents loaded: {len(documents)}")

# ==========================================
# STEP 3: SPLIT DOCUMENTS INTO CHUNKS
# ==========================================

print("\nSplitting documents into chunks...\n")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# ==========================================
# STEP 4: LOAD EMBEDDING MODEL
# ==========================================

print("\nLoading embedding model...\n")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# STEP 5: CREATE CHROMA VECTOR DATABASE
# ==========================================

print("\nCreating Chroma vector database...\n")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=VECTOR_DB_PATH
)

# Save database permanently


print("\nVector database created successfully!")

# ==========================================
# STEP 6: TEST VECTOR SEARCH
# ==========================================

print("\nTesting vector retrieval...\n")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

query = "Amazon graph interview questions"

results = retriever.invoke(query)

print(f"\nTop results for query: '{query}'\n")

for i, doc in enumerate(results, start=1):

    print(f"\nRESULT {i}")
    print("-" * 50)

    print(doc.page_content[:500])

    print("\n")