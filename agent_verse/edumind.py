import os
import json
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_community.chat_models import ChatLiteLLM
from langgraph.prebuilt import create_react_agent
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool
from urllib.request import urlopen

# ---------------- CONFIG ----------------
load_dotenv()
DATA_DIR = "edumind_data"
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
DOCS_FILE = os.path.join(DATA_DIR, "docs.pkl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast for embeddings

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- INIT MODELS ----------------
llm = ChatLiteLLM(
    model="ollama/deepseek-r1:1.5b",
    streaming=True,
    temperature=0.7
)

embed_model = SentenceTransformer(EMBEDDING_MODEL)

# ---------------- DATASET ----------------
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"

def load_squad_data():
    """Download SQuAD dataset and extract text paragraphs."""
    print("📥 Downloading SQuAD dataset...")
    with urlopen(SQUAD_URL) as f:
        squad_json = json.load(f)

    docs = []
    for article in squad_json["data"]:
        for paragraph in article["paragraphs"]:
            text = paragraph["context"]
            if text.strip():
                docs.append(text.strip())
    print(f"✅ Loaded {len(docs)} paragraphs from SQuAD.")
    return docs

# ---------------- FAISS VECTOR STORE ----------------
def build_faiss_index(docs):
    """Build FAISS index from document embeddings."""
    print("📚 Building EduMind vector store...")
    embeddings = embed_model.encode(docs, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    # Save index and docs
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)
    print(f"✅ FAISS index saved to {INDEX_FILE}")
    return index

def load_or_build_index():
    """Load FAISS index if exists, else build from SQuAD."""
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
        print("📂 Loading existing FAISS index and docs...")
        index = faiss.read_index(INDEX_FILE)
        with open(DOCS_FILE, "rb") as f:
            docs = pickle.load(f)
    else:
        docs = load_squad_data()
        index = build_faiss_index(docs)
    return index, docs

index, docs = load_or_build_index()

# ---------------- RAG TOOL ----------------
@tool
def edumind_agent(query: str, top_k: int = 3) -> str:
    """Answer questions using retrieval augmented generation (RAG)."""
    # Encode query
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    # Search top_k docs
    D, I = index.search(q_emb, top_k)
    retrieved_texts = [docs[i] for i in I[0]]
    context = "\n\n".join(retrieved_texts)

    # Prompt LLM with context
    prompt = f"""
You are EduMind, an AI learning assistant.

Answer the user question based on the following context:
{context}

User question: {query}

Answer concisely and clearly.
"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

# ---------------- OPTIONAL REACT AGENT ----------------
class ResponseFormat(BaseModel):
    result: str

agent = create_react_agent(
    model=llm,
    tools=[edumind_agent],
    response_format=ResponseFormat
)

# ---------------- TEST ----------------
if __name__ == "__main__":
    while True:
        q = input("💬 You: ")
        if q.lower() in ["exit", "quit"]:
            break
        answer = edumind_agent.invoke({"query": q})


        print("\n🧠 EduMind:\n", answer)
