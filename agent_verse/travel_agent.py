import os
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from langchain_litellm import ChatLiteLLM  # updated import
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# ---------------- CONFIG ----------------
DATA_DIR = "travel_data"
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
DOCS_FILE = os.path.join(DATA_DIR, "docs.pkl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- INIT MODELS ----------------
llm = ChatLiteLLM(
    model="ollama/deepseek-r1:1.5b",
    streaming=True,
    temperature=0.7
)

embed_model = SentenceTransformer(EMBEDDING_MODEL)

# ---------------- LOAD DATASETS ----------------
def load_travel_data():
    docs = []

    # 1️⃣ International tourists
    tourists_csv = r"C:\Users\User\OneDrive\Desktop\agentverse\data\Internation Torsists in Pakistan.csv"
    if os.path.exists(tourists_csv):
        df_tourists = pd.read_csv(tourists_csv)
        for _, row in df_tourists.iterrows():
            docs.append(f"In {row['Year']}, Pakistan had {row['Number of Toursits']} international tourists.")
    else:
        print(f"⚠️ Missing file: {tourists_csv}")

    # 2️⃣ Hotels
    hotels_csv = r"C:\Users\User\OneDrive\Desktop\agentverse\data\Pakistan_Tourism_Hotels.csv"
    if os.path.exists(hotels_csv):
        df_hotels = pd.read_csv(hotels_csv)
        for _, row in df_hotels.iterrows():
            docs.append(
                f"Hotel {row['Hotel_Name']} in {row['City']}, {row['Province']} has {row['Total_Rooms']} rooms, "
                f"average occupancy {row['Avg_Occupancy_%']}%, room rate {row['Room_Rate_PKR']} PKR, "
                f"amenities count {row['Amenities_Count']}, customer rating {row['Customer_Rating']}."
            )
    else:
        print(f"⚠️ Missing file: {hotels_csv}")

    # 3️⃣ Tourist Destinations
    dest_csv = r"C:\Users\User\OneDrive\Desktop\agentverse\data\Tourist Destinations.csv"
    if os.path.exists(dest_csv):
        df_dest = pd.read_csv(dest_csv)
        for _, row in df_dest.iterrows():
            docs.append(
                f"{row['Desc']} Located in {row['district']}, Category: {row['category']}, "
                f"Coordinates: ({row['latitude']}, {row['longitude']})."
            )
    else:
        print(f"⚠️ Missing file: {dest_csv}")

    print(f"✅ Loaded {len(docs)} travel documents")
    return docs

# ---------------- FAISS VECTOR STORE ----------------
def build_faiss_index(docs):
    if len(docs) == 0:
        raise ValueError("No documents to index. Check your CSV paths and contents.")

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
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(DOCS_FILE, "rb") as f:
            docs = pickle.load(f)
        print("📂 Loaded existing FAISS index and docs")
    else:
        docs = load_travel_data()
        index = build_faiss_index(docs)
    return index, docs

index, docs = load_or_build_index()

# ---------------- RAG TOOL ----------------
@tool
def travel_agent(query: str, top_k: int = 5) -> str:
    """Answer travel queries using RAG over Pakistan travel data."""
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    retrieved_texts = [docs[i] for i in I[0]]
    context = "\n\n".join(retrieved_texts)

    prompt = f"""
You are a professional travel assistant for Pakistan.

Answer the user's question based on the following context:

{context}

User question: {query}

Provide practical advice on places, hotels, attractions, travel stats, and tips.
"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

# ---------------- WRAPPER FOR COORDINATOR ----------------
def travel_planner(prompt: str) -> str:
    """Wrapper to call RAG travel_agent from coordinator."""
    return travel_agent.invoke({"query": prompt})

# ---------------- REACT AGENT ----------------
class ResponseFormat(BaseModel):
    result: str

agent = create_react_agent(
    model=llm,
    tools=[travel_agent],
    response_format=ResponseFormat
)

# ---------------- TEST ----------------
if __name__ == "__main__":
    print("🧳 Travel Agent ready!")
    while True:
        q = input("💬 You: ")
        if q.lower() in ["exit", "quit"]:
            break
        ans = travel_planner(q)
        print("\n🧳 Travel Agent:\n", ans)
