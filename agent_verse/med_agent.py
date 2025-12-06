import os
import pickle
import csv
import json
import faiss
import xml.etree.ElementTree as ET
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.tools import tool

# ---------------- CONFIG ----------------
PACKAGE_DIR = os.path.dirname(__file__)
PDF_DATA_DIR = os.path.join(PACKAGE_DIR, "data")      # optional PDF folder
MED_DATA_DIR = os.path.join(PACKAGE_DIR, "med_data")  # your medquad.csv, WikiMed.json, TREC xml
INDEX_DIR = os.path.join(PACKAGE_DIR, "med_index")
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(PDF_DATA_DIR, exist_ok=True)
os.makedirs(MED_DATA_DIR, exist_ok=True)

INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")
DOCS_FILE = os.path.join(INDEX_DIR, "docs.pkl")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # CPU-friendly
CHUNK_SIZE = 1200      # characters per chunk (~200-300 tokens)
CHUNK_OVERLAP = 300

# ---------------- MODELS ----------------
llm = ChatLiteLLM(model="ollama/deepseek-r1:1.5b", streaming=False, temperature=0)
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# ---------------- UTIL ----------------
def extract_text_from_pdf(path: str) -> List[Dict]:
    """Extract text by page, returning list of {'text','source','path','page'}."""
    out = []
    try:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                out.append({"text": text.strip(), "source": os.path.basename(path), "path": path, "page": i + 1})
    except Exception:
        pass
    return out

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Character-based chunking with overlap."""
    chunks = []
    start = 0
    length = len(text)
    if length <= size:
        return [text.strip()] if text.strip() else []
    while start < length:
        end = min(start + size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end - overlap > start else end
    return chunks

def extract_from_csv(path: str) -> List[Dict]:
    docs = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text_parts = []
                for key in ("question", "Question", "Q", "q", "summary", "Summary", "answer", "Answer", "context", "text", "Text"):
                    if key in row and row[key]:
                        text_parts.append(row[key])
                if not text_parts:
                    text_parts = [str(v) for v in row.values() if v]
                if text_parts:
                    text = " ".join(text_parts).strip()
                    docs.append({"text": text, "source": os.path.basename(path), "path": path, "page": i + 1, "chunk_id": 0})
    except Exception:
        pass
    return docs

def extract_from_json(path: str) -> List[Dict]:
    docs = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    text = None
                    for k in ("text", "content", "article", "body", "summary"):
                        if k in item and item[k]:
                            text = item[k]
                            break
                    if not text:
                        text = " ".join([str(v) for v in item.values() if isinstance(v, str)])
                    if text:
                        docs.append({"text": text, "source": os.path.basename(path), "path": path, "page": i + 1, "chunk_id": 0})
                elif isinstance(item, str):
                    docs.append({"text": item, "source": os.path.basename(path), "path": path, "page": i + 1, "chunk_id": 0})
        elif isinstance(data, dict):
            added = False
            for k, v in data.items():
                if isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            text = " ".join([str(val) for val in item.values() if isinstance(val, str)])
                            if text:
                                docs.append({"text": text, "source": os.path.basename(path), "path": path, "page": i + 1, "chunk_id": 0})
                                added = True
                elif isinstance(v, str) and len(v) > 50:
                    docs.append({"text": v, "source": os.path.basename(path), "path": path, "page": 0, "chunk_id": 0})
                    added = True
            if not added:
                text = json.dumps(data, ensure_ascii=False)
                docs.append({"text": text, "source": os.path.basename(path), "path": path, "page": 0, "chunk_id": 0})
    except Exception:
        pass
    return docs

def extract_from_xml(path: str) -> List[Dict]:
    docs = []
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        candidates = []
        for tag in root.iter():
            tagname = (tag.tag or "").lower()
            if any(k in tagname for k in ("question", "summary", "text", "body", "doc", "item")):
                text = "".join(tag.itertext()).strip()
                if text:
                    candidates.append(text)
        if not candidates:
            full = "".join(root.itertext()).strip()
            if full:
                candidates = [full]
        for i, c in enumerate(candidates):
            docs.append({"text": c, "source": os.path.basename(path), "path": path, "page": i + 1, "chunk_id": 0})
    except Exception:
        pass
    return docs

def load_med_data() -> List[Dict]:
    """Aggregate docs from PDFs in PDF_DATA_DIR and CSV/XML/JSON in MED_DATA_DIR."""
    docs = []

    # PDFs (if any)
    for root, _, files in os.walk(PDF_DATA_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                path = os.path.join(root, f)
                pages = extract_text_from_pdf(path)
                for p in pages:
                    chunks = chunk_text(p["text"])
                    for idx, c in enumerate(chunks):
                        docs.append({
                            "text": c,
                            "source": p["source"],
                            "path": p["path"],
                            "page": p["page"],
                            "chunk_id": idx
                        })

    # Structured files in med_data (CSV, JSON, XML, TXT)
    for root, _, files in os.walk(MED_DATA_DIR):
        for f in files:
            path = os.path.join(root, f)
            lower = f.lower()
            entries = []
            if lower.endswith(".csv"):
                entries = extract_from_csv(path)
            elif lower.endswith(".json"):
                entries = extract_from_json(path)
            elif lower.endswith(".xml"):
                entries = extract_from_xml(path)
            elif lower.endswith(".txt"):
                try:
                    with open(path, encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                    entries = [{"text": text, "source": os.path.basename(path), "path": path, "page": 0, "chunk_id": 0}]
                except Exception:
                    entries = []
            else:
                continue

            for e in entries:
                chunks = chunk_text(e["text"])
                if chunks:
                    for idx, c in enumerate(chunks):
                        docs.append({
                            "text": c,
                            "source": e.get("source", os.path.basename(path)),
                            "path": e.get("path", path),
                            "page": e.get("page", 0),
                            "chunk_id": idx
                        })
                else:
                    docs.append({
                        "text": e["text"],
                        "source": e.get("source", os.path.basename(path)),
                        "path": e.get("path", path),
                        "page": e.get("page", 0),
                        "chunk_id": 0
                    })

    if not docs:
        docs = [
            {"text": "Hypertension: lifestyle modification, monitoring, ACE inhibitors are common first-line agents.", "source": "sample", "path": "sample", "page": 0, "chunk_id": 0},
            {"text": "Type 2 diabetes: diet, exercise, metformin as first-line therapy for many patients.", "source": "sample", "path": "sample", "page": 0, "chunk_id": 1},
        ]
    return docs

# ---------------- INDEX ----------------
def build_faiss_index(docs: List[Dict]):
    if not docs:
        raise ValueError("No medical documents to index.")
    texts = [d["text"] for d in docs]
    emb = embed_model.encode(texts, convert_to_numpy=True)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)
    return index, docs

def load_or_build_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
        try:
            index = faiss.read_index(INDEX_FILE)
            with open(DOCS_FILE, "rb") as f:
                docs = pickle.load(f)
            return index, docs
        except Exception:
            pass
    docs = load_med_data()
    return build_faiss_index(docs)

index, docs = load_or_build_index()

# ---------------- RAG TOOL ----------------
@tool
def med_agent(query: str, top_k: int = 4) -> str:
    """
    Retrieve top_k passages and answer with citations.
    Include explicit disclaimer and encourage professional consultation.
    """
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    D, I = index.search(q_emb, top_k)
    retrieved = []
    for idx in I[0]:
        if idx < len(docs):
            retrieved.append(docs[idx])

    context_parts = []
    for r in retrieved:
        snippet = r["text"][:800].replace("\n", " ").strip()
        context_parts.append(snippet)
    context = "\n\n".join(context_parts) or "No retrieved documents."

    prompt = f"""
You are a careful, conservative medical assistant. Use ONLY the provided context passages to answer the user's question. If the context does not contain enough information to answer, say you don't have enough evidence and recommend consulting a licensed healthcare professional.

Context:
{context}

User question:
{query}

Answer concisely and finish with a brief safety disclaimer. Do NOT cite document sources or page numbers.
"""
    resp = llm.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)
    
    # Post-process: convert **text** patterns to bold HTML and add newlines before headings
    import re
    # Replace **heading**: patterns with <br/><b>heading</b>:
    text = re.sub(r'\*\*([^*]+)\*\*:', r'<br/><b>\1</b>:', text)
    
    disclaimer = ("\n\nDISCLAIMER: This information is for educational purposes only and is not medical advice. "
                  "For diagnosis or treatment consult a licensed healthcare professional. If this is an emergency contact local emergency services.")
    return text.strip() + disclaimer

# wrapper used by coordinator
    return med_agent(prompt)
def med_planner(prompt:
     str) -> str:
    """Call the tool safely whether `med_agent` is a StructuredTool or a plain function.

    - If `med_agent` is a StructuredTool (decorator from `langchain_core.tools`), call
      `.invoke({"query": prompt})` and return its content.
    - Otherwise, fall back to the direct implementation function `med_agent_impl`.
    """
    # Preferred: StructuredTool exposes `.invoke({...})`
    try:
        resp = med_agent.invoke({"query": prompt})
        return resp.content if hasattr(resp, "content") else str(resp)
    except AttributeError:
        # med_agent is not a StructuredTool; try to call a local implementation if present
        impl = globals().get("med_agent_impl") or globals().get("med_agent_impl")
        if impl and callable(impl):
            try:
                out = impl(prompt)
                return out if isinstance(out, str) else str(out)
            except Exception as e:
                return f"Error running med_agent_impl: {e}"
        return "med_agent is not callable and no med_agent_impl fallback found."
    except Exception as e:
        # Other failures when invoking the StructuredTool — try dynamic fallback impl
        impl = globals().get("med_agent_impl")
        if impl and callable(impl):
            try:
                out = impl(prompt)
                return out if isinstance(out, str) else str(out)
            except Exception as e2:
                return f"Error invoking med_agent: {e} / fallback error: {e2}"
        return f"Error invoking med_agent: {e} and no med_agent_impl fallback found."

# CLI for quick local testing
if __name__ == "__main__":
    print("Medical agent ready.")
    print("Place PDFs in:", PDF_DATA_DIR)
    print("Place CSV/XML/JSON in:", MED_DATA_DIR)
    print("Type 'exit' to quit.")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        print("\n" + med_planner(q) + "\n")