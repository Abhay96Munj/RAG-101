"""
Loaders for the heavy resources (LLM client, vector store, BM25 index).
Pure functions — no shared state lives here. The lifespan in app/main.py
calls these once at startup and stores the results in app.core.state.
"""
import os
import json
import faiss
from google import genai
from rank_bm25 import BM25Okapi
from app.core.config import GEMINI_MODEL, VECTOR_STORE_DIR, GEMINI_API_KEY


# LOAD THE LLM FOR TEXT GENERATION
def load_llm():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not found — set it in the .env file.")
    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"[OK] Gemini model ready: {GEMINI_MODEL}")
    return client


# LOAD THE VECTOR DB
def load_vector_store(dimension: int):
    """
    Load the FAISS index + chunk metadata from disk.

    `dimension` must come from the loaded embedding model
    (embed_model.get_sentence_embedding_dimension()) so an empty index
    always matches whatever model is configured — no hardcoded sizes.
    """
    index_path = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    chunks_path = os.path.join(VECTOR_STORE_DIR, "chunks.json")
    # 1. If files don't exist, build an empty index & list
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        print("[INFO] Vector store files not found. Initializing empty vector store.")
        return faiss.IndexFlatIP(dimension), []
    # 2. Try loading the files safely
    try:
        index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            chunks = json.loads(content) if content else []
    except Exception as e:
        print(f"[WARNING] Failed to load index, initializing empty: {e}")
        return faiss.IndexFlatIP(dimension), []
    return index, chunks


# BUILD BM25 INDEX FROM CHUNK TEXTS
def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Build a BM25 index over all chunk texts.
    Tokenises each chunk with simple whitespace splitting.
    Called once at startup (via lifespan) and again after each PDF ingest.
    """
    if not chunks:
        # Return an empty-safe BM25 over a placeholder so callers never crash
        return BM25Okapi([[""]])
    tokenised = [chunk["text"].lower().split() for chunk in chunks]
    print(f"[OK] BM25 index built over {len(tokenised)} chunks.")
    return BM25Okapi(tokenised)
