import os
import sys
import json
import faiss
import torch
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from app.core.config import EMBEDDING_MODEL, RERANKER_MODEL, GEMINI_MODEL, VECTOR_STORE_DIR, GEMINI_API_KEY, TOP_K, RERANK_TOP_N, SYSTEM_PROMPT, SENTENCE_OVERLAP, SENTENCES_PER_CHUNK, PDFS_DIR


load_dotenv()

# ── Shared state — populated once by the lifespan in app/main.py ─────────────
# These are None at import time and assigned during server startup.
# All service modules import from here to get the live references.
embed_model   = None
reranker      = None
llm_client    = None
vector_index  = None
vector_chunks = None
bm25_index    = None

gemini_key    = GEMINI_API_KEY
system_prompt = SYSTEM_PROMPT

# LOAD THE LLM FOR TEXT GENERATION
def load_llm():
    api_key = gemini_key
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not found in .env file.")
        sys.exit(1)
    client = genai.Client(api_key=api_key)
    print(f"[OK] Gemini model ready: {GEMINI_MODEL}")
    return client

# LOAD THE VECTOR DB
def load_vector_store():
    index_path = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    chunks_path = os.path.join(VECTOR_STORE_DIR, "chunks.json")
    # 1. If files don't exist, build an empty index & list
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        print("[INFO] Vector store files not found. Initializing empty vector store.")
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        chunks = []
        return index, chunks
    # 2. Try loading the files safely
    try:
        index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            chunks = json.loads(content) if content else []
    except Exception as e:
        print(f"[WARNING] Failed to load index, initializing empty: {e}")
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        chunks = []
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
