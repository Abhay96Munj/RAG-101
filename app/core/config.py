"""
Application Configuration
===========================
Reads settings from environment variables / .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
PDFS_DIR = os.path.join(BASE_DIR, "pdfs")

# ── Models ────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# ── Retrieval ─────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", 20))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 5))

# ── API Keys ──────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SENTENCES_PER_CHUNK = 5
SENTENCE_OVERLAP = 1

SYSTEM_PROMPT = """You are a concise question-answering assistant.
Answer based ONLY on the context provided.
If the context does not contain the answer, say: I don't have enough information.
Keep your response under 4 sentences."""