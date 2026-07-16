"""
Shared application state.

One RAGState instance holds every heavy resource (models, indexes, chunks).
It is populated exactly once by the lifespan in app/main.py, which also
attaches it to `app.state.rag` so request handlers can reach it the
idiomatic FastAPI way (request.app.state.rag). Services import the module
singleton directly because some callers (agent tools) have no Request.

Why a class instead of module-level globals?
- Tests can build a RAGState with fakes and swap it in.
- All mutable state lives in ONE named place instead of scattered globals.
- The lock that guards it lives next to the data it guards.
"""
import threading


class RAGState:
    def __init__(self):
        self.embed_model   = None   # SentenceTransformer
        self.reranker      = None   # CrossEncoder
        self.llm_client    = None   # google-genai Client
        self.vector_index  = None   # faiss.Index
        self.vector_chunks = None   # list[dict] — row i of the index ↔ chunks[i]
        self.bm25_index    = None   # BM25Okapi

        # Guards vector_index / vector_chunks / bm25_index.
        # Ingest mutates all three (in a worker thread) while chat queries
        # read them from other worker threads — without this lock a search
        # could see the FAISS index and the chunks list out of sync.
        self.lock = threading.Lock()


state = RAGState()
