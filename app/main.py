"""
FastAPI Application Entry Point
================================
Start the server:
    uvicorn app.main:app --reload --reload-dir app/api --reload-dir app/schemas

Using --reload-dir means ONLY changes to routers/schemas trigger a hot reload.
Changes to app/services (model loading, ingestion) require a manual restart,
which avoids unnecessarily reloading model weights and rebuilding BM25 on every save.
"""

from contextlib import asynccontextmanager
import os
import torch
from fastapi import FastAPI
from langfuse import get_client
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.api.v1 import chat, ingest, agent
from app.core.config import EMBEDDING_MODEL, RERANKER_MODEL
from app.core.state import state
from app.services.load_llm import load_llm, load_vector_store, build_bm25_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at server startup — loads all heavy resources into the shared
    RAGState. Nothing here re-runs on hot reload of router/schema files.
    """
    print("\n[*] Starting up — loading models and indexes...")

    # 1. Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""
    print(f"[OK] Device: {device.upper()}{gpu_name}")

    # 2. Embedding model
    print(f"[*] Loading embedding model: {EMBEDDING_MODEL}")
    state.embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # 3. Reranker
    print(f"[*] Loading reranker: {RERANKER_MODEL}")
    state.reranker = CrossEncoder(RERANKER_MODEL, device=device)

    # 4. LLM client (Gemini)
    state.llm_client = load_llm()

    # 5. Vector store (FAISS + chunks) — dimension comes from the live
    #    embedding model, so changing EMBEDDING_MODEL can't cause a mismatch.
    dimension = state.embed_model.get_embedding_dimension()
    state.vector_index, state.vector_chunks = load_vector_store(dimension)
    print(f"[OK] FAISS index loaded: {state.vector_index.ntotal} vectors")

    # 6. BM25 index
    state.bm25_index = build_bm25_index(state.vector_chunks)

    # Expose the state object the idiomatic FastAPI way too, so request
    # handlers can use request.app.state.rag if they prefer.
    app.state.rag = state

    # 7. Langfuse tracing — enabled purely by env vars; without keys every
    #    @observe decorator is a no-op and the app runs exactly as before.
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        print(f"[OK] Langfuse tracing -> {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")
    else:
        print("[INFO] Langfuse keys not set - tracing disabled.")

    print("[OK] All resources loaded. Server is ready.\n")
    yield
    # Shutdown — flush any traces still buffered in the Langfuse client
    get_client().shutdown()
    print("[*] Shutting down.")


app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API backed by FAISS + Gemini",
    version="1.0.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(chat.router,   prefix="/api/v1/chat",   tags=["Chat"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingest"])
app.include_router(agent.router,  prefix="/api/v1/agent",  tags=["Agent"])


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
