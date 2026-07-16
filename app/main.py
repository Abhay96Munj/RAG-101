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
import torch
import faiss
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from app.api.v1 import chat, ingest
from app.core.config import EMBEDDING_MODEL, RERANKER_MODEL, VECTOR_STORE_DIR
from app.services.load_llm import load_llm, load_vector_store, build_bm25_index
import app.services.load_llm as llm_state
import json, os


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at server startup — loads all heavy resources into app.state.
    Nothing here re-runs on hot reload of router/schema files.
    """
    print("\n[*] Starting up — loading models and indexes...")

    # 1. Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""
    print(f"[OK] Device: {device.upper()}{gpu_name}")

    # 2. Embedding model
    print(f"[*] Loading embedding model: {EMBEDDING_MODEL}")
    llm_state.embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # 3. Reranker
    print(f"[*] Loading reranker: {RERANKER_MODEL}")
    llm_state.reranker = CrossEncoder(RERANKER_MODEL, device=device)

    # 4. LLM client (Gemini)
    llm_state.llm_client = load_llm()

    # 5. Vector store (FAISS + chunks)
    llm_state.vector_index, llm_state.vector_chunks = load_vector_store()
    print(f"[OK] FAISS index loaded: {llm_state.vector_index.ntotal} vectors")

    # 6. BM25 index
    llm_state.bm25_index = build_bm25_index(llm_state.vector_chunks)

    print("[OK] All resources loaded. Server is ready.\n")
    yield
    # Shutdown — nothing to clean up
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


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
