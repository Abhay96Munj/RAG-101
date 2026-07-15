"""
FastAPI Application Entry Point
================================
Start the server:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from app.api.v1 import chat, ingest

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API backed by FAISS + Gemini",
    version="1.0.0",
)

# Register routers
app.include_router(chat.router,   prefix="/api/v1/chat",   tags=["Chat"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingest"])


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
