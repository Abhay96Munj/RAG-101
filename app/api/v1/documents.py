"""
Documents Router  —  GET /api/v1/documents
"""
from fastapi import APIRouter
from app.schemas.documents import DocumentsResponse
from app.services.knowledge_base import get_document_stats

router = APIRouter()


@router.get("")
async def list_documents() -> DocumentsResponse:
    # Just iterates the in-memory chunk list (no models, no I/O), so unlike
    # chat/ingest it doesn't need to be pushed off to a worker thread.
    docs = get_document_stats()
    return {
        "documents": docs,
        "total_chunks": sum(d["chunk_count"] for d in docs),
    }
