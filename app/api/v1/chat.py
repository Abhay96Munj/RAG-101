"""
Chat Router  —  POST /api/v1/chat/query
"""

from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import retrieve_chunks, rerank_chunks, generate_answer

router = APIRouter()


@router.post("/query")
async def query(body: ChatRequest) -> ChatResponse:
    question = body.question

    results = retrieve_chunks(question)
    reranked_result = rerank_chunks(question, results)

    context = "\n\n---\n\n".join([r["text"] for r in reranked_result])
    answer = generate_answer(context, question)
    return {
        "answer": answer,
        "sources": reranked_result
    }


@router.post("/reranked")
async def rerank(body: ChatRequest) -> ChatResponse:
    question = body.question

    results = retrieve_chunks(question)
    reranked_result = rerank_chunks(question, results)

    return {
        "answer": "this is only testing",
        "sources": reranked_result
    }
