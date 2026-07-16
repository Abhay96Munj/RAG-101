"""
Chat Router  —  POST /api/v1/chat/query
"""
import asyncio
from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.answer import hybrid_retrieve, rerank_chunks, generate_answer, retrieve_chunks

router = APIRouter()


@router.post("/query")
async def query(body: ChatRequest) -> ChatResponse:
    question = body.question

    # Run CPU-bound work in a thread so the async event loop stays unblocked
    results = await asyncio.to_thread(hybrid_retrieve, question)
    reranked_result = await asyncio.to_thread(rerank_chunks, question, results)

    context = "\n\n---\n\n".join([r["text"] for r in reranked_result])
    answer = await asyncio.to_thread(generate_answer, context, question)
    return {
        "answer": answer,
        "sources": reranked_result
    }


@router.post("/reranked")
async def rerank(body: ChatRequest) -> ChatResponse:
    question = body.question

    results = await asyncio.to_thread(hybrid_retrieve, question)
    reranked_result = await asyncio.to_thread(rerank_chunks, question, results)

    return {
        "answer": "this is only for testing",
        "sources": reranked_result
    }