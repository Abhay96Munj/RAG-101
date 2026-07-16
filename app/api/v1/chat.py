"""
Chat Router  —  POST /api/v1/chat/query
"""
import asyncio
from fastapi import APIRouter, HTTPException
from langfuse import observe
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.answer import hybrid_retrieve, rerank_chunks, generate_answer

router = APIRouter()


# @observe sits UNDER @router.post so FastAPI registers the wrapped function.
# It opens the root trace; every @observe'd function called from here
# (including via asyncio.to_thread, which copies contextvars) nests inside it.
@router.post("/query")
@observe(name="chat-query")
async def query(body: ChatRequest) -> ChatResponse:
    question = body.question

    # Run CPU-bound work in a thread so the async event loop stays unblocked
    results = await asyncio.to_thread(hybrid_retrieve, question, body.top_k)
    reranked_result = await asyncio.to_thread(rerank_chunks, question, results, body.rerank_top_n)

    # Every candidate fell below RERANK_SCORE_THRESHOLD — refuse here
    # instead of paying for a Gemini call with junk context.
    if not reranked_result:
        return {
            "answer": "I don't have enough information — nothing relevant to this question was found in the uploaded documents.",
            "sources": []
        }

    context = "\n\n---\n\n".join([r["text"] for r in reranked_result])
    try:
        answer = await asyncio.to_thread(generate_answer, context, question)
    except Exception as e:
        # A failed generation is an ERROR, not an answer — return 502 so
        # clients can tell the difference instead of parsing answer text.
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}")

    return {
        "answer": answer,
        "sources": reranked_result
    }


@router.post("/reranked")
async def rerank(body: ChatRequest) -> ChatResponse:
    question = body.question

    results = await asyncio.to_thread(hybrid_retrieve, question, body.top_k)
    reranked_result = await asyncio.to_thread(rerank_chunks, question, results, body.rerank_top_n)

    return {
        "answer": "this is only for testing",
        "sources": reranked_result
    }