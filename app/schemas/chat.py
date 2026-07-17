"""
Pydantic schemas for the Chat endpoints.
"""

from pydantic import BaseModel
from typing import List, Optional
from app.core.config import TOP_K, RERANK_TOP_N


class ChatRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    rerank_top_n: int = RERANK_TOP_N


class SourceChunk(BaseModel):
    source: str
    page_num: Optional[int] = None   # older chunks in the store may not have it
    chunk_index: int
    score: float
    rerank_score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    refused: bool = False            # True when nothing relevant was found — no LLM call was made
    query_id: Optional[str] = None   # set by /query; cross-references logs/traces like the agent's
