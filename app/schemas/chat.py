"""
Pydantic schemas for the Chat endpoints.
"""

from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 20
    rerank_top_n: Optional[int] = 5


class SourceChunk(BaseModel):
    source: str
    chunk_index: int
    score: float
    rerank_score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
