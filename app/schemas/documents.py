"""
Pydantic schemas for the Documents endpoint.
"""

from pydantic import BaseModel
from typing import List


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int
    pages: int   # distinct page numbers; 0 if older chunks lack page_num


class DocumentsResponse(BaseModel):
    documents: List[DocumentInfo]
    total_chunks: int
