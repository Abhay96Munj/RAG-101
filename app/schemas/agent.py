"""
Pydantic schemas for the Agent endpoint.
"""
from pydantic import BaseModel
from typing import List


class AgentRequest(BaseModel):
    question: str


class ToolCallLog(BaseModel):
    tool_name: str
    arguments: dict
    result: str


class AgentResponse(BaseModel):
    answer: str
    tool_calls: List[ToolCallLog]
    query_id: str
