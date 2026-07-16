"""
Agent Router  —  POST /api/v1/agent/query
"""
import asyncio
from fastapi import APIRouter
from app.schemas.agent import AgentRequest, AgentResponse
from app.services.agent import run_agent

router = APIRouter()


@router.post("/query")
async def agent_query(body: AgentRequest) -> AgentResponse:
    """
    Ask the agent a question. It will autonomously decide which tools
    to call, execute them, and return a synthesised answer along with
    the full tool call trace and a session ID for log cross-referencing.
    """
    result = await asyncio.to_thread(run_agent, body.question)
    return result
