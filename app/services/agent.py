"""
Agent loop — orchestrates multi-turn Gemini tool calling.

Flow per request:
  1. Open a log session
  2. Send question + tool declarations to Gemini
  3. If Gemini returns function calls → execute → append results → repeat
  4. When Gemini returns plain text → that is the final answer
  5. Close log session (flushes to JSONL)
"""
from google.genai import types
from langfuse import observe, get_client
from app.core.config import GEMINI_MODEL
from app.services.agent_logger import AgentSession
from app.services.tools import TOOL_DECLARATIONS, dispatch_tool
from app.core.state import state

MAX_ITERATIONS = 10   # safety cap to prevent runaway loops

AGENT_SYSTEM_PROMPT = """\
You are a helpful research assistant with access to the following tools:
- search_documents: search through uploaded PDF documents
- calculator: evaluate mathematical expressions
- get_current_datetime: get the current date and time
- list_knowledge_base: list all available documents

Rules:
- Always use the most appropriate tool to answer accurately.
- For document questions, use search_documents.
- For arithmetic or percentage questions, use calculator.
- For date/time questions, use get_current_datetime.
- To check what documents exist, use list_knowledge_base.
- You may call multiple tools if the question requires it.
- After receiving tool results, synthesise a clear, concise answer.
"""


@observe(as_type="generation", name="gemini-agent-turn", capture_input=False)
def _call_gemini(client, conversation, config):
    """One Gemini turn, traced as a generation with token usage."""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=conversation,
        config=config,
    )
    usage = getattr(response, "usage_metadata", None)
    if usage:
        get_client().update_current_generation(
            model=GEMINI_MODEL,
            usage_details={
                "input":  usage.prompt_token_count or 0,
                "output": usage.candidates_token_count or 0,
            },
        )
    return response


@observe(as_type="agent", name="agent-loop")
def run_agent(question: str) -> dict:
    """
    Run the full agent loop for a question.

    Returns:
        {
            "answer":     str,
            "tool_calls": list[dict],   # full trace of tool calls made
            "query_id":   str,          # session ID for cross-referencing logs
        }
    """
    session    = AgentSession(question)
    client     = state.llm_client
    tool_log   = []

    # Cross-reference: stamp the JSONL logger's query_id onto the Langfuse
    # trace, so a line in logs/agent_sessions.jsonl can be matched to its
    # trace in the UI (filter/search by metadata.query_id).
    get_client().update_current_span(metadata={"query_id": session.query_id})

    # Build initial conversation
    conversation: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=question)])
    ]

    config = types.GenerateContentConfig(
        system_instruction=AGENT_SYSTEM_PROMPT,
        tools=[TOOL_DECLARATIONS],
    )

    for iteration in range(1, MAX_ITERATIONS + 1):

        response = _call_gemini(client, conversation, config)

        response_content = response.candidates[0].content
        conversation.append(response_content)

        # Separate text (thinking) from function call parts
        thinking_parts: list[str] = []
        fn_calls: list = []
        for part in response_content.parts:
            if hasattr(part, "function_call") and part.function_call:
                fn_calls.append(part.function_call)
            elif hasattr(part, "text") and part.text:
                thinking_parts.append(part.text)

        thinking = " ".join(thinking_parts).strip()

        # No tool calls → Gemini is done, thinking IS the final answer
        if not fn_calls:
            final_answer = thinking or "I was unable to generate a final answer."
            session.close(final_answer)
            return {
                "answer":     final_answer,
                "tool_calls": tool_log,
                "query_id":   session.query_id,
            }

        # Execute every tool call Gemini requested in this turn
        function_response_parts: list[types.Part] = []
        for fn_call in fn_calls:
            tool_name   = fn_call.name
            tool_args   = dict(fn_call.args)
            tool_result = dispatch_tool(tool_name, tool_args)

            session.log_iteration(iteration, thinking, tool_name, tool_args, tool_result)
            tool_log.append({
                "tool_name": tool_name,
                "arguments": tool_args,
                "result":    tool_result,
            })

            function_response_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=tool_name,
                    response={"result": tool_result},
                )
            ))

        # Feed all tool results back to Gemini for the next turn
        conversation.append(
            types.Content(role="tool", parts=function_response_parts)
        )

    # Safety: hit the iteration cap
    final_answer = "Agent reached the maximum number of iterations without a final answer."
    session.close(final_answer)
    return {
        "answer":     final_answer,
        "tool_calls": tool_log,
        "query_id":   session.query_id,
    }
