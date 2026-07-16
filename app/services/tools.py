"""
Tool definitions for the RAG Agent.
Each tool has:
  1. A Python function that does the actual work
  2. A Gemini FunctionDeclaration describing it to the LLM
"""
import math
import ast
from datetime import datetime
from zoneinfo import ZoneInfo
from google.genai import types
from app.core.state import state
from app.services.answer import hybrid_retrieve, rerank_chunks


# ── Tool 1: Search Documents (RAG pipeline) ───────────────────────
def search_documents(query: str) -> str:
    """Search uploaded PDF documents using the full hybrid RAG pipeline."""
    if not state.vector_chunks:
        return "No documents have been uploaded yet."
    results = hybrid_retrieve(query)
    if not results:
        return "No relevant content found in the documents."
    reranked = rerank_chunks(query, results)
    chunks = [f"[Source: {r['source']}]\n{r['text']}" for r in reranked]
    return "\n\n---\n\n".join(chunks)


# ── Tool 2: Calculator ────────────────────────────────────────────
# Whitelist of safe names available inside expressions
_SAFE_NAMES = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
_SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})

_ALLOWED_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call,
    ast.Constant, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
)

def calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Supports arithmetic operators and all functions from Python's math module
    (sqrt, log, sin, cos, tan, ceil, floor, factorial, etc.).
    """
    try:
        tree = ast.parse(expression, mode="eval")
        # Whitelist-only AST walk — rejects anything dangerous
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_AST_NODES):
                return f"Error: unsupported operation in expression ('{type(node).__name__}')"
        result = eval(compile(tree, "<string>", "eval"), {"__builtins__": {}}, _SAFE_NAMES)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error evaluating expression: {e}"


# ── Tool 3: Get Current Datetime ──────────────────────────────────
def get_current_datetime(timezone: str = "Asia/Kolkata") -> str:
    """
    Return the current date and time in the given IANA timezone.
    Defaults to Asia/Kolkata (IST).
    """
    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
    except Exception:
        now = datetime.now()
        timezone = "Local"
    return now.strftime(f"Date: %A, %B %d, %Y | Time: %H:%M:%S | Timezone: {timezone}")


# ── Tool 4: List Knowledge Base ───────────────────────────────────
def list_knowledge_base() -> str:
    """List all PDF documents currently in the vector store with per-document chunk counts."""
    with state.lock:
        chunks = list(state.vector_chunks)
    if not chunks:
        return "The knowledge base is empty. No documents have been uploaded yet."

    doc_stats: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.get("source", "unknown")
        doc_stats[src] = doc_stats.get(src, 0) + 1

    lines = [f"Knowledge base: {len(doc_stats)} document(s), {len(chunks)} total chunks\n"]
    for doc, count in sorted(doc_stats.items()):
        lines.append(f"  - {doc}  ({count} chunk(s))")
    return "\n".join(lines)


# ── Gemini FunctionDeclarations ───────────────────────────────────
TOOL_DECLARATIONS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_documents",
        description=(
            "Search through the uploaded PDF documents to find relevant information. "
            "Use this for any factual question that might be answered by the user's documents."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="The search query to find relevant document passages."
                )
            },
            required=["query"]
        )
    ),
    types.FunctionDeclaration(
        name="calculator",
        description=(
            "Evaluate a mathematical expression. Supports +, -, *, /, **, % and all "
            "Python math functions: sqrt, log, sin, cos, tan, ceil, floor, factorial, etc."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "expression": types.Schema(
                    type=types.Type.STRING,
                    description="A valid Python math expression, e.g. '4850 * 0.15' or 'math.sqrt(144)'."
                )
            },
            required=["expression"]
        )
    ),
    types.FunctionDeclaration(
        name="get_current_datetime",
        description=(
            "Get the current date and time. Use this when the user asks about today's date, "
            "the current time, day of the week, or anything time-related."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "timezone": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "IANA timezone name, e.g. 'Asia/Kolkata', 'UTC', 'America/New_York'. "
                        "Defaults to Asia/Kolkata."
                    )
                )
            },
            required=[]
        )
    ),
    types.FunctionDeclaration(
        name="list_knowledge_base",
        description=(
            "List all PDF documents currently available in the knowledge base, "
            "along with how many chunks each document has been split into. "
            "Use this when the user asks what documents are available."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={},
            required=[]
        )
    ),
])

# ── Tool dispatcher ───────────────────────────────────────────────
TOOL_REGISTRY = {
    "search_documents":    lambda args: search_documents(**args),
    "calculator":          lambda args: calculator(**args),
    "get_current_datetime": lambda args: get_current_datetime(**args),
    "list_knowledge_base": lambda _: list_knowledge_base(),
}

def dispatch_tool(name: str, args: dict) -> str:
    """Call the named tool with the given arguments. Returns result as a string."""
    handler = TOOL_REGISTRY.get(name)
    if handler is None:
        return f"Error: unknown tool '{name}'"
    try:
        return handler(args)
    except Exception as e:
        return f"Error executing {name}: {e}"
