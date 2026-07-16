"""
Agent session logger.
Writes structured output to two destinations simultaneously:
  - Console  (human-readable, for live debugging)
  - logs/agent_sessions.jsonl  (one JSON object per session, for research)
"""
import json
import logging
import os
import time
from datetime import datetime, timezone
from uuid import uuid4

# ── Console logger ────────────────────────────────────────────────
# Custom formatter injects query_id into every log line
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s  [AGENT][%(query_id)s]  %(message)s",
    datefmt="%H:%M:%S"
))

logger = logging.getLogger("rag_agent")
if not logger.handlers:          # avoid duplicate handlers on hot reload
    logger.addHandler(_handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False         # don't bubble up to uvicorn's root logger

# ── JSONL file path ───────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_DIR  = os.path.join(_BASE_DIR, "logs")
JSONL_PATH = os.path.join(LOGS_DIR, "agent_sessions.jsonl")


class AgentSession:
    """
    Tracks a single agent run from first question to final answer.

    Usage:
        session = AgentSession(question)
        session.log_iteration(...)   # once per tool call
        session.close(final_answer)  # flushes to JSONL
    """

    def __init__(self, question: str):
        self.query_id   = uuid4().hex[:8]          # short readable ID
        self.question   = question
        self.timestamp  = datetime.now(timezone.utc).isoformat()
        self.iterations: list[dict] = []
        self._started   = time.monotonic()

        self._log(f'New session | Question: "{question}"')

    # ── Logging helpers ───────────────────────────────────────────

    def _log(self, msg: str):
        logger.info(msg, extra={"query_id": self.query_id})

    def log_iteration(
        self,
        iteration: int,
        thinking: str,
        tool_name: str,
        tool_args: dict,
        tool_result: str,
    ):
        """Log one tool call within the agent loop."""
        record = {
            "iteration":   iteration,
            "thinking":    thinking,
            "tool_name":   tool_name,
            "tool_args":   tool_args,
            "tool_result": tool_result,
        }
        self.iterations.append(record)

        # Console output — truncate long strings for readability
        if thinking:
            preview = thinking[:200] + ("..." if len(thinking) > 200 else "")
            self._log(f"[iter {iteration}] Thinking: {preview}")

        self._log(f"[iter {iteration}] Tool call : {tool_name}({json.dumps(tool_args)})")

        result_preview = tool_result[:300] + ("..." if len(tool_result) > 300 else "")
        self._log(f"[iter {iteration}] Tool result: {result_preview}")

    def close(self, final_answer: str):
        """Finalise the session and append the full record to the JSONL log file."""
        duration_ms = int((time.monotonic() - self._started) * 1000)
        self._log(
            f"Final answer after {len(self.iterations)} iteration(s) "
            f"[{duration_ms} ms]: {final_answer[:120]}{'...' if len(final_answer) > 120 else ''}"
        )

        # Build the full JSONL record
        record = {
            "query_id":         self.query_id,
            "timestamp":        self.timestamp,
            "question":         self.question,
            "iterations":       self.iterations,
            "final_answer":     final_answer,
            "total_iterations": len(self.iterations),
            "duration_ms":      duration_ms,
        }

        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
