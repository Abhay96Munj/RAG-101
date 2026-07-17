# RAG-101

A Retrieval-Augmented Generation (RAG) API built with FastAPI — upload PDFs, ask questions, get answers grounded in your documents with sources. Includes a tool-calling agent, an evaluation harness, and self-hosted [Langfuse](https://langfuse.com) observability.

Built as a learning project, so the code favors readability and the design decisions are documented in [`docs/work_history.md`](docs/work_history.md).

## How it works

```
Question
  ├── FAISS  (dense / semantic)   → top-10 chunks     BAAI/bge-small-en-v1.5
  └── BM25   (sparse / keyword)   → top-10 chunks
            ↓
    Reciprocal Rank Fusion (RRF, k=60)
            ↓
    CrossEncoder reranker → top-3                     ms-marco-MiniLM-L-6-v2
            ↓  (candidates below RERANK_SCORE_THRESHOLD are dropped;
            ↓   if nothing survives, the API refuses instead of calling the LLM)
    Gemini → answer + sources
```

Hybrid retrieval matters because the two retrievers fail differently: FAISS understands paraphrases but misses exact names and numbers; BM25 nails exact keywords but not meaning. RRF merges both, and the reranker restores precision.

There is also an **agent endpoint**: a multi-turn Gemini tool-calling loop with four tools — `search_documents` (the full RAG pipeline), `calculator`, `get_current_datetime`, and `list_knowledge_base`. Every agent session is logged to `logs/agent_sessions.jsonl` and traced in Langfuse (cross-referenced by `query_id`).

## Quick start (local)

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), a [Gemini API key](https://aistudio.google.com/apikey) (free tier works).

```powershell
git clone <this-repo> && cd RAG-101
uv sync
```

Create a `.env` file in the repo root:

```env
GEMINI_API_KEY=your-key-here

# Optional — Langfuse tracing (see Observability below). Without these,
# tracing is a no-op and the app runs normally.
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

Start the server:

```powershell
uv run uvicorn app.main:app --reload --reload-dir app/api --reload-dir app/schemas
```

First startup downloads the embedding and reranker models (~300 MB, one time). The `--reload-dir` flags mean only router/schema edits hot-reload — service changes need a manual restart, which avoids reloading model weights on every save. A CUDA GPU is used automatically if available (see `docs/work_history.md` for the CUDA torch install); otherwise it falls back to CPU.

Interactive API docs: http://localhost:8000/docs

## Using the API

**1. Upload a PDF** (repeat per document — re-uploading the same file duplicates its chunks):

```powershell
curl -F "file=@your_document.pdf" http://localhost:8000/api/v1/ingest/upload
```

**2. Ask a question:**

```powershell
curl -X POST http://localhost:8000/api/v1/chat/query `
  -H "Content-Type: application/json" `
  -d '{"question": "How much did the Sunseeker Resort cost to build?"}'
```

Returns an answer plus the source chunks (document, page number, scores). If nothing relevant is found, it answers "I don't have enough information" instead of hallucinating.

**3. Or ask the agent** (decides on its own which tools to use, can chain them):

```powershell
curl -X POST http://localhost:8000/api/v1/agent/query `
  -H "Content-Type: application/json" `
  -d '{"question": "What did the resort cost, and what is 15% of that?"}'
```

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/ingest/upload` | Upload a PDF; chunks, embeds, and indexes it |
| `POST /api/v1/chat/query` | RAG answer with sources |
| `POST /api/v1/chat/reranked` | Retrieval + rerank only (debugging, no LLM call) |
| `POST /api/v1/agent/query` | Tool-calling agent |
| `GET /health` | Liveness check |

## Running with Docker

The compose file has two modes, controlled by a profile:

```powershell
docker compose up -d                        # Langfuse observability stack only
docker compose --profile app up -d --build  # everything: Langfuse + the RAG API on :8000
```

The containerized API runs **CPU-only inference** (keeps the image at ~1.8 GB instead of ~8 GB with CUDA) and mounts `vector_store/`, `pdfs/`, and `logs/` from the host — so the container and a locally-run server share the same index. Model weights are cached in a named volume and downloaded only on first start.

Typical dev setup: run Langfuse in Docker, run uvicorn locally (hot reload + GPU). Use the full profile when you want to test the deployable artifact.

`docker compose stop` frees all memory and keeps data; `docker compose down` removes containers (data survives in named volumes).

## Observability (Langfuse, self-hosted)

Every request produces a nested trace — retrieval spans, rerank, and Gemini generations with real token usage:

```
chat-query                          agent-loop
├── hybrid-retrieve                 ├── gemini-agent-turn   [generation]
│   ├── faiss-dense    [retriever]  ├── tool:search_documents
│   └── bm25-sparse    [retriever]  │   └── hybrid-retrieve → rerank
├── rerank                          └── gemini-agent-turn   [generation]
└── gemini-answer      [generation]
```

Setup: `docker compose up -d`, open http://localhost:3000, create an account and project, copy the API keys into `.env`, restart the server. No keys in `.env` → tracing silently disabled, zero overhead.

Note for the Langfuse stack: Postgres is mapped to host port **5433** (5432 is often taken by a native install), and the whole stack needs roughly 3–4 GB of RAM.

## Evaluation harness

Measure retrieval and answer quality instead of tuning blind — see `eval/`:

```powershell
uv run python eval/run_eval.py          # retrieval: hit-rate + MRR per pipeline stage (free, no LLM)
uv run python eval/compare.py           # diff the two latest runs: config changes, regressions
uv run python eval/eval_generation.py   # end-to-end answers: keywords, refusals, LLM-judge faithfulness
```

The golden set (`eval/golden_set.json`) contains keyword, paraphrase, and negative questions with `(source, page_num)` ground truth. Each run saves a timestamped JSON with a full config snapshot, enabling the loop: **baseline → change one knob → re-run → compare**.

The generation eval respects the Gemini free-tier limit (15 requests/min) via a configurable delay (`--delay`, default 5 s).

## Configuration

All via environment variables (see `app/core/config.py`):
Please check Google AI Studio on how to create your Project and API key, also to check what models are available in your API Tier.
| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | — (required) | Gemini API key |
| `GEMINI_MODEL` | `gemini-flash-latest` | Generation model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Dense embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder reranker |
| `TOP_K` | `10` | Candidates per retriever |
| `RERANK_TOP_N` | `3` | Chunks passed to the LLM |
| `RERANK_SCORE_THRESHOLD` | `0.0` | Below this score, chunks are dropped; empty result → refusal |

## Project structure

```
RAG-101/
├── app/
│   ├── api/v1/          ← routers: chat, ingest, agent
│   ├── core/            ← config + shared RAGState (models, indexes, lock)
│   ├── schemas/         ← Pydantic request/response models
│   ├── services/        ← retrieval, rerank, generation, agent loop, tools, ingestion
│   └── main.py          ← lifespan (loads models once) + router registration
├── eval/                ← golden set, eval scripts, timestamped results
├── docs/work_history.md ← session-by-session build log with design rationale
├── Dockerfile           ← CPU-only image for the API
├── docker-compose.yml   ← Langfuse stack + rag-api (profile: "app")
├── pdfs/                ← uploaded PDFs (created at runtime)
├── vector_store/        ← FAISS index + chunk metadata (created at runtime)
└── logs/                ← agent session JSONL logs
```
