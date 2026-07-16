# RAG-101 — Work History

A running log of all work done on this project, updated as features are added.

---

## Session 1 — July 15–16, 2026

### Starting Point
Existing `rag_service.py` (monolith) split into 3 service files:
- `app/services/load_llm.py` — model loading + vector store
- `app/services/ingestion.py` — PDF processing pipeline
- `app/services/answer.py` — retrieval + reranking + generation

---

### Feature: BM25 Hybrid Retrieval

**Why:** Pure semantic search (FAISS) misses exact keyword matches — names, numbers, technical terms.

**What was built:**
- `load_llm.py` → added `build_bm25_index()` using `rank_bm25.BM25Okapi`
- `answer.py` → added `retrieve_bm25()`, `reciprocal_rank_fusion()`, `hybrid_retrieve()`
- `ingestion.py` → rebuilds BM25 index after every new PDF ingest
- `api/v1/chat.py` → both endpoints now call `hybrid_retrieve()` instead of `retrieve_chunks()`

**How it works:**
```
Query
  ├── FAISS (dense / semantic)  → top-10 chunks
  └── BM25  (sparse / keyword)  → top-10 chunks
           ↓
   Reciprocal Rank Fusion (RRF, k=60)
           ↓
     CrossEncoder Reranker → top-3
           ↓
       Gemini LLM → final answer
```

---

### Feature: Performance Improvements

| Change | Before | After |
|--------|--------|-------|
| Reranker model | `BAAI/bge-reranker-base` (1.1 GB) | `cross-encoder/ms-marco-MiniLM-L-6-v2` (66 MB) |
| TOP_K | 20 | 10 |
| RERANK_TOP_N | 5 | 3 |
| Event loop | Blocked by CPU inference | `asyncio.to_thread()` throughout |

---

### Feature: GPU Acceleration

**Machine:** NVIDIA GeForce GTX 1050 (4 GB VRAM), Driver 512.78, CUDA 11.6

**Steps taken:**
1. Detected CPU-only PyTorch (`torch==2.13.0+cpu`)
2. Reinstalled with CUDA support: `uv pip install torch --index-url https://download.pytorch.org/whl/cu118 --force-reinstall`
3. Installed version: `torch==2.7.1+cu118`
4. Added auto-detection in `load_llm.py`:
   ```python
   DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
   embed_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
   reranker    = CrossEncoder(RERANKER_MODEL, device=DEVICE)
   ```

**Result:** Both embedding model and reranker now run on GPU. Falls back to CPU automatically if CUDA is unavailable.

---

### Feature: FastAPI Lifespan + Smart Hot Reload

**Problem:** Every file save triggered uvicorn hot reload → models re-loaded from disk → slow.

**Solution:**
- Moved all heavy initialization into FastAPI's `lifespan()` context in `app/main.py`
- `load_llm.py` now only holds `None` placeholders at import time — populated once at startup
- Server command uses `--reload-dir` to only watch lightweight files:
  ```powershell
  uv run uvicorn app.main:app --reload --reload-dir app/api --reload-dir app/schemas
  ```

**Result:**
- Router/schema changes → hot reload (fast, no model re-init)
- Service/config changes → requires manual restart only
- New PDF upload → only FAISS + BM25 updated in memory (no model reload)

---

### Feature: Tool-Calling Agent

**What was built:**

| File | Purpose |
|------|---------|
| `app/services/tools.py` | 4 tool functions + Gemini `FunctionDeclaration` schemas |
| `app/services/agent_logger.py` | Console + JSONL structured logging |
| `app/services/agent.py` | Multi-turn Gemini tool-calling loop |
| `app/api/v1/agent.py` | `POST /api/v1/agent/query` endpoint |
| `app/schemas/agent.py` | `AgentRequest` / `AgentResponse` Pydantic schemas |

**The 4 tools:**

| Tool | What it does | Libraries |
|------|-------------|-----------|
| `search_documents` | Hybrid RAG search over uploaded PDFs | FAISS + BM25 (existing) |
| `calculator` | Safely evaluates math expressions via AST whitelist | `ast`, `math` (stdlib) |
| `get_current_datetime` | Returns current date/time for a given IANA timezone | `datetime`, `zoneinfo` (stdlib) |
| `list_knowledge_base` | Lists all uploaded docs + chunk counts | `llm_state` (in-memory) |

**Agent loop:**
```
User Query
    ↓
AgentSession opened (query_id, timestamp logged)
    ↓
Gemini + tool declarations
    ↓ thinking text captured + logged
    ↓ tool call(s) dispatched + results logged
    ↓ results fed back to Gemini
    ↓ repeat until plain-text response
Final answer → AgentSession closed → appended to logs/agent_sessions.jsonl
```

**Logging — two outputs:**
- **Console:** `HH:MM:SS  [AGENT][query_id]  message` for live debugging
- **`logs/agent_sessions.jsonl`:** one JSON object per session — includes thinking text, all tool calls + results, final answer, duration in ms

**Endpoint:**
```
POST /api/v1/agent/query
{ "question": "..." }

→ { "answer": "...", "tool_calls": [...], "query_id": "abc123" }
```

---

## Session 2 — July 16, 2026

Full code review of the app, then fixes for all 12 findings (bugs first, then design issues).

### Bug Fixes (review items #1–#5)

| # | Bug | Fix |
|---|-----|-----|
| 1 | `process_new_pdf` crashed on every upload (`filename.page` on a string); page numbers were unknowable anyway since all pages were concatenated before chunking | `ingestion.py` now extracts text **per page** (`extract_pages_from_pdf`) and chunks each page separately — every chunk carries a real 1-based `page_num` and a document-wide unique `chunk_index` |
| 2 | FAISS pads results with `-1` when the index has fewer than `top_k` vectors; the old `idx < len(chunks)` guard let `-1` through and silently returned the **last** chunk | Guard is now `0 <= idx < len(chunks)` in `retrieve_chunks` |
| 3 | `ChatRequest.top_k` / `rerank_top_n` were accepted but silently ignored | Both endpoints now pass them to `hybrid_retrieve` / `rerank_chunks`; fields changed from `Optional[int]` to `int` (a client sending `null` gets a 422 instead of a crash) |
| 4 | `/ingest/upload` ran embedding synchronously inside `async def`, freezing every other request for seconds | `process_new_pdf` now runs via `asyncio.to_thread` |
| 5 | Path traversal: `os.path.join(PDFS_DIR, file.filename)` with `..\..\evil.pdf` writes outside `pdfs/`; non-PDFs caused a 500 in `PdfReader` | Filename sanitized with `os.path.basename()`; non-`.pdf` uploads rejected with 400 |

Retrieval results and `SourceChunk` now also include `page_num` (`Optional[int]` — chunks ingested before Session 2 have `null`; delete `vector_store/` and re-upload to backfill).

### Design Fixes (review items #6–#12)

| # | Issue | Fix |
|---|-------|-----|
| 6 | Race condition — ingest mutates `vector_index` / `vector_chunks` / `bm25_index` in one worker thread while chat queries read them from others | Added `threading.Lock` to shared state. Ingest embeds outside the lock (slow part), then mutates all three structures + persists under it. Both retrievers search/read under the same lock |
| 7 | Non-atomic persistence — a crash mid-write left `index.faiss` / `chunks.json` half-written or out of sync | New `_persist_vector_store()`: writes to `.tmp` siblings, then `os.replace()` (atomic) over the real files |
| 8 | `rag_service.py` — dead duplicate of `load_llm.py` that loaded model weights at import time | Deleted |
| 9 | Hardcoded `dimension = 384` for empty index — silently broke if `EMBEDDING_MODEL` changed | `load_vector_store(dimension)` now takes the dimension from the live model: `embed_model.get_sentence_embedding_dimension()` |
| 10 | `generate_answer` returned `"[ERROR] ..."` strings with HTTP 200 — clients couldn't tell answers from failures | `generate_answer` now raises; `/chat/query` converts to **HTTP 502**. (Tool errors in `dispatch_tool` intentionally stay as strings — the agent LLM should see them and retry) |
| 11 | Shared state as scattered module globals in `load_llm.py` | New `app/core/state.py` with a `RAGState` class (all resources + the lock in one place) and a module singleton. Lifespan populates it and also attaches it as `app.state.rag` for idiomatic FastAPI access. `load_llm.py` is now pure loader functions |
| 12 | Hygiene — unused imports in `main.py`/`chat.py`/`answer.py`, stale comment block in `answer.py`, schema defaults (20/5) drifted from config (10/3), `sys.exit` on missing API key | All cleaned; `ChatRequest` defaults now come from config (`TOP_K`, `RERANK_TOP_N`); missing `GEMINI_API_KEY` raises `RuntimeError` instead of `sys.exit` |

**New file:** `app/core/state.py` — shared `RAGState` (models, indexes, chunks, lock)
**Deleted:** `app/services/rag_service.py`

**Verified:** all files compile; full app imports; OpenAPI schema still lists all 5 endpoints; per-page chunking smoke-tested against `pdfs/blackstone.pdf` (3 pages → 7 chunks, correct page numbers, unique chunk indexes).

### Known Remaining Ideas (from the same review, not yet done)
- [ ] BM25 tokenization is punctuation-blind (`"revenue."` ≠ `"revenue"`) — share a `re.findall(r"\w+", ...)` tokenizer between indexing and querying
- [ ] Duplicate ingestion — re-uploading the same PDF doubles its chunks
- [ ] RRF overwrites the dense result dict with the sparse one, so `score` means different things per chunk
- [x] Retrieval evaluation harness (golden Q→source/page pairs, hit-rate metric) — **done in Session 3**
- [ ] Citations in answers (pass source + page metadata into the prompt)
- [ ] Conversation history; streaming responses
- [ ] Calculator: cap expression size (`9**9**9` can hang the process)
- [ ] Consider `pydantic-settings` `BaseSettings` for typed config (already installed)

---

## Session 3 — July 16, 2026

Built the evaluation harness — the first way to *measure* whether any retrieval/generation change helps or hurts, instead of tuning blind.

### Feature: Eval Harness (`eval/`)

| File | What it does | Cost to run |
|------|-------------|-------------|
| `eval/golden_set.json` | 26 hand-authored questions with `(source, page_num)` ground truth | — |
| `eval/run_eval.py` | Retrieval eval: hit-rate + MRR at each pipeline stage | Free (no LLM calls) |
| `eval/compare.py` | Diffs two runs: config changes, per-stage deltas, per-question regressions | Free |
| `eval/eval_generation.py` | End-to-end answer eval: keyword containment, refusals, LLM-judge faithfulness | ~52 Gemini calls |
| `eval/runs/` | Timestamped JSON results, each with a full config snapshot | — |

**Golden set design:**
- Relevance matched on `(source, page_num)`, **not** `chunk_index` — page numbers survive re-chunking, chunk indexes don't
- 12 **keyword** questions (exact names/figures — BM25's home turf), 11 **paraphrase** questions (wording deliberately avoids the document's — dense retrieval must carry them), 3 **negative** questions (answer not in the corpus at all, incl. one *hard* negative that sounds exactly like an indexed doc)
- All 9 documents covered; grounded in the actual chunk text, not the PDFs

**Retrieval eval (`run_eval.py`):** replicates the app lifespan in-process (no server, no API key) and captures all four stages per question — `dense@10`, `bm25@10`, `fused@10` (RRF), `rerank@3` — because *where* a failure happens matters more than *that* it happened. Reports overall + per-type breakdown, lists rerank-stage misses, and prints top rerank scores for negatives.

**Generation eval (`eval_generation.py`):** runs the exact `POST /chat/query` flow (same context format), then grades: keyword containment (free), refusal detection (free), and faithfulness via a hand-rolled LLM-as-judge — what RAGAS automates. Handles the free-tier 15 req/min limit: 5 s default delay + 60 s cooldown-and-retry on 429.

**The experiment loop this enables:**
```
run_eval.py → baseline saved → change ONE knob → run_eval.py → compare.py
```

### Baseline results (81 chunks, 9 docs)

| Metric | Score |
|--------|-------|
| Retrieval hit-rate (all stages) | 1.00 |
| MRR dense/bm25 → fused → rerank | 0.96 → 0.91 → 1.00 |
| Keyword hit rate (generation) | 1.00 (23/23) |
| Faithfulness (LLM judge) | 1.00 (23/23) |
| False refusals | 0 |
| Correct refusals on negatives | 3/3 |

Honest reading: perfect scores are expected at this corpus size (page-level matching @10 over ~2–4-page docs is easy). The harness's job now is **regression detection**; it becomes discriminating as the corpus and question set grow.

**Findings along the way:**
- `blackstone.pdf` was ingested twice (14 chunks instead of 7) — the duplicate-ingestion issue from Session 2's list, observed in the wild. Fix: delete `vector_store/`, re-upload each PDF once, re-run eval, `compare.py` the two runs
- Fused MRR (0.91) dips *below* the individual retrievers (0.96) — RRF admits the other list's near-misses — and the reranker restores it to 1.00. Fusion buys recall; reranking buys precision
- Negative questions get rerank scores of −3.7 to −10.5 vs. well above 0 for real questions → a rerank-score threshold (~0) would let the pipeline say "nothing relevant found" instead of feeding wrong context to the LLM. Measurable with this harness when implemented

---

## Session 4 — July 16–17, 2026

Two things: acted on Session 3's biggest finding (the rerank-score threshold), and stood up **Langfuse** for LLM observability — self-hosted in Docker, with the whole pipeline instrumented.

### Feature: Rerank Score Threshold

**Why:** Session 3's eval showed negatives (questions the corpus can't answer) get CrossEncoder scores of **−3.7 to −10.5**, while real questions score well above 0. Without a threshold, the pipeline feeds confidently-wrong context to Gemini and hopes the prompt makes it refuse.

**What was built:**
- `config.py` → `RERANK_SCORE_THRESHOLD` (env-overridable, default `0.0`)
- `answer.py` → `rerank_chunks()` drops candidates below the threshold *before* the top-N cut
- `chat.py` → if nothing survives, `/query` refuses immediately — **no Gemini call is made** (saves a free-tier request and removes the chance of a hallucinated answer)
- `tools.py` → `search_documents` does the same for the agent path ("No relevant content found")
- `eval_generation.py` mirrors the behavior, and both eval scripts snapshot the threshold in their config, so `compare.py` can attribute any metric change to it

### Feature: Langfuse Observability (self-hosted)

**Why Langfuse over the JSONL logger:** the hand-rolled logger answers "what happened"; Langfuse answers "where did the time and tokens go" — nested spans with latency, token usage, and cost per request, browsable in a UI.

**Deployment — `docker-compose.yml` at repo root** (moved from a `langfuse/` subfolder so future services join one stack; `name: rag-101` pins the Compose project so volume names don't depend on folder layout):

| Service | Role |
|---------|------|
| `langfuse-web` (:3000) | UI + API — the only port exposed beyond localhost |
| `langfuse-worker` | Pulls raw events off the queue, writes them to ClickHouse |
| `postgres` (host :**5433** — native PostgreSQL owns 5432) | Transactional data: users, projects, API keys |
| `clickhouse` | Columnar OLAP store for the traces themselves |
| `minio` | S3-compatible blob storage for raw event payloads |
| `redis` | Queue between web and worker |

**Instrumentation — `@observe` decorators (Langfuse SDK 4.x, OTel-based):**

```
chat-query                          agent-loop  (metadata: query_id)
├── hybrid-retrieve                 ├── gemini-agent-turn   [generation]
│   ├── faiss-dense    [retriever]  ├── tool:search_documents  [tool]
│   └── bm25-sparse    [retriever]  │   ├── hybrid-retrieve → ...
├── rerank                          │   └── rerank
└── gemini-answer      [generation] └── gemini-agent-turn   [generation]
```

- Both Gemini call sites report real token usage (`usage_metadata` → `update_current_generation`)
- `dispatch_tool` renames its span to `tool:<name>` at runtime
- `@observe` sits **under** `@router.post` so FastAPI registers the wrapped function; spans nest correctly across `asyncio.to_thread` because contextvars are copied into the thread
- **Zero-config off-switch:** without `LANGFUSE_PUBLIC_KEY` in `.env`, every decorator is a no-op — the app runs exactly as before
- **Cross-reference to the old logger:** the agent stamps the JSONL `query_id` onto the trace as metadata, so a line in `logs/agent_sessions.jsonl` can be looked up in the Langfuse UI (and vice versa)

**`.env` additions:**
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000   # without this the SDK targets cloud.langfuse.com
```

**Deferred by choice:** containerizing the app itself — the torch + CUDA image is heavy, and the root-level compose file is already set up to receive it later.

---

## Tech Stack (as of Session 4)

| Component | Package | Purpose |
|-----------|---------|---------|
| API framework | `fastapi 0.139` | REST API |
| PDF parsing | `pypdf 6.14` | Extract text |
| Embeddings | `sentence-transformers 5.6` + `BAAI/bge-small-en-v1.5` | Dense vectors |
| Reranker | `sentence-transformers` + `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-score candidates |
| Vector store | `faiss-cpu 1.14` | ANN search |
| Keyword search | `rank-bm25 0.2` | Sparse retrieval |
| LLM | `google-genai 2.11` + `gemini-flash-latest` (override via `GEMINI_MODEL`) | Text generation + tool calling |
| ML backend | `torch 2.7.1+cu118` | GPU inference |
| Server | `uvicorn 0.51` | ASGI server |
| Observability | `langfuse 4.14` (SDK) + self-hosted Langfuse v3 via Docker Compose | Tracing, token usage, latency |

---

## Project Structure (as of Session 4)

```
RAG-101/
├── app/
│   ├── api/v1/
│   │   ├── agent.py       ← POST /api/v1/agent/query
│   │   ├── chat.py        ← POST /api/v1/chat/query + /reranked  (traced)
│   │   └── ingest.py      ← POST /api/v1/ingest/upload
│   ├── core/
│   │   ├── config.py      ← env vars + defaults (incl. RERANK_SCORE_THRESHOLD)
│   │   └── state.py       ← RAGState: models, indexes, chunks, thread lock
│   ├── schemas/
│   │   ├── agent.py
│   │   └── chat.py
│   ├── services/
│   │   ├── agent.py       ← agent loop (traced, query_id → Langfuse metadata)
│   │   ├── agent_logger.py← console + JSONL logging
│   │   ├── answer.py      ← hybrid_retrieve, rerank (+threshold), generate
│   │   ├── ingestion.py   ← per-page PDF → chunks → FAISS + BM25 (atomic persist)
│   │   ├── load_llm.py    ← pure loader functions (models, index, BM25)
│   │   └── tools.py       ← 4 agent tools (traced)
│   └── main.py            ← lifespan + router registration + Langfuse shutdown
├── docs/
│   └── work_history.md    ← this file
├── eval/
│   ├── golden_set.json    ← 26 questions with (source, page_num) ground truth
│   ├── run_eval.py        ← retrieval eval (hit-rate + MRR per stage, free)
│   ├── compare.py         ← diff two runs
│   ├── eval_generation.py ← end-to-end answer eval (Gemini + LLM judge)
│   └── runs/              ← timestamped results with config snapshots
├── logs/
│   └── agent_sessions.jsonl  ← created on first agent query
├── pdfs/                  ← uploaded PDFs (gitignored)
├── vector_store/          ← FAISS index + chunks.json (gitignored)
├── docker-compose.yml     ← Langfuse stack (project name: rag-101)
├── .env                   ← GEMINI_API_KEY, LANGFUSE_* keys, model overrides
└── pyproject.toml
```

---

## Future Ideas / Next Steps

- [ ] Add Ollama as an alternative LLM backend (local, no API key)
- [ ] Add a web search tool to the agent (DuckDuckGo, no API key needed)
- [ ] Persist conversation history for multi-turn chat sessions
- [ ] Add a Streamlit or Gradio UI on top of the API
- [ ] Experiment with larger embedding models for better retrieval accuracy
- [ ] Add a `DELETE /api/v1/ingest/{filename}` endpoint to remove documents
- [ ] Load-test the API and profile bottlenecks
