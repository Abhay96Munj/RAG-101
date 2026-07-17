# Frontend for the RAG-101 API — Design & Plan

**Date:** 2026-07-17
**Status:** Reviewed 2026-07-17 — backend recommendations approved for implementation (§6, resolved below); open questions answered (§11). Frontend not yet started.

## 1. Goal

A lightweight web UI for the existing FastAPI RAG backend covering its three user-facing capabilities:

1. **Chat** — ask a question, see the answer and its source chunks.
2. **Agent** — ask the tool-calling agent, see the answer and the tool-call trace.
3. **Documents** — upload PDFs and see what's in the knowledge base.

Constraints from the request: keep it simple; lightweight but a real JS app (not just HTML/CSS); frontend plan only — backend changes are *recommendations*, listed in §6.

## 2. Assumptions (made autonomously — please confirm on review)

- Single developer/local-use tool, no auth, no multi-user concerns — same audience as the API today.
- The backend is stateless per question (no conversation memory). The UI shows a session transcript for convenience, but each query is independent; the plan does not add multi-turn chat.
- English-only, desktop-first (usable on mobile, not optimized for it).
- Lives in this repo under `frontend/`.

## 3. Approaches considered

| Approach | Verdict |
|---|---|
| **A. Vite + React SPA (recommended)** | Small, fast dev loop, no server runtime. The API already exists — the frontend is pure client. Fits "lightweight but not just HTML/CSS". |
| B. Next.js | Rejected: SSR/routing/server components solve problems this app doesn't have. Heavier toolchain, extra Node server in Docker. |
| C. Streamlit / Gradio | Rejected: fastest to ship but limited control over source-chunk and tool-trace rendering, and it isn't a real frontend skill-fit for this project. |

**Recommendation: A.** Vite + React 19 + TypeScript.

## 4. Stack

| Concern | Choice | Why |
|---|---|---|
| Build | Vite | zero-config, instant HMR |
| UI | React 19 + TypeScript | typed against the Pydantic response shapes |
| Data fetching | SWR (GET) + thin `fetch` wrappers (POST) | request dedup/caching for the documents list (`client-swr-dedup`); mutations don't need a library |
| Styling | Tailwind CSS v4 | no component library; fast to build a clean two-column layout; dev-time dependency only |
| Markdown | `react-markdown`, lazy-loaded | Gemini answers contain markdown; loaded via `React.lazy` so it stays out of the initial bundle (`bundle-dynamic-imports`) |
| Routing | None — tab state in `App` | 3 views don't justify a router (YAGNI). Add one later only if deep-linking is needed. |

Total runtime deps: `react`, `react-dom`, `swr`, `react-markdown`. Nothing else.

## 5. Architecture

```
frontend/
├── src/
│   ├── api/
│   │   ├── client.ts        ← base URL + fetch wrapper (JSON, error normalization)
│   │   └── types.ts         ← TS mirrors of Pydantic schemas (ChatResponse, AgentResponse, …)
│   ├── views/
│   │   ├── ChatView.tsx     ← question box, transcript, per-answer sources
│   │   ├── AgentView.tsx    ← question box, transcript, per-answer tool-call trace
│   │   └── DocumentsView.tsx← upload dropzone + knowledge-base table
│   ├── components/
│   │   ├── QuestionForm.tsx ← shared input + submit + pending state
│   │   ├── SourceCard.tsx   ← collapsible chunk: doc, page, scores, text
│   │   ├── ToolCallCard.tsx ← tool name, arguments, result (collapsible)
│   │   ├── Answer.tsx       ← lazy markdown render, refusal styling
│   │   └── ErrorBanner.tsx  ← normalized API errors (FastAPI `detail`)
│   ├── App.tsx              ← header + tab switcher (Chat / Agent / Documents)
│   └── main.tsx
├── index.html
└── vite.config.ts           ← dev proxy: /api → http://localhost:8000
```

**Data flow:** view holds local state (question, in-flight flag, transcript array) → `api/client.ts` → FastAPI. No global store — the three views share nothing (state stays local; `rerender-*` rules mostly satisfied by structure rather than memoization).

**Dev networking:** Vite's dev proxy forwards `/api/*` to `:8000`, so **no CORS change is needed for development**. CORS (or serving the built bundle from FastAPI) only matters for a non-proxied deployment — see §6.

**Each view:**

- **ChatView** — POST `/api/v1/chat/query`. Transcript of Q→A cards; each answer card lists `sources` as collapsible `SourceCard`s showing `source`, `page_num`, `chunk_index`, `score`, `rerank_score`. Refusals (empty `sources`) render in a muted "no relevant documents" style. A 502 renders in `ErrorBanner` (the backend deliberately distinguishes LLM failure from refusal).
- **AgentView** — POST `/api/v1/agent/query`. Same transcript shape, but each answer shows the `tool_calls` array as an ordered trace (`ToolCallCard`) plus the `query_id` in small print for Langfuse cross-referencing. Agent calls can take tens of seconds → prominent pending state with elapsed-time counter.
- **DocumentsView** — POST `/api/v1/ingest/upload` (multipart) with per-file pending state; ingestion is slow (embedding), so the UI must stay honest: "uploading → indexing…" until the response returns `chunks_added`. Below it, the knowledge-base list — **blocked on the `GET /documents` endpoint (§6.2)**; until that exists, the view shows only upload + a running log of this session's uploads.

## 6. Recommended backend changes (NOT made — for discussion)

Ordered by need. Only 6.1–6.2 block a good v1; the rest are additive nice-to-haves.

1. **CORS or static hosting (required for any non-proxied deployment).** Today the API sets no CORS headers. Options: (a) add `CORSMiddleware` allowing the frontend origin, or (b) mount the built bundle via `StaticFiles` in `app/main.py` so frontend and API share an origin — (b) also gives a zero-extra-container Docker story. Recommend (b) for deployment, dev proxy for development, and skip CORS entirely.
2. **`GET /api/v1/documents` (required for the Documents view).** The logic already exists as the agent tool `list_knowledge_base()` (`app/services/tools.py:81`) but returns a human-formatted string. Recommend a small shared service function returning structured data — `[{filename, chunk_count, pages}]` — used by both the tool (formats it) and a new REST endpoint (returns JSON).
3. **Explicit refusal flag on `ChatResponse`.** A refusal is currently detectable only by matching the answer string or inferring from empty `sources`. Recommend an additive `refused: bool = False` field so the UI (and eval scripts) don't rely on prose. Non-breaking.
4. **`query_id` on `ChatResponse`.** `AgentResponse` has one; chat doesn't. Additive, enables trace cross-referencing and future feedback buttons.
5. **Duplicate-upload signal.** Re-uploading a PDF silently duplicates chunks (README warns, API doesn't). Recommend the upload response include `already_existed: bool` (or reject with 409) so the UI can warn.
6. **(v2) SSE streaming** for chat tokens and agent tool-call progress. Explicitly out of scope for v1 — spinners are enough at this stage; noted so response shapes above aren't designed in a way that blocks it.

## 7. React best practices applied (from `react-best-practices`)

- `client-swr-dedup` — SWR for the documents list; revalidate after an upload succeeds.
- `bundle-dynamic-imports` / `bundle-conditional` — `react-markdown` behind `React.lazy` + `Suspense`.
- `bundle-barrel-imports` — direct file imports, no `index.ts` barrels.
- `rendering-conditional-render` — ternaries, never `&&`, for transcript/pending/error branches.
- `rerender-functional-setstate` — transcript appends use functional updates so submit handlers stay stable.
- `rerender-no-inline-components`, `rendering-hoist-jsx` — cards defined at module level; static header/tab JSX hoisted.
- `js-early-exit`, `async-parallel` — client code is trivial; the only awaited work is single API calls, so no waterfall risk.

## 8. Error handling

- `api/client.ts` normalizes every non-2xx into `{status, message}` from FastAPI's `{detail}`; views render it in `ErrorBanner` — no raw fetch errors reach components.
- Network-down and 502 (LLM failure) are shown distinctly from refusals (which are 200s).
- Upload rejects non-PDF client-side before hitting the 400.

## 9. Testing

Deliberately light, matching the project's learning-project ethos: Vitest + React Testing Library for (a) `api/client.ts` error normalization and (b) one render test per view against mocked responses (answer with sources, refusal, tool trace). No e2e harness in v1.

## 10. Out of scope (YAGNI)

Auth, conversation memory/multi-turn, chat history persistence, streaming (v2), mobile polish, i18n, the `/chat/reranked` debug inspector, routing/deep links, state libraries.

## 11. Open questions — resolved 2026-07-17

1. **Separate dev-server-only tool.** The frontend runs via `npm run dev` with the Vite proxy; no StaticFiles mount, no CORS change needed (§6.1 requires no backend work).
2. **`GET /api/v1/documents` is in scope** — implemented per §6.2; the Documents view ships complete.
3. **Tailwind v4** (delegated decision).

Consequently §6.3–6.5 are being implemented (additive `refused` + `query_id` on `ChatResponse`; duplicate upload rejected with **409** rather than a warn-after-the-fact flag, since by response time the chunks would already be duplicated). §6.6 (SSE) remains v2.
