# RAG-101 Frontend

A lightweight Vite + React + TypeScript UI for the RAG API: Chat (answers with source chunks), Agent (tool-call traces), and Documents (upload + knowledge-base listing).

Design and decisions: [`docs/specs/2026-07-17-frontend-design.md`](../docs/specs/2026-07-17-frontend-design.md).

## Run (development)

Requires Node 20+. Start the FastAPI server on port 8000 first, then:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api/*` to `http://localhost:8000`, so no CORS configuration is needed anywhere.

## Scripts

| Command | Purpose |
|---|---|
| `npm run dev` | dev server with hot reload |
| `npm test` | Vitest unit/render tests |
| `npm run typecheck` | TypeScript check only |
| `npm run build` | typecheck + production bundle (not used for deployment yet — this is a dev-server-only tool by design) |
