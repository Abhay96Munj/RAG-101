from langfuse import observe, get_client
from app.core.config import (
    GEMINI_MODEL, TOP_K, RERANK_TOP_N, RERANK_SCORE_THRESHOLD, SYSTEM_PROMPT,
)
from app.core.state import state


# ── FAISS (dense / semantic) retrieval ───────────────────────────────────────
@observe(as_type="retriever", name="faiss-dense")
def retrieve_chunks(query: str, top_k: int = TOP_K) -> list:
    """Retrieve top-K chunks using dense vector similarity (FAISS)."""
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    # Embed OUTSIDE the lock — it's the slow part and touches no shared state.
    query_embedding = state.embed_model.encode(
        [prefixed], normalize_embeddings=True
    ).astype("float32")

    # Lock while searching + reading chunks so a concurrent ingest can't
    # leave the FAISS index and the chunks list out of sync under us.
    with state.lock:
        chunks = state.vector_chunks
        scores, indices = state.vector_index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            # FAISS pads with -1 when the index holds fewer than top_k vectors;
            # a plain `idx < len(chunks)` would let -1 through and silently
            # return the LAST chunk (Python negative indexing).
            if 0 <= idx < len(chunks):
                results.append({
                    "text":        chunks[idx]["text"],
                    "source":      chunks[idx]["source"],
                    "page_num":    chunks[idx].get("page_num"),
                    "chunk_index": chunks[idx]["chunk_index"],
                    "score":       float(scores[0][i])
                })
    return results


# ── BM25 (sparse / keyword) retrieval ────────────────────────────────────────
@observe(as_type="retriever", name="bm25-sparse")
def retrieve_bm25(query: str, top_k: int = TOP_K) -> list:
    """
    Retrieve top-K chunks using BM25 keyword matching.
    Great at catching exact names, numbers, and technical terms
    that dense embeddings might miss.
    """
    tokenised_query = query.lower().split()

    with state.lock:
        chunks = state.vector_chunks
        if not chunks:
            return []
        scores = state.bm25_index.get_scores(tokenised_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0 and idx < len(chunks):
                results.append({
                    "text":        chunks[idx]["text"],
                    "source":      chunks[idx]["source"],
                    "page_num":    chunks[idx].get("page_num"),
                    "chunk_index": chunks[idx]["chunk_index"],
                    "score":       float(scores[idx])
                })
    return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────
def reciprocal_rank_fusion(dense_results: list, sparse_results: list, k: int = 60) -> list:
    """
    Merge two ranked lists using Reciprocal Rank Fusion (RRF).

    Score for each chunk = sum of 1/(rank + k) across both lists.
    Chunks appearing highly in both lists get the highest combined score.

    k=60 is the standard constant from the original RRF paper (Cormack 2009).
    """
    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict]  = {}

    for rank, result in enumerate(dense_results):
        key = f"{result['source']}::{result['chunk_index']}"
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + 1 + k)
        chunk_map[key] = result

    for rank, result in enumerate(sparse_results):
        key = f"{result['source']}::{result['chunk_index']}"
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + 1 + k)
        chunk_map[key] = result

    sorted_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)
    merged = []
    for key in sorted_keys:
        chunk = chunk_map[key].copy()
        chunk["rrf_score"] = rrf_scores[key]
        merged.append(chunk)
    return merged


# ── Hybrid retrieval (FAISS + BM25 + RRF) ────────────────────────────────────
@observe(name="hybrid-retrieve")
def hybrid_retrieve(query: str, top_k: int = TOP_K) -> list:
    """
    Run both FAISS and BM25 retrieval, then merge with RRF.
    Returns a unified ranked list ready for reranking.
    """
    dense_results  = retrieve_chunks(query, top_k=top_k)
    sparse_results = retrieve_bm25(query, top_k=top_k)
    return reciprocal_rank_fusion(dense_results, sparse_results)


# ── Rerank ────────────────────────────────────────────────────────────────────
@observe(name="rerank")
def rerank_chunks(query: str, candidates: list, top_n: int = RERANK_TOP_N) -> list:
    """
    Re-score candidates with a CrossEncoder and keep the top-N.

    Candidates scoring below RERANK_SCORE_THRESHOLD are dropped entirely —
    the reranker is telling us they don't answer the query, and bad context
    is worse than no context. An empty return means "nothing relevant found"
    and callers should refuse instead of calling the LLM.
    """
    pairs  = [(query, c["text"]) for c in candidates]
    scores = state.reranker.predict(pairs)

    for cand, score in zip(candidates, scores):
        cand["rerank_score"] = float(score)

    reranked = [c for c in candidates if c["rerank_score"] >= RERANK_SCORE_THRESHOLD]
    reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


# ── LLM generation ────────────────────────────────────────────────────────────
@observe(as_type="generation", name="gemini-answer")
def generate_answer(context: str, question: str) -> str:
    """
    Generate the final answer with Gemini.

    Deliberately does NOT catch exceptions: a failed generation is an error,
    not an answer. The chat router converts it to an HTTP 502 so clients can
    tell "the model failed" apart from "here is your answer".
    """
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
    response = state.llm_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    # Report model + token usage to Langfuse so traces show cost/latency
    # per generation. No-op when tracing is disabled (no keys set).
    usage = getattr(response, "usage_metadata", None)
    if usage:
        get_client().update_current_generation(
            model=GEMINI_MODEL,
            usage_details={
                "input":  usage.prompt_token_count or 0,
                "output": usage.candidates_token_count or 0,
            },
        )
    return response.text.strip()
