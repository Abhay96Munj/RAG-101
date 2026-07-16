from sentence_transformers import CrossEncoder
from app.services.load_llm import llm_client, vector_index, vector_chunks, embed_model, bm25_index
from app.core.config import RERANKER_MODEL, GEMINI_MODEL, TOP_K, RERANK_TOP_N, SYSTEM_PROMPT

TOP_K = TOP_K
RERANK_TOP_N = RERANK_TOP_N
reranker = CrossEncoder(RERANKER_MODEL)
system_prompt = SYSTEM_PROMPT


# ── FAISS (dense / semantic) retrieval ───────────────────────────
def retrieve_chunks(query: str, model=embed_model, index=vector_index, chunks=vector_chunks, top_k=TOP_K):
    """Retrieve top-K chunks using dense vector similarity (FAISS)."""
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    query_embedding = model.encode([prefixed], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            results.append({
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "chunk_index": chunks[idx]["chunk_index"],
                "score": float(scores[0][i])
            })
    return results


# ── BM25 (sparse / keyword) retrieval ────────────────────────────
def retrieve_bm25(query: str, chunks=vector_chunks, bm25=bm25_index, top_k=TOP_K):
    """
    Retrieve top-K chunks using BM25 keyword matching.
    Great at catching exact names, numbers, and technical terms
    that dense embeddings might miss.
    """
    if not chunks:
        return []
    tokenised_query = query.lower().split()
    scores = bm25.get_scores(tokenised_query)
    # Get indices sorted by score descending
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0 and idx < len(chunks):  # skip zero-score chunks
            results.append({
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "chunk_index": chunks[idx]["chunk_index"],
                "score": float(scores[idx])
            })
    return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────
def reciprocal_rank_fusion(dense_results: list, sparse_results: list, k: int = 60) -> list:
    """
    Merge two ranked lists using Reciprocal Rank Fusion (RRF).

    Score for each chunk = sum of 1/(rank + k) across both lists.
    Chunks appearing highly in both lists get the highest combined score.

    k=60 is the standard constant from the original RRF paper (Cormack 2009).
    It dampens the impact of very high ranks and prevents any single list
    from dominating.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, result in enumerate(dense_results):
        key = f"{result['source']}::{result['chunk_index']}"
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + 1 + k)
        chunk_map[key] = result

    for rank, result in enumerate(sparse_results):
        key = f"{result['source']}::{result['chunk_index']}"
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + 1 + k)
        chunk_map[key] = result

    # Sort by combined RRF score descending
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    merged = []
    for key in sorted_keys:
        chunk = chunk_map[key].copy()
        chunk["rrf_score"] = rrf_scores[key]
        merged.append(chunk)
    return merged


# ── Hybrid retrieval (FAISS + BM25 + RRF) ────────────────────────
def hybrid_retrieve(query: str, top_k=TOP_K) -> list:
    """
    Run both FAISS and BM25 retrieval, then merge with RRF.
    Returns a unified ranked list ready for reranking.
    """
    dense_results  = retrieve_chunks(query, top_k=top_k)
    sparse_results = retrieve_bm25(query, top_k=top_k)
    return reciprocal_rank_fusion(dense_results, sparse_results)


# ── Rerank ────────────────────────────────────────────────────────
def rerank_chunks(query: str, candidates: list, top_n=RERANK_TOP_N):
    """Re-score candidates with a CrossEncoder and keep the top-N."""
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for cand, score in zip(candidates, scores):
        cand["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


# ── LLM generation ────────────────────────────────────────────────
def generate_answer(context: str, question: str):
    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}"
    try:
        response = llm_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] Generation failed: {e}"
