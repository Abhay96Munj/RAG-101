"""
Retrieval Evaluation Harness (phase 1 — retrieval only, no LLM)
================================================================
Runs every golden-set question through the four retrieval stages:

    1. dense   — FAISS vector search          (top TOP_K)
    2. bm25    — keyword search               (top TOP_K)
    3. fused   — Reciprocal Rank Fusion       (top TOP_K)
    4. rerank  — CrossEncoder                 (top RERANK_TOP_N)

and reports hit-rate / MRR per stage, plus a keyword-vs-paraphrase
breakdown. A chunk counts as relevant when its (source, page_num)
matches an entry in the question's "relevant" list — page-level ground
truth survives re-chunking, chunk indexes don't.

No Gemini calls are made, so this runs free, fast, and without an API key.

Usage (from the repo root):
    .venv\\Scripts\\python eval\\run_eval.py

Results print to the console AND are saved to eval/runs/<timestamp>.json
together with a snapshot of every retrieval knob, so two runs can be
compared long after you've forgotten what you changed.
"""
import os
import sys
import json
from datetime import datetime

# Allow `python eval/run_eval.py` from the repo root (puts the repo on sys.path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.core import config
from app.core.state import state
from app.services.load_llm import load_vector_store, build_bm25_index
from app.services.answer import (
    retrieve_chunks,
    retrieve_bm25,
    reciprocal_rank_fusion,
    rerank_chunks,
)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_SET_PATH = os.path.join(EVAL_DIR, "golden_set.json")
RUNS_DIR = os.path.join(EVAL_DIR, "runs")

STAGES = ["dense", "bm25", "fused", "rerank"]


def setup_state():
    """
    Load models + indexes into the shared RAGState.

    Mirrors the app lifespan in app/main.py, minus the Gemini client —
    retrieval eval never generates, so it needs no API key.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device.upper()}")
    print(f"[*] Loading embedding model: {config.EMBEDDING_MODEL}")
    state.embed_model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    print(f"[*] Loading reranker: {config.RERANKER_MODEL}")
    state.reranker = CrossEncoder(config.RERANKER_MODEL, device=device)

    dimension = state.embed_model.get_embedding_dimension()
    state.vector_index, state.vector_chunks = load_vector_store(dimension)
    state.bm25_index = build_bm25_index(state.vector_chunks)
    print(f"[OK] {state.vector_index.ntotal} vectors, {len(state.vector_chunks)} chunks loaded\n")


def run_stages(question: str) -> dict:
    """Run one question through the pipeline, capturing each stage's output."""
    dense  = retrieve_chunks(question, top_k=config.TOP_K)
    sparse = retrieve_bm25(question, top_k=config.TOP_K)
    fused  = reciprocal_rank_fusion(dense, sparse)
    # Production passes the FULL fused list to the reranker (see chat.py),
    # so the reranker gets the same candidates here. The fused *metric*
    # below is still measured at TOP_K for a fair @k comparison.
    reranked = rerank_chunks(question, fused, top_n=config.RERANK_TOP_N) if fused else []
    return {
        "dense":  dense,
        "bm25":   sparse,
        "fused":  fused[:config.TOP_K],
        "rerank": reranked,
    }


def score_stage(results: list, relevant: list) -> tuple[int, float]:
    """
    Return (hit, reciprocal_rank) for one stage.

    hit = 1 if any retrieved chunk's (source, page_num) matches the golden
    list; reciprocal rank = 1/rank of the FIRST relevant chunk (0 if none).
    """
    relevant_set = {(r["source"], r["page_num"]) for r in relevant}
    for rank, r in enumerate(results, start=1):
        if (r["source"], r.get("page_num")) in relevant_set:
            return 1, 1.0 / rank
    return 0, 0.0


def evaluate(golden_set: list) -> list:
    """Score every question; returns a per-question result list."""
    per_question = []
    for q in golden_set:
        stages = run_stages(q["question"])

        if q["type"] == "negative":
            # A negative question has no relevant chunks — the interesting
            # signal is the top rerank score, i.e. how confidently the
            # pipeline hands WRONG context to the LLM. (Today it always
            # hands over top-3; a future score threshold would change that,
            # and this number tells you where to set it.)
            top = stages["rerank"][0] if stages["rerank"] else None
            per_question.append({
                "id": q["id"],
                "type": "negative",
                "question": q["question"],
                "top_rerank_score": top["rerank_score"] if top else None,
                "top_rerank_source": f"{top['source']} p.{top['page_num']}" if top else None,
            })
            continue

        scores = {}
        for stage in STAGES:
            hit, rr = score_stage(stages[stage], q["relevant"])
            scores[stage] = {"hit": hit, "rr": round(rr, 4)}

        # For diagnosing misses: where did the top-ranked chunk come from?
        top = stages["rerank"][0] if stages["rerank"] else None
        per_question.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "scores": scores,
            "top_rerank_source": f"{top['source']} p.{top['page_num']}" if top else None,
        })
    return per_question


def aggregate(per_question: list) -> dict:
    """Mean hit-rate and MRR per stage, overall and per question type."""
    scored = [r for r in per_question if r["type"] != "negative"]

    def summarise(rows):
        n = len(rows)
        out = {}
        for stage in STAGES:
            hits = sum(r["scores"][stage]["hit"] for r in rows)
            mrr  = sum(r["scores"][stage]["rr"]  for r in rows) / n if n else 0.0
            out[stage] = {"hit_rate": round(hits / n, 4) if n else 0.0,
                          "mrr": round(mrr, 4), "n": n}
        return out

    return {
        "overall":    summarise(scored),
        "keyword":    summarise([r for r in scored if r["type"] == "keyword"]),
        "paraphrase": summarise([r for r in scored if r["type"] == "paraphrase"]),
    }


def print_report(agg: dict, per_question: list):
    cutoffs = {"dense": f"@{config.TOP_K}", "bm25": f"@{config.TOP_K}",
               "fused": f"@{config.TOP_K}", "rerank": f"@{config.RERANK_TOP_N}"}

    print("=" * 62)
    print("RETRIEVAL EVAL")
    print("=" * 62)
    for label in ["overall", "keyword", "paraphrase"]:
        block = agg[label]
        n = block["dense"]["n"]
        print(f"\n{label.upper()}  (n={n})")
        print(f"  {'stage':<12}{'hit-rate':>10}{'MRR':>10}")
        for stage in STAGES:
            s = block[stage]
            print(f"  {stage + cutoffs[stage]:<12}{s['hit_rate']:>10.2f}{s['mrr']:>10.2f}")

    # Questions the LLM would have received WRONG context for
    misses = [r for r in per_question
              if r["type"] != "negative" and r["scores"]["rerank"]["hit"] == 0]
    if misses:
        print(f"\nMISSES AT RERANK STAGE ({len(misses)}) - these reach the LLM with wrong context:")
        for r in misses:
            print(f"  {r['id']} [{r['type']}] {r['question']}")
            print(f"       top reranked chunk came from: {r['top_rerank_source']}")
    else:
        print("\nNo misses at the rerank stage.")

    negatives = [r for r in per_question if r["type"] == "negative"]
    if negatives:
        print("\nNEGATIVE QUESTIONS (answer not in corpus) - top rerank score fed to LLM:")
        for r in negatives:
            score = r["top_rerank_score"]
            score_str = f"{score:.2f}" if score is not None else "n/a"
            print(f"  {r['id']} score={score_str}  from {r['top_rerank_source']}")
            print(f"       {r['question']}")
        print("  (Lower = the reranker itself knows the context is bad. If these scores")
        print("   sit clearly below the scores on real questions, a threshold would let")
        print("   the pipeline say 'nothing relevant found' instead of feeding bad context.)")


def save_run(agg: dict, per_question: list) -> str:
    """Persist results + a full knob snapshot so runs stay comparable."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(RUNS_DIR, f"{stamp}.json")
    payload = {
        "timestamp": stamp,
        "config": {
            "embedding_model":     config.EMBEDDING_MODEL,
            "reranker_model":      config.RERANKER_MODEL,
            "top_k":               config.TOP_K,
            "rerank_top_n":        config.RERANK_TOP_N,
            "rerank_threshold":    config.RERANK_SCORE_THRESHOLD,
            "sentences_per_chunk": config.SENTENCES_PER_CHUNK,
            "sentence_overlap":    config.SENTENCE_OVERLAP,
            "rrf_k":               60,  # hardcoded default in answer.py
            "num_vectors":         int(state.vector_index.ntotal),
            "num_chunks":          len(state.vector_chunks),
        },
        "aggregate": agg,
        "per_question": per_question,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def main():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
    print(f"[*] Loaded {len(golden_set)} golden questions")

    setup_state()
    if not state.vector_chunks:
        print("[ERROR] Vector store is empty — ingest your PDFs first.")
        sys.exit(1)

    per_question = evaluate(golden_set)
    agg = aggregate(per_question)
    print_report(agg, per_question)

    path = save_run(agg, per_question)
    print(f"\n[OK] Run saved to {os.path.relpath(path, os.path.dirname(EVAL_DIR))}")


if __name__ == "__main__":
    main()
