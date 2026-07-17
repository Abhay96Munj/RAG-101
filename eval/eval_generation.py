"""
Generation Evaluation Harness (phase 2 — costs Gemini API calls)
=================================================================
Where run_eval.py stops at retrieval, this runs the FULL pipeline —
retrieve, rerank, build context, generate — and grades the final answer:

  1. keyword containment  (non-negative questions, free, deterministic)
       Does the answer contain any of the golden answer_keywords?
  2. refusal check        (negative questions, free, deterministic)
       The system prompt says to answer "I don't have enough information"
       when the context lacks the answer. Did the model actually refuse?
  3. faithfulness         (LLM-as-judge, one extra Gemini call per answer)
       Is every claim in the answer supported by the retrieved context?
       This is what frameworks like RAGAS automate — hand-rolled here so
       you can see exactly what such a judge does.

Each question costs 2 Gemini calls (answer + judge). 26 questions = 52
calls. The free tier allows 15 requests/minute, so the default --delay of
5s stays just under it; questions that still hit a 429 are retried after
a cooldown instead of being dropped.

Usage (from the repo root):
    .venv\\Scripts\\python eval\\eval_generation.py                 # full run
    .venv\\Scripts\\python eval\\eval_generation.py --limit 3      # first 3 only
    .venv\\Scripts\\python eval\\eval_generation.py --delay 6      # 6s between API calls

Results save to eval/runs/gen_<timestamp>.json.
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.core import config
from app.core.state import state
from app.services.load_llm import load_llm, load_vector_store, build_bm25_index
from app.services.answer import hybrid_retrieve, rerank_chunks, generate_answer

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_SET_PATH = os.path.join(EVAL_DIR, "golden_set.json")
RUNS_DIR = os.path.join(EVAL_DIR, "runs")

# Phrases that count as the model refusing to answer (per SYSTEM_PROMPT).
# Matched case-insensitively after normalising curly apostrophes.
REFUSAL_MARKERS = [
    "don't have enough information",
    "do not have enough information",
]

JUDGE_PROMPT = """You are grading a RAG system's answer for FAITHFULNESS.

Context that was given to the system:
---
{context}
---

Question: {question}
Answer given by the system: {answer}

Is every factual claim in the answer fully supported by the context above?
Ignore style; judge only whether the claims are grounded in the context.
Reply with exactly one word: YES or NO."""


def setup_state():
    """Full app lifespan equivalent — this one DOES need the Gemini client."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device.upper()}")
    print(f"[*] Loading embedding model: {config.EMBEDDING_MODEL}")
    state.embed_model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    print(f"[*] Loading reranker: {config.RERANKER_MODEL}")
    state.reranker = CrossEncoder(config.RERANKER_MODEL, device=device)
    state.llm_client = load_llm()
    dimension = state.embed_model.get_embedding_dimension()
    state.vector_index, state.vector_chunks = load_vector_store(dimension)
    state.bm25_index = build_bm25_index(state.vector_chunks)
    print(f"[OK] {state.vector_index.ntotal} vectors, {len(state.vector_chunks)} chunks loaded\n")


def normalise(text: str) -> str:
    return text.replace("’", "'").lower()


def is_refusal(answer: str) -> bool:
    norm = normalise(answer)
    return any(marker in norm for marker in REFUSAL_MARKERS)


def contains_keyword(answer: str, keywords: list[str]) -> bool:
    norm = normalise(answer)
    return any(normalise(kw) in norm for kw in keywords)


def judge_faithfulness(context: str, question: str, answer: str) -> bool:
    """One extra Gemini call: YES -> faithful, anything else -> not."""
    prompt = JUDGE_PROMPT.format(context=context, question=question, answer=answer)
    response = state.llm_client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
    )
    verdict = (response.text or "").strip().upper()
    return verdict.startswith("YES")


def evaluate_question(q: dict, delay: float) -> dict:
    """Full pipeline + grading for one golden question."""
    # Same flow as POST /api/v1/chat/query (chat.py), including the exact
    # context format, so this grades what production actually does.
    candidates = hybrid_retrieve(q["question"], top_k=config.TOP_K)
    reranked = rerank_chunks(q["question"], candidates, top_n=config.RERANK_TOP_N) if candidates else []
    context = "\n\n---\n\n".join([r["text"] for r in reranked])

    if not reranked:
        # Mirrors chat.py: below-threshold candidates -> refuse without
        # calling Gemini. Graded exactly like a model-generated refusal.
        answer = ("I don't have enough information — nothing relevant to "
                  "this question was found in the uploaded documents.")
    else:
        time.sleep(delay)
        answer = generate_answer(context, q["question"])

    result = {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "answer": answer,
        "refused": is_refusal(answer),
    }

    if q["type"] == "negative":
        # Grade: did the model correctly say "I don't know"?
        # A refusal is trivially faithful, so no judge call is needed.
        result["correct_refusal"] = result["refused"]
        if not result["refused"]:
            time.sleep(delay)
            result["faithful"] = judge_faithfulness(context, q["question"], answer)
        return result

    result["keyword_hit"] = contains_keyword(answer, q["answer_keywords"])
    if result["refused"]:
        # Refused despite the corpus holding the answer — retrieval or
        # prompt problem, and there is no claim for the judge to grade.
        result["faithful"] = None
    else:
        time.sleep(delay)
        result["faithful"] = judge_faithfulness(context, q["question"], answer)
    return result


def aggregate(results: list) -> dict:
    positives = [r for r in results if r["type"] != "negative" and "error" not in r]
    negatives = [r for r in results if r["type"] == "negative" and "error" not in r]
    judged    = [r for r in positives if r.get("faithful") is not None]
    errors    = [r for r in results if "error" in r]

    def rate(hits, total):
        return round(hits / total, 4) if total else None

    return {
        "n_positive":            len(positives),
        "n_negative":            len(negatives),
        "n_errors":              len(errors),
        "keyword_hit_rate":      rate(sum(r["keyword_hit"] for r in positives), len(positives)),
        "false_refusal_rate":    rate(sum(r["refused"] for r in positives), len(positives)),
        "faithfulness_rate":     rate(sum(r["faithful"] for r in judged), len(judged)),
        "n_judged":              len(judged),
        "correct_refusal_rate":  rate(sum(r["correct_refusal"] for r in negatives), len(negatives)),
    }


def print_report(agg: dict, results: list):
    print("=" * 62)
    print("GENERATION EVAL")
    print("=" * 62)
    print(f"  keyword hit rate:      {_fmt(agg['keyword_hit_rate'])}   (n={agg['n_positive']})")
    print(f"  faithfulness rate:     {_fmt(agg['faithfulness_rate'])}   (n={agg['n_judged']} judged)")
    print(f"  false refusal rate:    {_fmt(agg['false_refusal_rate'])}   (answer WAS in corpus)")
    print(f"  correct refusal rate:  {_fmt(agg['correct_refusal_rate'])}   (n={agg['n_negative']} negatives)")
    if agg["n_errors"]:
        print(f"  errors:                {agg['n_errors']} question(s) failed - see JSON")

    problems = [r for r in results if "error" not in r and (
        (r["type"] != "negative" and (not r["keyword_hit"] or r["refused"] or r.get("faithful") is False))
        or (r["type"] == "negative" and not r["correct_refusal"])
    )]
    if problems:
        print(f"\nPROBLEM ANSWERS ({len(problems)}):")
        for r in problems:
            flags = []
            if r["type"] == "negative":
                flags.append("answered instead of refusing")
            else:
                if not r["keyword_hit"]:
                    flags.append("missing expected keyword")
                if r["refused"]:
                    flags.append("false refusal")
                if r.get("faithful") is False:
                    flags.append("judge says unfaithful")
            print(f"  {r['id']} [{r['type']}] {', '.join(flags)}")
            print(f"       Q: {r['question']}")
            print(f"       A: {r['answer'][:160]}")
    else:
        print("\nNo problem answers.")


def _fmt(v) -> str:
    return f"{v:.2f}" if v is not None else " n/a"


def save_run(agg: dict, results: list) -> str:
    os.makedirs(RUNS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(RUNS_DIR, f"gen_{stamp}.json")
    payload = {
        "timestamp": stamp,
        "config": {
            "embedding_model": config.EMBEDDING_MODEL,
            "reranker_model":  config.RERANKER_MODEL,
            "gemini_model":    config.GEMINI_MODEL,
            "top_k":           config.TOP_K,
            "rerank_top_n":    config.RERANK_TOP_N,
            "rerank_threshold": config.RERANK_SCORE_THRESHOLD,
            "system_prompt":   config.SYSTEM_PROMPT,
            "num_chunks":      len(state.vector_chunks),
        },
        "aggregate": agg,
        "per_question": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generation eval (costs Gemini calls)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only run the first N questions (cheap smoke test)")
    parser.add_argument("--delay", type=float, default=5.0,
                        help="seconds to sleep before each Gemini call (free tier: 15/min)")
    args = parser.parse_args()

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
    if args.limit:
        golden_set = golden_set[:args.limit]
    print(f"[*] Running generation eval on {len(golden_set)} questions "
          f"(~{len(golden_set) * 2} Gemini calls, delay={args.delay}s)")

    setup_state()
    if not state.vector_chunks:
        print("[ERROR] Vector store is empty - ingest your PDFs first.")
        sys.exit(1)

    results = []
    for i, q in enumerate(golden_set, start=1):
        print(f"[{i}/{len(golden_set)}] {q['id']} ...", flush=True)
        for attempt in range(3):
            try:
                results.append(evaluate_question(q, args.delay))
                break
            except Exception as e:
                # Rate limit: the per-minute quota resets, so wait it out
                # and retry rather than dropping the question.
                if "RESOURCE_EXHAUSTED" in str(e) and attempt < 2:
                    print(f"    [WARN] rate limited, cooling down 60s "
                          f"(retry {attempt + 1}/2)", flush=True)
                    time.sleep(60)
                    continue
                # Anything else (or third strike): record and continue -
                # one bad question shouldn't kill the run.
                print(f"    [WARN] failed: {e}")
                results.append({"id": q["id"], "type": q["type"],
                                "question": q["question"], "error": str(e)})
                break

    agg = aggregate(results)
    print()
    print_report(agg, results)

    path = save_run(agg, results)
    print(f"\n[OK] Run saved to {os.path.relpath(path, os.path.dirname(EVAL_DIR))}")


if __name__ == "__main__":
    main()
