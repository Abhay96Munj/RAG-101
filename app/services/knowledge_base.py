"""
Knowledge base statistics — shared by the Documents endpoint and the
agent tool `list_knowledge_base`. Computes per-document stats from
state.vector_chunks; the callers decide how to present them (JSON vs
human-readable string).
"""
from app.core.state import state


def get_document_stats() -> list[dict]:
    """
    Return per-document stats for every distinct source file in the store:
        [{"filename": str, "chunk_count": int, "pages": int}, ...]
    sorted by filename. `pages` counts distinct page numbers — older chunks
    in the store may lack page_num, so those are simply not counted.
    """
    # Copy under the lock so a concurrent ingest can't mutate the list
    # while we iterate over it.
    with state.lock:
        chunks = list(state.vector_chunks)

    docs: dict[str, dict] = {}
    for chunk in chunks:
        src = chunk.get("source", "unknown")
        entry = docs.setdefault(src, {"filename": src, "chunk_count": 0, "page_nums": set()})
        entry["chunk_count"] += 1
        page_num = chunk.get("page_num")
        if page_num is not None:
            entry["page_nums"].add(page_num)

    return [
        {
            "filename":    docs[src]["filename"],
            "chunk_count": docs[src]["chunk_count"],
            "pages":       len(docs[src]["page_nums"]),
        }
        for src in sorted(docs)
    ]
