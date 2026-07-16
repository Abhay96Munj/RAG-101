from sentence_transformers import SentenceTransformer
import os
import re
import numpy as np
import json
import faiss
from app.core.config import SENTENCE_OVERLAP, SENTENCES_PER_CHUNK, VECTOR_STORE_DIR
from app.core.state import state
from app.services.load_llm import build_bm25_index
from pypdf import PdfReader


def _persist_vector_store(index, chunks: list[dict]):
    """
    Save the FAISS index + chunk metadata to disk ATOMICALLY.

    Write each file to a .tmp sibling first, then os.replace() it over the
    real file. os.replace is atomic, so a crash mid-write can never leave a
    half-written index.faiss or chunks.json on disk. (A crash between the
    two replace calls can still leave them one step apart — a real system
    would use a single transactional store — but the window shrinks from
    "the whole write" to "two syscalls".)
    """
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    index_path  = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    chunks_path = os.path.join(VECTOR_STORE_DIR, "chunks.json")

    faiss.write_index(index, index_path + ".tmp")
    with open(chunks_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    os.replace(index_path + ".tmp", index_path)
    os.replace(chunks_path + ".tmp", chunks_path)

# EXTRACT TEXT FROM PDF
def extract_pages_from_pdf(pdf_path: str) -> list[str]:
    """
    Read a PDF file and return the text of each page as a list.

    Keeping pages separate (instead of one big string) lets each chunk
    remember which page it came from — needed for citations like
    "see page 12 of report.pdf".
    """
    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]


def chunk_text(
    text: str,
    sentences_per_chunk: int = SENTENCES_PER_CHUNK,
    overlap: int = SENTENCE_OVERLAP
) -> list[str]:
    """
    Split text into chunks of N sentences with a sentence-level overlap.

    Why sentences instead of characters?
    - Character chunks cut mid-sentence, breaking meaning.
    - Sentence chunks keep each semantic unit intact, giving the retriever
      cleaner, more accurate context to match against a query.

    Steps:
      1. Use regex to split on sentence-ending punctuation (. ! ?)
      2. Group sentences into windows of `sentences_per_chunk`
      3. Slide forward by (sentences_per_chunk - overlap) each step
    """
    # Split on sentence boundaries, keeping the delimiter
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw if s.strip()]

    chunks = []
    step = max(1, sentences_per_chunk - overlap)
    for i in range(0, len(sentences), step):
        window = sentences[i : i + sentences_per_chunk]
        chunk = " ".join(window).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# ADD TO EXISTING DB
def build_vector_store(all_chunks: list[dict], model: SentenceTransformer):
    """
    Embed all chunks and save:
      - FAISS index (vector_store/index.faiss)
      - Chunk metadata (vector_store/chunks.json)
    """
    # Extract just the text for embedding
    texts = [chunk["text"] for chunk in all_chunks]
    
    print(f"  [*] Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    _persist_vector_store(index, all_chunks)

    print(f"  [OK] Saved FAISS index ({index.ntotal} vectors, {dimension}D)")
    print(f"  [OK] Saved chunk metadata to chunks.json")


def process_new_pdf(pdf_path: str, filename: str):
    # 1. Extract text page by page, then chunk each page separately
    #    so every chunk keeps a real page number.
    pages = extract_pages_from_pdf(pdf_path)

    new_chunks = []
    chunk_index = 0   # unique per-document index across all pages
    for page_num, page_text in enumerate(pages, start=1):
        for chunk in chunk_text(page_text):
            new_chunks.append({
                "text": chunk,
                "source": filename,
                "page_num": page_num,
                "chunk_index": chunk_index
            })
            chunk_index += 1

    if not new_chunks:
        return 0

    # 2. Embed ONLY the new chunks — done OUTSIDE the lock because it's the
    #    slow part (seconds) and touches no shared state.
    new_texts = [c["text"] for c in new_chunks]
    print(f"[*] Embedding {len(new_texts)} new chunks...")
    new_embeddings = state.embed_model.encode(new_texts, show_progress_bar=True, normalize_embeddings=True)
    new_embeddings = np.array(new_embeddings, dtype="float32")

    # 3. Mutate all three shared structures under the lock so a concurrent
    #    query can never see the FAISS index and chunk list out of sync.
    with state.lock:
        state.vector_index.add(new_embeddings)
        state.vector_chunks.extend(new_chunks)
        state.bm25_index = build_bm25_index(state.vector_chunks)

        # 4. Persist while still holding the lock — snapshotting a stable,
        #    consistent view of index + chunks to disk (atomic writes).
        _persist_vector_store(state.vector_index, state.vector_chunks)

    print(f"[OK] Added {len(new_chunks)} chunks to FAISS, BM25 rebuilt, changes saved.")
    return len(new_chunks)