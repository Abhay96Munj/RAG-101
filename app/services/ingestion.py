from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import re
import numpy as np
import json
import faiss
from app.core.config import SENTENCE_OVERLAP, SENTENCES_PER_CHUNK, VECTOR_STORE_DIR
from app.services.load_llm import vector_index, vector_chunks, embed_model, build_bm25_index
import app.services.load_llm as llm_state
from pypdf import PdfReader

# EXTRACT TEXT FROM PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    """Read a PDF file and return all its text as a single string."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


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
    
    # Save to disk
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "index.faiss"))
    with open(os.path.join(VECTOR_STORE_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"  [OK] Saved FAISS index ({index.ntotal} vectors, {dimension}D)")
    print(f"  [OK] Saved chunk metadata to chunks.json")


def process_new_pdf(pdf_path: str, filename: str):
    global vector_chunks, vector_index
    
    # 1. Extract text
    text = extract_text_from_pdf(pdf_path)
    
    # 2. Chunk text
    new_raw_chunks = chunk_text(text)
    if not new_raw_chunks:
        return 0
    
    # 3. Create metadata objects
    new_chunks = []
    for i, chunk in enumerate(new_raw_chunks):
        new_chunks.append({
            "text": chunk,
            "source": filename,
            "page_num": filename.page,
            "chunk_index": i
        })
        
    # 4. Embed ONLY the new chunks
    new_texts = [c["text"] for c in new_chunks]
    print(f"[*] Embedding {len(new_texts)} new chunks...")
    new_embeddings = embed_model.encode(new_texts, show_progress_bar=True, normalize_embeddings=True)
    new_embeddings = np.array(new_embeddings, dtype="float32")
    
    # 5. Add new embeddings directly to our active FAISS index
    vector_index.add(new_embeddings)
    
    # 6. Append new chunks to our active chunk metadata list
    vector_chunks.extend(new_chunks)
    
    # 7. Rebuild BM25 index so keyword search stays in sync
    llm_state.bm25_index = build_bm25_index(vector_chunks)
    
    # 8. Persist updated index and metadata list to disk
    index_path = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    chunks_path = os.path.join(VECTOR_STORE_DIR, "chunks.json")
    
    faiss.write_index(vector_index, index_path)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(vector_chunks, f, indent=2, ensure_ascii=False)
        
    print(f"[OK] Added {len(new_chunks)} chunks to FAISS, BM25 rebuilt, changes saved.")
    return len(new_chunks)