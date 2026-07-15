"""
RAG Service  —  business logic that will be imported by the routers.
Placeholder stubs — fill in when integrating rag/chat.py.
"""
import os
import sys
import json
import faiss
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.core.config import EMBEDDING_MODEL, RERANKER_MODEL, GEMINI_MODEL, VECTOR_STORE_DIR, GEMINI_API_KEY, TOP_K, RERANK_TOP_N, SYSTEM_PROMPT, SENTENCE_OVERLAP, SENTENCES_PER_CHUNK, PDFS_DIR

import re
import numpy as np
from pypdf import PdfReader

load_dotenv()

embed_model = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)
gemini_key = GEMINI_API_KEY
TOP_K = TOP_K
RERANK_TOP_N = RERANK_TOP_N
system_prompt = SYSTEM_PROMPT

# LOAD THE LLM FOR TEXT GENERATION
def load_llm():
    api_key = gemini_key
    if not api_key:
        print("[ERROR] GEMINI_API_KEY key not found in .env file.")
        sys.exit(1)
    client = genai.Client(api_key=api_key)
    print(f"[OK] Gemini model ready: {GEMINI_MODEL}")
    return client

# LOAD THE VECTOR DB
def load_vector_store():
    index_path = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    chunks_path = os.path.join(VECTOR_STORE_DIR, "chunks.json")
    # 1. If files don't exist, build an empty index & list
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        print("[INFO] Vector store files not found. Initializing empty vector store.")
        # BAAI/bge-small-en-v1.5 has a dimension size of 384
        dimension = 384 
        index = faiss.IndexFlatIP(dimension)
        chunks = []
        return index, chunks
    # 2. Try loading the files safely
    try:
        index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            chunks = json.loads(content) if content else []
    except Exception as e:
        print(f"[WARNING] Failed to load index, initializing empty: {e}")
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        chunks = []
    return index, chunks

# Call the helper functions
llm_client = load_llm()
vector_index, vector_chunks = load_vector_store()

# RETRIEVE RELEVANT CHUNKS FROM FAISS
def retrieve_chunks(query: str, model=embed_model, index=vector_index, chunks=vector_chunks, top_k=TOP_K):
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    query_embedding = model.encode([prefixed],normalize_embeddings=True).astype("float32")
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


# RANK THE CHUNKS
def rerank_chunks(query: str, candidates: list, top_n=RERANK_TOP_N):
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for cand, score in zip(candidates, scores):
        cand["rerank_score"] = float(score)

    # Sort by rerank score descending, keep top_n
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


# CALL LLM TO GENERATE THE ANSWER
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
            "chunk_index": i
        })
        
    # 4. Embed ONLY the new chunks
    new_texts = [c["text"] for c in new_chunks]
    print(f"  [*] Embedding {len(new_texts)} new chunks...")
    new_embeddings = embed_model.encode(new_texts, show_progress_bar=True, normalize_embeddings=True)
    new_embeddings = np.array(new_embeddings, dtype="float32")
    
    # 5. Add new embeddings directly to our active FAISS index
    vector_index.add(new_embeddings)
    
    # 6. Append new chunks to our active chunk metadata list
    vector_chunks.extend(new_chunks)
    
    # 7. Persist updated index and metadata list to disk
    index_path = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    chunks_path = os.path.join(VECTOR_STORE_DIR, "chunks.json")
    
    faiss.write_index(vector_index, index_path)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(vector_chunks, f, indent=2, ensure_ascii=False)
        
    print(f"  [OK] Added {len(new_chunks)} chunks to FAISS and saved changes.")
    return len(new_chunks)