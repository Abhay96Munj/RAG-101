"""
Ingest Router  —  POST /api/v1/ingest/upload
"""

import asyncio
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ingestion import process_new_pdf
from app.core.config import PDFS_DIR
router = APIRouter()


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    # basename() strips any directory parts a malicious client could send
    # (e.g. "..\\..\\evil.pdf"), so we can never write outside PDFS_DIR.
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    os.makedirs(PDFS_DIR, exist_ok=True)
    file_path = os.path.join(PDFS_DIR, filename)

    content = await file.read()
    with open(file_path, "wb") as buffer:
        buffer.write(content)

    # Embedding is CPU/GPU heavy — run in a thread so the event loop
    # (and every other request) isn't frozen for the duration.
    num_chunks = await asyncio.to_thread(process_new_pdf, file_path, filename)

    return {
        "message": "File Uploaded and Index Updated",
        "filename": filename,
        "chunks_added": num_chunks
    }
