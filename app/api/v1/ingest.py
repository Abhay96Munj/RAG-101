"""
Ingest Router  —  POST /api/v1/ingest/upload
"""

import os
from fastapi import APIRouter, UploadFile, File
from app.services.rag_service import process_new_pdf
from app.core.config import PDFS_DIR
router = APIRouter()


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs(PDFS_DIR, exist_ok=True)

    file_path = os.path.join(PDFS_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    num_chunks = process_new_pdf(file_path, file.filename)
    
    return {
        "message": "File Uploaded and Index Updated", 
        "filename": file.filename, 
        "chunks_added": num_chunks
    }
