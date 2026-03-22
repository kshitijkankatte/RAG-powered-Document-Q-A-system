import os
import uuid
import tempfile
import asyncio
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import aiofiles

from config import get_settings
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_chain import RAGChain

settings = get_settings()

# Global instances
vector_store: Optional[VectorStore] = None
rag_chain: Optional[RAGChain] = None
doc_processor: Optional[DocumentProcessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, rag_chain, doc_processor
    vector_store = VectorStore()
    rag_chain = RAGChain(vector_store)
    doc_processor = DocumentProcessor()
    print("✅ RAG system initialized")
    yield
    print("🔴 Shutting down RAG system")


app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload documents and ask questions powered by GPT-4o-mini + ChromaDB",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None
    k: Optional[int] = None
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    retrieved_chunks: int


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    total_chunks: int
    total_characters: int
    avg_chunk_size: int
    message: str


class StatsResponse(BaseModel):
    total_chunks: int
    total_documents: int


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "RAG Document Q&A API is running 🚀"}


@app.get("/health", tags=["Health"])
async def health():
    stats = vector_store.get_collection_stats()
    return {
        "status": "healthy",
        "vector_store": stats,
        "models": {
            "llm": settings.llm_model,
            "embedding": settings.embedding_model,
        },
    }


@app.post("/documents/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document (PDF, TXT, DOCX)."""
    allowed_types = {".pdf", ".txt", ".docx", ".doc"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_types)}",
        )

    doc_id = str(uuid.uuid4())

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = doc_processor.process_document(tmp_path, file.filename, doc_id)
        stats = doc_processor.get_document_stats(chunks)
        vector_store.add_documents(chunks)

        return UploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            total_chunks=stats["total_chunks"],
            total_characters=stats["total_characters"],
            avg_chunk_size=stats["avg_chunk_size"],
            message=f"✅ Successfully indexed '{file.filename}' into {stats['total_chunks']} chunks.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all indexed documents."""
    return vector_store.get_all_documents()


@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a document and its chunks from the vector store."""
    success = vector_store.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete document.")
    return {"message": f"Document {doc_id} deleted successfully."}


@app.post("/query", response_model=QueryResponse, tags=["QA"])
async def query_documents(request: QueryRequest):
    """Query documents using RAG (non-streaming)."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = await rag_chain.query(
        question=request.question,
        doc_id=request.doc_id,
        k=request.k,
    )
    return QueryResponse(**result)


@app.post("/query/stream", tags=["QA"])
async def stream_query(request: QueryRequest):
    """Query documents with streaming response."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async def event_generator():
        async for token in rag_chain.stream_query(
            question=request.question,
            doc_id=request.doc_id,
            k=request.k,
        ):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    """Get vector store statistics."""
    return vector_store.get_collection_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
