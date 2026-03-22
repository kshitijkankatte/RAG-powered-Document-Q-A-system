import os
import uuid
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

from config import get_settings

settings = get_settings()


class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_document(self, file_path: str, filename: str) -> List[Document]:
        """Load document based on file extension."""
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return loader.load()

    def process_document(
        self, file_path: str, filename: str, doc_id: str
    ) -> List[Document]:
        """Load and split document into chunks."""
        documents = self.load_document(file_path, filename)

        # Add metadata
        for doc in documents:
            doc.metadata.update({
                "doc_id": doc_id,
                "filename": filename,
                "source": filename,
            })

        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)

        # Add chunk IDs
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{i}"
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def get_document_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """Get statistics about processed document."""
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "avg_chunk_size": total_chars // len(chunks) if chunks else 0,
        }
