from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import chromadb

from config import get_settings

settings = get_settings()


class VectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self._vectorstore: Optional[Chroma] = None
        self._client: Optional[chromadb.PersistentClient] = None
        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and vector store."""
        # ✅ FIX: Use PersistentClient (chromadb >= 0.4+). The old
        # chromadb.Settings(chroma_db_impl="duckdb+parquet", ...) API was
        # removed and causes an immediate TypeError on startup.
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

        self._vectorstore = Chroma(
            client=self._client,
            collection_name=settings.collection_name,
            embedding_function=self.embeddings,
        )

    def add_documents(self, chunks: List[Document]) -> List[str]:
        """Add document chunks to vector store."""
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        self._vectorstore.add_documents(documents=chunks, ids=ids)
        return ids

    def similarity_search(
        self, query: str, k: int = None, doc_id: Optional[str] = None
    ) -> List[Document]:
        """Search for similar documents."""
        k = k or settings.max_retrieved_docs

        if doc_id:
            results = self._vectorstore.similarity_search(
                query, k=k, filter={"doc_id": doc_id}
            )
        else:
            results = self._vectorstore.similarity_search(query, k=k)

        return results

    def similarity_search_with_score(
        self, query: str, k: int = None, doc_id: Optional[str] = None
    ) -> List[tuple]:
        """Search with relevance scores."""
        k = k or settings.max_retrieved_docs

        if doc_id:
            results = self._vectorstore.similarity_search_with_score(
                query, k=k, filter={"doc_id": doc_id}
            )
        else:
            results = self._vectorstore.similarity_search_with_score(query, k=k)

        return results

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            # ✅ FIX: Use get_or_create_collection to avoid exception on empty store
            collection = self._client.get_or_create_collection(settings.collection_name)
            results = collection.get(where={"doc_id": doc_id})
            if results["ids"]:
                collection.delete(ids=results["ids"])
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get metadata for all unique documents."""
        try:
            # ✅ FIX: Use get_or_create_collection so this never raises on a
            # fresh / empty store (get_collection raises if it doesn't exist).
            collection = self._client.get_or_create_collection(settings.collection_name)
            results = collection.get()

            docs: Dict[str, Dict] = {}
            for metadata in results.get("metadatas", []):
                if metadata and "doc_id" in metadata:
                    doc_id = metadata["doc_id"]
                    if doc_id not in docs:
                        docs[doc_id] = {
                            "doc_id": doc_id,
                            "filename": metadata.get("filename", "Unknown"),
                            "chunk_count": 0,
                        }
                    docs[doc_id]["chunk_count"] += 1

            return list(docs.values())
        except Exception:
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            # ✅ FIX: Same — get_or_create_collection prevents crash on cold start
            collection = self._client.get_or_create_collection(settings.collection_name)
            count = collection.count()
            docs = self.get_all_documents()
            return {
                "total_chunks": count,
                "total_documents": len(docs),
            }
        except Exception:
            return {"total_chunks": 0, "total_documents": 0}