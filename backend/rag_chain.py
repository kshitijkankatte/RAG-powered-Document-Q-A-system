from typing import List, Dict, Any, Optional, AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import get_settings
from vector_store import VectorStore

settings = get_settings()

SYSTEM_PROMPT = """You are an expert document analyst and Q&A assistant. Your job is to answer questions accurately based on the provided document context.

Guidelines:
- Answer ONLY based on the provided context. Do not use outside knowledge.
- If the context doesn't contain enough information to answer, clearly state: "The provided documents don't contain sufficient information to answer this question."
- Be precise, concise, and cite relevant parts of the context when helpful.
- If asked for a summary, provide a structured, comprehensive overview.
- Format your responses clearly using markdown when appropriate.

Context from documents:
{context}
"""

HUMAN_PROMPT = "{question}"


class RAGChain:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.1,
            streaming=True,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self.output_parser = StrOutputParser()

    def _format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into context string."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f", Page {page + 1}" if page != "" else ""
            formatted.append(
                f"[Source {i}: {source}{page_info}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)

    async def query(
        self,
        question: str,
        doc_id: Optional[str] = None,
        k: int = None,
    ) -> Dict[str, Any]:
        """Query the RAG system and return answer with sources."""
        # Retrieve relevant chunks
        results_with_scores = self.vector_store.similarity_search_with_score(
            question, k=k, doc_id=doc_id
        )

        if not results_with_scores:
            return {
                "answer": "No relevant documents found. Please upload documents first.",
                "sources": [],
                "retrieved_chunks": 0,
            }

        docs = [doc for doc, _ in results_with_scores]
        scores = [float(score) for _, score in results_with_scores]

        context = self._format_context(docs)

        # Build and run chain
        chain = self.prompt | self.llm | self.output_parser
        answer = await chain.ainvoke({
            "context": context,
            "question": question,
        })

        # Build sources list
        sources = []
        seen = set()
        for doc, score in zip(docs, scores):
            filename = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page", None)
            key = f"{filename}_{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "filename": filename,
                    "page": page + 1 if page is not None else None,
                    "relevance_score": round(1 - score, 4),  # Convert distance to similarity
                    "excerpt": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                })

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(docs),
        }

    async def stream_query(
        self,
        question: str,
        doc_id: Optional[str] = None,
        k: int = None,
    ) -> AsyncGenerator[str, None]:
        """Stream the answer token by token."""
        results_with_scores = self.vector_store.similarity_search_with_score(
            question, k=k, doc_id=doc_id
        )

        if not results_with_scores:
            yield "No relevant documents found. Please upload documents first."
            return

        docs = [doc for doc, _ in results_with_scores]
        context = self._format_context(docs)

        chain = self.prompt | self.llm | self.output_parser
        async for chunk in chain.astream({
            "context": context,
            "question": question,
        }):
            yield chunk
