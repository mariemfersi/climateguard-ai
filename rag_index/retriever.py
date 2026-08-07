from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from agents.llm_client import get_llm
from config.settings import get_settings


@dataclass
class RetrievedChunk:
    text: str
    source: str
    title: str
    score: float
    chunk_index: int


class RAGRetriever:
    def __init__(self):
        self.settings = get_settings()
        self.llm = get_llm()
        self._client = chromadb.PersistentClient(
            path=self.settings.rag_persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.settings.rag_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        k = top_k or self.settings.rag_top_k
        results = self._collection.query(
            query_embeddings=[self.llm.embed_one(query)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        if not results["documents"] or not results["documents"][0]:
            return chunks
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            chunks.append(RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                title=meta.get("title", "unknown"),
                score=1.0 - float(dist),
                chunk_index=int(meta.get("chunk_index", 0)),
            ))
        return chunks

    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant documents retrieved."
        return "\n\n".join(
            f"[{i}] (source: {c.source} | title: {c.title} | score: {c.score:.3f})\n{c.text}"
            for i, c in enumerate(chunks, 1)
        )


_retriever = None
def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever