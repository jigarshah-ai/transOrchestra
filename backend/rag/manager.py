"""Centralized RAG resource initialization.

`RagManager` is a process-wide singleton responsible for one-time initialization of:
  - Embedding model
  - ChromaDB vectorstore
  - Reconstructed Document corpus (for BM25)
  - Hybrid + optional reranking retrievers
  - RAG chains

This prevents reloading ChromaDB and embedding models on every request.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from backend.config import settings
from backend.rag.embeddings import get_embedding_model
from backend.rag.pipeline import build_rag_chain
from backend.rag.retriever import build_hybrid_retriever, build_reranking_retriever
from backend.rag.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagResources:
    embedding_model: Embeddings
    vectorstore: object
    docs: List[Document]
    hybrid_retriever: BaseRetriever
    reranking_retriever: BaseRetriever
    hybrid_chain: object
    reranking_chain: object


class RagManager:
    """Lazy-initialized singleton to hold RAG resources."""

    _instance: Optional["RagManager"] = None
    _lock = asyncio.Lock()

    def __init__(self) -> None:
        self._resources: Optional[RagResources] = None

    @classmethod
    def instance(cls) -> "RagManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def get_resources(self) -> RagResources:
        """Return initialized resources; initialize once if needed."""
        if self._resources is not None:
            return self._resources

        async with self._lock:
            if self._resources is not None:
                return self._resources

            logger.info("RagManager initializing embedding model + vectorstore (one-time).")
            embedding_model = get_embedding_model(settings.EMBEDDING_MODEL)
            vectorstore = load_vectorstore(embedding_model)

            # Reconstruct the document corpus for BM25 from ChromaDB.
            raw = vectorstore._collection.get(include=["documents", "metadatas"])
            docs: List[Document] = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(raw["documents"], raw["metadatas"])
            ]

            hybrid = build_hybrid_retriever(vectorstore, docs)
            reranking = build_reranking_retriever(hybrid, docs)

            hybrid_chain = build_rag_chain(hybrid)
            reranking_chain = build_rag_chain(reranking)

            self._resources = RagResources(
                embedding_model=embedding_model,
                vectorstore=vectorstore,
                docs=docs,
                hybrid_retriever=hybrid,
                reranking_retriever=reranking,
                hybrid_chain=hybrid_chain,
                reranking_chain=reranking_chain,
            )
            logger.info("RagManager initialized successfully.")
            return self._resources

