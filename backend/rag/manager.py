"""Centralized, asyncio-safe RAG resource initialization (``RagManager``).

``RagManager`` is a **process-wide singleton** responsible for one-time
initialization of:

    - HuggingFace / OpenAI embedding backend (per ``settings.EMBEDDING_MODEL``)
    - Persisted ChromaDB vector store handle
    - Reconstructed in-memory ``Document`` list for BM25 + hybrid fusion
    - Hybrid and reranking retrievers plus their compiled LCEL chains

**Why singleton + lazy init**:
    Loading sentence-transformers and opening Chroma collections is expensive
    (seconds of CPU + disk I/O). Doing this per HTTP request would dominate
    latency. A single shared instance amortizes that cost across all traffic.

**Why an ``asyncio.Lock`` around initialization**:
    Concurrent first requests must not execute ``load_vectorstore`` twice in
    parallel. Chroma's embedded SQLite backend can throw lock errors or corrupt
    a collection when multiple processes/threads open the same path
    uncontended — the lock serializes the *first* build; afterwards reads are
    cheap.

This module intentionally contains **no** request-scoped state: only shared
read-only retrievers and chains after warmup.
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
    """Immutable bundle of handles produced during RagManager warmup.

    Attributes:
        embedding_model: LangChain-compatible embeddings for the active strategy.
        vectorstore: Loaded Chroma vector store instance.
        docs: Materialized chunk list mirroring Chroma rows (for BM25).
        hybrid_retriever: BM25+vector ensemble retriever.
        reranking_retriever: Hybrid retriever optionally wrapped with Cohere rerank.
        hybrid_chain: LCEL RAG chain bound to ``hybrid_retriever``.
        reranking_chain: LCEL RAG chain bound to ``reranking_retriever``.
    """

    embedding_model: Embeddings
    vectorstore: object
    docs: List[Document]
    hybrid_retriever: BaseRetriever
    reranking_retriever: BaseRetriever
    hybrid_chain: object
    reranking_chain: object


class RagManager:
    """Lazy-initialized singleton for embedding + vector + retriever warmup.

    The class method ``instance()`` returns a shared ``RagManager`` so FastAPI
    handlers, LangGraph nodes, and offline eval scripts all hit the same Chroma
    snapshot without duplicating heavy native libraries in memory.
    """

    _instance: Optional["RagManager"] = None
    _lock = asyncio.Lock()

    def __init__(self) -> None:
        """Create an empty manager shell; resources populate on first ``get_resources``."""
        self._resources: Optional[RagResources] = None

    @classmethod
    def instance(cls) -> "RagManager":
        """Return the process-global ``RagManager`` instance.

        Returns:
            RagManager: Shared singleton.

        Raises:
            None
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def get_resources(self) -> RagResources:
        """Return warmed-up ``RagResources``, initializing once under a lock.

        Returns:
            RagResources: Cached retrievers and chains.

        Raises:
            FileNotFoundError: When the vector store directory is missing
                (surfaced to callers so the API can instruct operators to ingest).
            Exception: Propagates embedding or Chroma failures during first init.

        Why double-checked locking:
            The outer ``if self._resources`` avoids acquiring the lock on every
            call after warmup; the inner check prevents duplicate initialization
            when two coroutines raced on first access.
        """
        if self._resources is not None:
            return self._resources

        async with self._lock:
            if self._resources is not None:
                return self._resources

            logger.info("RagManager initializing embedding model + vectorstore (one-time).")
            embedding_model = get_embedding_model(settings.EMBEDDING_MODEL)
            vectorstore = load_vectorstore(embedding_model)

            # BM25 needs the same logical chunks Chroma indexed — rebuild Documents
            # from the collection rather than re-parsing PDFs from disk.
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

