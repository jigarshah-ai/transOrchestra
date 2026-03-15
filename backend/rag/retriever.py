"""Retrieval strategy builders for TransOrchestra (vector, BM25, hybrid, reranker)."""

import logging
from typing import List

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from backend.config import COHERE_API_KEY, TOP_K_RERANK, TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)


def build_vector_retriever(vectorstore, top_k: int = None) -> BaseRetriever:
    """Return a simple similarity-search retriever over *vectorstore*."""
    k = top_k if top_k is not None else TOP_K_RETRIEVAL
    logger.info("Building vector retriever (top_k=%d)", k)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_bm25_retriever(docs: List[Document], top_k: int = None) -> BaseRetriever:
    """Return a keyword-based BM25 retriever built from *docs*."""
    try:
        from langchain_community.retrievers import BM25Retriever

        k = top_k if top_k is not None else TOP_K_RETRIEVAL
        logger.info("Building BM25 retriever (top_k=%d, corpus_size=%d)", k, len(docs))
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = k
        return retriever
    except Exception as exc:
        logger.error("Failed to build BM25 retriever: %s", exc)
        raise


def build_hybrid_retriever(vectorstore, docs: List[Document]) -> BaseRetriever:
    """Combine BM25 (weight=0.4) and vector search (weight=0.6) via EnsembleRetriever.

    Weight rationale:
    - BM25 excels at exact regulation code lookups like "Section 396.11"
    - Vector search excels at semantic safety questions
    Together they cover both precise and fuzzy query patterns.
    """
    try:
        bm25 = build_bm25_retriever(docs)
        vector = build_vector_retriever(vectorstore)

        logger.info("Building hybrid retriever (BM25 0.4 + Vector 0.6)")
        return EnsembleRetriever(
            retrievers=[bm25, vector],
            weights=[0.4, 0.6],
        )
    except Exception as exc:
        logger.error("Failed to build hybrid retriever: %s", exc)
        raise


def build_reranking_retriever(base_retriever: BaseRetriever, docs: List[Document]) -> BaseRetriever:
    """Wrap *base_retriever* with Cohere reranking via ContextualCompressionRetriever.

    Falls back to *base_retriever* gracefully when COHERE_API_KEY is not set.
    """
    if not COHERE_API_KEY:
        logger.warning(
            "COHERE_API_KEY not set — skipping reranker, using base retriever."
        )
        return base_retriever

    try:
        from langchain_cohere import CohereRerank

        compressor = CohereRerank(
            cohere_api_key=COHERE_API_KEY,
            top_n=TOP_K_RERANK,
            model="rerank-english-v3.0",
        )
        logger.info("Building reranking retriever (top_n=%d)", TOP_K_RERANK)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
    except Exception as exc:
        logger.error(
            "Failed to build reranking retriever (%s) — falling back to base retriever.", exc
        )
        return base_retriever
