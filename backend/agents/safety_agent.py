"""Safety agent node — RAG pipeline with optional Tavily corrective search fallback."""

import logging
from typing import List

from langchain_core.documents import Document

from backend.agents.state import AgentState
from backend.config import (
    CHROMA_PERSIST_DIR,
    RELEVANCE_SCORE_THRESHOLD,
    TAVILY_API_KEY,
)
from backend.rag.embeddings import get_embedding_model
from backend.rag.pipeline import (
    build_rag_chain,
    check_relevance_score,
    run_query_with_docs,
    run_query_with_web_context,
)
from backend.rag.retriever import build_hybrid_retriever, build_reranking_retriever
from backend.rag.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)


def _load_pipeline():
    """Load vectorstore, build hybrid + reranking retriever, and return (retriever, chain)."""
    embedding_model = get_embedding_model()
    vectorstore = load_vectorstore(embedding_model)

    # Reconstruct the document corpus for BM25 from ChromaDB.
    raw = vectorstore._collection.get(include=["documents", "metadatas"])
    docs: List[Document] = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

    hybrid = build_hybrid_retriever(vectorstore, docs)
    retriever = build_reranking_retriever(hybrid, docs)
    chain = build_rag_chain(retriever)
    return retriever, chain


def _run_tavily_search(query: str) -> List[Document]:
    """Execute a Tavily web search and return results as Document objects.

    Each result becomes a Document whose metadata 'source' is the page title
    (prefixed with [WEB]) and 'page' holds the URL — matching the same
    source-attribution format used for PDF chunks.
    """
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(query=query, max_results=3)
        docs: List[Document] = []
        for r in results.get("results", []):
            title = r.get("title", "Web result")
            content = r.get("content", "")
            url = r.get("url", "")
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": f"[WEB] {title}", "page": url},
                )
            )
        logger.info("Tavily returned %d web result(s) for query.", len(docs))
        return docs
    except Exception as exc:
        logger.error("Tavily search failed: %s", exc)
        return []


def safety_agent_node(state: AgentState) -> AgentState:
    """Run the hybrid RAG pipeline with optional corrective web search."""
    query = state.get("query", "")
    web_search_used = False

    try:
        retriever, chain = _load_pipeline()
    except FileNotFoundError:
        logger.warning("Vector store not found — returning fallback response.")
        fallback = (
            "The document knowledge base has not been initialised yet. "
            "Please upload PDF documents and click 'Ingest Documents' in the sidebar."
        )
        return {
            **state,
            "rag_answer": fallback,
            "rag_sources": [],
            "web_search_used": False,
            "final_answer": fallback,
        }
    except Exception as exc:
        logger.error("Safety agent pipeline load failed: %s", exc)
        error_msg = f"Pipeline load error: {exc}"
        return {**state, "rag_answer": error_msg, "rag_sources": [], "web_search_used": False, "final_answer": error_msg}

    # Corrective RAG: fall back to Tavily when local relevance is low.
    web_docs: List[Document] = []
    try:
        score = check_relevance_score(retriever, query)
        logger.info("Relevance score: %.3f (threshold: %.3f)", score, RELEVANCE_SCORE_THRESHOLD)

        if score < RELEVANCE_SCORE_THRESHOLD and TAVILY_API_KEY:
            logger.info("Low relevance score — triggering Tavily web search.")
            web_docs = _run_tavily_search(query)
            if web_docs:
                web_search_used = True
    except Exception as exc:
        logger.warning("Relevance check failed: %s — proceeding without web search.", exc)

    def _run_with_retriever(ret, chn) -> dict:
        """Run query through ret/chn, injecting web_docs into context when present."""
        if web_docs:
            # FIX: inject web results as real context documents, not into the query string.
            return run_query_with_web_context(ret, query, web_docs)
        return run_query_with_docs(chn, ret, query)

    # Run RAG — with automatic fallback to hybrid retriever if reranker fails (e.g. bad API key).
    try:
        result = _run_with_retriever(retriever, chain)
        return {
            **state,
            "rag_answer": result["answer"],
            "rag_sources": result["sources"],
            "web_search_used": web_search_used,
            "final_answer": result["answer"],
        }
    except Exception as exc:
        # If the reranker caused the failure (401, network, etc.) retry with hybrid only.
        if "401" in str(exc) or "rerank" in str(exc).lower() or "cohere" in str(exc).lower():
            logger.warning(
                "Reranker failed (%s) — retrying with hybrid retriever only.", exc
            )
            try:
                embedding_model = get_embedding_model()
                vectorstore = load_vectorstore(embedding_model)
                raw = vectorstore._collection.get(include=["documents", "metadatas"])
                docs_fallback: List[Document] = [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(raw["documents"], raw["metadatas"])
                ]
                fallback_retriever = build_hybrid_retriever(vectorstore, docs_fallback)
                fallback_chain = build_rag_chain(fallback_retriever)
                result = _run_with_retriever(fallback_retriever, fallback_chain)
                logger.info("Fallback to hybrid retriever succeeded.")
                return {
                    **state,
                    "rag_answer": result["answer"],
                    "rag_sources": result["sources"],
                    "web_search_used": web_search_used,
                    "final_answer": result["answer"],
                }
            except Exception as fallback_exc:
                logger.error("Fallback retriever also failed: %s", fallback_exc)
                error_msg = f"I encountered an error while processing your query: {fallback_exc}"
                return {
                    **state,
                    "rag_answer": error_msg,
                    "rag_sources": [],
                    "web_search_used": web_search_used,
                    "final_answer": error_msg,
                }

        logger.error("Safety agent RAG query failed: %s", exc)
        error_msg = f"I encountered an error while processing your query: {exc}"
        return {
            **state,
            "rag_answer": error_msg,
            "rag_sources": [],
            "web_search_used": web_search_used,
            "final_answer": error_msg,
        }
