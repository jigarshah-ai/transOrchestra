"""Safety agent node — RAG pipeline with optional Tavily corrective search fallback."""

import logging
from typing import List

from langchain_core.documents import Document
import anyio

from backend.agents.state import AgentState
from backend.config import settings
from backend.rag.pipeline import (
    check_relevance_score,
    run_query_with_docs,
    run_query_with_web_context,
)
from backend.rag.manager import RagManager

logger = logging.getLogger(__name__)

def _normalize_intent(intent: str) -> str:
    return (intent or "").strip().lower()


def _split_pdf_and_web_sources(sources: List[dict]) -> tuple[list[dict], list[dict]]:
    pdf_sources: list[dict] = []
    web_sources: list[dict] = []
    for s in sources or []:
        filename = str((s or {}).get("filename", ""))
        if filename.startswith("[WEB]"):
            web_sources.append(s)
        else:
            pdf_sources.append(s)
    return pdf_sources, web_sources


def _finalize_sources(
    *,
    sources: List[dict],
    intent: str,
    relevance_score: float | None,
    web_search_used: bool,
) -> List[dict]:
    """Return at most 3 citations, and suppress for irrelevant/general queries."""
    intent_norm = _normalize_intent(intent)

    # For greetings/admin/general queries, don't show citations.
    if intent_norm == "general":
        return []

    # If we didn't do web search and relevance is low, suppress sources entirely.
    if relevance_score is not None and relevance_score < settings.RELEVANCE_SCORE_THRESHOLD and not web_search_used:
        return []

    pdf_sources, web_sources = _split_pdf_and_web_sources(sources)

    if pdf_sources:
        return pdf_sources[:3]
    if web_sources:
        return web_sources[:3]
    return []


def _run_tavily_search(query: str) -> List[Document]:
    """Execute a Tavily web search and return results as Document objects.

    Each result becomes a Document whose metadata 'source' is the page title
    (prefixed with [WEB]) and 'page' holds the URL — matching the same
    source-attribution format used for PDF chunks.
    """
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.TAVILY_API_KEY)
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


async def safety_agent_node(state: AgentState) -> AgentState:
    """Run the hybrid RAG pipeline with optional corrective web search."""
    query = state.get("query", "")
    web_search_used = False
    intent = _normalize_intent(state.get("intent", ""))

    # For general greetings/admin questions, skip RAG/citations.
    if intent == "general":
        greeting = (
            "Hello! I’m TransOrchestra. "
            "How can I help with FMCSA/DOT safety, routes, or maintenance today?"
        )
        return {
            **state,
            "rag_answer": greeting,
            "rag_sources": [],
            "web_search_used": False,
            "final_answer": greeting,
            "relevance_score": None,
        }

    try:
        resources = await RagManager.instance().get_resources()
        # Use reranking retriever/chain as primary; it already falls back gracefully if key missing.
        retriever, chain = resources.reranking_retriever, resources.reranking_chain
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
    relevance_score = None
    try:
        score = check_relevance_score(retriever, query)
        relevance_score = score
        logger.info(
            "Relevance score: %.3f (threshold: %.3f)",
            score,
            settings.RELEVANCE_SCORE_THRESHOLD,
        )

        if score < settings.RELEVANCE_SCORE_THRESHOLD and settings.TAVILY_API_KEY:
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
        result = await anyio.to_thread.run_sync(_run_with_retriever, retriever, chain)
        final_sources = _finalize_sources(
            sources=result.get("sources", []),
            intent=intent,
            relevance_score=relevance_score,
            web_search_used=web_search_used,
        )
        return {
            **state,
            "rag_answer": result["answer"],
            "rag_sources": final_sources,
            "web_search_used": web_search_used,
            "final_answer": result["answer"],
            "relevance_score": relevance_score,
        }
    except Exception as exc:
        # If the reranker caused the failure (401, network, etc.) retry with hybrid only.
        if "401" in str(exc) or "rerank" in str(exc).lower() or "cohere" in str(exc).lower():
            logger.warning(
                "Reranker failed (%s) — retrying with hybrid retriever only.", exc
            )
            try:
                resources = await RagManager.instance().get_resources()
                fallback_retriever = resources.hybrid_retriever
                fallback_chain = resources.hybrid_chain
                result = await anyio.to_thread.run_sync(
                    _run_with_retriever, fallback_retriever, fallback_chain
                )
                logger.info("Fallback to hybrid retriever succeeded.")
                final_sources = _finalize_sources(
                    sources=result.get("sources", []),
                    intent=intent,
                    relevance_score=relevance_score,
                    web_search_used=web_search_used,
                )
                return {
                    **state,
                    "rag_answer": result["answer"],
                    "rag_sources": final_sources,
                    "web_search_used": web_search_used,
                    "final_answer": result["answer"],
                    "relevance_score": relevance_score,
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
                    "relevance_score": relevance_score,
                }

        logger.error("Safety agent RAG query failed: %s", exc)
        error_msg = f"I encountered an error while processing your query: {exc}"
        return {
            **state,
            "rag_answer": error_msg,
            "rag_sources": [],
            "web_search_used": web_search_used,
            "final_answer": error_msg,
            "relevance_score": relevance_score,
        }
