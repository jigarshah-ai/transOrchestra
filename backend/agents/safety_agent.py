"""Safety agent LangGraph node — hybrid RAG with corrective (Tavily) web fallback.

This module implements **Corrective RAG (CRAG)** in the narrow sense used here:
when local retrieval confidence (embedding similarity proxy) falls below a
configured threshold, the agent **augments** context with fresh web documents
instead of hallucinating from weak PDF matches. Web results are injected as
first-class ``Document`` objects so attribution stays parallel to PDF chunks.

``general`` intent is handled by ``general_agent_node`` in the graph, not this
module.
"""

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
    """Lowercase and trim intent labels for consistent comparisons.

    Args:
        intent: Raw intent string from ``AgentState`` (may be None-like).

    Returns:
        str: Normalized token used for branching and citation policy.

    Raises:
        None
    """
    return (intent or "").strip().lower()


def _split_pdf_and_web_sources(sources: List[dict]) -> tuple[list[dict], list[dict]]:
    """Partition mixed citations into PDF vs Tavily-derived rows.

    Web rows are identified by a ``[WEB]`` filename prefix applied when building
    ``Document`` metadata in ``_run_tavily_search`` — this keeps a single
    renderer in the UI while still letting us cap PDF citations preferentially.

    Args:
        sources: Heterogeneous source dicts from the RAG pipeline.

    Returns:
        tuple[list[dict], list[dict]]: ``(pdf_sources, web_sources)``.

    Raises:
        None
    """
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
    """Cap and filter citations for UX quality and trust.

    Args:
        sources: Raw source dicts produced by the RAG pipeline (PDF and/or web).
        intent: Dispatcher intent; ``general`` triggers full suppression if this
            pipeline is ever invoked with that intent (defensive).
        relevance_score: Local retrieval confidence from ``check_relevance_score``.
        web_search_used: Whether Tavily documents were merged into generation.

    Returns:
        list[dict]: Up to three sources — PDFs preferred over web rows.

    Raises:
        None

    Why suppress when relevance is low **and** web was not used:
        Showing arbitrary PDF snippets for a poor match implies a false sense of
        grounding; hiding citations signals 'answer may be weak' better than a
        misleading footnote. Once Tavily supplies web docs, we allow up to three
        web attributions because the model was actually conditioned on them.
    """
    intent_norm = _normalize_intent(intent)

    # General chit-chat should never surface regulatory "Sources" — it reads as
    # if a greeting were backed by FMCSA PDFs.
    if intent_norm == "general":
        return []

    # Threshold defined in settings (default ~0.6): below this, local chunks are
    # considered too weak to cite unless web augmentation occurred.
    if relevance_score is not None and relevance_score < settings.RELEVANCE_SCORE_THRESHOLD and not web_search_used:
        return []

    pdf_sources, web_sources = _split_pdf_and_web_sources(sources)

    if pdf_sources:
        return pdf_sources[:3]
    if web_sources:
        return web_sources[:3]
    return []


def _run_tavily_search(query: str) -> List[Document]:
    """Execute a Tavily web search and return results as ``Document`` objects.

    Each result becomes a ``Document`` whose metadata ``source`` is the page title
    (prefixed with ``[WEB]``) and ``page`` holds the URL — matching the same
    source-attribution shape used for PDF chunks so the UI renderer stays unified.

    Args:
        query: End-user text forwarded to Tavily (same as RAG query).

    Returns:
        list[Document]: Web documents to be passed into ``run_query_with_web_context``.
        Returns an empty list when the client errors or the key is invalid.

    Raises:
        None: Exceptions are logged and swallowed into an empty list.
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
    """Run hybrid + reranked RAG with Tavily augmentation on low local relevance.

    State Mutations:
        READS:
            - ``query`` — Primary retrieval and generation input.
            - ``intent`` — Drives citation policy in ``_finalize_sources`` (e.g.
              defensive ``general`` suppression).
        WRITES:
            - ``rag_answer`` / ``final_answer`` — Model output (or fallback text).
            - ``rag_sources`` — Filtered to max three, possibly empty.
            - ``web_search_used`` — True when Tavily docs were merged.
            - ``relevance_score`` — Float similarity score when check succeeds, else
              may remain unset on early exits.

    CRAG / threshold behavior:
        ``check_relevance_score`` estimates whether retrieved PDF chunks align with
        the question. When the score is **below** ``settings.RELEVANCE_SCORE_THRESHOLD``
        and ``TAVILY_API_KEY`` is configured, we fetch a small web corpus and call
        ``run_query_with_web_context`` so the LLM sees **both** grounded PDF text
        and fresh pages. This reduces hallucination on out-of-corpus questions
        while keeping PDFs as the default path when confidence is high.

    Args:
        state: ``AgentState`` including ``query`` and ``intent``.

    Returns:
        AgentState: Merged updates (spread with ``**state``) containing answers,
        sources, and telemetry flags.

    Raises:
        None: Exceptions are handled into user-visible error strings where
        possible; unexpected failures still propagate from ``anyio`` wrappers in
        rare cases.
    """
    query = state.get("query", "")
    web_search_used = False
    intent = _normalize_intent(state.get("intent", ""))

    try:
        resources = await RagManager.instance().get_resources()
        # Prefer reranked retrieval when Cohere is configured — the builder already
        # degrades to hybrid internally if keys are missing.
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

    # Corrective RAG gate: only hit Tavily when local similarity is weak *and* we
    # have an API key — otherwise stay on PDFs to control cost/latency.
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

        # Symmetric threshold with ``_finalize_sources``: sub-threshold means "do
        # not cite PDFs unless web augmented".
        if score < settings.RELEVANCE_SCORE_THRESHOLD and settings.TAVILY_API_KEY:
            logger.info("Low relevance score — triggering Tavily web search.")
            web_docs = _run_tavily_search(query)
            if web_docs:
                web_search_used = True
    except Exception as exc:
        logger.warning("Relevance check failed: %s — proceeding without web search.", exc)

    def _run_with_retriever(ret, chn) -> dict:
        """Run retrieval+generation, optionally merging ``web_docs`` into context.

        Args:
            ret: Retriever instance (reranked or hybrid fallback).
            chn: Pre-built LCEL chain bound to ``ret``.

        Returns:
            dict: Pipeline output containing at least ``answer`` and ``sources``.
        """
        if web_docs:
            # Inject web results as documents so the model attends to real snippets
            # instead of stuffing URLs into the user query (which hurts ranking).
            return run_query_with_web_context(ret, query, web_docs)
        return run_query_with_docs(chn, ret, query)

    # Primary path uses reranker; if Cohere rejects the call we rebuild with the
    # hybrid retriever only so user queries still succeed with slightly lower precision.
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
