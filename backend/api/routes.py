"""FastAPI route definitions for the TransOrchestra control-plane API.

Exposes JSON endpoints consumed by the Streamlit frontend and automation:

    - ``POST /query`` — Runs the LangGraph multi-agent system (dispatcher →
      specialized agents), then **additionally** invokes two comparison LLMs
      (``LLM_MODEL_OPEN`` / ``LLM_MODEL_CLOSED``) in parallel for portfolio-style
      observability. The graph answer remains the primary user-facing response.
    - ``POST /ingest`` — Chunks PDFs on disk paths provided by the client and
      rebuilds the Chroma collection used by RAG.
    - ``GET /health`` — Lightweight liveness for orchestrators.

The router assumes ``backend.main`` mounts it under a versioned prefix (e.g.
``/api/v1``).
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from backend.agents.graph import run_graph
from backend.config import settings
from backend.rag.embeddings import get_embedding_model
from backend.rag.loader import load_and_chunk_pdfs, load_single_pdf
from backend.rag.llm_factory import build_chat_llm
from backend.rag.vectorstore import build_vectorstore

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Payload for the ``/query`` endpoint.

    Attributes:
        query: Natural language user message.
        thread_id: LangGraph checkpointer thread key for conversational memory.
        image_base64: Optional document image; when set, dispatcher forces
            ``document_processing`` until the client clears it.
    """

    query: str
    thread_id: str = "default"
    image_base64: Optional[str] = None


class QueryResponse(BaseModel):
    """Structured response for ``/query`` including telemetry for the UI.

    Attributes:
        answer: Primary assistant answer from LangGraph's ``final_answer``.
        sources: Attributed citations (already capped/filtered server-side).
        intent: Normalized dispatcher intent string.
        agent_used: Which specialized node produced the answer.
        web_search_used: Whether Tavily augmented context.
        latency_ms: End-to-end handler latency including comparison LLMs.
        route_data: Present for navigator intents (map rendering).
        weather_data: Parallel weather summary for routes.
        relevance_score: Retrieval confidence from the safety agent when set.
        llm_model: Active primary model label from settings snapshot.
        embedding_model: Active embedding model label.
        llm_provider: ``openrouter`` vs ``openai`` discriminator.
        llm_comparison: Side-by-side outputs from open/closed comparison models.
        flow_data: Static topology + ``execution_path`` for Graphviz UI.
        extracted_doc_data: Structured vision extraction payload when applicable.
    """

    answer: str
    sources: list
    intent: str
    agent_used: str
    web_search_used: bool
    latency_ms: int
    route_data: Optional[Dict[str, Any]] = None
    weather_data: Optional[Dict[str, Any]] = None
    relevance_score: Optional[float] = None
    llm_model: str
    embedding_model: str
    llm_provider: str
    llm_comparison: Optional[Dict[str, Any]] = None
    flow_data: Optional[Dict[str, Any]] = None
    extracted_doc_data: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    """Payload for the ``/ingest`` endpoint.

    Attributes:
        pdf_paths: Absolute or relative paths readable by the API process.
    """

    pdf_paths: List[str]


class IngestResponse(BaseModel):
    """Acknowledgement and chunk counts after ingestion.

    Attributes:
        status: Human-readable status token (``success`` on happy path).
        chunks_created: Number of LangChain ``Document`` chunks embedded.
    """

    status: str
    chunks_created: int


class HealthResponse(BaseModel):
    """Minimal health probe response.

    Attributes:
        status: Liveness token.
        version: Static API version label.
        model: Primary LLM model name from settings (for quick sanity checks).
    """

    status: str
    version: str
    model: str


async def _invoke_model_for_compare(query: str, model_name: str) -> Dict[str, Any]:
    """Run a **standalone** chat completion for benchmarking, not the graph answer.

    Uses ``build_chat_llm`` so OpenRouter vs OpenAI routing matches the rest of
    the codebase. Failures are captured per-model so one bad endpoint does not
    break the primary ``run_graph`` response.

    Args:
        query: Original user question for the comparison prompt.
        model_name: Explicit OpenRouter/OpenAI model id override.

    Returns:
        dict: Keys ``model``, ``answer``, ``latency_ms``, ``error`` (``None`` on success).

    Raises:
        None: Exceptions are converted into the ``error`` field.
    """
    start = int(time.time() * 1000)
    try:
        llm = build_chat_llm(model=model_name)
        resp = await llm.ainvoke(
            [
                HumanMessage(
                    content=(
                        "Answer the user's logistics/safety question in 3-5 lines, "
                        "focused on practical guidance.\n\n"
                        f"Question: {query}"
                    )
                )
            ]
        )
        text = resp.content if hasattr(resp, "content") else str(resp)
        return {
            "model": model_name,
            "answer": text,
            "latency_ms": int(time.time() * 1000) - start,
            "error": None,
        }
    except Exception as exc:
        return {
            "model": model_name,
            "answer": "",
            "latency_ms": int(time.time() * 1000) - start,
            "error": str(exc),
        }


async def _compare_open_closed_models(query: str) -> Dict[str, Any]:
    """Fan out to open and closed comparison models via ``asyncio.gather``.

    Args:
        query: User text for both completions.

    Returns:
        dict: ``{"open": {...}, "closed": {...}}`` structures from
        ``_invoke_model_for_compare``.

    Raises:
        None: Individual model errors are stored inside each sub-dict.
    """
    open_task = _invoke_model_for_compare(query, settings.LLM_MODEL_OPEN)
    closed_task = _invoke_model_for_compare(query, settings.LLM_MODEL_CLOSED)
    open_out, closed_out = await asyncio.gather(open_task, closed_task)
    return {"open": open_out, "closed": closed_out}


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Execute LangGraph and attach dual-model comparison metadata.

    Args:
        request: Validated ``QueryRequest`` instance from the client.

    Returns:
        QueryResponse: Answer + citations + telemetry + ``llm_comparison``.

    Raises:
        HTTPException: 500 when ``run_graph`` or response assembly fails.
    """
    start_ms = int(time.time() * 1000)
    try:
        result = await run_graph(
            request.query,
            thread_id=request.thread_id,
            image_base64=request.image_base64,
        )
        comparison = await _compare_open_closed_models(request.query)
        latency_ms = int(time.time() * 1000) - start_ms
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            intent=result["intent"],
            agent_used=result.get("agent_used", ""),
            web_search_used=result["web_search_used"],
            latency_ms=latency_ms,
            route_data=result.get("route_data"),
            weather_data=result.get("weather_data"),
            relevance_score=result.get("relevance_score"),
            llm_model=result.get("llm_model", settings.LLM_MODEL),
            embedding_model=result.get("embedding_model", settings.EMBEDDING_MODEL),
            llm_provider=result.get("llm_provider", settings.LLM_PROVIDER),
            llm_comparison=comparison,
            flow_data=result.get("flow_data"),
            extracted_doc_data=result.get("extracted_doc_data"),
        )
    except Exception as exc:
        logger.error("/query endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {exc}")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest) -> IngestResponse:
    """Chunk PDFs from disk paths and rebuild the persisted Chroma index.

    Args:
        request: Paths previously saved by the Streamlit uploader or CI jobs.

    Returns:
        IngestResponse: Status + chunk count.

    Raises:
        HTTPException: 400 when no readable PDFs remain; 500 on unexpected errors.
    """
    try:
        all_docs = []
        for path in request.pdf_paths:
            if not os.path.isfile(path):
                logger.warning("Ingest: file not found — skipping: %s", path)
                continue
            all_docs.extend(load_single_pdf(path))

        if not all_docs:
            raise HTTPException(
                status_code=400,
                detail="No valid PDF files found at the provided paths.",
            )

        embedding_model = get_embedding_model(settings.EMBEDDING_MODEL)
        build_vectorstore(all_docs, embedding_model)

        logger.info("Ingestion complete: %d chunks from %d files", len(all_docs), len(request.pdf_paths))
        return IngestResponse(status="success", chunks_created=len(all_docs))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/ingest endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@router.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """Return service liveness and configured primary LLM name.

    Returns:
        HealthResponse: Static version plus ``settings.LLM_MODEL``.

    Raises:
        None
    """
    return HealthResponse(status="ok", version="1.0.0", model=settings.LLM_MODEL)
