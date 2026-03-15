"""FastAPI route definitions for the TransOrchestra API."""

import logging
import os
import time
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.agents.graph import run_graph
from backend.config import EMBEDDING_MODEL, LLM_MODEL
from backend.rag.embeddings import get_embedding_model
from backend.rag.loader import load_and_chunk_pdfs, load_single_pdf
from backend.rag.vectorstore import build_vectorstore

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Payload for the /query endpoint."""
    query: str
    thread_id: str = "default"


class QueryResponse(BaseModel):
    """Response shape returned by the /query endpoint."""
    answer: str
    sources: list
    intent: str
    web_search_used: bool
    latency_ms: int


class IngestRequest(BaseModel):
    """Payload for the /ingest endpoint."""
    pdf_paths: List[str]


class IngestResponse(BaseModel):
    """Response shape returned by the /ingest endpoint."""
    status: str
    chunks_created: int


class HealthResponse(BaseModel):
    """Response shape returned by the /health endpoint."""
    status: str
    version: str
    model: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Run *request.query* through the full multi-agent graph and return the result."""
    start_ms = int(time.time() * 1000)
    try:
        result = run_graph(request.query, thread_id=request.thread_id)
        latency_ms = int(time.time() * 1000) - start_ms
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            intent=result["intent"],
            web_search_used=result["web_search_used"],
            latency_ms=latency_ms,
        )
    except Exception as exc:
        logger.error("/query endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {exc}")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest) -> IngestResponse:
    """Ingest a list of PDF file paths, chunk them, and add to the vector store."""
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

        embedding_model = get_embedding_model(EMBEDDING_MODEL)
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
    """Simple liveness check."""
    return HealthResponse(status="ok", version="1.0.0", model=LLM_MODEL)
