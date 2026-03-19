"""TransOrchestra FastAPI application entry point."""

import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# config must be the first project import so LangSmith env vars are set
# before any LangChain module is loaded.
from backend.config import settings
from backend.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TransOrchestra API",
    description="Multi-agent logistics control plane powered by LangGraph + RAG.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event() -> None:
    """Log startup configuration on application boot."""
    logger.info("TransOrchestra API starting...")
    logger.info("Embedding model: %s", settings.EMBEDDING_MODEL)
    logger.info("LLM model: %s", settings.LLM_MODEL)
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
    project = os.getenv("LANGCHAIN_PROJECT", "transOrchestra")
    if tracing == "true":
        logger.info("LangSmith tracing: ENABLED → project '%s'", project)
    else:
        logger.info("LangSmith tracing: disabled (set LANGCHAIN_TRACING_V2=true to enable)")


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
