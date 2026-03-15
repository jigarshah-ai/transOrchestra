"""TransOrchestra FastAPI application entry point."""

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.config import EMBEDDING_MODEL, LLM_MODEL

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
    logger.info("Embedding model: %s", EMBEDDING_MODEL)
    logger.info("LLM model: %s", LLM_MODEL)


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
