"""Central configuration module — loads all settings from .env via python-dotenv."""

import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# ── LangSmith tracing ──────────────────────────────────────────────────────────
# Must be set in os.environ BEFORE any LangChain module is imported so that
# the LangChain callback machinery picks them up at import time.
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "transOrchestra")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv(
    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
)

logger = logging.getLogger(__name__)

# ── Core API keys ──────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
GOOGLE_MAPS_API_KEY: str = os.getenv("GOOGLE_MAPS_API_KEY", "")

# ── Storage ────────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# ── Model settings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "10"))
TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "3"))
RELEVANCE_SCORE_THRESHOLD: float = float(os.getenv("RELEVANCE_SCORE_THRESHOLD", "0.6"))


@dataclass
class Config:
    """Typed container for all runtime configuration values."""
    openai_api_key: str
    cohere_api_key: str
    tavily_api_key: str
    chroma_persist_dir: str
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    top_k_retrieval: int
    top_k_rerank: int
    relevance_score_threshold: float


def get_config() -> dict:
    """Return all configuration values as a plain dictionary."""
    _warn_missing_optional_keys()
    return {
        "openai_api_key": OPENAI_API_KEY,
        "cohere_api_key": COHERE_API_KEY,
        "tavily_api_key": TAVILY_API_KEY,
        "chroma_persist_dir": CHROMA_PERSIST_DIR,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k_retrieval": TOP_K_RETRIEVAL,
        "top_k_rerank": TOP_K_RERANK,
        "relevance_score_threshold": RELEVANCE_SCORE_THRESHOLD,
    }


def _warn_missing_optional_keys() -> None:
    """Log warnings for optional API keys that are not configured."""
    if not COHERE_API_KEY:
        logger.warning(
            "COHERE_API_KEY is not set. Reranking will fall back to base retriever."
        )
    if not TAVILY_API_KEY:
        logger.warning(
            "TAVILY_API_KEY is not set. Corrective web search will be disabled."
        )
    if not GOOGLE_MAPS_API_KEY:
        logger.warning(
            "GOOGLE_MAPS_API_KEY not set in .env — "
            "Navigator agent will use realistic mock route data."
        )


# Warn on import so misconfiguration surfaces early.
_warn_missing_optional_keys()
