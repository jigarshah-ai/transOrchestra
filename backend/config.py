"""Central configuration module.

Production-grade settings via Pydantic BaseSettings with validation.

IMPORTANT: This module is intentionally imported first (see `backend/main.py`) so
LangSmith environment variables are forwarded to `os.environ` before any LangChain
imports occur anywhere else in the codebase.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Validated runtime configuration loaded from environment / `.env`."""

    # Resolve `.env` reliably regardless of current working directory:
    # - Prefer `<repo_root>/.env`
    # - Also support a shared `Final_Project/.env` one level above the repo folder
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _ENV_CANDIDATES = (
        _REPO_ROOT / ".env",
        _REPO_ROOT.parent / ".env",
    )

    model_config = SettingsConfigDict(
        env_file=[str(p) for p in _ENV_CANDIDATES if p.exists()] or ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Core API keys ──────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key (required).")
    COHERE_API_KEY: str = Field(default="", description="Cohere key (optional).")
    TAVILY_API_KEY: str = Field(default="", description="Tavily key (optional).")
    GOOGLE_MAPS_API_KEY: str = Field(default="", description="Google Maps key (optional).")
    OPENWEATHERMAP_API_KEY: str = Field(default="", description="OpenWeatherMap key (optional).")
    # ── Google Gemini ──────────────────────────────────────────────────────────
    GOOGLE_API_KEY: str = Field(default="", description="Google Gemini API key (optional).")

    # ── Weather ────────────────────────────────────────────────────────────────
    WEATHER_UNITS: Literal["imperial", "metric"] = "imperial"

    # ── Storage ────────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: Path = Field(default=Path("./chroma_db"))

    # ── Model settings ─────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    LLM_MODEL: str = "gpt-4o-mini"

    # ── Chunking ───────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # ── Retrieval ──────────────────────────────────────────────────────────────
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 3
    RELEVANCE_SCORE_THRESHOLD: float = 0.6

    # ── LangSmith tracing ──────────────────────────────────────────────────────
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "transOrchestra"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"


    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API key (required when LLM_PROVIDER=openrouter).")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_PROVIDER: Literal["openai", "openrouter"] = "openai"
    LLM_MODEL_CLOSED: str = "openai/gpt-4o-mini"
    LLM_MODEL_OPEN: str = "meta-llama/Meta-Llama-3-8B-Instruct"


    @field_validator("CHROMA_PERSIST_DIR", mode="before")
    @classmethod
    def _coerce_path(cls, v):
        return Path(v) if isinstance(v, str) else v

    @field_validator("CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K_RETRIEVAL", "TOP_K_RERANK")
    @classmethod
    def _positive_ints(cls, v: int):
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @field_validator("RELEVANCE_SCORE_THRESHOLD")
    @classmethod
    def _threshold_range(cls, v: float):
        if not (0.0 <= v <= 1.0):
            raise ValueError("must be between 0.0 and 1.0")
        return v

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def _openai_key_format(cls, v: str):
        # Do not hard-fail at import time; allow the app to start and surface a clear
        # runtime error on endpoints that require the key. If provided, it should
        # at least look like a key.
        v = (v or "").strip()
        if v and len(v) < 20:
            raise ValueError("OPENAI_API_KEY looks too short to be valid")
        return v

    @field_validator("LLM_MODEL")
    @classmethod
    def _llm_model_non_empty(cls, v: str):
        if not v.strip():
            raise ValueError("LLM_MODEL must be non-empty")
        return v

    @field_validator("EMBEDDING_MODEL")
    @classmethod
    def _embedding_model_non_empty(cls, v: str):
        if not v.strip():
            raise ValueError("EMBEDDING_MODEL must be non-empty")
        return v


def _forward_langsmith_env(s: Settings) -> None:
    """Forward LangSmith vars to os.environ before any LangChain imports."""
    os.environ["LANGCHAIN_TRACING_V2"] = (s.LANGCHAIN_TRACING_V2 or "false")
    os.environ["LANGCHAIN_API_KEY"] = (s.LANGCHAIN_API_KEY or "")
    os.environ["LANGCHAIN_PROJECT"] = (s.LANGCHAIN_PROJECT or "transOrchestra")
    os.environ["LANGCHAIN_ENDPOINT"] = (s.LANGCHAIN_ENDPOINT or "https://api.smith.langchain.com")


def _warn_optional_keys(s: Settings) -> None:
    if not s.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY is not set. LLM-powered features will fail until configured.")
    if not s.COHERE_API_KEY:
        logger.warning("COHERE_API_KEY is not set. Reranking will fall back to base retriever.")
    if not s.TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY is not set. Corrective web search will be disabled.")
    if not s.GOOGLE_MAPS_API_KEY:
        logger.warning("GOOGLE_MAPS_API_KEY not set — Navigator uses realistic mock route data.")
    if not s.OPENWEATHERMAP_API_KEY:
        logger.warning("OPENWEATHERMAP_API_KEY not set — weather tool will return mock data.")
    if s.LLM_PROVIDER == "openrouter" and not (s.OPENROUTER_API_KEY or "").strip():
        logger.warning("LLM_PROVIDER=openrouter but OPENROUTER_API_KEY not set — LLM calls may fail.")


# Singleton settings object used across the codebase.
settings = Settings()
_forward_langsmith_env(settings)
_warn_optional_keys(settings)

