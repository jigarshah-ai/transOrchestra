from typing import Any

from langchain_openai import ChatOpenAI

from backend.config import settings


def build_chat_llm(model: str | None = None, **kwargs: Any) -> ChatOpenAI:
    """Return a ChatOpenAI client pointed to OpenAI or OpenRouter.

    When LLM_PROVIDER=openrouter and OPENROUTER_API_KEY is set, uses OpenRouter
    base URL and API key so you can use models like meta-llama/Meta-Llama-3-8B-Instruct.
    Otherwise uses direct OpenAI.

    Args:
        model: Override model name (e.g. for comparison). Uses settings.LLM_MODEL if None.
        **kwargs: Passed through to ChatOpenAI (e.g. streaming=False for RAG chains).
    """
    use_or = bool((settings.OPENROUTER_API_KEY or "").strip()) and settings.LLM_PROVIDER == "openrouter"
    model_name = model or settings.LLM_MODEL

    if use_or:
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            **kwargs,
        )
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        openai_api_key=settings.OPENAI_API_KEY,
        **kwargs,
    )