"""General agent LangGraph node — greetings and chit-chat without RAG."""

from __future__ import annotations

import logging

from backend.agents.state import AgentState

logger = logging.getLogger(__name__)

_GENERAL_GREETING = (
    "Hello! I'm TransOrchestra. "
    "How can I help with FMCSA/DOT safety, routes, or maintenance today?"
)


async def general_agent_node(state: AgentState) -> AgentState:
    """Return a friendly default response without retrieval or citations.

    State Mutations:
        READS:
            - ``state`` (merged back via ``**state``) — preserves messages/thread.
        WRITES:
            - ``rag_answer`` / ``final_answer`` — Fixed greeting text.
            - ``rag_sources`` — Empty.
            - ``web_search_used`` — False.
            - ``relevance_score`` — None.

    Args:
        state: Current ``AgentState``.

    Returns:
        AgentState: Merged dict with greeting fields set.

    Raises:
        None
    """
    logger.info("General agent: returning greeting (no RAG).")
    return {
        **state,
        "rag_answer": _GENERAL_GREETING,
        "rag_sources": [],
        "web_search_used": False,
        "final_answer": _GENERAL_GREETING,
        "relevance_score": None,
    }
