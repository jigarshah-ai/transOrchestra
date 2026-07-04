"""Maintenance agent LangGraph node — vehicle/mechanical RAG via shared pipeline."""

from __future__ import annotations

import logging

from backend.agents.safety_agent import safety_agent_node
from backend.agents.state import AgentState

logger = logging.getLogger(__name__)


async def maintenance_agent_node(state: AgentState) -> AgentState:
    """Delegate to the hybrid RAG pipeline (same as safety) for maintenance queries.

    Uses ``safety_agent_node`` because maintenance and safety share the same
    corpus, retrievers, and CRAG behavior; only the graph node id / UI label differ.

    Args:
        state: ``AgentState`` with ``intent`` typically ``maintenance_query``.

    Returns:
        AgentState: Output from ``safety_agent_node``.

    Raises:
        None: Delegates exception handling to ``safety_agent_node``.
    """
    logger.info("Maintenance agent: delegating to RAG pipeline.")
    return await safety_agent_node(state)
