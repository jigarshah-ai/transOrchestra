"""LangGraph orchestration graph for TransOrchestra multi-agent system.

This module defines the **control-plane topology** for the TransOrchestra LangGraph:

**Flow (high level)**:
    1. **START** → **dispatcher** — The entry node classifies intent (or forces
       ``document_processing`` when an image payload is present; see
       ``dispatcher_node``).
    2. **dispatcher** → *conditional edge* → one of:
       - **safety_agent** — Hybrid RAG + optional Tavily for ``safety_query``.
       - **maintenance_agent** — Same RAG stack as safety for ``maintenance_query``
         (separate node id for observability).
       - **general_agent** — Greetings / chit-chat for ``general`` (no RAG).
       - **navigator_agent** — Live routing, weather, and HazMat notes for
         ``route_query``.
       - **document_agent** — Vision-based structured extraction for
         ``document_processing``.
    3. **Specialized agent** → **END** — Each branch is a short DAG (no cycles);
       agents merge their outputs into ``AgentState`` and terminate.

**Why this shape**:
    - A single **dispatcher** keeps routing logic in one place and avoids every
      agent re-implementing intent detection.
    - **Fan-out from dispatcher** encodes product rules (which subsystem owns
      which question type) without hard-coding calls inside the UI.
    - **MemorySaver** checkpointing (see ``build_graph``) enables thread-scoped
      conversational memory keyed by ``thread_id`` in ``run_graph``.

**Visualization**:
    ``GRAPH_FLOW_NODES`` / ``GRAPH_FLOW_EDGES`` and ``get_graph_flow_data()`` feed
    the Streamlit UI so reviewers can see static topology plus the per-request
    ``execution_path``.
"""

import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.agents.dispatcher import dispatcher_node
from backend.agents.document_agent import document_agent_node
from backend.agents.general_agent import general_agent_node
from backend.agents.maintenance_agent import maintenance_agent_node
from backend.agents.navigator_agent import navigator_agent_node
from backend.agents.safety_agent import safety_agent_node
from backend.agents.state import AgentState
from backend.config import settings

logger = logging.getLogger(__name__)

# ── Graph flow visualization (nodes + edges + tools per node) ───────────────────
# These structures are UI-only metadata: they mirror the compiled graph so the
# frontend can render Graphviz without importing LangGraph internals.
GRAPH_FLOW_NODES: List[Dict[str, Any]] = [
    {"id": "__start__", "label": "Start", "tools": []},
    {"id": "dispatcher", "label": "Dispatcher", "tools": ["LLM (intent classification)"]},
    {
        "id": "safety_agent",
        "label": "Safety Agent",
        "tools": ["RAG (ChromaDB + BM25)", "Cohere Reranker", "Tavily (if low relevance)"],
    },
    {
        "id": "maintenance_agent",
        "label": "Maintenance Agent",
        "tools": ["RAG (same pipeline as safety)", "Tavily (if low relevance)"],
    },
    {
        "id": "general_agent",
        "label": "General Agent",
        "tools": ["Fixed greeting (no RAG)"],
    },
    {
        "id": "document_agent",
        "label": "Document Clerk",
        "tools": ["gpt-4o vision", "Structured extraction"],
    },
    {
        "id": "navigator_agent",
        "label": "Navigator Agent",
        "tools": ["LLM (location extraction)", "Google Maps", "Weather (MCP)"],
    },
    {"id": "__end__", "label": "End", "tools": []},
]

GRAPH_FLOW_EDGES: List[Dict[str, str]] = [
    {"from": "__start__", "to": "dispatcher", "label": ""},
    {"from": "dispatcher", "to": "safety_agent", "label": "safety_query"},
    {"from": "dispatcher", "to": "maintenance_agent", "label": "maintenance_query"},
    {"from": "dispatcher", "to": "general_agent", "label": "general"},
    {"from": "dispatcher", "to": "document_agent", "label": "document_processing"},
    {"from": "dispatcher", "to": "navigator_agent", "label": "route_query"},
    {"from": "safety_agent", "to": "__end__", "label": ""},
    {"from": "maintenance_agent", "to": "__end__", "label": ""},
    {"from": "general_agent", "to": "__end__", "label": ""},
    {"from": "document_agent", "to": "__end__", "label": ""},
    {"from": "navigator_agent", "to": "__end__", "label": ""},
]


def get_graph_flow_data() -> Dict[str, Any]:
    """Return static graph topology for UI visualization.

    Returns:
        Dict[str, Any]: Mapping with keys ``nodes`` and ``edges`` (lists), suitable
        for building a Graphviz DOT diagram alongside a per-run
        ``execution_path`` (injected in ``run_graph``).
    """
    return {"nodes": GRAPH_FLOW_NODES, "edges": GRAPH_FLOW_EDGES}


def route_after_dispatch(
    state: AgentState,
) -> str:
    """Map post-dispatcher ``AgentState.intent`` to the next graph node name.

    LangGraph invokes this after ``dispatcher_node``; the return value must match
    a key in ``add_conditional_edges``'s path map. Normalizing intent here avoids
    brittle routing when the LLM emits trailing whitespace or inconsistent case.

    Args:
        state: Current graph state; only ``intent`` is read for routing.

    Returns:
        str: Next node id: ``navigator_agent``, ``safety_agent``,
            ``maintenance_agent``, ``general_agent``, or ``document_agent``.
            Unknown intents fall back to ``safety_agent`` so odd classifier
            outputs still get a RAG attempt.

    Raises:
        None: This function does not raise.
    """

    intent = (state.get("intent", "general") or "").strip().lower()
    logger.info("Routing evaluated intent as: %s", intent)
    if intent == "route_query":
        return "navigator_agent"
    if intent == "safety_query":
        return "safety_agent"
    if intent == "maintenance_query":
        return "maintenance_agent"
    if intent == "document_processing":
        return "document_agent"
    if intent == "general":
        return "general_agent"
    # Unknown intent: RAG may still help; avoid sending arbitrary text to general-only greeting.
    return "safety_agent"


def build_graph():
    """Build, compile, and return the TransOrchestra LangGraph with checkpointing.

    Compiles a ``StateGraph(AgentState)`` with:
        - Six worker nodes: ``dispatcher``, ``safety_agent``, ``maintenance_agent``,
          ``general_agent``, ``document_agent``, ``navigator_agent``.
        - A conditional edge from ``dispatcher`` driven by ``route_after_dispatch``.
        - Linear edges from each agent to ``END``.
        - ``MemorySaver`` so ``ainvoke`` can persist thread state when a
          ``thread_id`` is supplied in the runnable config.

    Returns:
        object: Compiled LangGraph instance ready for ``ainvoke`` / ``invoke``.

    Raises:
        Exception: Propagates if node registration or compilation fails (e.g.
            invalid state schema).
    """
    graph = StateGraph(AgentState)

    graph.add_node("dispatcher", dispatcher_node)
    graph.add_node("safety_agent", safety_agent_node)
    graph.add_node("maintenance_agent", maintenance_agent_node)
    graph.add_node("general_agent", general_agent_node)
    graph.add_node("document_agent", document_agent_node)
    graph.add_node("navigator_agent", navigator_agent_node)

    graph.add_edge(START, "dispatcher")
    graph.add_conditional_edges(
        "dispatcher",
        route_after_dispatch,
        {
            "safety_agent": "safety_agent",
            "maintenance_agent": "maintenance_agent",
            "general_agent": "general_agent",
            "navigator_agent": "navigator_agent",
            "document_agent": "document_agent",
            END: END,
        },
    )
    graph.add_edge("safety_agent", END)
    graph.add_edge("maintenance_agent", END)
    graph.add_edge("general_agent", END)
    graph.add_edge("document_agent", END)
    graph.add_edge("navigator_agent", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("TransOrchestra LangGraph compiled successfully.")
    return compiled


# Module-level singleton compiled graph — avoids rebuilding the graph on every
# API request (compilation is not free; checkpoint wiring is stable per process).
_graph = None


def _get_graph():
    """Return the process-wide compiled graph, lazily building it once.

    Returns:
        object: The singleton compiled LangGraph from ``build_graph``.

    Raises:
        Exception: Propagates from ``build_graph`` on first build failure.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def run_graph(query: str, thread_id: str = "default", image_base64: str | None = None) -> dict:
    """Run one end-to-end multi-agent turn and normalize outputs for the API layer.

    Seeds ``AgentState`` with a single ``HumanMessage``, optional ``image_base64``
    (consumed by the dispatcher / document path), and default empty fields so
    downstream nodes can rely on keys existing.

    Args:
        query: End-user text (or placeholder when an image drives document flow).
        thread_id: Checkpointer thread id for conversational continuity.
        image_base64: Optional raw or data-URL base64 payload for vision extraction.

    Returns:
        dict: API-oriented payload including ``answer``, ``sources``, ``intent``,
            ``agent_used``, telemetry (models, relevance), ``flow_data`` with
            ``execution_path``, and optional ``route_data`` / ``weather_data`` /
            ``extracted_doc_data``.

    Raises:
        Exception: Logs and re-raises if ``ainvoke`` fails (caller maps to HTTP 500).
    """
    compiled = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "image_data": image_base64,
        "intent": "",
        "vehicle_id": "",
        "cargo_type": "",
        "route_constraints": {},
        "rag_answer": "",
        "rag_sources": [],
        "web_search_used": False,
        "final_answer": "",
        "extracted_doc_data": None,
        "thread_id": thread_id,
        "route_data":   None,
        "weather_data": None,
        "relevance_score": None,
    }

    try:
        final_state = await compiled.ainvoke(initial_state, config=config)
        intent = (final_state.get("intent", "general") or "").strip().lower()
        if intent == "route_query":
            agent_used = "navigator_agent"
            execution_path: List[str] = ["__start__", "dispatcher", "navigator_agent", "__end__"]
        elif intent == "document_processing":
            agent_used = "document_agent"
            execution_path = ["__start__", "dispatcher", "document_agent", "__end__"]
        elif intent == "safety_query":
            agent_used = "safety_agent"
            execution_path = ["__start__", "dispatcher", "safety_agent", "__end__"]
        elif intent == "maintenance_query":
            agent_used = "maintenance_agent"
            execution_path = ["__start__", "dispatcher", "maintenance_agent", "__end__"]
        elif intent == "general":
            agent_used = "general_agent"
            execution_path = ["__start__", "dispatcher", "general_agent", "__end__"]
        else:
            agent_used = "safety_agent"
            execution_path = ["__start__", "dispatcher", "safety_agent", "__end__"]

        return {
            "answer":          final_state.get("final_answer", "No answer generated."),
            "sources":         final_state.get("rag_sources", []),
            "intent":          intent,
            "agent_used":      agent_used,
            "web_search_used": final_state.get("web_search_used", False),
            "route_data":      final_state.get("route_data"),
            "weather_data":    final_state.get("weather_data"),
            "relevance_score": final_state.get("relevance_score"),
            "extracted_doc_data": final_state.get("extracted_doc_data"),
            "llm_model":       settings.LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_provider":    settings.LLM_PROVIDER,
            "flow_data":       {
                **get_graph_flow_data(),
                "execution_path": execution_path,
            },
        }
    except Exception as exc:
        logger.error("Graph execution failed for thread '%s': %s", thread_id, exc)
        raise
