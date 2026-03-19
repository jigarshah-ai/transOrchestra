"""LangGraph orchestration graph for TransOrchestra multi-agent system."""

import logging
from typing import Any, Dict, List, Literal

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.agents.dispatcher import dispatcher_node
from backend.agents.document_agent import document_agent_node
from backend.agents.navigator_agent import navigator_agent_node
from backend.agents.safety_agent import safety_agent_node
from backend.agents.state import AgentState
from backend.config import settings

logger = logging.getLogger(__name__)

# ── Graph flow visualization (nodes + edges + tools per node) ───────────────────
GRAPH_FLOW_NODES: List[Dict[str, Any]] = [
    {"id": "__start__", "label": "Start", "tools": []},
    {"id": "dispatcher", "label": "Dispatcher", "tools": ["LLM (intent classification)"]},
    {
        "id": "safety_agent",
        "label": "Safety Agent",
        "tools": ["RAG (ChromaDB + BM25)", "Cohere Reranker", "Tavily (if low relevance)"],
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
    {"from": "dispatcher", "to": "safety_agent", "label": "safety / general / maintenance"},
    {"from": "dispatcher", "to": "document_agent", "label": "document_processing"},
    {"from": "dispatcher", "to": "navigator_agent", "label": "route_query"},
    {"from": "safety_agent", "to": "__end__", "label": ""},
    {"from": "document_agent", "to": "__end__", "label": ""},
    {"from": "navigator_agent", "to": "__end__", "label": ""},
]


def get_graph_flow_data() -> Dict[str, Any]:
    """Return graph topology (nodes, edges) for UI visualization."""
    return {"nodes": GRAPH_FLOW_NODES, "edges": GRAPH_FLOW_EDGES}


def route_after_dispatch(
    state: AgentState,
) -> str:
    """Conditional edge: map classified intent to the appropriate agent node."""

    intent = (state.get("intent", "general") or "").strip().lower()
    logger.info("Routing evaluated intent as: %s", intent)
    if intent == "route_query":
        return "navigator_agent"
    if intent in ("safety_query", "maintenance_query"):
        return "safety_agent"
    if intent == "document_processing":
        return "document_agent"
    if intent == "general":
        # No dedicated general agent exists; fall back to safety_agent.
        return "safety_agent"
    # Unknown intent: end early (safer than guessing another agent).
    return "safety_agent"


def build_graph():
    """Build, compile, and return the TransOrchestra LangGraph with MemorySaver checkpointing."""
    graph = StateGraph(AgentState)

    graph.add_node("dispatcher", dispatcher_node)
    graph.add_node("safety_agent", safety_agent_node)
    graph.add_node("document_agent", document_agent_node)
    graph.add_node("navigator_agent", navigator_agent_node)

    graph.add_edge(START, "dispatcher")
    graph.add_conditional_edges(
        "dispatcher",
        route_after_dispatch,
        {
            "safety_agent": "safety_agent",
            "navigator_agent": "navigator_agent",
            "document_agent": "document_agent",
            END: END,
        },
    )
    graph.add_edge("safety_agent", END)
    graph.add_edge("document_agent", END)
    graph.add_edge("navigator_agent", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("TransOrchestra LangGraph compiled successfully.")
    return compiled


# Module-level singleton compiled graph.
_graph = None


def _get_graph():
    """Return the singleton compiled graph, building it on first call."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def run_graph(query: str, thread_id: str = "default", image_base64: str | None = None) -> dict:
    """Invoke the full multi-agent graph for *query* and return a structured result dict."""
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
        elif intent in ("safety_query", "maintenance_query"):
            agent_used = "safety_agent"
            execution_path = ["__start__", "dispatcher", "safety_agent", "__end__"]
        elif intent == "general":
            agent_used = "safety_agent"
            execution_path = ["__start__", "dispatcher", "safety_agent", "__end__"]
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
