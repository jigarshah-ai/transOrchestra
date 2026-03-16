"""LangGraph orchestration graph for TransOrchestra multi-agent system."""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.agents.dispatcher import dispatcher_node
from backend.agents.navigator_agent import navigator_agent_node
from backend.agents.safety_agent import safety_agent_node
from backend.agents.state import AgentState

logger = logging.getLogger(__name__)


def route_after_dispatch(
    state: AgentState,
) -> Literal["safety_agent", "navigator_agent"]:
    """Conditional edge: map classified intent to the appropriate agent node."""
    intent = state.get("intent", "general")
    if intent == "route_query":
        return "navigator_agent"
    # safety_query, maintenance_query, and general all go to safety_agent.
    return "safety_agent"


def build_graph():
    """Build, compile, and return the TransOrchestra LangGraph with MemorySaver checkpointing."""
    graph = StateGraph(AgentState)

    graph.add_node("dispatcher", dispatcher_node)
    graph.add_node("safety_agent", safety_agent_node)
    graph.add_node("navigator_agent", navigator_agent_node)

    graph.add_edge(START, "dispatcher")
    graph.add_conditional_edges(
        "dispatcher",
        route_after_dispatch,
        {
            "safety_agent": "safety_agent",
            "navigator_agent": "navigator_agent",
        },
    )
    graph.add_edge("safety_agent", END)
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


def run_graph(query: str, thread_id: str = "default") -> dict:
    """Invoke the full multi-agent graph for *query* and return a structured result dict."""
    compiled = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "intent": "",
        "vehicle_id": "",
        "cargo_type": "",
        "route_constraints": {},
        "rag_answer": "",
        "rag_sources": [],
        "web_search_used": False,
        "final_answer": "",
        "thread_id": thread_id,
        "route_data":   None,
        "weather_data": None,
    }

    try:
        final_state = compiled.invoke(initial_state, config=config)
        return {
            "answer":          final_state.get("final_answer", "No answer generated."),
            "sources":         final_state.get("rag_sources", []),
            "intent":          final_state.get("intent", "unknown"),
            "web_search_used": final_state.get("web_search_used", False),
            "route_data":      final_state.get("route_data"),
            "weather_data":    final_state.get("weather_data"),
        }
    except Exception as exc:
        logger.error("Graph execution failed for thread '%s': %s", thread_id, exc)
        raise
