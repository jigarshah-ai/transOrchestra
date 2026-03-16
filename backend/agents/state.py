"""LangGraph state schema for the TransOrchestra multi-agent system."""

from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state passed between every node in the TransOrchestra LangGraph."""

    messages: Annotated[list, add_messages]
    query: str
    intent: str
    vehicle_id: str
    cargo_type: str
    route_constraints: Dict[str, Any]
    rag_answer: str
    rag_sources: List[Dict[str, Any]]
    web_search_used: bool
    final_answer: str
    thread_id: str
    route_data: Optional[Dict[str, Any]]  # populated by navigator_agent; None for all other intents
