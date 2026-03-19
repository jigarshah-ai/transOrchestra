"""Dispatcher node — classifies incoming queries and routes to the correct agent."""

import logging
from enum import Enum
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from backend.agents.state import AgentState
from backend.config import settings

logger = logging.getLogger(__name__)

# 1. Force the LLM to choose from this exact list
class IntentCategory(str, Enum):
    SAFETY = "safety_query"
    ROUTE = "route_query"
    MAINTENANCE = "maintenance_query"
    GENERAL = "general"

# 2. The LLM reads these descriptions to understand the nuance
class IntentClassification(BaseModel):
    intent: IntentCategory = Field(
        description=(
            "Classify the user's intent into exactly one category:\n"
            "- safety_query: DOT laws, regulations, compliance, hours of service.\n"
            "- route_query: Map directions, live traffic, travel time, weather delays, locations.\n"
            "- maintenance_query: Truck repairs, engine fault codes, tire pressure, mechanical.\n"
            "- general: Greetings, HR, administrative questions."
        )
    )

async def dispatcher_node(state: AgentState) -> AgentState:
    """Extract the last user message, classify intent using structured output, and update state."""
    try:
        # Initialize the LLM
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        
        # Force structured JSON output
        structured_llm = llm.with_structured_output(IntentClassification)

        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        else:
            query = state.get("query", "")

        # Simple prompt (the descriptions above do the heavy work)
        prompt = f"Classify this logistics query: '{query}'"
        
        response = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        
        intent = response.intent.value
        logger.info("Dispatcher classified query as: '%s'", intent)

        return {"query": query, "intent": intent}
    except Exception as exc:
        logger.error("Dispatcher node failed: %s — defaulting intent to 'general'", exc)
        return {"query": state.get("query", ""), "intent": "general"}