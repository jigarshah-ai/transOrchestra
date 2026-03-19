"""Dispatcher node — classifies incoming queries and routes to the correct agent."""

import logging

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from backend.agents.state import AgentState
from backend.config import settings

logger = logging.getLogger(__name__)

INTENT_CATEGORIES = ("safety_query", "route_query", "maintenance_query", "general")

CLASSIFICATION_PROMPT = (
    "Classify this logistics query into exactly one category. "
    "Return only the category name, nothing else.\n"
    "Categories: safety_query, route_query, maintenance_query, general\n"
    "Query: {query}"
)


async def dispatcher_node(state: AgentState) -> AgentState:
    """Extract the last user message, classify intent, and update state."""
    try:
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY,
        )

        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        else:
            query = state.get("query", "")

        prompt = CLASSIFICATION_PROMPT.format(query=query)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw_intent = response.content.strip().lower()

        intent = raw_intent if raw_intent in INTENT_CATEGORIES else "general"
        logger.info("Dispatcher classified query as: '%s'", intent)

        return {
            **state,
            "query": query,
            "intent": intent,
        }
    except Exception as exc:
        logger.error("Dispatcher node failed: %s — defaulting intent to 'general'", exc)
        return {
            **state,
            "query": state.get("query", ""),
            "intent": "general",
        }
