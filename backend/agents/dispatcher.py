"""Dispatcher LangGraph node — intent classification for TransOrchestra.

The dispatcher is the **sole routing authority** after START: it either classifies
free-text with a structured-output LLM or **short-circuits** to document flow
when binary image data is already present in state (see ``dispatcher_node``).
"""

import logging
from enum import Enum
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from backend.agents.state import AgentState
from backend.config import settings

logger = logging.getLogger(__name__)

# Enum-backed labels constrain the classifier to a closed set LangGraph routing
# can depend on (open strings would invite drift and broken edges).
class IntentCategory(str, Enum):
    """Closed set of routing labels consumed by ``route_after_dispatch``."""

    SAFETY = "safety_query"
    ROUTE = "route_query"
    MAINTENANCE = "maintenance_query"
    GENERAL = "general"
    DOCUMENT = "document_processing"

# Field descriptions are embedded in the JSON schema sent to the model — they
# act as soft rules so "hours of service" maps to safety, not route.
class IntentClassification(BaseModel):
    """LLM-facing schema for structured intent classification."""

    intent: IntentCategory = Field(
        description=(
            "Classify the user's intent into exactly one category:\n"
            "- safety_query: DOT laws, regulations, compliance, hours of service.\n"
            "- route_query: Map directions, live traffic, travel time, weather delays, locations.\n"
            "- maintenance_query: Truck repairs, engine fault codes, tire pressure, mechanical.\n"
            "- document_processing: Use when the user uploads an image, document, receipt, or Bill of Lading for processing.\n"
            "- general: Greetings, HR, administrative questions."
        )
    )

async def dispatcher_node(state: AgentState) -> AgentState:
    """Classify user intent or force document-processing when an image is attached.

    State Mutations:
        READS:
            - ``messages`` — Last message content is treated as the user query when
              present (LangGraph chat history pattern).
            - ``query`` — Fallback text if the message list is empty.
            - ``image_data`` — When non-empty, **skips** the classifier LLM entirely
              so uploads cannot be mislabeled as ``general``/``route_query``.
        WRITES:
            - ``query`` — Echoed user text, or the fixed prompt
              ``"Process uploaded document."`` for the document branch.
            - ``intent`` — One of ``IntentCategory`` string values for downstream
              ``route_after_dispatch``.

    Why bypass the LLM when ``image_data`` is present:
        Vision/document handling is a different modality; asking a text classifier
        "what is this image" without reliable image input would be noisy and could
        strand uploads in the wrong agent. A deterministic override guarantees
        the graph reaches ``document_agent`` and keeps latency predictable.

    Args:
        state: Current ``AgentState`` including messages and optional image payload.

    Returns:
        AgentState: **Partial** update dict ``{"query": ..., "intent": ...}`` only
        (LangGraph merges updates; do not spread full state here).

    Raises:
        None: Exceptions are caught; on failure intent defaults to ``"general"``.
    """
    try:
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        else:
            query = state.get("query", "")

        # If an image is present, bypass LLM intent classification completely.
        if state.get("image_data"):
            logger.info("Dispatcher detected image upload — routing to document_agent.")
            return {"query": "Process uploaded document.", "intent": IntentCategory.DOCUMENT.value}

        # Structured output binds the model to ``IntentClassification`` (Pydantic),
        # yielding a parseable label without fragile JSON prompting.
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        structured_llm = llm.with_structured_output(IntentClassification)

        # Short user-facing prompt; category nuance lives in the schema descriptions.
        prompt = f"Classify this logistics query: '{query}'"
        
        response = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        
        intent = response.intent.value
        logger.info("Dispatcher classified query as: '%s'", intent)

        return {"query": query, "intent": intent}
    except Exception as exc:
        logger.error("Dispatcher node failed: %s — defaulting intent to 'general'", exc)
        return {"query": state.get("query", ""), "intent": "general"}