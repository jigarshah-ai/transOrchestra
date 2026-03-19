"""Document Agent — multimodal (vision) logistics extraction."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from backend.agents.state import AgentState
from backend.config import settings

logger = logging.getLogger(__name__)


class DocumentExtraction(BaseModel):
    document_type: str
    origin: str
    destination: str
    weight: str
    freight_class: str
    summary: str


def _to_data_url(image_data: str) -> str:
    """Convert raw base64 into a data URL.

    If the string already looks like a data URL, return it unchanged.
    """
    img = (image_data or "").strip()
    if img.startswith("data:"):
        return img
    # Default to jpeg; most image encodings still decode fine even if the mime is slightly off.
    return f"data:image/jpeg;base64,{img}"


async def document_agent_node(state: AgentState) -> Dict[str, Any]:
    """Extract structured logistics data from an uploaded image."""
    image_data: Optional[str] = state.get("image_data")
    query: str = state.get("query", "")

    if not image_data:
        return {
            "extracted_doc_data": None,
            "final_answer": (
                "I didn’t receive an image to extract from. "
                "Please upload a Bill of Lading (BOL) or receipt image and try again."
            ),
        }

    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=settings.OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(DocumentExtraction)

    # Vision payload: text + image_url (base64).
    prompt_text = f"Extract logistics data from this image. {query}".strip()
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": _to_data_url(image_data)}},
        ]
    )

    try:
        extraction: DocumentExtraction = await structured_llm.ainvoke([message])
        extracted_dict = extraction.model_dump()

        final_answer = (
            f"Document extraction complete.\n\n"
            f"- Document type: {extracted_dict.get('document_type')}\n"
            f"- Origin: {extracted_dict.get('origin')}\n"
            f"- Destination: {extracted_dict.get('destination')}\n"
            f"- Weight: {extracted_dict.get('weight')}\n"
            f"- Freight class: {extracted_dict.get('freight_class')}\n"
            f"\nSummary: {extracted_dict.get('summary')}"
        )

        logger.info("Document extraction succeeded (doc_type=%s).", extracted_dict.get("document_type"))
        return {
            "extracted_doc_data": extracted_dict,
            "final_answer": final_answer,
        }
    except Exception as exc:
        logger.error("Document extraction failed: %s", exc)
        return {
            "extracted_doc_data": None,
            "final_answer": (
                "I couldn’t extract the logistics fields from this image. "
                "Please upload a clearer photo/scan (high contrast, full document visible) and try again."
            ),
        }

