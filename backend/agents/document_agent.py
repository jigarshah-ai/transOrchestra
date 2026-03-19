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


def _normalize_base64(image_data: str) -> str:
    """Return raw base64 without any `data:*;base64,` prefix."""
    img = (image_data or "").strip()
    if img.startswith("data:"):
        # Expected format: data:image/jpeg;base64,<base64>
        _, _, tail = img.partition("base64,")
        return tail.strip()
    return img


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

    # Vision payload: OpenAI requires `data:<mime>;base64,<base64>` formatting.
    base64_str = _normalize_base64(image_data)
    image_url = f"data:image/jpeg;base64,{base64_str}"

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "You are extracting structured logistics fields from a shipping document "
                    "(e.g., Bill of Lading / BOL, receipt, or packing slip).\n\n"
                    "Extract the following fields exactly and return them in the required schema.\n\n"
                    "Rules:\n"
                    "- If a field is not visible or cannot be confidently read, return \"N/A\".\n"
                    "- document_type: use one of {\"Bill of Lading\", \"Receipt\", \"Packing Slip\", \"Other\"}.\n"
                    "- origin and destination: prefer City + State (or nearest readable location).\n"
                    "- weight: return the numeric weight + unit if present (e.g., \"4250 lb\" or \"1,920 kg\").\n"
                    "- freight_class: return the class as readable text (e.g., \"Class 70\"), or \"N/A\".\n\n"
                    f"User note / what they care about: {query}\n"
                ),
            },
            {"type": "image_url", "image_url": {"url": image_url}},
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

