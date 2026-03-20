"""Document Agent — multimodal (vision) logistics extraction for TransOrchestra.

Uses OpenAI-compatible vision + structured output to turn shipping paperwork
images into a small Pydantic schema suitable for UI metrics and auditing.
"""

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
    """Structured fields extracted from a logistics document image.

    Attributes:
        document_type: One of the constrained labels requested in the prompt.
        origin: Ship-from location text as read from the document.
        destination: Ship-to location text as read from the document.
        weight: Human-readable weight string including units when visible.
        freight_class: Freight class text, or the literal N/A when absent.
        summary: Short natural-language recap for operators.
    """

    document_type: str
    origin: str
    destination: str
    weight: str
    freight_class: str
    summary: str


def _normalize_base64(image_data: str) -> str:
    """Strip a data-URL prefix so callers may pass either raw or data-URL base64.

    Args:
        image_data: Base64 string, optionally prefixed with ``data:*;base64,``.

    Returns:
        str: Payload portion only, suitable for re-wrapping as a vision URL.

    Raises:
        None
    """
    img = (image_data or "").strip()
    if img.startswith("data:"):
        # Expected format: data:image/jpeg;base64,<base64>
        _, _, tail = img.partition("base64,")
        return tail.strip()
    return img


async def document_agent_node(state: AgentState) -> Dict[str, Any]:
    """Extract structured logistics fields from an uploaded document image.

    State Mutations:
        READS:
            - ``image_data`` — Required; without it the node returns a user-facing
              error in ``final_answer``.
            - ``query`` — Optional user note appended to the vision prompt for
              context (e.g., 'confirm weight').
        WRITES:
            - ``extracted_doc_data`` — Dict from ``DocumentExtraction.model_dump()``
              on success, else ``None``.
            - ``final_answer`` — Human-readable summary for chat UIs.

    Why ``data:image/jpeg;base64,...`` wrapping:
        The OpenAI multimodal message format expects an image URL object; raw
        base64 without a MIME prefix is rejected by the API.

    Args:
        state: ``AgentState`` carrying base64 image data and optional query hint.

    Returns:
        Dict[str, Any]: Partial state update with ``extracted_doc_data`` and
        ``final_answer`` keys only.

    Raises:
        None: Vision/LLM failures are caught and surfaced as a graceful
        ``final_answer`` string with ``extracted_doc_data`` cleared.
    """
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

    # Vision payload must be a proper data URL; we normalize first because the
    # frontend may send either bare base64 or a full data-URL string.
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

