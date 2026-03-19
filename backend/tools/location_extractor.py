"""Location extraction tool (structured output).

Uses OpenAI chat model structured outputs for reliable parsing.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from backend.config import settings
from backend.rag.llm_factory import build_chat_llm

logger = logging.getLogger(__name__)


class LocationInfo(BaseModel):
    origin: str = Field(default="", description="Origin city/state, e.g. 'Chicago, IL'")
    destination: str = Field(default="", description="Destination city/state, e.g. 'Detroit, MI'")
    found: bool = Field(default=False, description="True if both locations were identified")
    confidence: Literal["high", "low"] = "low"

    @field_validator("origin", "destination")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @field_validator("found")
    @classmethod
    def _compute_found(cls, v: bool, info):
        # If origin + destination are present, found must be True.
        data = info.data
        origin = (data.get("origin") or "").strip()
        dest = (data.get("destination") or "").strip()
        return bool(origin and dest)


async def extract_locations(query: str) -> LocationInfo:
    """Extract origin and destination from a route query using structured output."""
    llm = build_chat_llm().with_structured_output(LocationInfo)

    prompt = (
        "You are a location extractor for a logistics routing system.\n"
        "Extract the route origin and destination from the user's query.\n\n"
        "Rules:\n"
        "- Prefer 'City, State' format when possible (e.g., 'Chicago, IL')\n"
        "- If the query does not clearly include BOTH origin and destination, leave them blank\n"
        "- Do not guess ambiguous cities (e.g. 'Springfield')\n"
        f"\nUser query: {query}"
    )

    try:
        result: LocationInfo = await llm.ainvoke(prompt)
        # Ensure found/confidence are consistent.
        if result.origin and result.destination:
            conf = "high" if result.confidence not in ("high", "low") else result.confidence
            result = LocationInfo(
                origin=result.origin,
                destination=result.destination,
                found=True,
                confidence=conf,
            )
        else:
            result = LocationInfo(origin="", destination="", found=False, confidence="low")

        logger.info(
            "Extracted: '%s' → '%s' (confidence: %s)",
            result.origin,
            result.destination,
            result.confidence,
        )
        return result
    except Exception as exc:
        logger.error("Location extraction failed: %s", exc)
        return LocationInfo(origin="", destination="", found=False, confidence="low")
