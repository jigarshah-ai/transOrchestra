"""
Extracts origin and destination city/state from natural language queries.
Handles varied phrasings like:
  "route from Chicago to Detroit"
  "how long to drive from Houston TX to Dallas"
  "I need to get to Miami from NYC"
"""
import json
import logging
from typing import Dict

from langchain_openai import ChatOpenAI

from backend.config import LLM_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)


def extract_locations(query: str) -> Dict:
    """Use an LLM to extract origin and destination from a route query.

    Returns:
        dict with keys:
            origin (str): extracted origin city/state
            destination (str): extracted destination city/state
            found (bool): True if both locations were identified
            confidence (str): "high" | "low"
    """
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

    prompt = f"""You are a location extractor for a logistics system.
Extract the origin and destination from the query below.

Rules:
- Return ONLY valid JSON, no explanation, no markdown fences
- Use full "City, State" format when possible (e.g. "Chicago, IL")
- If only a city is mentioned, infer the state if it is obvious
- If you cannot find both locations, set found to false

Return this exact JSON structure:
{{
  "origin": "City, State",
  "destination": "City, State",
  "found": true,
  "confidence": "high"
}}

Or if not found:
{{
  "origin": "",
  "destination": "",
  "found": false,
  "confidence": "low"
}}

Query: {query}

JSON:"""

    raw = ""
    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown fences if the model adds them anyway.
        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.split("```")[0]

        result: Dict = json.loads(raw.strip())

        if "origin" not in result or "destination" not in result:
            return {"origin": "", "destination": "", "found": False, "confidence": "low"}

        logger.info(
            "Extracted: '%s' → '%s' (confidence: %s)",
            result.get("origin"),
            result.get("destination"),
            result.get("confidence", "unknown"),
        )
        return result

    except json.JSONDecodeError as exc:
        logger.error("JSON parse error in location extractor: %s | raw: %s", exc, raw)
        return {"origin": "", "destination": "", "found": False, "confidence": "low"}
    except Exception as exc:
        logger.error("Location extraction failed: %s", exc)
        return {"origin": "", "destination": "", "found": False, "confidence": "low"}
