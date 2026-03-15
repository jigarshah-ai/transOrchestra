"""Navigator agent node — route analysis with simulated data (stub for Google Maps integration)."""

import logging
import re

from backend.agents.state import AgentState

logger = logging.getLogger(__name__)

# Common city pairs with approximate distances and drive times for realistic mock data.
_ROUTE_LOOKUP = {
    ("chicago", "detroit"): ("281 miles", "4h 23min", "Illinois and Indiana"),
    ("new york", "boston"): ("215 miles", "3h 45min", "New York and Connecticut"),
    ("los angeles", "san francisco"): ("381 miles", "5h 30min", "California"),
    ("dallas", "houston"): ("239 miles", "3h 30min", "Texas"),
    ("miami", "orlando"): ("236 miles", "3h 45min", "Florida"),
    ("seattle", "portland"): ("174 miles", "2h 45min", "Washington and Oregon"),
    ("denver", "salt lake city"): ("371 miles", "5h 15min", "Colorado and Utah"),
    ("atlanta", "nashville"): ("249 miles", "3h 50min", "Georgia and Tennessee"),
}


def _extract_cities(query: str):
    """Attempt to extract origin and destination from the query string."""
    patterns = [
        r"from\s+([A-Za-z ,]+?)\s+to\s+([A-Za-z ,]+?)(?:\s|$|\.|\?)",
        r"([A-Za-z ,]+?)\s+to\s+([A-Za-z ,]+?)(?:\s|$|\.|\?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            origin = match.group(1).strip().rstrip(",")
            destination = match.group(2).strip().rstrip(",")
            return origin, destination
    return None, None


def navigator_agent_node(state: AgentState) -> AgentState:
    """Return a realistic mock route analysis response."""
    query = state.get("query", "")
    origin, destination = _extract_cities(query)

    if origin and destination:
        key = (origin.lower(), destination.lower())
        if key in _ROUTE_LOOKUP:
            distance, drive_time, states = _ROUTE_LOOKUP[key]
        else:
            # Generic fallback for unlisted city pairs.
            distance = "~350 miles (estimated)"
            drive_time = "~5h 30min (estimated)"
            states = "multiple states"

        response = (
            f"Route analysis: {origin.title()} → {destination.title()}\n"
            f"Estimated distance: {distance}\n"
            f"Estimated drive time: {drive_time} (current traffic)\n\n"
            f"Compliance note: This route crosses {states}. "
            "Ensure HazMat placards meet all applicable state requirements "
            "per 49 CFR 172.504. Verify Hours-of-Service compliance under "
            "49 CFR 395.3 before departure.\n\n"
            "⚠️  Note: This is simulated routing data. "
            "Google Maps / HERE Maps integration is planned for v2.0."
        )
    else:
        response = (
            "I can help with route analysis. Please specify your origin and destination "
            "in the format: 'Route from [City A] to [City B]'.\n\n"
            "Example: 'Route from Chicago, IL to Detroit, MI'\n\n"
            "I can provide:\n"
            "• Estimated distance and drive time\n"
            "• Multi-state compliance notes (HazMat, HOS)\n"
            "• Weight restriction alerts\n\n"
            "⚠️  Note: This is simulated routing data. "
            "Google Maps / HERE Maps integration is planned for v2.0."
        )

    logger.info("Navigator agent processed route query: origin=%s, dest=%s", origin, destination)

    return {
        **state,
        "final_answer": response,
    }
