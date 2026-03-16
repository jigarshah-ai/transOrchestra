"""Navigator Agent — Live Google Maps routing with per-state HazMat compliance."""

import logging
from typing import Dict

from backend.agents.state import AgentState
from backend.tools.location_extractor import extract_locations
from backend.tools.maps_tool import get_route

logger = logging.getLogger(__name__)


def navigator_agent_node(state: AgentState) -> AgentState:
    """Real navigator node:

    1. Extract origin + destination from the user query via LLM.
    2. Fetch a live driving route from Google Maps (or mock fallback).
    3. Build a compliance-aware markdown answer.
    4. Store route_data in state so Streamlit can render the folium map.
    """
    query = state.get("query", "")
    logger.info("Navigator agent processing: %s", query[:80])

    # ── Step 1: extract locations ──────────────────────────────────────────────
    locations: Dict = extract_locations(query)

    if not locations.get("found"):
        return {
            **state,
            "final_answer": (
                "I couldn't identify a clear origin and destination in your query.\n\n"
                "Please include both, for example:\n"
                "- *'Route from Chicago, IL to Detroit, MI'*\n"
                "- *'How long to drive from Houston TX to Dallas TX?'*"
            ),
            "route_data":      None,
            "rag_sources":     [],
            "web_search_used": False,
        }

    origin = locations["origin"]
    destination = locations["destination"]
    logger.info("Routing: %r → %r", origin, destination)

    # ── Step 2: fetch route ────────────────────────────────────────────────────
    route = get_route(origin, destination)

    # ── Step 3: build markdown answer ─────────────────────────────────────────
    mock_badge = (
        "\n> ⚠️ *Simulated route — add `GOOGLE_MAPS_API_KEY` to `.env` "
        "for live Google Maps data.*\n"
        if route.get("mock_data") else ""
    )

    states_str = (
        ", ".join(route["states_crossed"]) if route["states_crossed"] else "unknown"
    )

    compliance_section = ""
    if route["compliance_notes"]:
        permit_states = [
            n["state"] for n in route["compliance_notes"] if n.get("permit_required")
        ]
        compliance_section = "\n\n**⚠️ HazMat Compliance Alerts:**\n"
        for note in route["compliance_notes"]:
            permit_flag = " *(permit required)*" if note["permit_required"] else ""
            compliance_section += (
                f"\n**{note['state']}{permit_flag}**\n"
                f"{note['summary']}\n"
                f"> {note['detail']}\n"
            )
        if permit_states:
            compliance_section += (
                f"\n> **Action required:** Obtain HazMat permits for: "
                f"{', '.join(permit_states)} before departure.\n"
            )

    answer = (
        f"**Route: {route['origin']} → {route['destination']}**\n"
        f"{mock_badge}\n"
        f"| Detail | Value |\n"
        f"|---|---|\n"
        f"| Distance | {route['distance_text']} ({route['distance_miles']} mi) |\n"
        f"| Est. drive time | {route['duration_text']} |\n"
        f"| With current traffic | {route['duration_in_traffic']} |\n"
        f"| States crossed | {states_str} |\n"
        f"{compliance_section}\n"
        f"---\n"
        f"*Per 49 CFR 392.9, verify cargo securement before departure. "
        f"Confirm HazMat placards meet all state requirements per 49 CFR 172.504.*"
    )

    logger.info(
        "Navigator: %.1f mi, %s, %d compliance notes",
        route["distance_miles"],
        route["duration_in_traffic"],
        len(route["compliance_notes"]),
    )

    return {
        **state,
        "final_answer":    answer,
        "route_data":      route,
        "rag_sources":     [],
        "web_search_used": False,
    }
