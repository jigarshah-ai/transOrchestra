"""Navigator Agent — live routing, weather context, and HazMat compliance notes.

Combines structured location extraction, synchronous Maps SDK work offloaded to a
thread pool, and async weather aggregation so route answers stay responsive under
FastAPI's event loop.
"""

import logging
from typing import Dict

import anyio

from backend.agents.state import AgentState
from backend.tools.location_extractor import extract_locations
from backend.tools.maps_tool import get_route
from backend.tools.weather_tool import get_route_weather

logger = logging.getLogger(__name__)


async def navigator_agent_node(state: AgentState) -> AgentState:
    """Plan a route, enrich with weather, and persist artifacts for the Streamlit map.

    State Mutations:
        READS:
            - ``query`` — Natural-language origin/destination text for geocoding
              via ``extract_locations``.
        WRITES:
            - ``final_answer`` — Markdown table + weather + optional tool warnings +
              HazMat section for the user.
            - ``route_data`` — Structured polyline and compliance metadata for
              Folium (``None`` when locations cannot be parsed).
            - ``weather_data`` — Endpoint weather summary + safety level for the UI
              card.
            - ``relevance_score`` — Explicitly cleared to ``None`` (not used on this
              branch; avoids leaking a prior turn's RAG score).
            - ``rag_sources`` / ``web_search_used`` — Reset for this branch so mixed
              sessions do not show stale citations beside a map.

    Why ``anyio.to_thread.run_sync`` for ``get_route``:
        The Google Maps client path is synchronous; running it directly would block
        the asyncio loop and stall concurrent API requests.

    Why merge ``**state`` on return:
        Preserves accumulated fields (e.g., ``messages``, ``thread_id``) while
        overriding navigator outputs — LangGraph merges dict returns into state.

    Args:
        state: Current ``AgentState`` including the user ``query``.

    Returns:
        AgentState: Merged state dict including route and weather payloads or a
        concise failure message when parsing fails.

    Raises:
        None: Tool errors are embedded in ``route_data`` / ``weather_data`` via
        ``api_error`` keys when available.
    """
    query = state.get("query", "")
    logger.info("Navigator agent processing: %s", query[:80])

    # ── Step 1: extract locations ──────────────────────────────────────────────
    locations = await extract_locations(query)

    if not locations.found:
        return {
            **state,
            "final_answer": (
                "I couldn't identify a clear origin and destination in your query.\n\n"
                "Please include both, for example:\n"
                "- *'Route from Chicago, IL to Detroit, MI'*\n"
                "- *'How long to drive from Houston TX to Dallas TX?'*"
            ),
            "route_data":      None,
            "weather_data":    None,
            "relevance_score": None,
            "rag_sources":     [],
            "web_search_used": False,
        }

    origin = locations.origin
    destination = locations.destination
    logger.info("Routing: %r → %r", origin, destination)

    # ── Step 2: fetch route ────────────────────────────────────────────────────
    # googlemaps is sync; run in a worker thread to avoid blocking the event loop.
    route = await anyio.to_thread.run_sync(get_route, origin, destination)

    # Weather is async-native (HTTP/MCP stack) — keeps parity with FastAPI without
    # nested ``asyncio.run`` calls that would break inside a running loop.
    weather = await get_route_weather(origin, destination)
    logger.info(
        "Route weather: %s (mock=%s)", weather["overall_safety"], weather["is_mock"]
    )

    # ── Step 3: build markdown answer ─────────────────────────────────────────
    mock_badge = (
        "\n> ⚠️ *Simulated route — add `GOOGLE_MAPS_API_KEY` to `.env` "
        "for live Google Maps data.*\n"
        if route.get("mock_data") else ""
    )

    states_str = (
        ", ".join(route["states_crossed"]) if route["states_crossed"] else "unknown"
    )

    # Weather section
    mock_weather_badge = " *(mock data)*" if weather["is_mock"] else ""
    weather_emoji = {
        "CLEAR":     "✅",
        "ADVISORY":  "🔵",
        "CAUTION":   "⚠️",
        "DANGEROUS": "🚨",
    }.get(weather["overall_safety"], "❓")

    weather_section = (
        f"\n\n**{weather_emoji} Weather Assessment{mock_weather_badge}**\n"
        f"{weather['raw_text']}"
    )

    # Degrade gracefully: maps/weather may be missing API keys or hit quota;
    # inline notices explain *why* the UI shows mock or partial data.
    tool_warnings = []
    if route.get("api_error"):
        tool_warnings.append(f"- **Maps**: {route['api_error']}")
    if weather.get("api_error"):
        tool_warnings.append(f"- **Weather**: {weather['api_error']}")
    tool_warning_section = ""
    if tool_warnings:
        tool_warning_section = "\n\n**Tool availability notice**\n" + "\n".join(tool_warnings)

    # HazMat compliance section
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
        f"{weather_section}"
        f"{tool_warning_section}"
        f"{compliance_section}\n"
        f"---\n"
        f"*Per 49 CFR 392.14, drivers must reduce speed or pull over in hazardous "
        f"weather conditions. Per 49 CFR 392.9, verify cargo securement before departure.*"
    )

    logger.info(
        "Navigator: %.1f mi, %s, %d compliance notes, weather=%s",
        route["distance_miles"],
        route["duration_in_traffic"],
        len(route["compliance_notes"]),
        weather["overall_safety"],
    )

    return {
        **state,
        "final_answer":    answer,
        "route_data":      route,
        "weather_data":    weather,
        "relevance_score": None,
        "rag_sources":     [],
        "web_search_used": False,
    }
