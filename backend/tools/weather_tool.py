"""
Weather tool for TransOrchestra LangGraph agents.

Synchronous wrappers around the async MCP weather server functions.
Agents call get_weather() / get_route_weather() as regular Python functions.

nest_asyncio is applied once at import time so asyncio.run() works correctly
inside FastAPI's running event loop without raising RuntimeError.
"""

import asyncio
import logging
from typing import Dict

import nest_asyncio

from backend.config import OPENWEATHERMAP_API_KEY, WEATHER_UNITS
from backend.mcp_servers.weather_server import (
    _fetch_route_weather,
    _fetch_weather,
    _mock_route_weather,
    _mock_weather,
)

# Allow asyncio.run() inside an already-running event loop (FastAPI / uvicorn).
nest_asyncio.apply()

logger = logging.getLogger(__name__)


def get_weather(location: str, units: str = None) -> Dict:
    """Get current weather for a single location.

    Args:
        location: City name e.g. "Chicago, IL"
        units:    "imperial" (°F / mph) or "metric" (°C / kph)

    Returns:
        dict with keys:
            location (str)
            raw_text (str)       — full formatted weather summary
            safety_level (str)   — CLEAR | ADVISORY | CAUTION | DANGEROUS
            is_mock (bool)       — True when OPENWEATHERMAP_API_KEY is missing
    """
    units = units or WEATHER_UNITS

    try:
        if OPENWEATHERMAP_API_KEY:
            raw     = asyncio.run(_fetch_weather(location, units))
            is_mock = False
        else:
            raw     = _mock_weather(location, units)
            is_mock = True

        safety_level = _parse_safety_level(raw)
        return {"location": location, "raw_text": raw, "safety_level": safety_level, "is_mock": is_mock}

    except Exception as exc:
        logger.error("Weather fetch error for %r: %s", location, exc)
        return {
            "location":     location,
            "raw_text":     _mock_weather(location, units),
            "safety_level": "CLEAR",
            "is_mock":      True,
        }


def get_route_weather(origin: str, destination: str, units: str = None) -> Dict:
    """Get weather at both route endpoints with a combined safety assessment.

    Args:
        origin:      Origin city e.g. "Chicago, IL"
        destination: Destination city e.g. "Detroit, MI"
        units:       "imperial" or "metric"

    Returns:
        dict with keys:
            origin, destination (str)
            raw_text (str)          — full formatted route weather summary
            overall_safety (str)    — worst safety level of the two endpoints
            has_warning (bool)      — True when CAUTION or DANGEROUS
            is_mock (bool)
    """
    units = units or WEATHER_UNITS

    try:
        if OPENWEATHERMAP_API_KEY:
            raw     = asyncio.run(_fetch_route_weather(origin, destination, units))
            is_mock = False
        else:
            raw     = _mock_route_weather(origin, destination, units)
            is_mock = True

        overall = _parse_overall_safety(raw)
        return {
            "origin":         origin,
            "destination":    destination,
            "raw_text":       raw,
            "overall_safety": overall,
            "has_warning":    overall in ("CAUTION", "DANGEROUS"),
            "is_mock":        is_mock,
        }

    except Exception as exc:
        logger.error("Route weather error %r → %r: %s", origin, destination, exc)
        return {
            "origin":         origin,
            "destination":    destination,
            "raw_text":       _mock_route_weather(origin, destination, units),
            "overall_safety": "CLEAR",
            "has_warning":    False,
            "is_mock":        True,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_safety_level(raw: str) -> str:
    """Extract the safety level keyword from a formatted weather string."""
    for level in ("DANGEROUS", "CAUTION", "ADVISORY", "CLEAR"):
        if level in raw:
            return level
    return "CLEAR"


def _parse_overall_safety(raw: str) -> str:
    """Extract the OVERALL ROUTE ASSESSMENT level from a route weather string."""
    for level in ("DANGEROUS", "CAUTION", "ADVISORY", "CLEAR"):
        if f"OVERALL ROUTE ASSESSMENT: {level}" in raw:
            return level
    return "CLEAR"
