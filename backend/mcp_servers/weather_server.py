"""
TransOrchestra MCP Weather Server.

Exposes two MCP tools to the agent ecosystem:
  get_current_weather(location)      — current conditions + driving safety
  get_weather_route_summary(origin, destination) — both endpoints + overall assessment

Run standalone for testing:
  python backend/mcp_servers/weather_server.py

Called programmatically by backend/tools/weather_tool.py via direct
async function invocation (in-process, no stdio transport needed).
"""

import asyncio
import logging
from typing import List

import httpx
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from backend.config import settings

logger = logging.getLogger(__name__)

OWM_BASE = "https://api.openweathermap.org/data/2.5"

# ── MCP server instance ────────────────────────────────────────────────────────
app = Server("transOrchestra-weather")


# ── Tool declarations ──────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> List[types.Tool]:
    """Declare the two available MCP tools."""
    return [
        types.Tool(
            name="get_current_weather",
            description=(
                "Get current weather conditions for a city. "
                "Returns temperature, humidity, wind speed, visibility, "
                "and a driving safety assessment. "
                "Use before routing to check driving conditions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name e.g. 'Chicago, IL' or 'Detroit, MI, US'",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["imperial", "metric"],
                        "description": "imperial=°F/mph  metric=°C/kph",
                        "default": "imperial",
                    },
                },
                "required": ["location"],
            },
        ),
        types.Tool(
            name="get_weather_route_summary",
            description=(
                "Get weather at both origin and destination of a route. "
                "Returns conditions at each endpoint plus an overall driving "
                "safety assessment. Use for route planning to flag hazardous weather."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "origin":      {"type": "string", "description": "Origin city e.g. 'Chicago, IL'"},
                    "destination": {"type": "string", "description": "Destination city e.g. 'Detroit, MI'"},
                    "units": {
                        "type": "string",
                        "enum": ["imperial", "metric"],
                        "default": "imperial",
                    },
                },
                "required": ["origin", "destination"],
            },
        ),
    ]


# ── Tool dispatcher ────────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """Route MCP tool calls to the appropriate handler."""
    try:
        if name == "get_current_weather":
            result = await _fetch_weather(
                location=arguments["location"],
                units=arguments.get("units", settings.WEATHER_UNITS),
            )
            return [types.TextContent(type="text", text=result)]

        elif name == "get_weather_route_summary":
            result = await _fetch_route_weather(
                origin=arguments["origin"],
                destination=arguments["destination"],
                units=arguments.get("units", settings.WEATHER_UNITS),
            )
            return [types.TextContent(type="text", text=result)]

        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as exc:
        logger.error("MCP tool error (%s): %s", name, exc)
        return [types.TextContent(type="text", text=f"Weather data unavailable: {exc}")]


# ── Core fetch functions (called directly by weather_tool.py) ─────────────────

async def _fetch_weather(location: str, units: str = "imperial") -> str:
    """Fetch current weather from OpenWeatherMap and return a formatted string."""
    if not settings.OPENWEATHERMAP_API_KEY:
        return _mock_weather(location, units)

    unit_label  = "°F" if units == "imperial" else "°C"
    speed_label = "mph" if units == "imperial" else "km/h"

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{OWM_BASE}/weather",
            params={"q": location, "appid": settings.OPENWEATHERMAP_API_KEY, "units": units},
        )
        resp.raise_for_status()
        data = resp.json()

    w    = data["weather"][0]
    main = data["main"]
    wind = data.get("wind", {})
    vis  = data.get("visibility", 10000)
    sys  = data.get("sys", {})

    safety = _assess_driving_safety(
        weather_id=w["id"],
        wind_speed=wind.get("speed", 0),
        visibility_m=vis,
        units=units,
    )

    return (
        f"Current weather — {data['name']}, {sys.get('country', '')}\n"
        f"Condition: {w['description'].title()}\n"
        f"Temperature: {main['temp']}{unit_label} (feels like {main['feels_like']}{unit_label})\n"
        f"Humidity: {main['humidity']}%\n"
        f"Wind: {wind.get('speed', 0)} {speed_label} {_wind_direction(wind.get('deg', 0))}\n"
        f"Visibility: {round(vis / 1000, 1)} km\n"
        f"Driving safety: {safety['level']} — {safety['message']}"
    )


async def _fetch_route_weather(
    origin: str, destination: str, units: str = "imperial"
) -> str:
    """Fetch weather at both route endpoints and produce a combined safety summary."""
    if not settings.OPENWEATHERMAP_API_KEY:
        return _mock_route_weather(origin, destination, units)

    unit_label  = "°F" if units == "imperial" else "°C"
    speed_label = "mph" if units == "imperial" else "km/h"

    async with httpx.AsyncClient(timeout=10.0) as client:
        o_resp, d_resp = await asyncio.gather(
            client.get(f"{OWM_BASE}/weather",
                       params={"q": origin,      "appid": settings.OPENWEATHERMAP_API_KEY, "units": units}),
            client.get(f"{OWM_BASE}/weather",
                       params={"q": destination, "appid": settings.OPENWEATHERMAP_API_KEY, "units": units}),
        )

    o = o_resp.json()
    d = d_resp.json()

    o_w = o["weather"][0]
    d_w = d["weather"][0]

    o_safety = _assess_driving_safety(o_w["id"], o["wind"].get("speed", 0), o.get("visibility", 10000), units)
    d_safety = _assess_driving_safety(d_w["id"], d["wind"].get("speed", 0), d.get("visibility", 10000), units)
    worst    = o_safety if o_safety["score"] >= d_safety["score"] else d_safety

    return (
        f"Route weather summary\n\n"
        f"ORIGIN — {o['name']}\n"
        f"  {o_w['description'].title()} · {o['main']['temp']}{unit_label} · "
        f"Wind {o['wind'].get('speed', 0)} {speed_label}\n"
        f"  Driving: {o_safety['level']} — {o_safety['message']}\n\n"
        f"DESTINATION — {d['name']}\n"
        f"  {d_w['description'].title()} · {d['main']['temp']}{unit_label} · "
        f"Wind {d['wind'].get('speed', 0)} {speed_label}\n"
        f"  Driving: {d_safety['level']} — {d_safety['message']}\n\n"
        f"OVERALL ROUTE ASSESSMENT: {worst['level']}\n"
        f"{worst['advice']}"
    )


# ── Driving safety classifier ──────────────────────────────────────────────────

def _assess_driving_safety(
    weather_id: int,
    wind_speed: float,
    visibility_m: int,
    units: str,
) -> dict:
    """Map OpenWeatherMap condition codes to a 4-level driving safety assessment.

    OWM condition IDs: https://openweathermap.org/weather-conditions
    Levels: CLEAR (0) → ADVISORY (1) → CAUTION (2) → DANGEROUS (3)
    """
    speed_high = 40 if units == "imperial" else 64
    speed_med  = 25 if units == "imperial" else 40

    level  = "CLEAR"
    score  = 0
    msg    = "Favorable driving conditions."
    advice = "No weather-related delays expected."

    # Thunderstorm (2xx)
    if 200 <= weather_id <= 232:
        level, score = "DANGEROUS", 3
        msg    = "Thunderstorm active. Pull over if lightning present."
        advice = (
            "Per FMCSA safety guidelines, consider delaying departure. "
            "Thunderstorms significantly reduce visibility and road traction."
        )
    # Heavy rain (501–531)
    elif 501 <= weather_id <= 531:
        level, score = "CAUTION", 2
        msg    = "Heavy rain — reduced visibility and hydroplaning risk."
        advice = (
            "Reduce speed per 49 CFR 392.14. Increase following distance. "
            "HazMat carriers should verify load securement before proceeding."
        )
    # Light rain / drizzle (3xx, 500)
    elif (300 <= weather_id <= 321) or weather_id == 500:
        level, score = "ADVISORY", 1
        msg    = "Light rain — slightly reduced road grip."
        advice = "Moderate speed reduction recommended. Monitor conditions."
    # Heavy snow (602, 621, 622)
    elif weather_id in (602, 621, 622):
        level, score = "DANGEROUS", 3
        msg    = "Heavy snow — severe driving hazard."
        advice = (
            "FMCSA regulations require driver judgment on hazardous conditions "
            "(49 CFR 392.14). Heavy snow may warrant route delay. "
            "Check state DOT road condition reports before proceeding."
        )
    # Snow (6xx except heavy)
    elif 600 <= weather_id <= 622:
        level, score = "CAUTION", 2
        msg    = "Snow — reduced traction and visibility."
        advice = (
            "Reduce speed, increase following distance. "
            "Verify tire chains if required by state regulations."
        )
    # Fog (741)
    elif weather_id == 741:
        level, score = "CAUTION", 2
        msg    = "Fog — significantly reduced visibility."
        advice = (
            "Per 49 CFR 392.14, if visibility drops below safe level, "
            "driver must pull off road. Use low beams and fog lights."
        )
    # Extreme atmosphere (762=ash, 771=squall, 781=tornado)
    elif weather_id in (762, 771, 781):
        level, score = "DANGEROUS", 3
        msg    = "Extreme atmospheric event — do not drive."
        advice = "Delay departure. Contact dispatch immediately."
    # General atmosphere (7xx)
    elif 700 <= weather_id <= 799:
        level, score = "ADVISORY", 1
        msg    = "Reduced visibility conditions."
        advice = "Use headlights and reduce speed in low-visibility areas."

    # Wind override
    spd_unit = "mph" if units == "imperial" else "kph"
    if wind_speed >= speed_high and score < 3:
        level, score = "DANGEROUS", 3
        msg    = f"Extreme wind ({wind_speed} {spd_unit})."
        advice = (
            "High-profile vehicles (semis, tankers) are at rollover risk. "
            "Per FMCSA, driver must exercise extreme caution or delay. "
            "Check state DOT wind advisories."
        )
    elif wind_speed >= speed_med and score < 2:
        level  = "ADVISORY"
        score  = max(score, 1)
        msg    = f"Strong winds ({wind_speed} {spd_unit}). High-profile vehicles use caution."
        advice = "Reduce speed and increase lateral clearance in open areas."

    # Visibility override
    if visibility_m < 1000 and score < 2:
        level, score = "CAUTION", 2
        msg    = f"Low visibility ({visibility_m} m)."
        advice = "Reduce speed significantly. Use hazard lights if appropriate."

    return {"level": level, "score": score, "message": msg, "advice": advice}


def _wind_direction(degrees: float) -> str:
    """Convert wind bearing in degrees to an 8-point compass label."""
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(degrees / 45) % 8]


# ── Mock data fallbacks ────────────────────────────────────────────────────────

def _mock_weather(location: str, units: str) -> str:
    unit_label  = "°F" if units == "imperial" else "°C"
    speed_label = "mph" if units == "imperial" else "km/h"
    return (
        f"Current weather — {location} *(mock data)*\n"
        f"Condition: Partly Cloudy\n"
        f"Temperature: 58{unit_label} (feels like 54{unit_label})\n"
        f"Humidity: 62%\n"
        f"Wind: 12 {speed_label} NW\n"
        f"Visibility: 16.1 km\n"
        f"Driving safety: CLEAR — Favorable driving conditions."
    )


def _mock_route_weather(origin: str, destination: str, units: str) -> str:
    unit_label  = "°F" if units == "imperial" else "°C"
    speed_label = "mph" if units == "imperial" else "km/h"
    return (
        f"Route weather summary *(mock data)*\n\n"
        f"ORIGIN — {origin}\n"
        f"  Partly Cloudy · 58{unit_label} · Wind 12 {speed_label}\n"
        f"  Driving: CLEAR — Favorable driving conditions.\n\n"
        f"DESTINATION — {destination}\n"
        f"  Light Rain · 52{unit_label} · Wind 18 {speed_label}\n"
        f"  Driving: ADVISORY — Light rain — slightly reduced road grip.\n\n"
        f"OVERALL ROUTE ASSESSMENT: ADVISORY\n"
        f"Moderate speed reduction recommended near destination. Monitor conditions."
    )


# ── Standalone entry point ─────────────────────────────────────────────────────

async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(_main())
