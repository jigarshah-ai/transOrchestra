"""
Google Maps Routes API integration for TransOrchestra.
Provides live routing, distance, travel time, polyline coords,
US state detection, and per-state HazMat compliance notes.
Falls back gracefully to mock data if API key is missing.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

from backend.config import settings

logger = logging.getLogger(__name__)

# ── US state bounding boxes (lat_min, lat_max, lng_min, lng_max) ──────────────
US_STATE_BOUNDS: Dict[str, tuple] = {
    "AL": (30.14, 35.01, -88.47, -84.89),
    "AZ": (31.33, 37.00, -114.82, -109.04),
    "CA": (32.53, 42.01, -124.41, -114.13),
    "CO": (36.99, 41.00, -109.05, -102.04),
    "FL": (24.39, 31.00, -87.63, -79.97),
    "GA": (30.36, 35.00, -85.61, -80.78),
    "IA": (40.37, 43.50, -96.64, -90.14),
    "IL": (36.97, 42.51, -91.51, -87.02),
    "IN": (37.77, 41.76, -88.10, -84.78),
    "KS": (36.99, 40.00, -102.05, -94.59),
    "KY": (36.50, 39.15, -89.57, -81.96),
    "LA": (28.92, 33.02, -94.04, -88.82),
    "MI": (41.70, 48.26, -90.42, -82.41),
    "MN": (43.50, 49.38, -97.24, -89.49),
    "MO": (35.99, 40.61, -95.77, -89.10),
    "MS": (30.17, 35.01, -91.65, -88.10),
    "NC": (33.84, 36.59, -84.32, -75.46),
    "NE": (39.99, 43.00, -104.05, -95.31),
    "NJ": (38.79, 41.36, -75.57, -73.89),
    "NY": (40.50, 45.01, -79.76, -71.86),
    "OH": (38.40, 42.32, -84.82, -80.52),
    "OK": (33.62, 37.00, -103.00, -94.43),
    "PA": (39.72, 42.27, -80.52, -74.69),
    "TN": (34.98, 36.68, -90.31, -81.65),
    "TX": (25.84, 36.50, -106.65, -93.51),
    "VA": (36.54, 39.47, -83.68, -75.17),
    "WI": (42.49, 47.08, -92.89, -86.25),
}

US_STATE_NAMES: Dict[str, str] = {
    "AL": "Alabama",       "AZ": "Arizona",        "CA": "California",
    "CO": "Colorado",      "FL": "Florida",         "GA": "Georgia",
    "IA": "Iowa",          "IL": "Illinois",        "IN": "Indiana",
    "KS": "Kansas",        "KY": "Kentucky",        "LA": "Louisiana",
    "MI": "Michigan",      "MN": "Minnesota",       "MO": "Missouri",
    "MS": "Mississippi",   "NC": "North Carolina",  "NE": "Nebraska",
    "NJ": "New Jersey",    "NY": "New York",        "OH": "Ohio",
    "OK": "Oklahoma",      "PA": "Pennsylvania",    "TN": "Tennessee",
    "TX": "Texas",         "VA": "Virginia",        "WI": "Wisconsin",
}

# ── Per-state HazMat compliance rules ─────────────────────────────────────────
HAZMAT_RULES: Dict[str, Dict] = {
    "IL": {
        "summary": "Illinois requires IDOT HazMat carrier registration.",
        "detail": (
            "Tunnel and route restrictions apply on I-90/94 through Chicago. "
            "Per 625 ILCS 5/18b-105, carriers must register with IDOT annually. "
            "Certain explosives prohibited on specified Chicago metro routes."
        ),
        "permit_required": True,
    },
    "IN": {
        "summary": "Indiana follows federal 49 CFR HazMat rules.",
        "detail": (
            "No additional state HazMat permit required for interstate through traffic. "
            "Intrastate carriers must comply with IC 8-2.1-24. "
            "I-90 Toll Road has specific oversized/HazMat lane rules."
        ),
        "permit_required": False,
    },
    "MI": {
        "summary": "Michigan requires a state HazMat permit for intrastate transport.",
        "detail": (
            "Michigan DNR permit required for Class 1 explosives on state routes. "
            "Ambassador Bridge and Mackinac Bridge have HazMat crossing restrictions. "
            "Contact MDOT for tunnel route alternatives in Detroit metro."
        ),
        "permit_required": True,
    },
    "OH": {
        "summary": "Ohio requires Tier II chemical pre-notification.",
        "detail": (
            "Per Ohio Admin Code 4501:2-1, certain HazMat require pre-notification "
            "to OEPA. I-90 lakefront tunnels near Cleveland restrict certain classes. "
            "Ohio Turnpike follows federal 49 CFR routing rules."
        ),
        "permit_required": False,
    },
    "TX": {
        "summary": "Texas requires a TxDOT oversize/HazMat permit.",
        "detail": (
            "TxDOT requires permit for loads >80,000 lbs or certain HazMat classes. "
            "Houston metro has mandatory HazMat routing through Beltway 8. "
            "Per Texas Transportation Code Ch. 623, annual permit required."
        ),
        "permit_required": True,
    },
    "CA": {
        "summary": "California has the strictest HazMat and emissions rules.",
        "detail": (
            "CARB ATCM requires 2010+ engine model year for drayage trucks. "
            "Bay Bridge, Caldecott Tunnel ban certain HazMat classes. "
            "CHP permit required for Class 1.1-1.3 explosives. "
            "Caltrans Route 58 and Route 138 have specific HazMat restrictions."
        ),
        "permit_required": True,
    },
    "NY": {
        "summary": "New York City has extensive HazMat routing restrictions.",
        "detail": (
            "NYPD HazMat permit required for transport through NYC five boroughs. "
            "Lincoln Tunnel, Holland Tunnel, and Brooklyn Battery Tunnel restrict "
            "flammable liquids and explosives. Per NYC Admin Code 24-505."
        ),
        "permit_required": True,
    },
    "PA": {
        "summary": "Pennsylvania Turnpike has specific HazMat lane restrictions.",
        "detail": (
            "PennDOT requires HazMat carriers to use I-81 corridor for "
            "Class 1 explosives (not PA Turnpike tunnels). "
            "Lehigh Tunnel and Allegheny Mountain Tunnel restrict certain classes."
        ),
        "permit_required": False,
    },
    "FL": {
        "summary": "Florida requires carrier registration for bulk HazMat.",
        "detail": (
            "FHSMV requires Florida HazMat permit for intrastate carriers. "
            "Port of Miami and Port Everglades have specific routing requirements. "
            "Sunshine Skyway Bridge restricts tanker vehicles in high winds."
        ),
        "permit_required": True,
    },
    "GA": {
        "summary": "Georgia follows federal 49 CFR with GDOT registration.",
        "detail": (
            "GDOT requires carrier registration under O.C.G.A. 46-7-85. "
            "Atlanta metro I-285 perimeter has HazMat truck hour restrictions "
            "during peak traffic (7–9 am, 4–7 pm weekdays)."
        ),
        "permit_required": False,
    },
}

# ── Approximate state center coordinates for map markers ──────────────────────
STATE_CENTERS: Dict[str, tuple] = {
    "AL": (32.8, -86.8),  "AZ": (34.3, -111.1), "CA": (36.8, -119.4),
    "CO": (39.0, -105.5), "FL": (28.5, -81.4),  "GA": (32.6, -83.4),
    "IA": (42.0, -93.5),  "IL": (40.0, -89.2),  "IN": (40.3, -86.1),
    "KS": (38.7, -98.4),  "KY": (37.8, -84.3),  "LA": (31.2, -91.8),
    "MI": (44.3, -85.4),  "MN": (46.4, -93.1),  "MO": (38.5, -92.5),
    "MS": (32.7, -89.7),  "NC": (35.5, -79.4),  "NE": (41.5, -99.9),
    "NJ": (40.1, -74.5),  "NY": (42.9, -75.5),  "OH": (40.4, -82.8),
    "OK": (35.6, -96.9),  "PA": (41.2, -77.2),  "TN": (35.9, -86.4),
    "TX": (31.5, -99.3),  "VA": (37.8, -78.2),  "WI": (44.3, -89.8),
}


# ── Google Maps client factory ─────────────────────────────────────────────────

def _get_client():
    """Return an initialised googlemaps.Client or None if key is missing."""
    if not settings.GOOGLE_MAPS_API_KEY:
        return None
    try:
        import googlemaps
        return googlemaps.Client(key=settings.GOOGLE_MAPS_API_KEY)
    except Exception as exc:
        logger.error("Google Maps client init error: %s", exc)
        return None


# ── Public route function ──────────────────────────────────────────────────────

def get_route(origin: str, destination: str) -> Dict:
    """Fetch a driving route between origin and destination.

    Returns a dict with keys:
        success (bool)
        origin, destination (str) — formatted addresses
        distance_miles (float)
        distance_text, duration_text, duration_in_traffic (str)
        polyline_coords (list of [lat, lng])
        start_location, end_location (dict with lat/lng)
        states_crossed (list of state abbreviation strings)
        compliance_notes (list of compliance dicts)
        steps_count (int)
        mock_data (bool) — True when using fallback mock route
    """
    client = _get_client()

    if client is None:
        logger.warning("No Google Maps API key — using mock route data.")
        mock = _build_mock_route(origin, destination)
        mock["api_error"] = "Google Maps API key not configured"
        return mock

    try:
        import googlemaps.convert

        directions = client.directions(
            origin=origin,
            destination=destination,
            mode="driving",
            departure_time=datetime.now(),
            alternatives=False,
            units="imperial",
        )

        if not directions:
            logger.warning("No directions found: %r → %r", origin, destination)
            mock = _build_mock_route(origin, destination)
            mock["api_error"] = "No directions returned by Google Maps"
            return mock

        route = directions[0]
        leg = route["legs"][0]

        raw_polyline = route["overview_polyline"]["points"]
        decoded = googlemaps.convert.decode_polyline(raw_polyline)
        coords: List[List[float]] = [[p["lat"], p["lng"]] for p in decoded]

        states = _detect_states(coords)
        compliance_notes = _build_compliance_notes(states)

        duration_traffic = leg.get("duration_in_traffic", {}).get(
            "text", leg["duration"]["text"]
        )

        return {
            "success":             True,
            "origin":              leg["start_address"],
            "destination":         leg["end_address"],
            "distance_miles":      round(leg["distance"]["value"] * 0.000621371, 1),
            "distance_text":       leg["distance"]["text"],
            "duration_text":       leg["duration"]["text"],
            "duration_in_traffic": duration_traffic,
            "polyline_coords":     coords,
            "start_location":      leg["start_location"],
            "end_location":        leg["end_location"],
            "states_crossed":      states,
            "compliance_notes":    compliance_notes,
            "steps_count":         len(leg["steps"]),
            "mock_data":           False,
        }

    except Exception as exc:
        logger.error("Google Maps API call failed: %s", exc)
        # Provide mock data so UI stays usable, but include an error for the agent to explain.
        mock = _build_mock_route(origin, destination)
        mock["api_error"] = f"Maps API currently unavailable: {exc}"
        return mock


# ── Internal helpers ───────────────────────────────────────────────────────────

def _detect_states(coords: List[List[float]]) -> List[str]:
    """Sample route coordinates to detect which US states are crossed.

    Samples every 8th point for efficiency on long routes.
    """
    detected: List[str] = []
    sample = coords[::8] if len(coords) >= 8 else coords

    for point in sample:
        lat, lng = point[0], point[1]
        for code, (lat_min, lat_max, lng_min, lng_max) in US_STATE_BOUNDS.items():
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                if code not in detected:
                    detected.append(code)
    return detected


def _build_compliance_notes(states: List[str]) -> List[Dict]:
    """Build a HazMat compliance note dict for each detected state that has rules."""
    notes = []
    for code in states:
        if code in HAZMAT_RULES:
            rule = HAZMAT_RULES[code]
            notes.append({
                "state_code":      code,
                "state":           US_STATE_NAMES.get(code, code),
                "summary":         rule["summary"],
                "detail":          rule["detail"],
                "permit_required": rule["permit_required"],
                "center":          list(STATE_CENTERS.get(code, (39.5, -98.35))),
            })
    return notes


def _build_mock_route(origin: str, destination: str) -> Dict:
    """Return a realistic Chicago → Detroit mock route.

    Used when GOOGLE_MAPS_API_KEY is not configured.
    Polyline approximates I-94 E between the two cities.
    """
    coords: List[List[float]] = [
        [41.8781, -87.6298],
        [41.9200, -87.3500],
        [41.6764, -86.2520],
        [41.6833, -85.9833],
        [41.7500, -85.5000],
        [41.9000, -84.8000],
        [42.1000, -84.2000],
        [42.2314, -83.5000],
        [42.3314, -83.0458],
    ]
    states = ["IL", "IN", "MI"]
    return {
        "success":             True,
        "origin":              origin or "Chicago, IL, USA",
        "destination":         destination or "Detroit, MI, USA",
        "distance_miles":      281.4,
        "distance_text":       "281 miles",
        "duration_text":       "4 hours 12 mins",
        "duration_in_traffic": "4 hours 35 mins (with current traffic)",
        "polyline_coords":     coords,
        "start_location":      {"lat": 41.8781, "lng": -87.6298},
        "end_location":        {"lat": 42.3314, "lng": -83.0458},
        "states_crossed":      states,
        "compliance_notes":    _build_compliance_notes(states),
        "steps_count":         12,
        "mock_data":           True,
    }
