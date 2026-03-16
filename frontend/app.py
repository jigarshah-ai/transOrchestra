"""TransOrchestra Streamlit frontend — Logistics Control Tower."""

import os
import sys
from pathlib import Path
from uuid import uuid4

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

# Allow importing backend config from anywhere.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.config import EMBEDDING_MODEL

BACKEND_URL = "http://localhost:8000/api/v1"
DATA_DIR = project_root / "data"

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TransOrchestra — Logistics Control Tower",
    page_icon="🚛",
    layout="wide",
)

# ── Session state initialisation ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())


# ── Folium map helper ──────────────────────────────────────────────────────────

def render_route_map(route_data: dict) -> None:
    """Render an interactive folium map for the given route.

    Shows:
    - Blue route polyline with distance/time tooltip
    - Green origin marker and red destination marker
    - Orange compliance warning markers per state (with HazMat detail popups)
    Only called when route_data is present — never for RAG answers.
    """
    if not route_data or not route_data.get("polyline_coords"):
        return

    coords = route_data["polyline_coords"]
    if not coords:
        return

    mid = coords[len(coords) // 2]
    m = folium.Map(location=[mid[0], mid[1]], zoom_start=7, tiles="CartoDB positron")

    # Route polyline
    folium.PolyLine(
        locations=coords,
        color="#185FA5",
        weight=5,
        opacity=0.85,
        tooltip=(
            f"{route_data['distance_text']}  ·  "
            f"{route_data['duration_in_traffic']}"
        ),
    ).add_to(m)

    # Origin marker (green)
    start = route_data["start_location"]
    folium.Marker(
        location=[start["lat"], start["lng"]],
        popup=folium.Popup(
            f"<b>Origin</b><br>{route_data['origin']}", max_width=240
        ),
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)

    # Destination marker (red)
    end = route_data["end_location"]
    folium.Marker(
        location=[end["lat"], end["lng"]],
        popup=folium.Popup(
            f"<b>Destination</b><br>{route_data['destination']}", max_width=240
        ),
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(m)

    # Compliance warning markers (orange) per state
    for note in route_data.get("compliance_notes", []):
        center_pt = note.get("center", [39.5, -98.35])
        permit_txt = " — PERMIT REQUIRED" if note.get("permit_required") else ""
        popup_html = (
            f"<b>{note['state']}{permit_txt}</b><br>"
            f"{note['summary']}<br><br>"
            f"<small>{note['detail'][:200]}…</small>"
        )
        folium.Marker(
            location=center_pt,
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color="orange", icon="warning-sign", prefix="glyphicon"),
        ).add_to(m)

    # Auto-fit bounds to the full route
    lats = [c[0] for c in coords]
    lngs = [c[1] for c in coords]
    m.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])

    mock_note = (
        " *(mock data — set `GOOGLE_MAPS_API_KEY` for live routing)*"
        if route_data.get("mock_data") else ""
    )
    st.caption(
        f"🗺️ {route_data['distance_text']}  ·  "
        f"{route_data['duration_in_traffic']}{mock_note}"
    )
    st_folium(m, height=420, use_container_width=True)

    permit_needed = [
        n["state"] for n in route_data.get("compliance_notes", [])
        if n.get("permit_required")
    ]
    if permit_needed:
        st.warning(
            f"⚠️ **Permit required in: {', '.join(permit_needed)}** — "
            "review compliance notes in the route details above."
        )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader(
        "Upload regulatory PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload FMCSA/DOT regulatory documents to query against.",
    )

    if st.button("Ingest Documents", type="primary", disabled=not uploaded_files):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for uf in uploaded_files:
            dest = DATA_DIR / uf.name
            dest.write_bytes(uf.read())
            saved_paths.append(str(dest))

        with st.spinner("Ingesting documents into ChromaDB…"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/ingest",
                    json={"pdf_paths": saved_paths},
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()
                st.success(
                    f"✅ Ingested {len(saved_paths)} document(s) "
                    f"({data['chunks_created']} chunks)"
                )
            except requests.exceptions.ConnectionError:
                st.warning(
                    "⚠️ Backend not running. Start with:\n"
                    "`uvicorn backend.main:app --reload`"
                )
            except Exception as exc:
                st.error(f"Ingestion error: {exc}")

    st.divider()
    st.caption(f"**Active embedding:** {EMBEDDING_MODEL}")
    st.caption("**Vector store:** ChromaDB (local)")

    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid4())
        st.rerun()

# ── Main area header ───────────────────────────────────────────────────────────
st.title("🚛 TransOrchestra Control Tower")
st.subheader(
    "Ask about FMCSA regulations, routes, HazMat compliance, vehicle maintenance"
)
st.divider()

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            # Folium map — only for route queries
            if msg.get("route_data"):
                render_route_map(msg["route_data"])

            # Source citations
            sources = msg.get("sources", [])
            if sources:
                with st.expander("📄 Sources"):
                    for src in sources:
                        filename = src.get("filename", "unknown")
                        page = src.get("page", "?")
                        st.markdown(f"- **{filename}** — page {page}")

            # Status badges
            badge_cols = st.columns([1, 1, 4])
            intent = msg.get("intent", "")
            if intent:
                with badge_cols[0]:
                    st.markdown(
                        f"<span style='background:#2563eb;color:white;padding:2px 8px;"
                        f"border-radius:4px;font-size:0.75rem'>Intent: {intent}</span>",
                        unsafe_allow_html=True,
                    )
            if msg.get("web_search_used"):
                with badge_cols[1]:
                    st.markdown(
                        "<span style='background:#16a34a;color:white;padding:2px 8px;"
                        "border-radius:4px;font-size:0.75rem'>🌐 Web search used</span>",
                        unsafe_allow_html=True,
                    )
            latency = msg.get("latency_ms")
            if latency:
                with badge_cols[2]:
                    st.caption(f"⏱ {latency} ms")

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask about regulations, routes, HazMat, maintenance…",
    key="chat_input",
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing query…"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/query",
                    json={
                        "query": user_input,
                        "thread_id": st.session_state.thread_id,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("answer", "No answer returned.")
                route_data = data.get("route_data")
                sources = data.get("sources", [])
                intent = data.get("intent", "")
                web_search_used = data.get("web_search_used", False)
                latency_ms = data.get("latency_ms")

                st.markdown(answer)

                # Folium map — only rendered when navigator returned route_data
                if route_data:
                    render_route_map(route_data)

                # Source citations
                if sources:
                    with st.expander("📄 Sources"):
                        for src in sources:
                            filename = src.get("filename", "unknown")
                            page = src.get("page", "?")
                            st.markdown(f"- **{filename}** — page {page}")

                # Status badges
                badge_cols = st.columns([1, 1, 4])
                if intent:
                    with badge_cols[0]:
                        st.markdown(
                            f"<span style='background:#2563eb;color:white;padding:2px 8px;"
                            f"border-radius:4px;font-size:0.75rem'>Intent: {intent}</span>",
                            unsafe_allow_html=True,
                        )
                if web_search_used:
                    with badge_cols[1]:
                        st.markdown(
                            "<span style='background:#16a34a;color:white;padding:2px 8px;"
                            "border-radius:4px;font-size:0.75rem'>🌐 Web search used</span>",
                            unsafe_allow_html=True,
                        )
                if latency_ms:
                    with badge_cols[2]:
                        st.caption(f"⏱ {latency_ms} ms")

                # Persist message including route_data for map re-render on rerun
                st.session_state.messages.append({
                    "role":            "assistant",
                    "content":         answer,
                    "sources":         sources,
                    "intent":          intent,
                    "web_search_used": web_search_used,
                    "latency_ms":      latency_ms,
                    "route_data":      route_data,
                })

            except requests.exceptions.ConnectionError:
                warning_msg = (
                    "⚠️ Backend not running. Start with:\n\n"
                    "```\nuvicorn backend.main:app --reload\n```"
                )
                st.warning(warning_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": warning_msg}
                )
            except Exception as exc:
                error_msg = f"❌ Error: {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

    st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "TransOrchestra v1.0 | Powered by LangGraph + RAG | "
    "Capstone Project — Analytics Vidya GenAI Pinnacle"
)
