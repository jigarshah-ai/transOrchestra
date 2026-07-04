"""TransOrchestra Streamlit frontend — logistics control tower UI.

This module is a **thin presentation layer** over the FastAPI backend:

    - Renders chat history, compliance maps (Folium), weather cards, dual LLM
      comparison expanders, and Graphviz agent-flow diagrams returned as JSON.
    - Manages ``st.session_state`` for thread ids, chat transcripts, and a
      **consume-once** image pipeline: base64 is attached only while
      ``image_ready_for_extraction`` is true so text questions after a document
      upload are not re-routed to the vision agent indefinitely.

Run locally with::

    streamlit run frontend/app.py

Ensure the API is available at ``BACKEND_URL`` (default ``http://localhost:8000/api/v1``).
"""

import base64
import hashlib
import os
import sys
from pathlib import Path
from uuid import uuid4

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

# Streamlit executes this file as a script — prepend repo root so `backend.*`
# imports resolve identically to FastAPI's PYTHONPATH layout.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.config import settings

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

def render_weather_card(weather_data: dict) -> None:
    """Render a colour-coded weather card with safety level indicator.

    Safety levels:
      CLEAR     → green card   ✅
      ADVISORY  → blue card    🔵
      CAUTION   → amber card   ⚠️
      DANGEROUS → red card     🚨

    Shown alongside the folium map for route_query responses only.

    Args:
        weather_data: Navigator payload containing ``raw_text``, ``overall_safety``,
            and optional ``is_mock`` flag.

    Returns:
        None: Renders inline via ``st.markdown``.

    Raises:
        None
    """
    if not weather_data or not weather_data.get("raw_text"):
        return

    safety = weather_data.get("overall_safety", "CLEAR")
    cfg = {
        "CLEAR":     {"bg": "#EAF3DE", "border": "#639922", "icon": "✅", "label": "Clear"},
        "ADVISORY":  {"bg": "#E6F1FB", "border": "#185FA5", "icon": "🔵", "label": "Advisory"},
        "CAUTION":   {"bg": "#FAEEDA", "border": "#BA7517", "icon": "⚠️",  "label": "Caution"},
        "DANGEROUS": {"bg": "#FCEBEB", "border": "#A32D2D", "icon": "🚨", "label": "Dangerous"},
    }.get(safety, {"bg": "#EAF3DE", "border": "#639922", "icon": "✅", "label": "Clear"})

    mock_note = " *(mock — set OPENWEATHERMAP_API_KEY for live data)*" \
                if weather_data.get("is_mock") else ""

    st.markdown(
        f"""<div style="
            background:{cfg['bg']};
            border-left:4px solid {cfg['border']};
            border-radius:8px;
            padding:12px 16px;
            margin:8px 0 12px 0;
        ">
        <b>{cfg['icon']} Weather: {cfg['label']}{mock_note}</b>
        <pre style="margin:8px 0 0 0;font-size:12px;background:transparent;
                    border:none;white-space:pre-wrap;">{weather_data['raw_text']}</pre>
        </div>""",
        unsafe_allow_html=True,
    )


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


def render_model_comparison(model_comparison: dict) -> None:
    """Render side-by-side open vs closed model outputs from ``llm_comparison``.

    Args:
        model_comparison: Backend dict with ``open`` / ``closed`` sub-results.

    Returns:
        None

    Raises:
        None
    """
    if not model_comparison:
        return
    open_out = model_comparison.get("open", {})
    closed_out = model_comparison.get("closed", {})
    if not open_out and not closed_out:
        return

    with st.expander("🧪 Open vs Closed Model Outputs"):
        left, right = st.columns(2)

        with left:
            st.markdown("**Open Model**")
            st.caption(
                f"Model: `{open_out.get('model', 'unknown')}`  ·  "
                f"Latency: `{open_out.get('latency_ms', 0)} ms`"
            )
            if open_out.get("error"):
                st.error(f"Error: {open_out['error']}")
            else:
                st.markdown(open_out.get("answer", "_No output_"))

        with right:
            st.markdown("**Closed Model**")
            st.caption(
                f"Model: `{closed_out.get('model', 'unknown')}`  ·  "
                f"Latency: `{closed_out.get('latency_ms', 0)} ms`"
            )
            if closed_out.get("error"):
                st.error(f"Error: {closed_out['error']}")
            else:
                st.markdown(closed_out.get("answer", "_No output_"))


def _dot_escape(s: str) -> str:
    """Escape characters that would break Graphviz string literals.

    Args:
        s: Raw label fragment.

    Returns:
        str: DOT-safe escaped string.

    Raises:
        None
    """
    return (s.replace("\\", "\\\\").replace('"', '\\"') if s else "")


def render_flow_viz(flow_data: dict) -> None:
    """Render a compact Graphviz diagram with the active execution path highlighted.

    Args:
        flow_data: Must include ``nodes``, ``edges``, and ``execution_path`` from API.

    Returns:
        None

    Raises:
        None
    """
    if not flow_data:
        return
    nodes = flow_data.get("nodes", [])
    edges = flow_data.get("edges", [])
    execution_path = set(flow_data.get("execution_path", []))

    if not nodes and not edges:
        return

    # Tight Graphviz directives keep the diagram legible inside Streamlit without
    # excessive vertical whitespace (portfolio/demo UX requirement).
    lines = [
        "digraph G {",
        "  rankdir=LR;",
        "  size=\"6,2\";", # Restrict the overall physical size of the image
        "  graph [margin=0, pad=0.1, ranksep=0.3, nodesep=0.1];",
        "  node [shape=box, style=rounded, margin=0.05, height=0.3, fontsize=10, fontname=\"Helvetica\"];",
        "  edge [fontname=\"Helvetica\", fontsize=8, arrowsize=0.5];",
    ]
    for n in nodes:
        nid = _dot_escape(n.get("id", ""))
        label = _dot_escape(n.get("label", nid))
        tools = n.get("tools", [])
        if tools:
            tool_str = "\\n".join(_dot_escape(t) for t in tools[:1])
            label = f"{label}\\n({tool_str})"
        if nid in execution_path:
            lines.append(
                f'  "{nid}" [label="{label}", style="filled", fillcolor="#bae6fd", color="#2563eb", penwidth=1];'
            )
        else:
            lines.append(f'  "{nid}" [label="{label}"];')
    for e in edges:
        fr = _dot_escape(e.get("from", ""))
        to = _dot_escape(e.get("to", ""))
        lbl = _dot_escape(e.get("label", ""))
        if lbl:
            lines.append(f'  "{fr}" -> "{to}" [label="{lbl}"];')
        else:
            lines.append(f'  "{fr}" -> "{to}";')
    lines.append("}")

    dot = "\n".join(lines)
    
    # Do not expand this by default either to save vertical space
    with st.expander("🔄 Agent flow (this request)", expanded=False):
        st.caption("Filled nodes = path taken. Labels on edges = intent branch.")
        # Removed use_container_width=True so it respects the size="6,2" rule above
        st.graphviz_chart(dot)

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
    st.caption(f"**Active embedding:** {settings.EMBEDDING_MODEL}")
    st.caption("**Vector store:** ChromaDB (local)")

    st.divider()
    st.header("📷 Intelligent Document Clerk (Image)")
    doc_image = st.file_uploader(
        "Upload BOL / receipt image (PNG/JPG/JPEG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help="When you upload an image, the backend will extract structured logistics fields using gpt-4o vision.",
    )
    if doc_image:
        image_bytes = doc_image.getvalue()
        st.session_state.image_filename = getattr(doc_image, "name", "uploaded_image")
        st.session_state.image_bytes_preview = image_bytes

        image_sha = hashlib.sha256(image_bytes).hexdigest()
        prev_sha = st.session_state.get("image_sha256")
        if prev_sha != image_sha:
            # Fresh file bytes → allow one backend round-trip with image payload.
            st.session_state.image_sha256 = image_sha
            st.session_state.image_ready_for_extraction = True

        if st.session_state.get("image_ready_for_extraction", False):
            st.session_state.image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        else:
            # Consume-once: clear payload so dispatcher stops forcing document flow.
            st.session_state.image_base64 = None
        # Streamlit uses `use_column_width` for images (older versions don't support `use_container_width`)
        st.image(image_bytes, caption=st.session_state.image_filename, use_column_width=True)
        st.caption("Image ready for extraction — ask your question below.")
    else:
        # Keep any previously uploaded image in session unless user clears the conversation.
        st.session_state.setdefault("image_base64", None)
        st.session_state.setdefault("image_bytes_preview", None)
        st.session_state.setdefault("image_ready_for_extraction", False)
        st.session_state.setdefault("image_sha256", None)

    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid4())
        st.session_state.image_base64 = None
        st.session_state.image_filename = None
        st.session_state.image_bytes_preview = None
        st.session_state.image_ready_for_extraction = False
        st.session_state.image_sha256 = None
        st.rerun()

# ── Main area header ───────────────────────────────────────────────────────────
st.title("🚛 TransOrchestra Control Tower")
st.subheader(
    "Ask about FMCSA regulations, routes, HazMat compliance, vehicle maintenance"
)
st.divider()

# If an image was uploaded in the sidebar, show a small preview in the chat area too.
if st.session_state.get("image_bytes_preview"):
    fname = st.session_state.get("image_filename", "uploaded_image")
    st.caption(f"📷 Document attached: {fname} (will be extracted on your next message)")
    st.image(st.session_state["image_bytes_preview"], caption="Uploaded document preview", width=220)

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            msg_ = msg
            # Folium map — only for route queries
            if msg_.get("route_data"):
                render_route_map(msg_["route_data"])

            # Weather card — shown alongside map for route queries
            if msg_.get("weather_data"):
                render_weather_card(msg_["weather_data"])
            if msg_.get("llm_comparison"):
                render_model_comparison(msg_["llm_comparison"])
            if msg_.get("flow_data"):
                render_flow_viz(msg_["flow_data"])
            if msg_.get("extracted_doc_data"):
                doc_data = msg_["extracted_doc_data"]
                with st.expander("📄 Document Extraction Results", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Origin", doc_data.get("origin", "N/A"))
                        st.metric("Weight", doc_data.get("weight", "N/A"))
                    with col2:
                        st.metric("Destination", doc_data.get("destination", "N/A"))
                        st.metric("Freight Class", doc_data.get("freight_class", "N/A"))
                    st.caption(f"**Document Type:** {doc_data.get('document_type', 'N/A')}")
                    st.caption(f"**Summary:** {doc_data.get('summary', '')}")

            # Source citations
            sources = (msg_.get("sources", []) or [])[:3]
            if sources:
                with st.expander("📄 Sources"):
                    for src in sources:
                        filename = src.get("filename", "unknown")
                        page = src.get("page", "?")
                        st.markdown(f"- **{filename}** — page {page}")

            # Status badges
            badge_cols = st.columns([1, 1, 1, 1, 2])
            intent = msg.get("intent", "")
            if intent:
                with badge_cols[0]:
                    st.markdown(
                        f"<span style='background:#2563eb;color:white;padding:2px 8px;"
                        f"border-radius:4px;font-size:0.75rem'>Intent: {intent}</span>",
                        unsafe_allow_html=True,
                    )
            agent_used = msg.get("agent_used", "")
            if agent_used:
                with badge_cols[1]:
                    st.markdown(
                        f"<span style='background:#0f766e;color:white;padding:2px 8px;"
                        f"border-radius:4px;font-size:0.75rem'>Agent: {agent_used}</span>",
                        unsafe_allow_html=True,
                    )
            if msg.get("web_search_used"):
                with badge_cols[2]:
                    st.markdown(
                        "<span style='background:#16a34a;color:white;padding:2px 8px;"
                        "border-radius:4px;font-size:0.75rem'>🌐 Web search used</span>",
                        unsafe_allow_html=True,
                    )
            rel = msg.get("relevance_score")
            if rel is not None:
                with badge_cols[3]:
                    st.markdown(
                        f"<span style='background:#111827;color:white;padding:2px 8px;"
                        f"border-radius:4px;font-size:0.75rem'>Relevance: {rel:.3f}</span>",
                        unsafe_allow_html=True,
                    )
            latency = msg.get("latency_ms")
            if latency:
                with badge_cols[4]:
                    st.caption(f"⏱ {latency} ms")

            # Model + provider info (small caption line)
            llm_model = msg.get("llm_model")
            embed_model = msg.get("embedding_model")
            llm_provider = msg.get("llm_provider", settings.LLM_PROVIDER)
            if llm_model or embed_model:
                provider_label = "OpenRouter" if llm_provider == "openrouter" else "OpenAI"
                st.caption(
                    f"Model: `{llm_model or 'unknown'}`  ·  Provider: `{provider_label}`  ·  "
                    f"Embeddings: `{embed_model or 'unknown'}`"
                )

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
                        "image_base64": st.session_state.get("image_base64"),
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("answer", "No answer returned.")
                route_data = data.get("route_data")
                weather_data = data.get("weather_data")
                sources = data.get("sources", [])
                intent = data.get("intent", "")
                agent_used = data.get("agent_used", "")
                web_search_used = data.get("web_search_used", False)
                latency_ms = data.get("latency_ms")
                relevance_score = data.get("relevance_score")
                llm_model = data.get("llm_model")
                embedding_model = data.get("embedding_model")
                llm_provider = data.get("llm_provider")
                model_comparison = data.get("llm_comparison")
                flow_data = data.get("flow_data")
                extracted_doc_data = data.get("extracted_doc_data")

                st.markdown(answer)

                # Folium map — only rendered when navigator returned route_data
                if route_data:
                    render_route_map(route_data)

                # Weather card — only rendered for route queries
                if weather_data:
                    render_weather_card(weather_data)
                if model_comparison:
                    render_model_comparison(model_comparison)
                if flow_data:
                    render_flow_viz(flow_data)
                if extracted_doc_data:
                    with st.expander("📄 Document Extraction Results", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Origin", extracted_doc_data.get("origin", "N/A"))
                            st.metric("Weight", extracted_doc_data.get("weight", "N/A"))
                        with col2:
                            st.metric("Destination", extracted_doc_data.get("destination", "N/A"))
                            st.metric(
                                "Freight Class",
                                extracted_doc_data.get("freight_class", "N/A"),
                            )
                        st.caption(
                            f"**Document Type:** {extracted_doc_data.get('document_type', 'N/A')}"
                        )
                        st.caption(f"**Summary:** {extracted_doc_data.get('summary', '')}")
                    # Consume-once: after successful extraction, don't route future queries to document_agent
                    # unless the user uploads a new image.
                    st.session_state.image_base64 = None
                    st.session_state.image_ready_for_extraction = False

                # Source citations
                sources = (sources or [])[:3]
                if sources:
                    with st.expander("📄 Sources"):
                        for src in sources:
                            filename = src.get("filename", "unknown")
                            page = src.get("page", "?")
                            st.markdown(f"- **{filename}** — page {page}")

                # Status badges
                badge_cols = st.columns([1, 1, 1, 1, 3])
                if intent:
                    with badge_cols[0]:
                        st.markdown(
                            f"<span style='background:#2563eb;color:white;padding:2px 8px;"
                            f"border-radius:4px;font-size:0.75rem'>Intent: {intent}</span>",
                            unsafe_allow_html=True,
                        )
                if agent_used:
                    with badge_cols[1]:
                        st.markdown(
                            f"<span style='background:#0f766e;color:white;padding:2px 8px;"
                            f"border-radius:4px;font-size:0.75rem'>Agent: {agent_used}</span>",
                            unsafe_allow_html=True,
                        )
                if web_search_used:
                    with badge_cols[2]:
                        st.markdown(
                            "<span style='background:#16a34a;color:white;padding:2px 8px;"
                            "border-radius:4px;font-size:0.75rem'>🌐 Web search used</span>",
                            unsafe_allow_html=True,
                        )
                if relevance_score is not None:
                    with badge_cols[3]:
                        st.markdown(
                            f"<span style='background:#111827;color:white;padding:2px 8px;"
                            f"border-radius:4px;font-size:0.75rem'>Relevance: {float(relevance_score):.3f}</span>",
                            unsafe_allow_html=True,
                        )
                if latency_ms:
                    with badge_cols[4]:
                        st.caption(f"⏱ {latency_ms} ms")

                if llm_model or embedding_model:
                    provider_label = "OpenRouter" if llm_provider == "openrouter" else "OpenAI"
                    st.caption(
                        f"Model: `{llm_model or 'unknown'}`  ·  Provider: `{provider_label}`  ·  "
                        f"Embeddings: `{embedding_model or 'unknown'}`"
                    )

                # Persist message with route + weather data for re-render on Streamlit rerun
                st.session_state.messages.append({
                    "role":            "assistant",
                    "content":         answer,
                    "sources":         sources,
                    "intent":          intent,
                    "agent_used":      agent_used,
                    "web_search_used": web_search_used,
                    "latency_ms":      latency_ms,
                    "route_data":      route_data,
                    "weather_data":    weather_data,
                    "relevance_score": relevance_score,
                    "llm_model":       llm_model,
                    "embedding_model": embedding_model,
                    "llm_provider":    llm_provider,
                    "llm_comparison":  model_comparison,
                    "extracted_doc_data": extracted_doc_data,
                    "flow_data":       flow_data,
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
