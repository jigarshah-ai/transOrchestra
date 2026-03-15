"""TransOrchestra Streamlit frontend — Logistics Control Tower."""

import os
import sys
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st

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

        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]

            # Source attribution expander.
            sources = meta.get("sources", [])
            if sources:
                with st.expander("📄 Sources"):
                    for src in sources:
                        filename = src.get("filename", "unknown")
                        page = src.get("page", "?")
                        st.markdown(f"- **{filename}** — page {page}")

            # Status badges.
            badge_cols = st.columns([1, 1, 4])
            with badge_cols[0]:
                intent = meta.get("intent", "")
                if intent:
                    st.markdown(
                        f"<span style='background:#2563eb;color:white;padding:2px 8px;"
                        f"border-radius:4px;font-size:0.75rem'>Intent: {intent}</span>",
                        unsafe_allow_html=True,
                    )
            with badge_cols[1]:
                if meta.get("web_search_used"):
                    st.markdown(
                        "<span style='background:#16a34a;color:white;padding:2px 8px;"
                        "border-radius:4px;font-size:0.75rem'>🌐 Web search used</span>",
                        unsafe_allow_html=True,
                    )
            with badge_cols[2]:
                latency = meta.get("latency_ms")
                if latency:
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
                st.markdown(answer)

                meta = {
                    "sources": data.get("sources", []),
                    "intent": data.get("intent", ""),
                    "web_search_used": data.get("web_search_used", False),
                    "latency_ms": data.get("latency_ms"),
                }

                sources = meta["sources"]
                if sources:
                    with st.expander("📄 Sources"):
                        for src in sources:
                            filename = src.get("filename", "unknown")
                            page = src.get("page", "?")
                            st.markdown(f"- **{filename}** — page {page}")

                badge_cols = st.columns([1, 1, 4])
                with badge_cols[0]:
                    if meta["intent"]:
                        st.markdown(
                            f"<span style='background:#2563eb;color:white;padding:2px 8px;"
                            f"border-radius:4px;font-size:0.75rem'>Intent: {meta['intent']}</span>",
                            unsafe_allow_html=True,
                        )
                with badge_cols[1]:
                    if meta["web_search_used"]:
                        st.markdown(
                            "<span style='background:#16a34a;color:white;padding:2px 8px;"
                            "border-radius:4px;font-size:0.75rem'>🌐 Web search used</span>",
                            unsafe_allow_html=True,
                        )
                with badge_cols[2]:
                    if meta["latency_ms"]:
                        st.caption(f"⏱ {meta['latency_ms']} ms")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "meta": meta}
                )

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
