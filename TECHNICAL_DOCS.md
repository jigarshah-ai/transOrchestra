# TransOrchestra — Technical Documentation

> Deep-dive reference for developers, evaluators, and capstone reviewers.

---

## Table of Contents

1. [Configuration Reference](#1-configuration-reference)
2. [Module Reference — RAG Pipeline](#2-module-reference--rag-pipeline)
3. [Module Reference — Agents](#3-module-reference--agents)
4. [Module Reference — Tools](#4-module-reference--tools)
5. [Module Reference — MCP Servers](#5-module-reference--mcp-servers)
6. [Module Reference — API](#6-module-reference--api)
7. [LangGraph State & Flow](#7-langgraph-state--flow)
8. [Embedding Model Comparison](#8-embedding-model-comparison)
9. [Retrieval Strategy Deep-Dive](#9-retrieval-strategy-deep-dive)
10. [Corrective RAG Flow](#10-corrective-rag-flow)
11. [Navigator Agent — Google Maps + Weather Flow](#11-navigator-agent--google-maps--weather-flow)
12. [Observability & Monitoring — LangSmith](#12-observability--monitoring--langsmith)
13. [Evaluation Framework](#13-evaluation-framework)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Configuration Reference

All configuration is loaded via **Pydantic Settings** (`BaseSettings`) in `backend/config.py` and exposed as a singleton `settings` object.

`.env` discovery is production-hardened for your workspace layout:
- Prefer `transOrchestra/.env`
- Also supports `Final_Project/.env` (one level above the repo)

| Variable | Default | Required | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | — | ✅ Yes | OpenAI API key for LLM + optional embeddings |
| `COHERE_API_KEY` | — | ❌ Optional | Enables Cohere reranking. Falls back to hybrid if missing |
| `TAVILY_API_KEY` | — | ❌ Optional | Enables corrective web search. Disabled if missing |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | — | Directory where ChromaDB persists to disk |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | — | Default embedding model. Prefix `text-embedding` to use OpenAI |
| `LLM_MODEL` | `gpt-4o-mini` | — | OpenAI chat model for all LLM calls |
| `CHUNK_SIZE` | `512` | — | Max tokens per document chunk |
| `CHUNK_OVERLAP` | `50` | — | Token overlap between consecutive chunks |
| `TOP_K_RETRIEVAL` | `10` | — | Number of documents fetched by vector/BM25 retriever |
| `TOP_K_RERANK` | `3` | — | Number of documents after Cohere reranking |
| `RELEVANCE_SCORE_THRESHOLD` | `0.6` | — | Minimum avg score before Tavily fallback triggers |
| `GOOGLE_MAPS_API_KEY` | — | ❌ Optional | Google Maps Directions API key. Mock route used if missing |
| `OPENWEATHERMAP_API_KEY` | — | ❌ Optional | OpenWeatherMap API key. Mock weather card used if missing |
| `WEATHER_UNITS` | `imperial` | — | `imperial` = °F/mph — `metric` = °C/kph |
| `LANGCHAIN_TRACING_V2` | `false` | ❌ Optional | Set `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | — | ❌ Optional | LangSmith API key (free at smith.langchain.com) |
| `LANGCHAIN_PROJECT` | `transOrchestra` | — | LangSmith project name for trace grouping |
| `LANGCHAIN_ENDPOINT` | `https://api.smith.langchain.com` | — | LangSmith ingest endpoint |

---

## 2. Module Reference — RAG Pipeline

### `backend/rag/loader.py`

**`load_and_chunk_pdfs(pdf_dir: str) → List[Document]`**
- Scans `pdf_dir` for all `.pdf` files
- Loads each with `PyPDFLoader` (pypdf backend)
- Splits with `RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)`
- Injects `metadata["source"] = filename` and `metadata["page"] = page_number` on every chunk
- Logs file count and total chunks
- Raises `FileNotFoundError` if directory does not exist

**`load_single_pdf(file_path: str) → List[Document]`**
- Same logic as above but for a single file path
- Used by the Streamlit sidebar upload flow and the `/ingest` API endpoint

---

### `backend/rag/embeddings.py`

**`get_embedding_model(model_name: str = None) → Embeddings`**

Routing logic:
```
model_name starts with "text-embedding"  →  OpenAIEmbeddings(model=model_name)
anything else                            →  HuggingFaceEmbeddings(model_name, device="cpu", normalize=True)
model_name is None                       →  reads EMBEDDING_MODEL from settings
```

**`compare_embeddings(query: str, docs: List[Document]) → dict`**
- Embeds query + docs with both BGE and OpenAI
- Returns top-3 cosine-similarity results from each
- Output shape: `{"bge": [...], "openai": [...]}`
- Used for the capstone embedding comparison experiment

---

### `backend/rag/vectorstore.py`

ChromaDB collection name: `transOrchestra_docs`

**`build_vectorstore(docs, embedding_model) → Chroma`**
- Creates/overwrites ChromaDB collection
- Persists to `CHROMA_PERSIST_DIR`
- Returns populated Chroma instance

**`load_vectorstore(embedding_model) → Chroma`**
- Loads existing collection from disk
- Raises `FileNotFoundError("No vector store found. Run scripts/ingest.py first.")` if missing

**`get_or_create_vectorstore(docs, embedding_model) → Chroma`**
- Tries `load_vectorstore` first
- Falls back to `build_vectorstore` if not found
- Used by the FastAPI startup event

---

### `backend/rag/retriever.py`

**`build_vector_retriever(vectorstore, top_k=TOP_K_RETRIEVAL) → BaseRetriever`**
- `vectorstore.as_retriever(search_kwargs={"k": top_k})`
- Standard cosine similarity search in ChromaDB

**`build_bm25_retriever(docs, top_k=TOP_K_RETRIEVAL) → BM25Retriever`**
- `BM25Retriever.from_documents(docs)` from `langchain_community`
- Corpus is the full document set from ChromaDB (reconstructed at load time)

**`build_hybrid_retriever(vectorstore, docs) → EnsembleRetriever`**
- Weights: BM25=0.4, Vector=0.6
- Reciprocal Rank Fusion merges the two ranked lists

**`build_reranking_retriever(base_retriever, docs) → BaseRetriever`**
- Wraps any retriever with `ContextualCompressionRetriever`
- Compressor: `CohereRerank(model="rerank-english-v3.0", top_n=TOP_K_RERANK)`
- **Graceful fallback**: returns `base_retriever` unchanged if `COHERE_API_KEY` is empty

---

### `backend/rag/pipeline.py`

**`build_rag_chain(retriever) → RunnableSequence`**

LCEL chain structure:
```python
chain = (
    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    | ChatPromptTemplate(system=SYSTEM_PROMPT, human="{context}\n\nQuestion: {question}")
    | ChatOpenAI(model=LLM_MODEL, temperature=0)
    | StrOutputParser()
)
```

System prompt enforces:
- Answer only from provided context documents
- Always cite document and section
- Return "I don't have enough information" if context is insufficient
- Never fabricate regulation codes

**`run_query_with_docs(chain, retriever, query) → dict`**
- Invokes chain for the answer
- Separately invokes retriever to collect source metadata
- Returns: `{"answer": str, "sources": [{"filename": str, "page": int}], "num_docs_retrieved": int}`

**`check_relevance_score(retriever, query) → float`**
- Unwraps `EnsembleRetriever` to access the underlying `Chroma` vector store
- Calls `similarity_search_with_relevance_scores(query, k=5)`
- Returns mean score (0.0–1.0)
- Returns 1.0 on failure so the corrective fallback doesn't trigger unnecessarily

---

## 3. Module Reference — Agents

### `backend/agents/state.py`

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # LangGraph message history
    query: str                                # current user query
    image_data: Optional[str]                # base64-encoded image for document processing
    intent: str                               # dispatcher output
    vehicle_id: str                           # optional conversation context
    cargo_type: str                           # optional conversation context
    route_constraints: dict                   # optional conversation context
    rag_answer: str                           # answer from RAG pipeline
    rag_sources: list                         # source citations
    web_search_used: bool                     # Tavily was triggered
    final_answer: str                         # final response to user
    extracted_doc_data: Optional[dict]      # structured extraction result for document_processing
    thread_id: str                            # MemorySaver thread key
    route_data: Optional[dict]                # navigator_agent route + polyline; None for all other intents
    weather_data: Optional[dict]              # MCP weather server result; None for non-route intents
    relevance_score: Optional[float]          # safety_agent relevance score (0.0–1.0); None for route intents
```

---

### `backend/agents/dispatcher.py`

**`dispatcher_node(state: AgentState) → AgentState`** *(async)*

- Extracts last `HumanMessage` from `state["messages"]`
- If `state.get("image_data")` is present, bypasses intent classification and hard-sets:
  - `state["intent"] = "document_processing"` (routes to `document_agent`)
- Calls `ChatOpenAI` with classification prompt using `await llm.ainvoke(...)`:
  ```
  Classify this logistics query into exactly one category.
  Return only the category name, nothing else.
  Categories: safety_query, route_query, maintenance_query, document_processing, general
  Query: {query}
  ```
- Validates response is one of the categories (defaults to `general` if not)
- Sets `state["intent"]`

---

### `backend/agents/safety_agent.py`

**`safety_agent_node(state: AgentState) → AgentState`**

Flow (async, production optimized):
```
1. If `intent == "general"`, skip RAG and return a greeting (no citations)
2. Get resources from RagManager (one-time init: embeddings + Chroma + retrievers + chains)
2. check_relevance_score(retriever, query)
   ├── score >= threshold → proceed with RAG
   └── score < threshold AND TAVILY_API_KEY set
       └── run Tavily search → inject web results as Document context
3. run_query_with_docs(chain, retriever, query)  (runs in worker thread)
4. On Cohere 401/reranker failure:
   └── retry with RagManager hybrid-only retriever + chain
5. Set state["rag_answer"], state["rag_sources"], state["web_search_used"], state["final_answer"], state["relevance_score"]
   - `rag_sources` is limited to max 3 citations for readability
   - low-relevance + no-web-search suppresses citations entirely
```

Handles 3 failure modes gracefully:
- **Vector store missing**: returns helpful "Please ingest documents" message
- **Cohere 401**: retries with hybrid-only retriever
- **Any other exception**: returns error message with details

---

### `backend/agents/navigator_agent.py`

**`navigator_agent_node(state: AgentState) → AgentState`**

Live routing node with Google Maps and MCP weather integration. Execution flow:

```
1. extract_locations(query)                  ← structured output (`LocationInfo`) via `with_structured_output`
2. get_route(origin, destination)            ← Google Maps or mock fallback (runs in worker thread)
2b. get_route_weather(origin, destination)   ← async MCP weather server or mock fallback
    → weather emoji + safety level badge for the answer
3. Build markdown answer table               ← distance, time, traffic ETA, states
4. Inject weather_section                    ← weather assessment block with FMCSA ref
5. Build compliance_section                  ← per-state HazMat rules and permit flags
6. Set state["route_data"] + state["weather_data"]
```

Returns `route_data=None` and `weather_data=None` if location extraction fails (polite error message returned instead).

Sets four state fields:
- `state["final_answer"]` — markdown route table + weather section + compliance alerts
- `state["route_data"]` — full route dict for Streamlit folium map rendering
- `state["weather_data"]` — weather dict for Streamlit colour-coded safety card
- `state["rag_sources"]` — always `[]` (routes do not use the vector store)

---

### `backend/agents/document_agent.py`

**`document_agent_node(state: AgentState) → dict`**

Vision-based "Intelligent Document Clerk" that extracts structured logistics fields from an uploaded image (BOL/receipt).

Workflow:
```
1. If state["image_data"] is missing:
   → returns extracted_doc_data=None and a helpful final_answer.
2. Creates `ChatOpenAI(model="gpt-4o", temperature=0)` with `.with_structured_output(DocumentExtraction)`
3. Sends a vision message:
   - text prompt: "Extract logistics data from this image"
   - image_url: {"url": "data:image/jpeg;base64,<...>"}
4. Returns:
   - extracted_doc_data = parsed structured fields
   - final_answer = friendly summary

Note: the Streamlit frontend uses a consume-once pattern for the uploaded image (after the first successful extraction, it clears `image_base64` so later queries don't get forced into document_processing).
```

Output model `DocumentExtraction` fields:
- `document_type` (str)
- `origin` (str)
- `destination` (str)
- `weight` (str)
- `freight_class` (str)
- `summary` (str)

---

### `backend/agents/graph.py`

**`build_graph() → CompiledGraph`**

```python
graph = StateGraph(AgentState)
graph.add_node("dispatcher", dispatcher_node)
graph.add_node("safety_agent", safety_agent_node)
graph.add_node("document_agent", document_agent_node)
graph.add_node("navigator_agent", navigator_agent_node)

graph.add_edge(START, "dispatcher")
graph.add_conditional_edges("dispatcher", route_after_dispatch, {
    "safety_agent": "safety_agent",
    "maintenance_agent": "maintenance_agent",
    "general_agent": "general_agent",
    "navigator_agent": "navigator_agent",
    "document_agent": "document_agent",
})
graph.add_edge("safety_agent", END)
graph.add_edge("maintenance_agent", END)
graph.add_edge("general_agent", END)
graph.add_edge("document_agent", END)
graph.add_edge("navigator_agent", END)

return graph.compile(checkpointer=MemorySaver())
```

Routing function `route_after_dispatch`:
- `"route_query"` → `"navigator_agent"`
- `"safety_query"` → `"safety_agent"`
- `"maintenance_query"` → `"maintenance_agent"` (delegates to same RAG pipeline as safety)
- `"document_processing"` → `"document_agent"`
- `"general"` → `"general_agent"` (greeting; no RAG)
- unknown / other → `"safety_agent"` (defensive RAG fallback)

**`run_graph(query, thread_id="default", image_base64=None) → dict`**
- Builds initial `AgentState` with `HumanMessage(query)` and `image_data=image_base64`
- Invokes with `{"configurable": {"thread_id": thread_id}}`
- `MemorySaver` persists message history per thread_id
- Returns `{"answer", "sources", "intent", "agent_used", "web_search_used", "route_data", "weather_data", "relevance_score", "llm_model", "embedding_model"}`
- and (when applicable) `extracted_doc_data` + `flow_data`

---

## 4. Module Reference — Tools

### `backend/tools/location_extractor.py`

**`extract_locations(query: str) → LocationInfo`** *(async, structured output)*

Uses `ChatOpenAI(...).with_structured_output(LocationInfo)` and `await llm.ainvoke(...)` to guarantee reliable parsing (no manual `json.loads` or fence stripping).

Return shape:
```python
{
    "origin":      "Chicago, IL",
    "destination": "Detroit, MI",
    "found":       True,
    "confidence":  "high"  # or "low"
}
```

Handles varied phrasings:
- `"route from Chicago to Detroit"`
- `"how long to drive from Houston TX to Dallas"`
- `"I need to get to Miami from NYC"`

Falls back to `found=False` on JSON parse errors or LLM failures — the navigator node handles this gracefully.

---

### `backend/tools/maps_tool.py`

**`get_route(origin: str, destination: str) → dict`**

Main routing function. Tries the Google Maps Directions API first; falls back to mock data if key is missing or API call fails.

**Robust failure behaviour:** When Google Maps is unavailable, it returns mock route data **plus** an `api_error` string (e.g. `"Maps API currently unavailable: ..."`). The Navigator agent surfaces this in the answer so the user understands what happened.

Return shape:
```python
{
    "success":             True,
    "origin":              "Chicago, IL, USA",        # formatted address from Maps
    "destination":         "Detroit, MI, USA",
    "distance_miles":      281.4,
    "distance_text":       "281 miles",
    "duration_text":       "4 hours 12 mins",
    "duration_in_traffic": "4 hours 35 mins (with current traffic)",
    "polyline_coords":     [[41.87, -87.62], ...],   # list of [lat, lng]
    "start_location":      {"lat": 41.8781, "lng": -87.6298},
    "end_location":        {"lat": 42.3314, "lng": -83.0458},
    "states_crossed":      ["IL", "IN", "MI"],
    "compliance_notes":    [...],                     # see below
    "steps_count":         12,
    "mock_data":           False                      # True when using fallback
}
```

**`_detect_states(coords) → List[str]`**

Samples polyline coordinates every 8th point and checks each against 27 US state bounding boxes. Returns ordered list of state abbreviations along the route.

**`_build_compliance_notes(states) → List[dict]`**

For each state code that has an entry in `HAZMAT_RULES`, returns a compliance note dict:
```python
{
    "state_code":      "IL",
    "state":           "Illinois",
    "summary":         "Illinois requires IDOT HazMat carrier registration.",
    "detail":          "Tunnel and route restrictions apply on I-90/94...",
    "permit_required": True,
    "center":          [40.0, -89.2]   # map marker location
}
```

**`_build_mock_route(origin, destination) → dict`**

Returns a hardcoded Chicago → Detroit route along I-94 with 9 real coordinate points. Always produces a valid, renderable map. States covered: IL, IN, MI.

**Data coverage:**

| Dataset | Coverage |
|---|---|
| State bounding boxes | 27 US states |
| HazMat rule database | 10 states (IL, IN, MI, OH, TX, CA, NY, PA, FL, GA) |
| State center coordinates | 27 US states (for map markers) |

---

> See **Section 5** for the full MCP weather server and `weather_tool.py` documentation.

---

## 5. Module Reference — MCP Servers

### `backend/mcp_servers/weather_server.py`

This module implements a **Model Context Protocol (MCP) server** that exposes two weather tools to the agent system. It can be used in two modes:

1. **In-process** — called directly by `weather_tool.py` via async function imports (no stdio transport, no subprocess).
2. **Standalone** — run as a real MCP stdio server for testing: `python backend/mcp_servers/weather_server.py`

#### Declared MCP tools

**`get_current_weather(location, units="imperial")`**
- Fetches current conditions for a single city from OpenWeatherMap `/weather` endpoint
- Returns: `temperature`, `feels_like`, `humidity`, `wind speed + direction`, `visibility`, `driving safety level + message`

**`get_weather_route_summary(origin, destination, units="imperial")`**
- Fetches conditions at both route endpoints concurrently (via `asyncio.gather`)
- Computes worst-case `overall_safety` across origin + destination
- Returns combined formatted text block suitable for injecting into the chat answer

#### Driving safety classifier — `_assess_driving_safety()`

Maps OpenWeatherMap condition IDs to a 4-level safety scale:

| Level | Score | OWM condition codes | Wind threshold | Visibility |
|---|---|---|---|---|
| `CLEAR` | 0 | Clear sky, few clouds | < 25 mph | > 1 km |
| `ADVISORY` | 1 | Drizzle (3xx), light rain (500), mist/haze (7xx), wind 25–40 mph | 25–40 mph | — |
| `CAUTION` | 2 | Heavy rain (501–531), snow (6xx), fog (741), wind 40+ mph, visibility < 1 km | 40+ mph | < 1 km |
| `DANGEROUS` | 3 | Thunderstorm (2xx), heavy snow (602/621/622), tornado/ash/squall (762/771/781) | Extreme | — |

The classifier also includes the relevant **FMCSA regulation reference** (49 CFR 392.14) in the advisory text for each non-CLEAR level.

#### Mock data fallbacks

`_mock_weather(location, units)` and `_mock_route_weather(origin, destination, units)` return realistic pre-canned responses (partly cloudy at origin, light rain at destination) tagged with `*(mock data)*`. These are called automatically when `OPENWEATHERMAP_API_KEY` is not set.

#### Helper utilities

| Function | Purpose |
|---|---|
| `_wind_direction(degrees)` | Converts 0–360° to 8-point compass (N/NE/E/SE/S/SW/W/NW) |
| `_assess_driving_safety()` | Returns `{level, score, message, advice}` dict |

---

### `backend/tools/weather_tool.py`

Async-native LangGraph-callable wrapper around the MCP weather server functions (no `asyncio.run`, no `nest_asyncio`).

**`get_weather(location, units=None) → dict`**

```python
{
    "location":     "Chicago, IL",
    "raw_text":     str,          # full formatted weather summary
    "safety_level": str,          # CLEAR | ADVISORY | CAUTION | DANGEROUS
    "is_mock":      bool          # True when OPENWEATHERMAP_API_KEY not set
}
```

**`get_route_weather(origin, destination, units=None) → dict`**

```python
{
    "origin":         str,
    "destination":    str,
    "raw_text":       str,          # full route weather summary text
    "overall_safety": str,          # worst level across both endpoints
    "has_warning":    bool,         # True when CAUTION or DANGEROUS
    "is_mock":        bool
}
```

Both functions return `api_error: Optional[str]` when the upstream API fails so the agent can explain the degradation gracefully.

---

## 6. Module Reference — API

### `backend/api/routes.py`

All routes are mounted at `/api/v1/` by `backend/main.py`.

#### `POST /query`

| Field | Type | Description |
|---|---|---|
| `query` | str | The user's question |
| `thread_id` | str | Conversation thread ID for memory persistence (default: "default") |
| `image_base64` | Optional[str] | Base64-encoded image for document extraction (BOL/receipt). When present, dispatcher routes to `document_agent`. |

Response includes `latency_ms` calculated from request start to response.

Additional production telemetry fields returned for UI transparency:
- `llm_model`
- `llm_provider`
- `embedding_model`
- `llm_comparison` (open vs closed model outputs + latency + error)
- `agent_used`
- `relevance_score` (safety agent only)
- `flow_data` (LangGraph nodes/edges + execution_path for visualization)
- `extracted_doc_data` (document_processing only)

Error handling: returns HTTP 500 with `{"detail": "Query processing failed: {reason}"}` on exceptions.

#### `POST /ingest`

| Field | Type | Description |
|---|---|---|
| `pdf_paths` | List[str] | Absolute or relative paths to PDF files on the server |

Note: In the Streamlit app, files are first saved to `./data/` then their paths are passed here.

#### `GET /health`

Always returns 200. Used by load balancers and monitoring.

---

### `backend/main.py`

CORS is configured to allow only:
- `http://localhost:8501`
- `http://127.0.0.1:8501`

This restricts the API to the local Streamlit frontend in development. For production deployment, update `allow_origins` to your actual frontend domain.

---

## 7. LangGraph State & Flow

### Thread-level Memory

`MemorySaver` stores the full `AgentState` keyed by `thread_id`. Each Streamlit session gets a unique `uuid4()` thread ID, so conversations are isolated between browser sessions.

The `messages` field uses `Annotated[list, add_messages]` which applies LangGraph's `add_messages` reducer — new messages are appended rather than replacing the list, creating a persistent conversation history.

### State Mutation Pattern

Each node receives the full `AgentState` dict and returns a **new dict** containing only the keys it wants to update.

LangGraph merges these updates into the running state. This lets some nodes return partial updates (for example, the dispatcher returns only `{"query": ..., "intent": ...}`).

```python
return {"intent": intent}
```
This avoids accidental field overwrites and keeps node logic focused on only what it computes.

### Async execution

All nodes are async and the graph is executed with `compiled.ainvoke(...)`. Sync-heavy calls (Google Maps client + RAG chain `.invoke`) run in worker threads to avoid blocking the FastAPI event loop.

### UI flow visualization (`flow_data`)

For each query, `run_graph()` also returns `flow_data` describing the LangGraph nodes/edges plus the `execution_path` taken (based on the evaluated intent). The Streamlit UI renders this via Graphviz to show exactly which agent node handled the request.

---

## 8. Embedding Model Comparison

### BGE-small-en-v1.5 (default)

- **Size**: 33M parameters, ~130 MB download
- **Device**: CPU (no GPU required)
- **Normalization**: L2-normalized embeddings (cosine similarity = dot product)
- **Strengths**: Fast, free, runs locally, good at regulatory/technical text
- **Weaknesses**: Slower first load (model download), slightly lower quality than OpenAI

### OpenAI text-embedding-3-small (comparison)

- **Dimensions**: 1536
- **Cost**: ~$0.02 per 1M tokens (very cheap)
- **Strengths**: Highest quality embeddings, fast API, no local GPU/CPU overhead
- **Weaknesses**: API calls (latency, cost, requires internet), not free

### How to switch

Change `EMBEDDING_MODEL` in `.env`:

```env
# Use BGE (default, local, free)
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Use OpenAI (API, paid)
EMBEDDING_MODEL=text-embedding-3-small
```

> **Important**: After switching embedding models, delete `chroma_db/` and re-run `scripts/ingest.py`. The vector space is incompatible between models.

### Running the comparison experiment

```python
from backend.rag.embeddings import compare_embeddings
from backend.rag.loader import load_and_chunk_pdfs

docs = load_and_chunk_pdfs("./data")
results = compare_embeddings("What are the hours of service rules?", docs[:20])

print("BGE top-3:", results["bge"])
print("OpenAI top-3:", results["openai"])
```

---

## 9. Retrieval Strategy Deep-Dive

### Why Hybrid Retrieval?

FMCSA regulatory documents have two distinct query patterns:

1. **Exact code lookups**: "What does Section 396.11 require?"  
   → BM25 wins here (exact keyword match on "396.11")

2. **Semantic safety questions**: "When does a driver need to rest?"  
   → Vector search wins here (semantic understanding of "rest" = "off duty" = "hours limit")

The hybrid ensemble combines both, weighted 40/60 toward vector (semantic queries are more common in practice).

### BM25 Weight Justification (0.4)

BM25 is a sparse retriever — it scores 0 for any document not containing the exact query terms. At weight 0.4, it provides strong signal for exact code citations without penalizing documents that use synonymous phrasing.

### Reranking with Cohere

Cohere's cross-encoder model (`rerank-english-v3.0`) reads the full query and each document chunk together — unlike the bi-encoder approach used for indexing. This joint reasoning produces more accurate relevance scores but is slower (one API call per query, not per document). It reduces top-10 results to top-3, dramatically improving precision for the final answer.

---

## 10. Corrective RAG Flow

```
Query arrives
    │
    ▼
check_relevance_score(retriever, query)
    │
    ├── score >= 0.6 ──────────────────────► run_query_with_docs(chain, retriever, query)
    │                                              │
    │                                              ▼
    │                                         Answer + sources returned
    │
    └── score < 0.6 AND TAVILY_API_KEY set
        │
        ▼
    TavilyClient.search(query, max_results=3)
        │
        ▼
    Each result → Document(
        page_content = result["content"],
        metadata = {
            "source": "[WEB] {title}",
            "page":   result["url"]
        }
    )
        │
        ▼
    run_query_with_web_context(retriever, query, web_docs)
        ├── retriever.invoke(query)   ← clean query, gets local ChromaDB docs
        ├── all_docs = web_docs + chroma_docs
        ├── context = _format_docs(all_docs)  ← web docs appear first in context
        └── llm.invoke(prompt(context, question))
        │
        ▼
    Answer grounded in web + local context
    Sources include [WEB] entries alongside PDF citations
        │
        ▼
    Set web_search_used = True in state
```

**Key design principle:** Tavily results are converted to `Document` objects and injected directly into the LLM `context` slot. The original query string is never modified. This ensures the retriever always searches ChromaDB with the clean user question, and the LLM sees both web and local evidence as properly attributed context chunks.

The relevance threshold (default 0.6) is configurable via `RELEVANCE_SCORE_THRESHOLD`. Lower values mean web search triggers more often; higher values mean the system relies more on indexed documents.

---

## 11. Navigator Agent — Google Maps + Weather Flow

### End-to-end flow

```
POST /api/v1/query  {"query": "Route from Chicago IL to Detroit MI"}
    │
    ▼
LangGraph dispatcher_node
    ChatOpenAI classifies → intent = "route_query"
    │
    ▼
navigator_agent_node
    │
    ├─ Step 1: extract_locations("Route from Chicago IL to Detroit MI")
    │       ChatOpenAI.with_structured_output(LocationInfo) → LocationInfo(origin="Chicago, IL", destination="Detroit, MI", found=True)
    │
    ├─ Step 2: get_route("Chicago, IL", "Detroit, MI")
    │       ├── GOOGLE_MAPS_API_KEY set?
    │       │     YES → client.directions(origin, destination, mode="driving",
    │       │                             departure_time=now(), units="imperial")
    │       │           → decode polyline → 9-400 [lat,lng] coords
    │       │     NO  → _build_mock_route() → 9 hardcoded I-94 coords
    │       │
    │       ├── _detect_states(coords)
    │       │     samples every 8th coord → checks 27 state bounding boxes
    │       │     → ["IL", "IN", "MI"]
    │       │
    │       └── _build_compliance_notes(["IL","IN","MI"])
    │             → 3 compliance dicts (IL: permit_required=True,
    │                                   IN: permit_required=False,
    │                                   MI: permit_required=True)
    │
    ├─ Step 2b: get_route_weather("Chicago, IL", "Detroit, MI")
    │       ├── weather_tool.py (async-native)
    │       │     └── await _fetch_route_weather(...)
    │       │           └── MCP weather server
    │       │                 ├── OPENWEATHERMAP_API_KEY set?
    │       │                 │     YES → asyncio.gather(
    │       │                 │              OWM /weather?q=Chicago,IL,
    │       │                 │              OWM /weather?q=Detroit,MI
    │       │                 │           )
    │       │                 │           → _assess_driving_safety() for each city
    │       │                 │           → worst = max(origin_score, dest_score)
    │       │                 │     NO  → _mock_route_weather() → realistic mock text
    │       │                 └── returns formatted route weather summary string
    │       └── overall_safety = "CLEAR" | "ADVISORY" | "CAUTION" | "DANGEROUS"
    │             weather emoji selected: ✅ / 🔵 / ⚠️ / 🚨
    │
    ├─ Step 3: Build markdown answer
    │     Route table + weather section + HazMat alerts + 49 CFR reminders
    │
    └─ Return AgentState with:
          final_answer  = markdown string
          route_data    = full route dict (polyline, states, compliance, etc.)
          weather_data  = {overall_safety, has_warning, raw_text, is_mock}
    │
    ▼
FastAPI QueryResponse
    answer       = markdown
    route_data   = full route dict
    weather_data = weather dict
    intent       = "route_query"
    │
    ▼
Streamlit frontend/app.py
    st.markdown(answer)               ← renders route table + weather + compliance text
    render_route_map(route_data)      ← renders folium map
        ├── folium.Map(CartoDB positron tiles)
        ├── PolyLine(coords, color="#185FA5")      ← blue route line
        ├── Marker(start, icon=green play)         ← origin
        ├── Marker(end,   icon=red flag)           ← destination
        └── Marker(state_center, icon=orange !)   ← per compliance note
    st_folium(m, height=420)
    render_weather_card(weather_data) ← colour-coded safety card below map
        ├── CLEAR     → green card  ✅
        ├── ADVISORY  → blue card   🔵
        ├── CAUTION   → amber card  ⚠️
        └── DANGEROUS → red card    🚨
    st.warning("Permit required in: Illinois, Michigan")
```

### AgentState fields used by Navigator

| Field | Set by | Read by |
|---|---|---|
| `query` | `graph.py` initial state | `navigator_agent_node` |
| `intent` | `dispatcher_node` | `route_after_dispatch` routing |
| `final_answer` | `navigator_agent_node` | FastAPI → Streamlit chat bubble |
| `route_data` | `navigator_agent_node` | FastAPI → Streamlit `render_route_map()` |
| `weather_data` | `navigator_agent_node` (via MCP) | FastAPI → Streamlit `render_weather_card()` |
| `rag_sources` | Set to `[]` by navigator | Source expander (empty for route queries) |
| `web_search_used` | Set to `False` by navigator | Web search badge (hidden for routes) |

### route_data schema

```python
{
    # Identity
    "success":             bool,
    "mock_data":           bool,          # True = no API key / API error

    # Addresses
    "origin":              str,           # formatted by Google Maps
    "destination":         str,

    # Distances and times
    "distance_miles":      float,
    "distance_text":       str,           # "281 miles"
    "duration_text":       str,           # "4 hours 12 mins"
    "duration_in_traffic": str,           # traffic-aware, or same as above

    # Map rendering
    "polyline_coords":     List[List[float]],  # [[lat, lng], ...]
    "start_location":      {"lat": float, "lng": float},
    "end_location":        {"lat": float, "lng": float},

    # Compliance
    "states_crossed":      List[str],     # ["IL", "IN", "MI"]
    "compliance_notes": [
        {
            "state_code":      str,
            "state":           str,       # full name
            "summary":         str,
            "detail":          str,
            "permit_required": bool,
            "center":          [float, float]  # [lat, lng] for map marker
        }
    ],
    "steps_count":         int,
}
```

### weather_data schema

```python
{
    # Identity
    "origin":         str,          # as passed to get_route_weather()
    "destination":    str,

    # Summary
    "raw_text":       str,          # full formatted route weather text block
    "overall_safety": str,          # CLEAR | ADVISORY | CAUTION | DANGEROUS
    "has_warning":    bool,         # True when CAUTION or DANGEROUS
    "is_mock":        bool,         # True when OPENWEATHERMAP_API_KEY not set
}
```

`render_weather_card(weather_data)` in `frontend/app.py` uses `overall_safety` to select the card background colour and icon, then renders `raw_text` verbatim in a `<pre>` block inside a styled `<div>`. If `is_mock=True`, a note is appended pointing the user to add their API key.

---

### Folium map layers

| Layer | folium object | Colour | Condition |
|---|---|---|---|
| Route line | `PolyLine` | `#185FA5` (blue) | Always |
| Origin | `Marker` | green (`fa:play`) | Always |
| Destination | `Marker` | red (`fa:flag`) | Always |
| State compliance | `Marker` | orange (`glyphicon:warning-sign`) | One per `compliance_notes` entry |
| Bounds | `fit_bounds` | — | Always — auto-zooms to route |

---

## 12. Observability & Monitoring — LangSmith

### Overview

TransOrchestra integrates **LangSmith** for full-stack observability of the multi-agent RAG pipeline. LangChain's built-in callback system auto-instruments every LLM call, retriever invocation, and agent node transition — no decorators or manual logging required in individual modules.

LangSmith is configured in a single place (`backend/config.py`) and activated via four environment variables in `.env`.

---

### How It Works — Implementation Detail

`backend/config.py` is imported as the **first project module** in `backend/main.py`. It loads settings from `.env` using Pydantic Settings, then immediately forwards the LangSmith variables into `os.environ` before any LangChain module is imported:

```python
# backend/config.py  (runs before any langchain import)
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "transOrchestra")
os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
```

When `LANGCHAIN_TRACING_V2=true`, LangChain's `LangSmithCallbackHandler` is registered globally and every `chain.invoke()`, `retriever.invoke()`, and `llm.invoke()` call in the codebase is traced automatically.

---

### Trace Structure

A single query from the Streamlit UI produces a trace tree with the following shape:

```
LangGraph  (root span — full query latency)
│
├── dispatcher                        ~1.0 s
│   └── ChatOpenAI / gpt-4o-mini      ~0.5 s, ~55 tokens
│       Input:  classification prompt + user query
│       Output: "safety_query" | "route_query" | ...
│
├── route_after_dispatch              ~0.0 s  (routing decision, no LLM call)
│
└── safety_agent                      ~5.1 s, ~1.4K tokens
    │
    ├── Retriever                     ~0.03 s  (hybrid ensemble)
    │   ├── BM25Retriever             ~0.00 s
    │   └── VectorStoreRetriever      ~0.03 s
    │
    └── ChatOpenAI / gpt-4o-mini      ~1.35 s, ~1.4K tokens
        Input:  system prompt + retrieved context + question
        Output: final answer with source citations
```

If Tavily web search fires (low relevance score), an additional `TavilySearch` span appears between the Retriever and the final LLM call.

---

### What Each Span Captures

| Span | Inputs recorded | Outputs recorded | Metrics |
|---|---|---|---|
| LangGraph root | Full `AgentState` dict | Final state | Total latency |
| dispatcher | Classification prompt | Raw LLM text + parsed intent | Latency, tokens |
| route_after_dispatch | intent string | target node name | Latency |
| safety_agent | query, retriever config | answer, sources, web_search_used | Latency, tokens |
| Retriever | query string | List of retrieved Document objects | Latency |
| BM25Retriever | query string | BM25-ranked documents | Latency |
| VectorStoreRetriever | query embedding | Cosine-similarity ranked docs | Latency |
| gpt-4o-mini (answer) | Full formatted prompt with context | Answer text | Latency, tokens, cost |

---

### LangSmith Dashboard Features Used

| Feature | How to access | What it shows |
|---|---|---|
| **Traces** | Tracing → transOrchestra | Full run list with input preview and latency |
| **Trace detail** | Click any run | Expandable tree of all spans |
| **Input / Output** | Right panel tabs | Full prompt and response text per node |
| **Metadata** | Right panel → Metadata tab | Model name, temperature, thread_id |
| **Threads** | Threads tab | Conversation history grouped by thread_id |
| **Runs** | Runs tab | Flat list of all individual LLM calls |

---

### Environment Variables Reference

```env
# Enable tracing
LANGCHAIN_TRACING_V2=true

# Your LangSmith API key — free tier available
# Get it at: https://smith.langchain.com/settings
LANGCHAIN_API_KEY=lsv2_pt_...

# Project name — traces appear under this name in the UI
LANGCHAIN_PROJECT=transOrchestra

# API endpoint (default, no need to change)
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### Startup confirmation

When tracing is enabled, the FastAPI startup log outputs:
```
INFO  backend.main — LangSmith tracing: ENABLED → project 'transOrchestra'
```

When disabled:
```
INFO  backend.main — LangSmith tracing: disabled (set LANGCHAIN_TRACING_V2=true to enable)
```

---

### Disabling Tracing

Set `LANGCHAIN_TRACING_V2=false` in `.env`. No data is sent, no API calls are made, and there is zero performance overhead.

---

## 13. Evaluation Framework

### 10.1 Ragas Metrics Explained

**Faithfulness** measures whether every claim in the generated answer can be traced back to the retrieved context chunks. A score of 1.0 means every statement is grounded; 0.0 means the model hallucinated all of it.

**Answer Relevancy** measures whether the answer actually addresses the question asked. A high-faithfulness but low-relevancy answer would be one that is factually grounded but answers a different question.

### 10.2 Evaluation Pipeline

```
eval_set.json (15 Q&A pairs)
    │
    ▼
For each question:
    run_query_with_docs(chain, retriever, question)
        → answer (str)
        → contexts (list of retrieved chunk texts)
    │
    ▼
Build Ragas Dataset:
    question   : str
    answer     : str
    contexts   : List[str]
    ground_truth: str
    │
    ▼
ragas.evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    │
    ▼
Print results table + save to eval/results.json
```

### 10.3 Interpreting Results

| Score Range | Interpretation |
|---|---|
| 0.9–1.0 | Excellent — answers are faithful and relevant |
| 0.7–0.9 | Good — minor hallucination or relevance issues |
| 0.5–0.7 | Acceptable — some grounding problems |
| < 0.5 | Poor — significant hallucination or irrelevance |

Typical causes of low faithfulness in this domain:
- LLM "fills in" regulation details not present in the indexed chunks
- Insufficient PDF coverage (missing the specific regulation being asked about)
- Chunk boundaries cutting off important context

Typical causes of low answer relevancy:
- The intent classifier routed to the wrong agent
- The query is too vague for the indexed corpus

---

## Ragas Results Snapshot

Latest Ragas evaluation on the current `eval/eval_set.json`:

| Retriever Strategy | Faithfulness | Answer Relevancy |
|---|---:|---:|
| Hybrid + Reranker | 0.7319 | 0.9294 |

## 14. Troubleshooting

### `chroma-hnswlib` build error on Windows

```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Fix:**
```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```
Reopen PowerShell after install completes.

---

### `No module named 'langchain_core'`

The pip install was aborted before completing (usually due to the chromadb build error above). Fix the chromadb issue first, then:

```powershell
pip install -r requirements.txt
```

---

### `langchain-chroma==X.X.X` not found

`langchain-chroma` has non-sequential version numbers. Verify available versions:

```powershell
pip index versions langchain-chroma
```

The project uses `0.2.0` which requires `chromadb<0.6.0`. Use `chromadb==0.5.18`.

---

### Cohere 401 Unauthorized

Your `COHERE_API_KEY` in `.env` is incorrect or expired.

1. Go to https://dashboard.cohere.com/api-keys
2. Copy your current Trial key
3. Update `.env`, save, restart uvicorn

The system automatically falls back to hybrid retriever on 401 errors — queries will still work but without reranking.

---

### Ragas evaluation returns N/A for all metrics

Two possible causes:

1. **Column name mismatch** — Ragas 0.1.21 requires singular keys (`question`, `answer`, `ground_truth`). This is already fixed in the current `eval/run_ragas.py`.

2. **All answers are errors** — If the Cohere key is invalid and the fallback also fails, all answers become `"Error: ..."` strings. Ragas cannot evaluate error strings. Fix the Cohere key or clear it entirely (fallback will use hybrid-only).

---

### ChromaDB telemetry errors in logs

```
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
```

This is a known bug in `posthog` (the analytics library ChromaDB uses for telemetry) version mismatch with chromadb 0.5.18. It is **completely harmless** — ChromaDB works perfectly despite this log message. It is suppressed in `eval/run_ragas.py` via `warnings.filterwarnings`.

To suppress it in the FastAPI server logs as well, add to `backend/main.py`:

```python
import logging
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
```

---

### `Number of requested results 10 is greater than number of elements in index 8`

You have fewer than 10 chunks in your vector store. ChromaDB automatically adjusts `n_results` down to the available count — this is informational, not an error. Ingest more PDFs to resolve.

---

### LangSmith traces not appearing

Check in order:

1. **Confirm tracing is enabled** — startup log must say `ENABLED`, not `disabled`
2. **Check the API key** — go to [smith.langchain.com/settings](https://smith.langchain.com/settings), copy the key exactly, paste into `.env` with no trailing spaces
3. **Check import order** — `backend/config.py` must be the first project import in `backend/main.py` (before `routes`). The env vars must be set before any `langchain` module is imported.
4. **Check project name** — in the LangSmith UI, use the **Tracing** → **All projects** view if you don't see `transOrchestra` listed immediately
5. **Firewall / proxy** — LangSmith sends traces to `https://api.smith.langchain.com`. If your network blocks outbound HTTPS, traces won't arrive. Check with `curl https://api.smith.langchain.com`

If the key is wrong, LangSmith silently drops traces (it doesn't crash the app). Set `LANGCHAIN_TRACING_V2=false` if you want to stop sending data while debugging the key.

---

### Folium map does not appear after a route query

1. Confirm the query was classified as `route_query` — check the **Intent** badge in the chat. If it shows `safety_query`, the dispatcher misclassified it; try phrasing with "Route from … to …".
2. Check FastAPI logs for `Location extraction failed` — this means the LLM could not parse city names from the query. Include both origin and destination explicitly.
3. Confirm `streamlit-folium==0.22.0` is installed: `pip show streamlit-folium`.
4. If the map renders blank (grey tiles), check your internet connection — folium loads CartoDB map tiles from the web.

---

### Google Maps API returns empty directions

Common causes:
- **Directions API not enabled** — go to [console.cloud.google.com](https://console.cloud.google.com) → APIs & Services → Enable "Directions API"
- **Billing not set up** — Google Maps requires a billing account (has a generous free tier)
- **Key restrictions** — if the key has IP or HTTP referer restrictions, API calls from a server may be blocked. Use an unrestricted key for local development.

The system falls back to mock data automatically on any API failure, so the app continues to work.

---

### Weather card does not appear after a route query

1. Confirm the query was classified as `route_query` — check the **Intent badge** in the chat. Only route queries populate `weather_data`.
2. Ensure the backend is on the latest refactor — `weather_tool.py` is async-native (no nest_asyncio needed). Restart uvicorn after upgrading.
3. Check FastAPI logs for `Weather fetch error` — this means the OWM API call failed. The tool should fall back to mock data automatically; if you see an error card, check the `OPENWEATHERMAP_API_KEY` value in `.env`.
4. If the card renders but shows `*(mock data)*`, add your `OPENWEATHERMAP_API_KEY` to `.env` and restart Uvicorn.

---

### OpenWeatherMap API returns 401 Unauthorized

Your `OPENWEATHERMAP_API_KEY` in `.env` is incorrect or not yet activated. New keys can take up to 2 hours to activate after signup. In the meantime, the mock data fallback renders correctly — no user-visible error occurs.

---

### OpenWeatherMap API returns 404 for a city

The city name was not recognized by the OWM geocoder. Use the `City, StateCode, CountryCode` format:

```
Chicago, IL, US     ✅ correct
Chigago, IL         ❌ typo — returns 404
Springfield         ❌ ambiguous — multiple matches
```

The `location_extractor.py` LLM typically provides clean city/state pairs, so this is rare in practice.

---

### LangSmith tracing causes slow queries

LangSmith trace submission is **asynchronous** — it does not block the response path. Queries should not be measurably slower with tracing enabled. If you observe slowness, it is likely unrelated to LangSmith (check OpenAI API latency or Tavily search time in the trace detail view instead).

---

*TransOrchestra v1.0 — Analytics Vidya GenAI Pinnacle Capstone*
