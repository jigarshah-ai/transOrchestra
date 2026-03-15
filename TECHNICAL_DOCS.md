# TransOrchestra — Technical Documentation

> Deep-dive reference for developers, evaluators, and capstone reviewers.

---

## Table of Contents

1. [Configuration Reference](#1-configuration-reference)
2. [Module Reference — RAG Pipeline](#2-module-reference--rag-pipeline)
3. [Module Reference — Agents](#3-module-reference--agents)
4. [Module Reference — API](#4-module-reference--api)
5. [LangGraph State & Flow](#5-langgraph-state--flow)
6. [Embedding Model Comparison](#6-embedding-model-comparison)
7. [Retrieval Strategy Deep-Dive](#7-retrieval-strategy-deep-dive)
8. [Corrective RAG Flow](#8-corrective-rag-flow)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Configuration Reference

All configuration is loaded from `.env` via `backend/config.py`. No values are hardcoded anywhere in the codebase.

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
model_name is None                       →  reads EMBEDDING_MODEL from config
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
    intent: str                               # dispatcher output
    vehicle_id: str                           # optional conversation context
    cargo_type: str                           # optional conversation context
    route_constraints: dict                   # optional conversation context
    rag_answer: str                           # answer from RAG pipeline
    rag_sources: list                         # source citations
    web_search_used: bool                     # Tavily was triggered
    final_answer: str                         # final response to user
    thread_id: str                            # MemorySaver thread key
```

---

### `backend/agents/dispatcher.py`

**`dispatcher_node(state: AgentState) → AgentState`**

- Extracts last `HumanMessage` from `state["messages"]`
- Calls `ChatOpenAI` with classification prompt:
  ```
  Classify this logistics query into exactly one category.
  Return only the category name, nothing else.
  Categories: safety_query, route_query, maintenance_query, general
  Query: {query}
  ```
- Validates response is one of the 4 categories (defaults to `general` if not)
- Sets `state["intent"]`

---

### `backend/agents/safety_agent.py`

**`safety_agent_node(state: AgentState) → AgentState`**

Flow:
```
1. Load pipeline (vectorstore → hybrid → reranker → RAG chain)
2. check_relevance_score(retriever, query)
   ├── score >= threshold → proceed with RAG
   └── score < threshold AND TAVILY_API_KEY set
       └── run Tavily search → prepend [WEB SOURCE] to query
3. run_query_with_docs(chain, retriever, query)
4. On Cohere 401/reranker failure:
   └── retry with fresh hybrid-only retriever + chain
5. Set state["rag_answer"], state["rag_sources"], state["web_search_used"], state["final_answer"]
```

Handles 3 failure modes gracefully:
- **Vector store missing**: returns helpful "Please ingest documents" message
- **Cohere 401**: retries with hybrid-only retriever
- **Any other exception**: returns error message with details

---

### `backend/agents/navigator_agent.py`

**`navigator_agent_node(state: AgentState) → AgentState`**

Currently a stub returning realistic simulated data. Includes:
- City pair extraction via regex
- Pre-loaded route lookup table (8 common US city pairs)
- HOS compliance note for multi-state routes
- HazMat placard reminder per 49 CFR 172.504
- Clear disclaimer that data is simulated

**Planned v2.0**: Google Maps Distance Matrix API or HERE Routing API integration.

---

### `backend/agents/graph.py`

**`build_graph() → CompiledGraph`**

```python
graph = StateGraph(AgentState)
graph.add_node("dispatcher", dispatcher_node)
graph.add_node("safety_agent", safety_agent_node)
graph.add_node("navigator_agent", navigator_agent_node)

graph.add_edge(START, "dispatcher")
graph.add_conditional_edges("dispatcher", route_after_dispatch, {
    "safety_agent": "safety_agent",
    "navigator_agent": "navigator_agent",
})
graph.add_edge("safety_agent", END)
graph.add_edge("navigator_agent", END)

return graph.compile(checkpointer=MemorySaver())
```

Routing function `route_after_dispatch`:
- `"route_query"` → `"navigator_agent"`
- Everything else → `"safety_agent"`

**`run_graph(query, thread_id="default") → dict`**
- Builds initial `AgentState` with `HumanMessage(query)`
- Invokes with `{"configurable": {"thread_id": thread_id}}`
- `MemorySaver` persists message history per thread_id
- Returns `{"answer", "sources", "intent", "web_search_used"}`

---

## 4. Module Reference — API

### `backend/api/routes.py`

All routes are mounted at `/api/v1/` by `backend/main.py`.

#### `POST /query`

| Field | Type | Description |
|---|---|---|
| `query` | str | The user's question |
| `thread_id` | str | Conversation thread ID for memory persistence (default: "default") |

Response includes `latency_ms` calculated from request start to response.

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

## 5. LangGraph State & Flow

### Thread-level Memory

`MemorySaver` stores the full `AgentState` keyed by `thread_id`. Each Streamlit session gets a unique `uuid4()` thread ID, so conversations are isolated between browser sessions.

The `messages` field uses `Annotated[list, add_messages]` which applies LangGraph's `add_messages` reducer — new messages are appended rather than replacing the list, creating a persistent conversation history.

### State Mutation Pattern

Each node receives the full `AgentState` dict and returns a **new dict** with updated fields:

```python
return {
    **state,           # preserve all existing fields
    "intent": intent,  # override only what this node computed
}
```

This ensures no node accidentally clears fields set by a previous node.

---

## 6. Embedding Model Comparison

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

## 7. Retrieval Strategy Deep-Dive

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

## 8. Corrective RAG Flow

```
Query arrives
    │
    ▼
check_relevance_score(retriever, query)
    │
    ├── score >= 0.6 ──────────────────────► Run RAG chain directly
    │
    └── score < 0.6 AND TAVILY_API_KEY set
        │
        ▼
    TavilyClient.search(query, max_results=3)
        │
        ▼
    Prepend web results as:
    [WEB SOURCE] {title}
    {content}
    URL: {url}
    
    Original question: {query}
        │
        ▼
    Run RAG chain with enriched query
        │
        ▼
    Set web_search_used = True in state
```

The relevance threshold (default 0.6) is configurable via `RELEVANCE_SCORE_THRESHOLD`. Lower values mean web search triggers more often; higher values mean the system relies more on indexed documents.

---

## 9. Evaluation Framework

### Ragas Metrics Explained

**Faithfulness** measures whether every claim in the generated answer can be traced back to the retrieved context chunks. A score of 1.0 means every statement is grounded; 0.0 means the model hallucinated all of it.

**Answer Relevancy** measures whether the answer actually addresses the question asked. A high-faithfulness but low-relevancy answer would be one that is factually grounded but answers a different question.

### Evaluation Pipeline

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

### Interpreting Results

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

## 10. Troubleshooting

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

*TransOrchestra v1.0 — Analytics Vidya GenAI Pinnacle Capstone*
