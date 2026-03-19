"""RAG chain construction and query execution for TransOrchestra."""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough

from backend.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are TransOrchestra, an AI assistant for transportation and logistics \
safety compliance. You help fleet dispatchers, safety officers, and drivers \
understand FMCSA/DOT regulations and logistics operations.

RULES:
- Answer ONLY based on the provided context documents
- Always cite the specific document and section you are referencing
- If the context does not contain enough information, say: \
"I don't have enough information in my documents to answer this reliably."
- Never fabricate regulation codes or legal requirements
- Keep answers concise and actionable for field use"""


def _format_docs(docs: List[Document]) -> str:
    """Concatenate retrieved documents into a single context string."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source: {source} | Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain(retriever: BaseRetriever):
    """Build and return the full RAG chain: retrieve → format → prompt → LLM → parse."""
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0,
            streaming=False,
            openai_api_key=settings.OPENAI_API_KEY,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Context documents:\n\n{context}\n\nQuestion: {question}",
                ),
            ]
        )

        chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("RAG chain built successfully with model '%s'", settings.LLM_MODEL)
        return chain
    except Exception as exc:
        logger.error("Failed to build RAG chain: %s", exc)
        raise


def run_query(chain, query: str) -> dict:
    """Execute *query* through *chain* and return the answer.

    Source attribution is not available through this convenience wrapper because
    the retriever reference is not retained by the compiled LCEL chain in
    LangChain v0.2+.  Use ``run_query_with_docs`` when you need source metadata.
    """
    try:
        answer = chain.invoke(query)
        return {
            "answer": answer,
            "sources": [],
            "num_docs_retrieved": 0,
        }
    except Exception as exc:
        logger.error("RAG query failed: %s", exc)
        raise


def run_query_with_docs(chain, retriever: BaseRetriever, query: str) -> dict:
    """Execute *query* and explicitly pass *retriever* for source attribution."""
    try:
        answer = chain.invoke(query)

        try:
            docs: List[Document] = retriever.invoke(query)
        except Exception:
            docs = []

        seen: set = set()
        sources = []
        for doc in docs:
            key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0))
            if key not in seen:
                seen.add(key)
                sources.append({"filename": key[0], "page": key[1]})

        return {
            "answer": answer,
            "sources": sources,
            "num_docs_retrieved": len(docs),
        }
    except Exception as exc:
        logger.error("RAG query (with docs) failed: %s", exc)
        raise


def run_query_with_web_context(
    retriever: BaseRetriever,
    query: str,
    web_docs: List[Document],
) -> dict:
    """Execute *query* combining ChromaDB results with *web_docs* as extra context.

    Web documents are injected directly into the context slot — they are never
    used as input to the retriever, which solves the corrective-RAG bug where
    Tavily content was previously prepended to the query string and lost.
    """
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0,
            streaming=False,
            openai_api_key=settings.OPENAI_API_KEY,
        )

        # Retrieve ChromaDB docs using the clean original query.
        try:
            chroma_docs: List[Document] = retriever.invoke(query)
        except Exception:
            chroma_docs = []

        # Web docs first so GPT sees them as priority context.
        all_docs = web_docs + chroma_docs
        context = _format_docs(all_docs)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "Context documents:\n\n{context}\n\nQuestion: {question}"),
            ]
        )
        messages = prompt.format_messages(context=context, question=query)
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        seen: set = set()
        sources = []
        for doc in all_docs:
            key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0))
            if key not in seen:
                seen.add(key)
                sources.append({"filename": key[0], "page": key[1]})

        logger.info(
            "Web-context RAG: %d web doc(s) + %d ChromaDB doc(s) → answer generated.",
            len(web_docs),
            len(chroma_docs),
        )
        return {
            "answer": answer,
            "sources": sources,
            "num_docs_retrieved": len(all_docs),
        }
    except Exception as exc:
        logger.error("RAG query with web context failed: %s", exc)
        raise


def check_relevance_score(retriever: BaseRetriever, query: str) -> float:
    """Retrieve docs for *query* and return the mean similarity score.

    Used by the corrective RAG logic to decide whether to trigger web search.
    Returns 0.0 on any failure so the caller can safely threshold-compare.
    """
    try:
        from langchain_chroma import Chroma

        # Unwrap EnsembleRetriever to get the underlying vector store retriever.
        base = retriever
        if hasattr(retriever, "retrievers"):
            for r in retriever.retrievers:  # type: ignore[attr-defined]
                if hasattr(r, "vectorstore"):
                    base = r
                    break

        if not hasattr(base, "vectorstore"):
            # Cannot compute score for non-vector retriever (e.g. pure BM25).
            return 1.0

        vs: Chroma = base.vectorstore  # type: ignore[attr-defined]
        results = vs.similarity_search_with_relevance_scores(query, k=5)
        if not results:
            return 0.0
        avg_score = sum(score for _, score in results) / len(results)
        logger.debug("Relevance score for query '%s': %.3f", query[:60], avg_score)
        return avg_score
    except Exception as exc:
        logger.warning("Could not compute relevance score: %s — defaulting to 1.0", exc)
        return 1.0
