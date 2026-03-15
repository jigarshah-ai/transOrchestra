"""Embedding model factory and comparison utilities for TransOrchestra."""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.config import EMBEDDING_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)


def get_embedding_model(model_name: str = None) -> Embeddings:
    """Return the appropriate LangChain Embeddings object for *model_name*.

    Defaults to the EMBEDDING_MODEL env variable when *model_name* is None.
    OpenAI models are identified by the 'text-embedding' prefix; everything
    else is loaded via HuggingFaceEmbeddings (sentence-transformers).
    """
    if model_name is None:
        model_name = EMBEDDING_MODEL

    if model_name.startswith("text-embedding"):
        logger.info("Loading OpenAI embedding model: %s", model_name)
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
        except Exception as exc:
            logger.error("Failed to load OpenAI embeddings (%s): %s", model_name, exc)
            raise

    logger.info(
        "Loading HuggingFace embedding model: %s  (this may take 30-60 s on first run)",
        model_name,
    )
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        logger.error("Failed to load HuggingFace embeddings (%s): %s", model_name, exc)
        raise


def compare_embeddings(query: str, docs: List[Document]) -> dict:
    """Embed *query* with both BGE and OpenAI and return top-3 results from each.

    Returns a dict with keys 'bge' and 'openai', each containing a list of
    {'text': str, 'score': float} dicts sorted by descending similarity.
    """
    import numpy as np

    bge_model = get_embedding_model("BAAI/bge-small-en-v1.5")
    openai_model = get_embedding_model("text-embedding-3-small")

    doc_texts = [d.page_content for d in docs]

    def _top3(emb_model: Embeddings) -> List[dict]:
        try:
            q_vec = emb_model.embed_query(query)
            d_vecs = emb_model.embed_documents(doc_texts)
            q_arr = np.array(q_vec)
            d_arr = np.array(d_vecs)
            scores = (d_arr @ q_arr) / (
                np.linalg.norm(d_arr, axis=1) * np.linalg.norm(q_arr) + 1e-9
            )
            top_idx = np.argsort(scores)[::-1][:3]
            return [{"text": doc_texts[i][:200], "score": float(scores[i])} for i in top_idx]
        except Exception as exc:
            logger.error("Embedding comparison failed: %s", exc)
            return []

    return {
        "bge": _top3(bge_model),
        "openai": _top3(openai_model),
    }
