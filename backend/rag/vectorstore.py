"""ChromaDB vector store management for TransOrchestra."""

import logging
import os
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.config import CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

COLLECTION_NAME = "transOrchestra_docs"


def build_vectorstore(docs: List[Document], embedding_model: Embeddings):
    """Create and persist a ChromaDB collection from *docs*.

    Returns the populated Chroma instance.
    """
    try:
        from langchain_chroma import Chroma

        logger.info(
            "Building vector store with %d documents → collection '%s'",
            len(docs),
            COLLECTION_NAME,
        )
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        logger.info(
            "Vector store persisted to %s (%d documents added)",
            CHROMA_PERSIST_DIR,
            len(docs),
        )
        return vectorstore
    except Exception as exc:
        logger.error("Failed to build vector store: %s", exc)
        raise


def load_vectorstore(embedding_model: Embeddings):
    """Load an existing ChromaDB collection from disk.

    Raises FileNotFoundError with a helpful message if the store is missing.
    """
    if not os.path.isdir(CHROMA_PERSIST_DIR):
        raise FileNotFoundError(
            f"No vector store found at '{CHROMA_PERSIST_DIR}'. "
            "Run scripts/ingest.py first to create it."
        )

    try:
        from langchain_chroma import Chroma

        logger.info("Loading vector store from %s", CHROMA_PERSIST_DIR)
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        count = vectorstore._collection.count()
        logger.info("Vector store loaded — %d documents in collection", count)
        return vectorstore
    except FileNotFoundError:
        raise
    except Exception as exc:
        logger.error("Failed to load vector store: %s", exc)
        raise


def get_or_create_vectorstore(docs: List[Document], embedding_model: Embeddings):
    """Load the vector store if it exists, otherwise build it from *docs*.

    Intended for the FastAPI startup event.
    """
    try:
        return load_vectorstore(embedding_model)
    except FileNotFoundError:
        logger.info("No existing vector store found — building from provided documents.")
        if not docs:
            logger.warning(
                "No documents supplied to get_or_create_vectorstore. "
                "The vector store will be empty until ingest.py is run."
            )
        return build_vectorstore(docs, embedding_model)
