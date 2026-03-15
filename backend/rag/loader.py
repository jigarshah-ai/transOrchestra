"""Document loading and chunking utilities for TransOrchestra."""

import argparse
import logging
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


def load_and_chunk_pdfs(pdf_dir: str) -> List[Document]:
    """Load every PDF in *pdf_dir*, split into chunks, and return all Document objects."""
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_docs: List[Document] = []
    for filename in pdf_files:
        file_path = os.path.join(pdf_dir, filename)
        try:
            raw_docs = PyPDFLoader(file_path).load()
            chunks = splitter.split_documents(raw_docs)
            for chunk in chunks:
                chunk.metadata["source"] = filename
                chunk.metadata["page"] = chunk.metadata.get("page", 0)
            all_docs.extend(chunks)
            logger.info("Loaded %s → %d chunks", filename, len(chunks))
        except Exception as exc:
            logger.error("Failed to load %s: %s", filename, exc)

    logger.info(
        "Ingestion complete — %d files, %d total chunks", len(pdf_files), len(all_docs)
    )
    return all_docs


def load_single_pdf(file_path: str) -> List[Document]:
    """Load and chunk a single PDF file — used by the Streamlit sidebar upload."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    try:
        raw_docs = PyPDFLoader(file_path).load()
        chunks = splitter.split_documents(raw_docs)
        filename = os.path.basename(file_path)
        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["page"] = chunk.metadata.get("page", 0)
        logger.info("Loaded single file %s → %d chunks", filename, len(chunks))
        return chunks
    except Exception as exc:
        logger.error("Failed to load %s: %s", file_path, exc)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Load and chunk PDFs for TransOrchestra")
    parser.add_argument("--pdf_dir", default="./data", help="Directory containing PDF files")
    args = parser.parse_args()

    docs = load_and_chunk_pdfs(args.pdf_dir)
    print(f"\n{'='*60}")
    print(f"Total chunks created: {len(docs)}")
    print(f"{'='*60}")
    print("\nSample chunks (first 3):")
    for i, doc in enumerate(docs[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
        print(f"Content preview: {doc.page_content[:200]}...")
