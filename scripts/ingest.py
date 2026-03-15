"""Standalone ingestion script — load PDFs and build the ChromaDB vector store."""

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is importable when running the script directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.config import EMBEDDING_MODEL, get_config
from backend.rag.embeddings import get_embedding_model
from backend.rag.loader import load_and_chunk_pdfs
from backend.rag.vectorstore import build_vectorstore


def main() -> None:
    """Parse arguments, ingest PDFs, and build the vector store."""
    parser = argparse.ArgumentParser(
        description="TransOrchestra — document ingestion pipeline"
    )
    parser.add_argument(
        "--pdf_dir",
        default="./data",
        help="Directory containing PDF files to ingest (default: ./data)",
    )
    parser.add_argument(
        "--embedding_model",
        default=None,
        help=(
            "Embedding model to use. Defaults to EMBEDDING_MODEL from config "
            f"(currently: {EMBEDDING_MODEL})"
        ),
    )
    args = parser.parse_args()

    cfg = get_config()
    model_name = args.embedding_model or cfg["embedding_model"]

    print(f"\n{'='*60}")
    print("TransOrchestra — Document Ingestion")
    print(f"{'='*60}")
    print(f"PDF directory  : {args.pdf_dir}")
    print(f"Embedding model: {model_name}")
    print(f"Vector store   : {cfg['chroma_persist_dir']}")
    print(f"{'='*60}\n")

    t0 = time.time()

    print("Step 1/3 — Loading and chunking PDF files...")
    docs = load_and_chunk_pdfs(args.pdf_dir)
    if not docs:
        print("⚠️  No documents were loaded. Add PDF files to the data/ directory and retry.")
        sys.exit(1)

    # Count unique source files.
    sources = {d.metadata.get("source", "unknown") for d in docs}
    print(f"  ✓ Loaded {len(sources)} file(s), {len(docs)} chunks total\n")

    print("Step 2/3 — Initialising embedding model...")
    embedding_model = get_embedding_model(model_name)
    print(f"  ✓ Model ready: {model_name}\n")

    print("Step 3/3 — Building and persisting ChromaDB vector store...")
    build_vectorstore(docs, embedding_model)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅  Ingestion complete in {elapsed:.1f}s")
    print(f"   Files processed : {len(sources)}")
    print(f"   Chunks created  : {len(docs)}")
    print(f"   Vector store    : {cfg['chroma_persist_dir']}")
    print(f"{'='*60}")
    print("\nNext step → start the API server:")
    print("  uvicorn backend.main:app --reload")
    print("\nThen launch the Streamlit frontend:")
    print("  streamlit run frontend/app.py\n")


if __name__ == "__main__":
    main()
