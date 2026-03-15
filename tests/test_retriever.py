"""Unit tests for backend.rag.retriever."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


@pytest.fixture()
def sample_docs():
    """Minimal corpus for retriever tests."""
    return [
        Document(
            page_content="Section 396.11 DVIR requirements for commercial drivers.",
            metadata={"source": "fmcsa.pdf", "page": 1},
        ),
        Document(
            page_content="HazMat placards must comply with 49 CFR 172.504.",
            metadata={"source": "hazmat.pdf", "page": 3},
        ),
        Document(
            page_content="Air brake pressure must reach 100 psi within 45 seconds.",
            metadata={"source": "brakes.pdf", "page": 7},
        ),
    ]


@pytest.fixture()
def mock_vectorstore():
    """Return a lightweight mock Chroma instance."""
    vs = MagicMock()
    vs.as_retriever.return_value = MagicMock()
    return vs


class TestBuildVectorRetriever:
    """Tests for build_vector_retriever."""

    def test_returns_retriever(self, mock_vectorstore):
        from backend.rag.retriever import build_vector_retriever

        retriever = build_vector_retriever(mock_vectorstore, top_k=5)
        mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
        assert retriever is not None

    def test_uses_default_top_k(self, mock_vectorstore):
        from backend.config import TOP_K_RETRIEVAL
        from backend.rag.retriever import build_vector_retriever

        build_vector_retriever(mock_vectorstore)
        mock_vectorstore.as_retriever.assert_called_once_with(
            search_kwargs={"k": TOP_K_RETRIEVAL}
        )


class TestBuildBm25Retriever:
    """Tests for build_bm25_retriever."""

    def test_builds_successfully(self, sample_docs):
        from backend.rag.retriever import build_bm25_retriever

        retriever = build_bm25_retriever(sample_docs, top_k=3)
        assert retriever is not None
        assert retriever.k == 3

    def test_empty_docs_raises(self):
        """BM25Retriever raises on empty corpus."""
        from backend.rag.retriever import build_bm25_retriever

        with pytest.raises(Exception):
            build_bm25_retriever([], top_k=5)


class TestBuildHybridRetriever:
    """Tests for build_hybrid_retriever."""

    def test_returns_ensemble_retriever(self, mock_vectorstore, sample_docs):
        from backend.rag.retriever import build_hybrid_retriever

        retriever = build_hybrid_retriever(mock_vectorstore, sample_docs)
        assert retriever is not None
        assert hasattr(retriever, "retrievers")
        assert len(retriever.retrievers) == 2

    def test_weights_sum_to_one(self, mock_vectorstore, sample_docs):
        from backend.rag.retriever import build_hybrid_retriever

        retriever = build_hybrid_retriever(mock_vectorstore, sample_docs)
        total = sum(retriever.weights)
        assert abs(total - 1.0) < 1e-6


class TestBuildRerankingRetriever:
    """Tests for build_reranking_retriever."""

    def test_falls_back_without_cohere_key(self, sample_docs, mock_vectorstore):
        """When COHERE_API_KEY is empty the function should return the base retriever."""
        from backend.rag.retriever import build_reranking_retriever

        base = MagicMock()
        with patch("backend.rag.retriever.COHERE_API_KEY", ""):
            result = build_reranking_retriever(base, sample_docs)
        assert result is base

    def test_wraps_with_cohere_key(self, sample_docs, mock_vectorstore):
        """When COHERE_API_KEY is set, returns a ContextualCompressionRetriever."""
        base = MagicMock()
        with patch("backend.rag.retriever.COHERE_API_KEY", "test-key-123"):
            with patch("backend.rag.retriever.CohereRerank"):
                with patch("backend.rag.retriever.ContextualCompressionRetriever") as mock_ccr:
                    mock_ccr.return_value = MagicMock()
                    from backend.rag.retriever import build_reranking_retriever

                    result = build_reranking_retriever(base, sample_docs)
                    assert mock_ccr.called
