"""Unit tests for backend.rag.loader."""

import os
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


def _make_fake_pdf(directory: str, filename: str = "test.pdf") -> str:
    """Write a placeholder text file in *directory* and return its path.

    We mock PyPDFLoader so the actual file does not need to be a valid PDF.
    """
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("placeholder")
    return path


@pytest.fixture()
def sample_docs():
    """Return a small list of Document objects mimicking loaded PDF chunks."""
    return [
        Document(
            page_content="Section 396.11 requires drivers to submit a DVIR at day end.",
            metadata={"source": "fmcsa.pdf", "page": 1},
        ),
        Document(
            page_content="HazMat placard 1203 indicates gasoline per 49 CFR 172.504.",
            metadata={"source": "fmcsa.pdf", "page": 5},
        ),
        Document(
            page_content="Drivers may drive at most 11 hours after 10 hours off duty.",
            metadata={"source": "hours_of_service.pdf", "page": 2},
        ),
    ]


class TestLoadAndChunkPdfs:
    """Tests for load_and_chunk_pdfs."""

    def test_raises_on_missing_directory(self):
        from backend.rag.loader import load_and_chunk_pdfs

        with pytest.raises(FileNotFoundError, match="PDF directory not found"):
            load_and_chunk_pdfs("/nonexistent/path/xyz")

    def test_returns_empty_list_when_no_pdfs(self, tmp_path):
        from backend.rag.loader import load_and_chunk_pdfs

        result = load_and_chunk_pdfs(str(tmp_path))
        assert result == []

    def test_metadata_injected_correctly(self, tmp_path, sample_docs):
        """Loader should attach 'source' and 'page' keys to every chunk."""
        _make_fake_pdf(str(tmp_path))

        with patch("backend.rag.loader.PyPDFLoader") as mock_loader_cls:
            mock_instance = MagicMock()
            mock_instance.load.return_value = [
                Document(page_content="Raw text from PDF page 1.", metadata={"page": 0})
            ]
            mock_loader_cls.return_value = mock_instance

            from backend.rag.loader import load_and_chunk_pdfs

            docs = load_and_chunk_pdfs(str(tmp_path))

        for doc in docs:
            assert "source" in doc.metadata
            assert "page" in doc.metadata

    def test_chunk_count_is_positive(self, tmp_path):
        """Splitting a non-trivial page should produce at least one chunk."""
        _make_fake_pdf(str(tmp_path))

        long_text = " ".join(["word"] * 600)  # > default CHUNK_SIZE
        with patch("backend.rag.loader.PyPDFLoader") as mock_loader_cls:
            mock_instance = MagicMock()
            mock_instance.load.return_value = [
                Document(page_content=long_text, metadata={"page": 0})
            ]
            mock_loader_cls.return_value = mock_instance

            from backend.rag.loader import load_and_chunk_pdfs

            docs = load_and_chunk_pdfs(str(tmp_path))

        assert len(docs) >= 1


class TestLoadSinglePdf:
    """Tests for load_single_pdf."""

    def test_raises_on_missing_file(self):
        from backend.rag.loader import load_single_pdf

        with pytest.raises(FileNotFoundError):
            load_single_pdf("/nonexistent/file.pdf")

    def test_returns_documents(self, tmp_path):
        pdf_path = _make_fake_pdf(str(tmp_path))

        with patch("backend.rag.loader.PyPDFLoader") as mock_loader_cls:
            mock_instance = MagicMock()
            mock_instance.load.return_value = [
                Document(page_content="Single page content.", metadata={"page": 0})
            ]
            mock_loader_cls.return_value = mock_instance

            from backend.rag.loader import load_single_pdf

            docs = load_single_pdf(pdf_path)

        assert isinstance(docs, list)
        assert len(docs) >= 1
        assert docs[0].metadata["source"] == "test.pdf"
