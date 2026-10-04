"""Tests for the current PDF ingestion service API."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.ingestion.ingestion import IngestionService
from utils.config import settings
from utils.schemas import DocumentType


class MemoryRegistry:
    """Small in-memory registry for ingestion unit tests."""

    def __init__(self) -> None:
        self.document_hashes: set[str] = set()

    def exists(self, document_hash: str) -> bool:
        return document_hash in self.document_hashes

    def save(self, document: object) -> None:
        self.document_hashes.add(document.document_hash)


def create_service(registry: MemoryRegistry | None = None) -> IngestionService:
    """Create the service without connecting to Pinecone."""
    return IngestionService(vector_store=object(), registry=registry)


def test_ingestion_service_initialization() -> None:
    service = create_service()

    assert isinstance(service, IngestionService)
    assert service.get_storage_directory(DocumentType.RESUME) == Path(
        settings.RESUME_DIRECTORY
    )
    assert service.get_storage_directory(DocumentType.JD) == Path(settings.JD_DIRECTORY)


def test_valid_pdf_file_passes_validation(tmp_path: Path) -> None:
    pdf_file = tmp_path / "resume.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n")

    assert create_service().validate_file(pdf_file) is None


def test_non_pdf_file_is_rejected(tmp_path: Path) -> None:
    text_file = tmp_path / "resume.txt"
    text_file.write_text("Python developer", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        create_service().validate_file(text_file)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="File does not exist"):
        create_service().validate_file(tmp_path / "missing.pdf")


def test_file_hash_is_deterministic(tmp_path: Path) -> None:
    file_path = tmp_path / "resume.pdf"
    file_path.write_bytes(b"%PDF-1.4\ncandidate")
    service = create_service()

    first_hash = service.calculate_hash(file_path)
    second_hash = service.calculate_hash(file_path)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_duplicate_detection_uses_document_registry(tmp_path: Path) -> None:
    registry = MemoryRegistry()
    service = create_service(registry)
    document_hash = "abc123"

    assert service.is_duplicate(document_hash) is False
    registry.document_hashes.add(document_hash)
    assert service.is_duplicate(document_hash) is True


def test_save_file_creates_the_configured_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resume_directory = tmp_path / "resumes"
    monkeypatch.setattr(settings, "RESUME_DIRECTORY", str(resume_directory))
    source = tmp_path / "candidate.pdf"
    source.write_bytes(b"%PDF-1.4\ncandidate")

    stored_path = create_service().save_file(source, DocumentType.RESUME)

    assert stored_path == resume_directory / source.name
    assert stored_path.read_bytes() == source.read_bytes()


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    file_path = tmp_path / "candidate.docx"
    file_path.write_bytes(b"candidate")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        create_service().validate_file(file_path)
