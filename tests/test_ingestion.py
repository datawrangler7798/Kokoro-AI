"""
Tests for Kokoro ingestion module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.ingestion.ingestion import (
    FileIngestionService,
)


# ============================================================
# Helpers
# ============================================================


def create_service(tmp_path: Path) -> FileIngestionService:
    """
    Create an ingestion service using temporary directories.
    """

    return FileIngestionService(
        resume_directory=tmp_path / "resumes",
        jd_directory=tmp_path / "jds",
    )


# ============================================================
# Initialization
# ============================================================


def test_ingestion_service_initialization(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    assert service is not None
    assert service.resume_directory == (
        tmp_path / "resumes"
    )
    assert service.jd_directory == (
        tmp_path / "jds"
    )


# ============================================================
# File Validation
# ============================================================


def test_valid_pdf_file(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    pdf_file = tmp_path / "resume.pdf"

    # Minimal PDF header for validation testing.
    pdf_file.write_bytes(
        b"%PDF-1.4\n"
    )

    assert service.validate_file(
        pdf_file
    ) is True


def test_non_pdf_file_rejected(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    txt_file = tmp_path / "resume.txt"

    txt_file.write_text(
        "Python developer",
        encoding="utf-8",
    )

    assert service.validate_file(
        txt_file
    ) is False


def test_missing_file_rejected(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    missing_file = (
        tmp_path / "missing.pdf"
    )

    assert service.validate_file(
        missing_file
    ) is False


# ============================================================
# File Hash
# ============================================================


def test_file_hash_is_deterministic(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    file_path = tmp_path / "resume.pdf"

    file_path.write_bytes(
        b"%PDF-1.4\ncandidate"
    )

    first_hash = service.get_file_hash(
        file_path
    )

    second_hash = service.get_file_hash(
        file_path
    )

    assert first_hash == second_hash
    assert len(first_hash) > 0


# ============================================================
# Duplicate Detection
# ============================================================


def test_duplicate_file_detection(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    file_path = tmp_path / "resume.pdf"

    file_path.write_bytes(
        b"%PDF-1.4\ncandidate"
    )

    file_hash = service.get_file_hash(
        file_path
    )

    assert (
        service.is_duplicate(
            file_hash
        )
        is False
    )

    service.register_hash(
        file_hash
    )

    assert (
        service.is_duplicate(
            file_hash
        )
        is True
    )


# ============================================================
# File Size
# ============================================================


def test_file_size(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    file_path = tmp_path / "resume.pdf"

    content = b"%PDF-1.4\ncandidate"

    file_path.write_bytes(content)

    assert service.get_file_size(
        file_path
    ) == len(content)


# ============================================================
# Directory Preparation
# ============================================================


def test_directories_are_created(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    service.ensure_directories()

    assert service.resume_directory.exists()
    assert service.jd_directory.exists()


# ============================================================
# Unsupported Extension
# ============================================================


def test_unsupported_extension(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    file_path = tmp_path / "candidate.docx"

    file_path.write_bytes(
        b"candidate"
    )

    assert service.validate_file(
        file_path
    ) is False


# ============================================================
# Empty File
# ============================================================


def test_empty_file_rejected(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    file_path = tmp_path / "empty.pdf"

    file_path.write_bytes(
        b""
    )

    assert service.validate_file(
        file_path
    ) is False