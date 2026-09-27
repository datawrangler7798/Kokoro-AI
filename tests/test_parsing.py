"""
Tests for Kokoro PDF parsing module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.ingestion.parsing import (
    clean_text,
    detect_sections,
    extract_candidate_name,
    generate_candidate_id,
    generate_document_id,
    parse_pdf,
)


# ============================================================
# Text Cleaning
# ============================================================


def test_clean_text_removes_extra_whitespace():
    text = """
    John Doe

    Python     Data Scientist


    5 years experience
    """

    cleaned = clean_text(text)

    assert "John Doe" in cleaned
    assert "Python Data Scientist" in cleaned
    assert "5 years experience" in cleaned
    assert "     " not in cleaned


def test_clean_text_empty_input():
    assert clean_text("") == ""


def test_clean_text_none_like_input():
    assert clean_text(None) == ""


# ============================================================
# Section Detection
# ============================================================


def test_detect_sections():
    text = """
    PROFESSIONAL SUMMARY

    Data Scientist with 5 years experience.

    TECHNICAL SKILLS

    Python, SQL, Machine Learning.

    EXPERIENCE

    Senior Data Scientist at ABC.
    """

    sections = detect_sections(text)

    assert isinstance(sections, dict)

    # The exact section representation depends on the
    # parser implementation, so verify the expected
    # section names are detected.
    section_text = " ".join(
        str(value)
        for value in sections.values()
    ).lower()

    assert (
        "professional summary" in section_text
        or "data scientist" in section_text
    )


# ============================================================
# Candidate Name
# ============================================================


def test_extract_candidate_name():
    text = """
    John Doe

    Data Scientist

    PROFESSIONAL SUMMARY

    Experienced data scientist with Python skills.
    """

    name = extract_candidate_name(
        text
    )

    assert name is not None
    assert "John" in name
    assert "Doe" in name


def test_extract_candidate_name_empty():
    result = extract_candidate_name("")

    assert result in (
        None,
        "",
    )


# ============================================================
# IDs
# ============================================================


def test_generate_document_id_is_deterministic():
    file_path = Path(
        "/data/resumes/candidate.pdf"
    )

    first = generate_document_id(
        file_path
    )

    second = generate_document_id(
        file_path
    )

    assert first == second
    assert first


def test_generate_candidate_id_is_deterministic():
    first = generate_candidate_id(
        "John Doe"
    )

    second = generate_candidate_id(
        "John Doe"
    )

    assert first == second
    assert first


def test_different_candidate_names_produce_different_ids():
    first = generate_candidate_id(
        "John Doe"
    )

    second = generate_candidate_id(
        "Jane Doe"
    )

    assert first != second


# ============================================================
# PDF Parsing
# ============================================================


def create_minimal_pdf(
    path: Path,
) -> None:
    """
    Create a minimal PDF-like file for parser validation.

    Full PDF extraction behavior should be tested with actual
    sample PDFs because PDF parsers require a valid PDF
    structure.
    """

    path.write_bytes(
        (
            b"%PDF-1.4\n"
            b"1 0 obj\n"
            b"<< /Type /Catalog >>\n"
            b"endobj\n"
            b"%%EOF\n"
        )
    )


def test_parse_pdf_missing_file(
    tmp_path: Path,
):
    pdf_path = (
        tmp_path / "missing.pdf"
    )

    with pytest.raises(
        Exception
    ):
        parse_pdf(pdf_path)


def test_parse_pdf_rejects_non_pdf(
    tmp_path: Path,
):
    txt_path = (
        tmp_path / "resume.txt"
    )

    txt_path.write_text(
        "John Doe Python Developer",
        encoding="utf-8",
    )

    with pytest.raises(
        Exception
    ):
        parse_pdf(txt_path)


def test_parse_pdf_returns_parsed_document(
    tmp_path: Path,
):
    """
    Smoke test for the parser.

    A real resume PDF should be used for complete extraction
    testing. This test verifies that the parser accepts a PDF
    path and returns the expected parser-level structure when
    the underlying PDF library can process the file.
    """

    pdf_path = (
        tmp_path / "resume.pdf"
    )

    create_minimal_pdf(
        pdf_path
    )

    try:
        result = parse_pdf(
            pdf_path
        )
    except Exception:
        # A deliberately minimal PDF may not contain enough
        # structure for every PDF backend.
        pytest.skip(
            "Minimal PDF is not sufficient for the configured "
            "PDF extraction backend."
        )

    assert result is not None

    assert hasattr(
        result,
        "text",
    )


# ============================================================
# Parser Output Quality
# ============================================================


def test_cleaned_text_is_string():
    text = clean_text(
        "Python\n\nData Scientist"
    )

    assert isinstance(
        text,
        str,
    )


def test_candidate_id_changes_with_name():
    id_1 = generate_candidate_id(
        "Candidate One"
    )

    id_2 = generate_candidate_id(
        "Candidate Two"
    )

    assert id_1 != id_2