"""
PDF parsing and chunking utilities for Kokoro AI.

Responsibilities:
- Validate PDF files
- Extract text using PyPDF
- Normalize extracted text
- Create LangChain Documents
- Split documents into retrieval-friendly chunks

This module does NOT:
- Generate embeddings
- Write to Pinecone
- Build the BM25 index
- Call Gemini

Those responsibilities belong to the ingestion/indexing layers.
"""

from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path
from typing import BinaryIO

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from utils.logger import logger
from utils.utils import normalize_text


# ============================================================
# Constants
# ============================================================

ALLOWED_PDF_EXTENSION = ".pdf"

DEFAULT_CHUNK_SIZE = 800

DEFAULT_CHUNK_OVERLAP = 120

MIN_EXTRACTED_TEXT_LENGTH = 20


# ============================================================
# PDF Validation
# ============================================================


def validate_pdf(
    pdf_data: bytes,
    filename: str | None = None,
) -> None:
    """
    Validate that the supplied bytes represent a readable PDF.

    Parameters
    ----------
    pdf_data:
        Raw PDF bytes.

    filename:
        Optional filename used to validate the extension.

    Raises
    ------
    ValueError
        If the input is invalid or cannot be read as a PDF.
    """

    if not pdf_data:
        raise ValueError("PDF file is empty.")

    if filename:
        extension = Path(filename).suffix.lower()

        if extension != ALLOWED_PDF_EXTENSION:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                "Only PDF files are supported."
            )

    # Basic PDF signature check.
    if not pdf_data.startswith(b"%PDF"):
        raise ValueError(
            "Invalid PDF file signature."
        )

    try:
        reader = PdfReader(
            io.BytesIO(pdf_data)
        )

        if not reader.pages:
            raise ValueError(
                "PDF does not contain any pages."
            )

    except Exception as exc:
        raise ValueError(
            "Unable to read the supplied PDF."
        ) from exc


# ============================================================
# Unicode Normalization
# ============================================================


def normalize_unicode(text: str) -> str:
    """
    Apply safe Unicode normalization.

    This helps normalize visually equivalent Unicode
    representations without aggressively modifying the
    document's content.

    Parameters
    ----------
    text:
        Extracted PDF text.

    Returns
    -------
    str
        Unicode-normalized text.
    """

    if not text:
        return ""

    return unicodedata.normalize(
        "NFKC",
        text,
    )


# ============================================================
# PDF Text Extraction
# ============================================================


def extract_pdf_text(
    pdf_data: bytes,
) -> list[dict[str, object]]:
    """
    Extract text from every PDF page.

    Parameters
    ----------
    pdf_data:
        Raw PDF bytes.

    Returns
    -------
    list[dict]
        Page-level extracted text.

        Example:
        [
            {
                "page_number": 1,
                "text": "Resume content..."
            }
        ]

    Raises
    ------
    ValueError
        If the PDF cannot be read or contains no useful text.
    """

    validate_pdf(pdf_data)

    try:
        reader = PdfReader(
            io.BytesIO(pdf_data)
        )

        pages: list[dict[str, object]] = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            raw_text = page.extract_text() or ""

            text = normalize_unicode(
                raw_text
            )

            text = normalize_text(
                text
            )

            if text:
                pages.append(
                    {
                        "page_number": page_number,
                        "text": text,
                    }
                )

        if not pages:
            raise ValueError(
                "No extractable text was found in the PDF. "
                "The document may be scanned/image-only."
            )

        total_text = " ".join(
            str(page["text"])
            for page in pages
        )

        if len(total_text.strip()) < MIN_EXTRACTED_TEXT_LENGTH:
            raise ValueError(
                "The PDF contains insufficient extractable text."
            )

        logger.info(
            "PDF text extraction completed | pages=%d",
            len(pages),
        )

        return pages

    except ValueError:
        raise

    except Exception as exc:
        logger.exception(
            "PDF text extraction failed."
        )

        raise ValueError(
            "Failed to extract text from PDF."
        ) from exc


# ============================================================
# Page Text Preparation
# ============================================================


def prepare_page_text(
    text: str,
) -> str:
    """
    Perform safe preprocessing on extracted page text.

    The preprocessing intentionally avoids:
    - aggressive lowercasing
    - stop-word removal
    - stemming
    - punctuation removal

    This is important for resumes because information such as:
    Python, C++, SQL, .NET, AWS, GCP, dates, percentages,
    and job titles should remain readable.

    Parameters
    ----------
    text:
        Raw extracted page text.

    Returns
    -------
    str
        Cleaned text.
    """

    if not text:
        return ""

    text = normalize_unicode(
        text
    )

    text = normalize_text(
        text
    )

    # Normalize common PDF line-break artifacts.
    text = re.sub(
        r"-\n(?=\w)",
        "",
        text,
    )

    # Convert single line breaks occurring inside a sentence
    # into spaces while preserving paragraph boundaries.
    text = re.sub(
        r"(?<!\n)\n(?!\n)",
        " ",
        text,
    )

    # Collapse excessive spaces again after line processing.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Keep a maximum of two consecutive newlines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# LangChain Documents
# ============================================================


def create_documents(
    pages: list[dict[str, object]],
    document_id: str,
    document_type: str,
    source_file: str,
    document_hash: str,
    candidate_id: str | None = None,
    candidate_name: str | None = None,
) -> list[Document]:
    """
    Convert page-level text into LangChain Documents.

    Each page initially becomes one Document. Chunking is
    performed separately by `split_documents()`.

    Parameters
    ----------
    pages:
        Output from `extract_pdf_text()`.

    document_id:
        Unique document identifier.

    document_type:
        Type of document, e.g. "resume" or "jd".

    source_file:
        Original filename.

    document_hash:
        SHA-256 hash of the source PDF.

    candidate_id:
        Optional candidate identifier.

    candidate_name:
        Optional candidate name.

    Returns
    -------
    list[Document]
        LangChain Documents with metadata.
    """

    documents: list[Document] = []

    for page in pages:
        page_number = int(
            page["page_number"]
        )

        text = str(
            page["text"]
        )

        text = prepare_page_text(
            text
        )

        if not text:
            continue

        metadata = {
            "document_id": document_id,
            "document_type": document_type,
            "source_file": source_file,
            "document_hash": document_hash,
            "page_number": page_number,
        }

        if candidate_id:
            metadata["candidate_id"] = candidate_id

        if candidate_name:
            metadata["candidate_name"] = candidate_name

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    if not documents:
        raise ValueError(
            "No usable documents were created from the PDF."
        )

    return documents


# ============================================================
# Document Chunking
# ============================================================


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split Documents into retrieval-friendly chunks.

    Uses RecursiveCharacterTextSplitter so that the splitter
    attempts to preserve natural boundaries before splitting
    smaller units.

    Parameters
    ----------
    documents:
        Page-level LangChain Documents.

    chunk_size:
        Maximum target chunk size.

    chunk_overlap:
        Number of overlapping characters between chunks.

    Returns
    -------
    list[Document]
        Chunked LangChain Documents.
    """

    if not documents:
        raise ValueError(
            "documents cannot be empty."
        )

    if chunk_size < 1:
        raise ValueError(
            "chunk_size must be >= 1."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        length_function=len,
    )

    chunks = splitter.split_documents(
        documents
    )

    # Add deterministic chunk IDs.
    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata['document_id']}_chunk_{index}"
        )

        chunk.metadata["chunk_index"] = index

    logger.info(
        "Document chunking completed | "
        "documents=%d | chunks=%d | "
        "chunk_size=%d | overlap=%d",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )

    return chunks


# ============================================================
# Complete Parsing Pipeline
# ============================================================


def parse_pdf(
    pdf_data: bytes,
    document_id: str,
    document_type: str,
    source_file: str,
    document_hash: str,
    candidate_id: str | None = None,
    candidate_name: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Execute the complete PDF parsing pipeline.

    Pipeline:

        PDF bytes
            ↓
        Validation
            ↓
        Text extraction
            ↓
        Unicode normalization
            ↓
        Safe preprocessing
            ↓
        LangChain Documents
            ↓
        Recursive chunking
            ↓
        Chunked Documents

    Parameters
    ----------
    pdf_data:
        Raw PDF bytes.

    document_id:
        Unique document ID.

    document_type:
        "resume" or "jd".

    source_file:
        Original PDF filename.

    document_hash:
        SHA-256 hash of the PDF.

    candidate_id:
        Optional candidate ID.

    candidate_name:
        Optional candidate name.

    chunk_size:
        Chunk size for recursive splitting.

    chunk_overlap:
        Chunk overlap.

    Returns
    -------
    list[Document]
        Chunked LangChain Documents.
    """

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_pdf(
        pdf_data=pdf_data,
        filename=source_file,
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    pages = extract_pdf_text(
        pdf_data
    )

    # --------------------------------------------------------
    # Create LangChain Documents
    # --------------------------------------------------------

    documents = create_documents(
        pages=pages,
        document_id=document_id,
        document_type=document_type,
        source_file=source_file,
        document_hash=document_hash,
        candidate_id=candidate_id,
        candidate_name=candidate_name,
    )

    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    chunks = split_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    logger.info(
        "PDF parsing pipeline completed | "
        "document_id=%s | type=%s | chunks=%d",
        document_id,
        document_type,
        len(chunks),
    )

    return chunks