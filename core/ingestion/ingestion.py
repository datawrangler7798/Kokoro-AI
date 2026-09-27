"""
Document ingestion orchestration for Kokoro AI.

Responsibilities:
- Validate uploaded PDF
- Calculate SHA-256 document hash
- Detect duplicate documents
- Save source PDF locally
- Parse and chunk the document
- Prepare the ingestion result for indexing

This module intentionally does not implement:
- Embedding generation
- Pinecone operations
- BM25 indexing

Those responsibilities belong to the retrieval/indexing layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from langchain_core.documents import Document

from core.ingestion.parsing import parse_pdf
from utils.logger import logger
from utils.utils import calculate_sha256


# ============================================================
# Constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RESUME_DIR = DATA_DIR / "resumes"

JD_DIR = DATA_DIR / "jds"

DocumentType = Literal["resume", "jd"]


# ============================================================
# Result Schema
# ============================================================


@dataclass
class IngestionResult:
    """
    Result returned after successful document ingestion.

    Attributes
    ----------
    document_id:
        Unique ID assigned to the uploaded document.

    document_type:
        Either "resume" or "jd".

    source_file:
        Original filename.

    stored_path:
        Local path where the original PDF was stored.

    document_hash:
        SHA-256 hash of the original PDF.

    chunks:
        Parsed and chunked LangChain Documents.

    is_duplicate:
        Whether the uploaded document was already present.
    """

    document_id: str

    document_type: DocumentType

    source_file: str

    stored_path: Path

    document_hash: str

    chunks: list[Document]

    is_duplicate: bool = False


# ============================================================
# Directory Helpers
# ============================================================


def _get_storage_directory(
    document_type: DocumentType,
) -> Path:
    """
    Return the local storage directory for a document type.
    """

    if document_type == "resume":
        directory = RESUME_DIR

    elif document_type == "jd":
        directory = JD_DIR

    else:
        raise ValueError(
            f"Unsupported document type: {document_type}"
        )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


# ============================================================
# Filename Validation
# ============================================================


def _validate_filename(
    filename: str,
) -> str:
    """
    Validate and sanitize an uploaded filename.

    Only the filename itself is retained; directory traversal
    components are discarded.
    """

    if not filename or not filename.strip():
        raise ValueError(
            "Filename cannot be empty."
        )

    safe_name = Path(
        filename.strip()
    ).name

    if not safe_name:
        raise ValueError(
            "Invalid filename."
        )

    if Path(safe_name).suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    return safe_name


# ============================================================
# Duplicate Detection
# ============================================================


def _find_existing_document_by_hash(
    storage_directory: Path,
    document_hash: str,
) -> Path | None:
    """
    Search locally stored PDFs for a matching SHA-256 hash.

    This is intentionally implemented as a local filesystem
    check for the current Kokoro development architecture.

    For a larger production deployment, this lookup can later
    be replaced by a persistent document registry without
    changing the parsing layer.
    """

    if not storage_directory.exists():
        return None

    for pdf_path in storage_directory.glob("*.pdf"):

        try:
            existing_bytes = pdf_path.read_bytes()

            existing_hash = calculate_sha256(
                existing_bytes
            )

            if existing_hash == document_hash:
                return pdf_path

        except OSError:
            # A single unreadable file should not prevent
            # ingestion of a new document.
            logger.warning(
                "Unable to inspect existing file during "
                "duplicate check | file=%s",
                pdf_path.name,
            )

    return None


# ============================================================
# Local Document Storage
# ============================================================


def _save_pdf(
    pdf_data: bytes,
    storage_directory: Path,
    filename: str,
) -> Path:
    """
    Save the original PDF to local storage.

    A unique prefix is added to avoid accidental overwriting
    of files having the same filename.
    """

    unique_prefix = uuid4().hex[:12]

    stored_filename = (
        f"{unique_prefix}_{filename}"
    )

    output_path = (
        storage_directory / stored_filename
    )

    try:
        output_path.write_bytes(
            pdf_data
        )

    except OSError as exc:
        raise RuntimeError(
            f"Unable to save PDF: {filename}"
        ) from exc

    return output_path


# ============================================================
# Candidate Metadata
# ============================================================


def _validate_document_type(
    document_type: str,
) -> DocumentType:
    """
    Validate the supported document type.
    """

    normalized = document_type.strip().lower()

    if normalized not in {"resume", "jd"}:
        raise ValueError(
            "document_type must be either 'resume' or 'jd'."
        )

    return normalized  # type: ignore[return-value]


# ============================================================
# Main Ingestion Function
# ============================================================


def ingest_document(
    pdf_data: bytes,
    filename: str,
    document_type: DocumentType,
    candidate_id: str | None = None,
    candidate_name: str | None = None,
    document_id: str | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> IngestionResult:
    """
    Ingest a single resume or JD PDF.

    Complete flow:

        PDF bytes
            ↓
        Filename validation
            ↓
        SHA-256 hash
            ↓
        Duplicate check
            ↓
        Local storage
            ↓
        PDF parsing
            ↓
        Text preprocessing
            ↓
        LangChain Document
            ↓
        Chunking
            ↓
        IngestionResult

    Parameters
    ----------
    pdf_data:
        Raw PDF bytes.

    filename:
        Original PDF filename.

    document_type:
        "resume" or "jd".

    candidate_id:
        Optional candidate identifier.
        Normally provided for resumes.

    candidate_name:
        Optional candidate name.

    document_id:
        Optional externally supplied document ID.
        If omitted, one is generated.

    chunk_size:
        Target chunk size.

    chunk_overlap:
        Chunk overlap.

    Returns
    -------
    IngestionResult
        Parsed and chunked document information.

    Raises
    ------
    ValueError
        For invalid input or duplicate documents.

    RuntimeError
        For storage failures.
    """

    # --------------------------------------------------------
    # Validate basic inputs
    # --------------------------------------------------------

    if not pdf_data:
        raise ValueError(
            "PDF data cannot be empty."
        )

    safe_filename = _validate_filename(
        filename
    )

    validated_type = _validate_document_type(
        document_type
    )

    # --------------------------------------------------------
    # Generate document ID
    # --------------------------------------------------------

    final_document_id = (
        document_id
        or f"{validated_type}_{uuid4().hex}"
    )

    # --------------------------------------------------------
    # Calculate SHA-256
    # --------------------------------------------------------

    document_hash = calculate_sha256(
        pdf_data
    )

    logger.info(
        "Document ingestion started | "
        "document_id=%s | type=%s | file=%s",
        final_document_id,
        validated_type,
        safe_filename,
    )

    # --------------------------------------------------------
    # Determine storage directory
    # --------------------------------------------------------

    storage_directory = _get_storage_directory(
        validated_type
    )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    existing_file = _find_existing_document_by_hash(
        storage_directory=storage_directory,
        document_hash=document_hash,
    )

    if existing_file is not None:

        logger.info(
            "Duplicate document detected | "
            "document_id=%s | existing_file=%s",
            final_document_id,
            existing_file.name,
        )

        # Parse the existing document so the caller receives
        # the same usable chunk structure as a new upload.
        try:
            existing_pdf_data = (
                existing_file.read_bytes()
            )

        except OSError as exc:
            raise RuntimeError(
                "Duplicate document was found, but the "
                "stored PDF could not be read."
            ) from exc

        chunks = parse_pdf(
            pdf_data=existing_pdf_data,
            document_id=final_document_id,
            document_type=validated_type,
            source_file=existing_file.name,
            document_hash=document_hash,
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return IngestionResult(
            document_id=final_document_id,
            document_type=validated_type,
            source_file=existing_file.name,
            stored_path=existing_file,
            document_hash=document_hash,
            chunks=chunks,
            is_duplicate=True,
        )

    # --------------------------------------------------------
    # Save new document
    # --------------------------------------------------------

    stored_path = _save_pdf(
        pdf_data=pdf_data,
        storage_directory=storage_directory,
        filename=safe_filename,
    )

    # --------------------------------------------------------
    # Parse and chunk
    # --------------------------------------------------------

    try:
        chunks = parse_pdf(
            pdf_data=pdf_data,
            document_id=final_document_id,
            document_type=validated_type,
            source_file=safe_filename,
            document_hash=document_hash,
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    except Exception:
        # If parsing fails after storage, remove the newly
        # created file so we don't leave an unusable document
        # behind.
        try:
            stored_path.unlink(
                missing_ok=True
            )
        except OSError:
            logger.exception(
                "Failed to clean up stored PDF after "
                "parsing failure | path=%s",
                stored_path,
            )

        raise

    # --------------------------------------------------------
    # Log successful ingestion
    # --------------------------------------------------------

    logger.info(
        "Document ingestion completed | "
        "document_id=%s | type=%s | chunks=%d",
        final_document_id,
        validated_type,
        len(chunks),
    )

    return IngestionResult(
        document_id=final_document_id,
        document_type=validated_type,
        source_file=safe_filename,
        stored_path=stored_path,
        document_hash=document_hash,
        chunks=chunks,
        is_duplicate=False,
    )


# ============================================================
# Batch Ingestion
# ============================================================


def ingest_documents(
    documents: list[tuple[bytes, str]],
    document_type: DocumentType,
    candidate_ids: list[str | None] | None = None,
    candidate_names: list[str | None] | None = None,
) -> list[IngestionResult]:
    """
    Ingest multiple PDF documents.

    Parameters
    ----------
    documents:
        List of tuples:
            (pdf_bytes, filename)

    document_type:
        "resume" or "jd".

    candidate_ids:
        Optional candidate IDs corresponding to documents.

    candidate_names:
        Optional candidate names corresponding to documents.

    Returns
    -------
    list[IngestionResult]
        Successful ingestion results.

    Notes
    -----
    Documents are processed independently. If one document
    fails, the exception is raised rather than silently
    continuing, because ingestion failures should be visible
    to the caller.
    """

    if not documents:
        raise ValueError(
            "documents cannot be empty."
        )

    if candidate_ids is not None and len(
        candidate_ids
    ) != len(documents):

        raise ValueError(
            "candidate_ids length must match documents length."
        )

    if candidate_names is not None and len(
        candidate_names
    ) != len(documents):

        raise ValueError(
            "candidate_names length must match documents length."
        )

    results: list[IngestionResult] = []

    for index, (
        pdf_data,
        filename,
    ) in enumerate(documents):

        candidate_id = (
            candidate_ids[index]
            if candidate_ids
            else None
        )

        candidate_name = (
            candidate_names[index]
            if candidate_names
            else None
        )

        result = ingest_document(
            pdf_data=pdf_data,
            filename=filename,
            document_type=document_type,
            candidate_id=candidate_id,
            candidate_name=candidate_name,
        )

        results.append(result)

    logger.info(
        "Batch ingestion completed | "
        "type=%s | documents=%d",
        document_type,
        len(results),
    )

    return results