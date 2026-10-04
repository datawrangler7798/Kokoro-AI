"""
core/ingestion/parsing.py

PDF parsing and text preprocessing for Kokoro.

Responsibilities:
    - Extract text from PDF files using PyPDF.
    - Normalize extracted text.
    - Remove PDF extraction noise safely.
    - Preserve resume meaning, headings, dates and acronyms.
    - Detect common resume sections.
    - Extract basic candidate metadata.
    - Build LangChain Document objects.
    - Provide parsed document content to the ingestion layer.

Important:
    This module does NOT:
        - Generate embeddings
        - Call Pinecone
        - Update BM25
        - Perform retrieval
        - Call Gemini
        - Perform chunking/indexing orchestration
        - Import Streamlit

The ingestion layer will use the parsed output produced here.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document as LangChainDocument
from pypdf import PdfReader

from utils.logger import get_logger
from utils.schemas import DocumentType
from utils.utils import calculate_file_hash, clean_string

# ============================================================
# LOGGER
# ============================================================

logger = get_logger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_SECTION_NAMES = {
    "professional summary": "professional_summary",
    "summary": "summary",
    "profile": "profile",
    "objective": "objective",
    "technical skills": "technical_skills",
    "technical skill": "technical_skills",
    "skills": "skills",
    "core skills": "skills",
    "key skills": "skills",
    "software proficiency": "software_proficiency",
    "software skills": "software_proficiency",
    "professional experience": "professional_experience",
    "work experience": "professional_experience",
    "experience": "professional_experience",
    "employment history": "professional_experience",
    "education": "education",
    "educational qualification": "education",
    "academic qualification": "education",
    "certifications": "certifications",
    "certification": "certifications",
    "achievements": "achievements",
    "awards": "achievements",
    "projects": "projects",
    "professional projects": "projects",
    "references": "references",
    "reference": "references",
}


# ============================================================
# PDF EXTRACTION
# ============================================================


def extract_pdf_text(
    file_path: str | Path,
) -> str:
    """
    Extract text from all pages of a PDF.

    Args:
        file_path:
            Path to the PDF file.

    Returns:
        Raw extracted PDF text.

    Raises:
        FileNotFoundError:
            If the PDF does not exist.

        ValueError:
            If the PDF cannot be read or contains no text.

        Exception:
            For unexpected PDF parsing errors.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        raise ValueError(f"PDF path is not a file: {path}")

    logger.info(
        "Starting PDF text extraction | file=%s",
        path.name,
    )

    try:
        reader = PdfReader(str(path))

        extracted_pages: list[str] = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            try:
                page_text = page.extract_text() or ""

            except Exception as exc:
                logger.warning(
                    "Failed to extract page %d | " "file=%s | error=%s",
                    page_number,
                    path.name,
                    str(exc),
                )

                page_text = ""

            if page_text.strip():
                extracted_pages.append(page_text)

        raw_text = "\n\n".join(extracted_pages)

        if not raw_text.strip():
            raise ValueError(f"No extractable text found in PDF: " f"{path.name}")

        logger.info(
            "PDF extraction completed | file=%s " "| pages=%d | characters=%d",
            path.name,
            len(reader.pages),
            len(raw_text),
        )

        return raw_text

    except ValueError:
        raise

    except Exception as exc:
        logger.exception(
            "PDF extraction failed | file=%s",
            path.name,
        )

        raise ValueError(f"Failed to parse PDF: {path.name}") from exc


# ============================================================
# UNICODE NORMALIZATION
# ============================================================


def normalize_unicode(
    text: str,
) -> str:
    """
    Normalize Unicode characters.

    NFC is used so that visually equivalent Unicode
    representations are normalized while preserving
    meaningful text.
    """

    if not text:
        return ""

    return unicodedata.normalize(
        "NFC",
        text,
    )


# ============================================================
# CONTROL CHARACTER REMOVAL
# ============================================================


def remove_control_characters(
    text: str,
) -> str:
    """
    Remove null/control characters while preserving:

        - newline
        - carriage return
        - tab

    This follows the architecture requirement to remove
    extraction noise without destroying resume meaning.
    """

    if not text:
        return ""

    cleaned_characters: list[str] = []

    for character in text:
        if character in (
            "\n",
            "\r",
            "\t",
        ):
            cleaned_characters.append(character)
            continue

        category = unicodedata.category(character)

        if category.startswith("C"):
            continue

        cleaned_characters.append(character)

    return "".join(cleaned_characters)


# ============================================================
# PDF ARTIFACT CLEANING
# ============================================================


def fix_pdf_artifacts(
    text: str,
) -> str:
    """
    Fix common PDF text extraction artifacts.

    The function intentionally performs only conservative
    transformations.

    It does NOT:
        - aggressively lowercase text
        - remove punctuation
        - remove stopwords
        - stem words
        - remove dates
        - remove acronyms

    Those transformations could damage resume evidence.
    """

    if not text:
        return ""

    # --------------------------------------------------------
    # Remove soft hyphen characters.
    # --------------------------------------------------------

    text = text.replace(
        "\u00ad",
        "",
    )

    # --------------------------------------------------------
    # Normalize non-breaking spaces.
    # --------------------------------------------------------

    text = text.replace(
        "\u00a0",
        " ",
    )

    # --------------------------------------------------------
    # Repair words broken across lines.
    #
    # Example:
    #
    #   Gener-
    #   ative AI
    #
    # becomes:
    #
    #   Generative AI
    # --------------------------------------------------------

    text = re.sub(
        r"(?<=\w)-\s*\n\s*(?=\w)",
        "",
        text,
    )

    # --------------------------------------------------------
    # Normalize CRLF / CR.
    # --------------------------------------------------------

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    # --------------------------------------------------------
    # Remove excessive spaces around newlines.
    # --------------------------------------------------------

    text = re.sub(
        r"[ \t]+\n",
        "\n",
        text,
    )

    text = re.sub(
        r"\n[ \t]+",
        "\n",
        text,
    )

    return text


# ============================================================
# TEXT PREPROCESSING
# ============================================================


def preprocess_text(
    text: str,
) -> str:
    """
    Perform light resume/JD text preprocessing.

    Processing includes:

        1. Unicode normalization
        2. Control-character removal
        3. Conservative PDF artifact correction
        4. Whitespace normalization

    The original meaning and readable structure are preserved.
    """

    if not text:
        return ""

    text = normalize_unicode(text)

    text = remove_control_characters(text)

    text = fix_pdf_artifacts(text)

    # Use the shared utility for general whitespace
    # normalization while preserving line structure.
    lines = text.splitlines()

    cleaned_lines: list[str] = []

    for line in lines:
        line = line.strip()

        if not line:
            cleaned_lines.append("")
            continue

        line = re.sub(
            r"[ \t]+",
            " ",
            line,
        )

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Collapse excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# SECTION DETECTION
# ============================================================


def normalize_section_name(
    section_name: str,
) -> str:
    """
    Normalize a section heading into a canonical name.
    """

    cleaned = clean_string(section_name)

    cleaned = cleaned.lower()

    cleaned = re.sub(
        r"[:\-]+$",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    )

    return SUPPORTED_SECTION_NAMES.get(
        cleaned,
        cleaned.replace(
            " ",
            "_",
        ),
    )


def is_section_heading(
    line: str,
) -> bool:
    """
    Determine whether a line is likely to be a resume section
    heading.

    Known section names are preferred.

    A conservative heuristic is also used for headings that
    are written in uppercase.
    """

    if not line:
        return False

    cleaned = line.strip()

    if not cleaned:
        return False

    normalized = cleaned.lower().rstrip(":").strip()

    if normalized in SUPPORTED_SECTION_NAMES:
        return True

    # --------------------------------------------------------
    # Uppercase heading heuristic.
    #
    # Example:
    # PROFESSIONAL EXPERIENCE
    # EDUCATION
    # CERTIFICATIONS
    # --------------------------------------------------------

    letters = [character for character in cleaned if character.isalpha()]

    if not letters:
        return False

    uppercase_ratio = sum(character.isupper() for character in letters) / len(letters)

    if uppercase_ratio >= 0.85 and len(cleaned) <= 80 and len(cleaned.split()) <= 8:
        return True

    return False


def extract_sections(
    text: str,
) -> dict[str, str]:
    """
    Split parsed text into logical resume sections.

    Returns:

        {
            "professional_summary": "...",
            "technical_skills": "...",
            "professional_experience": "...",
            "education": "..."
        }

    Unknown headings are preserved using a normalized heading
    name rather than being discarded.
    """

    if not text.strip():
        return {}

    lines = text.splitlines()

    sections: dict[str, list[str]] = {}

    current_section = "general"

    sections[current_section] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            sections[current_section].append("")

            continue

        if is_section_heading(stripped):
            current_section = normalize_section_name(stripped)

            if current_section not in sections:
                sections[current_section] = []

            continue

        sections[current_section].append(stripped)

    result: dict[str, str] = {}

    for section, section_lines in sections.items():
        section_text = "\n".join(section_lines).strip()

        if section_text:
            result[section] = section_text

    return result


# ============================================================
# CANDIDATE NAME EXTRACTION
# ============================================================


def _looks_like_name(
    line: str,
) -> bool:
    """
    Conservative check for a candidate name.

    This is intentionally heuristic.

    The parser should not assume every first line is a name.
    """

    if not line:
        return False

    line = line.strip()

    if len(line) > 100:
        return False

    if any(character.isdigit() for character in line):
        return False

    words = line.split()

    if not 2 <= len(words) <= 5:
        return False

    blocked_terms = {
        "resume",
        "curriculum vitae",
        "cv",
        "profile",
        "summary",
        "professional summary",
        "technical skills",
        "skills",
        "experience",
        "education",
        "objective",
    }

    if line.lower() in blocked_terms:
        return False

    alpha_count = sum(character.isalpha() for character in line)

    if alpha_count < 3:
        return False

    return True


def extract_candidate_name(
    text: str,
) -> Optional[str]:
    """
    Extract a candidate name from the beginning of the
    document.

    The parser checks the first meaningful lines rather than
    assuming the first line is always the candidate name.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Only inspect a small number of leading lines.
    for line in lines[:10]:
        if _looks_like_name(line):
            return line

    return None


# ============================================================
# DOCUMENT ID
# ============================================================


def create_document_id(
    file_path: str | Path,
    document_hash: Optional[str] = None,
) -> str:
    """
    Create a deterministic document ID.

    The preferred source is the document SHA-256 hash.

    If a hash is not supplied, the file hash is calculated.
    """

    path = Path(file_path)

    if document_hash is None:
        document_hash = calculate_file_hash(path)

    return f"document_{document_hash[:16]}"


# ============================================================
# CANDIDATE ID
# ============================================================


def create_candidate_id(
    candidate_name: Optional[str],
    document_hash: str,
) -> str:
    """
    Create a stable candidate identifier.

    Candidate name is preferred when available.

    Document hash is included as a fallback to guarantee that
    an ID can still be created when the candidate name cannot
    be extracted.
    """

    if candidate_name:
        candidate = candidate_name.lower().strip()

        candidate = re.sub(
            r"[^a-z0-9]+",
            "_",
            candidate,
        )

        candidate = candidate.strip("_")

        if candidate:
            return candidate

    return f"candidate_{document_hash[:16]}"


# ============================================================
# PARSED DOCUMENT RESULT
# ============================================================


class ParsedDocument:
    """
    Internal parsing result.

    This object keeps parsing output separate from retrieval
    and indexing concerns.

    Attributes:
        document_id:
            Unique document identifier.

        document_type:
            Resume or JD.

        candidate_id:
            Candidate identifier when applicable.

        candidate_name:
            Candidate name when detected.

        source_file:
            Original filename.

        document_hash:
            SHA-256 hash.

        text:
            Full normalized document text.

        sections:
            Section-aware representation.

        page_count:
            Number of pages extracted from the PDF.
    """

    def __init__(
        self,
        document_id: str,
        document_type: DocumentType,
        candidate_id: Optional[str],
        candidate_name: Optional[str],
        source_file: str,
        document_hash: str,
        text: str,
        sections: dict[str, str],
        page_count: int,
    ) -> None:
        self.document_id = document_id
        self.document_type = document_type
        self.candidate_id = candidate_id
        self.candidate_name = candidate_name
        self.source_file = source_file
        self.document_hash = document_hash
        self.text = text
        self.sections = sections
        self.page_count = page_count

    def to_dict(self) -> dict:
        """
        Convert the parsed document to a dictionary.
        """

        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "source_file": self.source_file,
            "document_hash": self.document_hash,
            "text": self.text,
            "sections": self.sections,
            "page_count": self.page_count,
        }


# ============================================================
# MAIN PARSER
# ============================================================


def parse_pdf(
    file_path: str | Path,
    document_type: DocumentType = DocumentType.RESUME,
    document_id: Optional[str] = None,
    document_hash: Optional[str] = None,
) -> ParsedDocument:
    """
    Parse a PDF into a structured ParsedDocument.

    Processing:

        PDF
         ↓
        PyPDF extraction
         ↓
        Unicode normalization
         ↓
        Control-character removal
         ↓
        PDF artifact correction
         ↓
        Whitespace normalization
         ↓
        Candidate metadata extraction
         ↓
        Section detection
         ↓
        ParsedDocument

    Args:
        file_path:
            Path to the PDF.

        document_type:
            RESUME or JD.

        document_id:
            Optional existing document ID.

        document_hash:
            Optional existing SHA-256 hash.

    Returns:
        ParsedDocument
    """

    path = Path(file_path)

    logger.info(
        "Starting document parsing | file=%s",
        path.name,
    )

    # --------------------------------------------------------
    # Calculate document hash when not already supplied.
    # --------------------------------------------------------

    if document_hash is None:
        document_hash = calculate_file_hash(path)

    # --------------------------------------------------------
    # Create document ID when not already supplied.
    # --------------------------------------------------------

    if document_id is None:
        document_id = create_document_id(
            path,
            document_hash,
        )

    # --------------------------------------------------------
    # Extract PDF text.
    # --------------------------------------------------------

    raw_text = extract_pdf_text(path)

    # --------------------------------------------------------
    # Preprocess.
    # --------------------------------------------------------

    processed_text = preprocess_text(raw_text)

    if not processed_text:
        raise ValueError(f"PDF contains no usable text: " f"{path.name}")

    # --------------------------------------------------------
    # Candidate metadata.
    # --------------------------------------------------------

    candidate_name = extract_candidate_name(processed_text)

    candidate_id: Optional[str] = None

    if document_type == DocumentType.RESUME:
        candidate_id = create_candidate_id(
            candidate_name,
            document_hash,
        )

    # --------------------------------------------------------
    # Section parsing.
    # --------------------------------------------------------

    sections = extract_sections(processed_text)

    # --------------------------------------------------------
    # Determine page count.
    # --------------------------------------------------------

    try:
        reader = PdfReader(str(path))

        page_count = len(reader.pages)

    except Exception:
        page_count = 0

    parsed_document = ParsedDocument(
        document_id=document_id,
        document_type=document_type,
        candidate_id=candidate_id,
        candidate_name=candidate_name,
        source_file=path.name,
        document_hash=document_hash,
        text=processed_text,
        sections=sections,
        page_count=page_count,
    )

    logger.info(
        "Document parsing completed | "
        "document_id=%s | candidate_id=%s "
        "| sections=%d | characters=%d",
        document_id,
        candidate_id,
        len(sections),
        len(processed_text),
    )

    return parsed_document


# ============================================================
# LANGCHAIN DOCUMENT CONVERSION
# ============================================================


def to_langchain_documents(
    parsed_document: ParsedDocument,
) -> list[LangChainDocument]:
    """
    Convert a ParsedDocument into LangChain Document objects.

    One Document is created for each detected section.

    This preserves section information in metadata and allows
    the ingestion layer to apply the configured text splitter
    later.

    If no sections are detected, the complete document is
    returned as one LangChain Document.
    """

    documents: list[LangChainDocument] = []

    base_metadata = {
        "document_id": (parsed_document.document_id),
        "document_type": (
            parsed_document.document_type.value
            if hasattr(
                parsed_document.document_type,
                "value",
            )
            else str(parsed_document.document_type)
        ),
        "candidate_id": (parsed_document.candidate_id),
        "candidate_name": (parsed_document.candidate_name),
        "source_file": (parsed_document.source_file),
        "document_hash": (parsed_document.document_hash),
        "page_count": (parsed_document.page_count),
    }

    # --------------------------------------------------------
    # Section-aware Documents.
    # --------------------------------------------------------

    if parsed_document.sections:
        for section_name, section_text in parsed_document.sections.items():
            metadata = {
                **base_metadata,
                "section": section_name,
            }

            content = f"{section_name.replace('_', ' ').title()}\n" f"{section_text}"

            documents.append(
                LangChainDocument(
                    page_content=content,
                    metadata=metadata,
                )
            )

    # --------------------------------------------------------
    # Fallback when no sections are detected.
    # --------------------------------------------------------

    else:
        documents.append(
            LangChainDocument(
                page_content=(parsed_document.text),
                metadata={
                    **base_metadata,
                    "section": "general",
                },
            )
        )

    logger.debug(
        "Converted parsed document to LangChain "
        "Documents | document_id=%s | documents=%d",
        parsed_document.document_id,
        len(documents),
    )

    return documents


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================


def parse_to_langchain_documents(
    file_path: str | Path,
    document_type: DocumentType = DocumentType.RESUME,
    document_id: Optional[str] = None,
    document_hash: Optional[str] = None,
) -> tuple[ParsedDocument, list[LangChainDocument],]:
    """
    Parse a PDF and immediately convert it to LangChain
    Documents.

    Returns:

        (
            ParsedDocument,
            list[LangChainDocument]
        )

    This is a convenience function for ingestion.py.
    """

    parsed_document = parse_pdf(
        file_path=file_path,
        document_type=document_type,
        document_id=document_id,
        document_hash=document_hash,
    )

    langchain_documents = to_langchain_documents(parsed_document)

    return (
        parsed_document,
        langchain_documents,
    )


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "ParsedDocument",
    "extract_pdf_text",
    "normalize_unicode",
    "remove_control_characters",
    "fix_pdf_artifacts",
    "preprocess_text",
    "normalize_section_name",
    "is_section_heading",
    "extract_sections",
    "extract_candidate_name",
    "create_document_id",
    "create_candidate_id",
    "parse_pdf",
    "to_langchain_documents",
    "parse_to_langchain_documents",
]
