"""
core/ingestion/ingestion.py

Kokoro document ingestion orchestration.

Responsibilities:
    - Validate PDF files.
    - Calculate SHA-256 document hashes.
    - Detect duplicate documents.
    - Save PDFs into the local data directory.
    - Parse PDFs.
    - Create section-aware LangChain Documents.
    - Split documents into retrieval chunks.
    - Prepare chunks for embedding/indexing.
    - Coordinate Pinecone and BM25 indexing through injectable services.
    - Support single and batch ingestion.
    - Report individual document failures.
    - Log ingestion progress and errors.

Important:
    This module orchestrates ingestion.

    It does NOT contain:
        - Pinecone implementation
        - BM25 implementation
        - Gemini implementation
        - Streamlit UI code

Those components are kept replaceable.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.ingestion.parsing import (
    ParsedDocument,
    parse_pdf,
    to_langchain_documents,
)

from utils.config import settings
from utils.logger import get_logger
from utils.schemas import (
    DocumentType,
    IngestionStatus,
)
from utils.utils import (
    calculate_file_hash,
    get_file_extension,
    get_file_size_bytes,
    sanitize_filename,
    validate_file_size,
)


# ============================================================
# LOGGER
# ============================================================

logger = get_logger(__name__)


# ============================================================
# SERVICE PROTOCOLS
# ============================================================


class VectorStoreProtocol(Protocol):
    """
    Interface expected from the future Pinecone vector-store
    implementation.

    vector_store.py will implement this interface.
    """

    def index_documents(
        self,
        documents: list[LangChainDocument],
    ) -> Any:
        ...


class HybridIndexerProtocol(Protocol):
    """
    Interface expected from the future hybrid/BM25 indexer.

    hybrid_indexer.py will implement this interface.
    """

    def add_documents(
        self,
        documents: list[LangChainDocument],
    ) -> Any:
        ...


class RegistryProtocol(Protocol):
    """
    Optional document registry interface.

    A registry implementation can persist ingestion state.
    """

    def exists(
        self,
        document_hash: str,
    ) -> bool:
        ...

    def save(
        self,
        document: ParsedDocument,
    ) -> Any:
        ...


# ============================================================
# INGESTION SERVICE
# ============================================================


class IngestionService:
    """
    Main Kokoro ingestion service.

    The service is deliberately dependency-injected so that
    Pinecone, BM25 and registry implementations can be changed
    without rewriting the ingestion pipeline.

    Example:

        service = IngestionService(
            vector_store=vector_store,
            hybrid_indexer=hybrid_indexer,
        )

        result = service.ingest_file(
            "data/resumes/Angela_Lewis.pdf"
        )
    """

    def __init__(
        self,
        vector_store: Optional[
            VectorStoreProtocol
        ] = None,
        hybrid_indexer: Optional[
            HybridIndexerProtocol
        ] = None,
        registry: Optional[
            RegistryProtocol
        ] = None,
    ) -> None:

        self.vector_store = vector_store
        self.hybrid_indexer = hybrid_indexer
        self.registry = registry

        self._splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True,
            )
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_file(
        self,
        file_path: str | Path,
    ) -> None:
        """
        Validate a file before ingestion.

        Checks:
            - file exists
            - file is a regular file
            - extension is allowed
            - file size is within configured limit
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        extension = get_file_extension(
            path
        ).lower()

        allowed_extensions = (
            settings.get_allowed_extensions()
        )

        if extension not in allowed_extensions:

            raise ValueError(
                f"Unsupported file extension "
                f"'{extension}'. "
                f"Allowed: {allowed_extensions}"
            )

        file_size = get_file_size_bytes(
            path
        )

        validate_file_size(
            file_size,
            settings.MAX_FILE_SIZE_MB,
        )

    # ========================================================
    # HASH
    # ========================================================

    def calculate_hash(
        self,
        file_path: str | Path,
    ) -> str:
        """
        Calculate SHA-256 hash for duplicate detection.
        """

        return calculate_file_hash(
            file_path
        )

    # ========================================================
    # DUPLICATE CHECK
    # ========================================================

    def is_duplicate(
        self,
        document_hash: str,
    ) -> bool:
        """
        Determine whether a document has already been
        ingested.

        If a registry is configured, use it.

        Otherwise, the caller can use the hash to perform
        duplicate detection externally.
        """

        if self.registry is None:
            return False

        try:

            return bool(
                self.registry.exists(
                    document_hash
                )
            )

        except Exception:

            logger.exception(
                "Document registry duplicate check failed."
            )

            raise

    # ========================================================
    # SAVE FILE
    # ========================================================

    def get_storage_directory(
        self,
        document_type: DocumentType,
    ) -> Path:
        """
        Return the local storage directory for the document.
        """

        if document_type == DocumentType.RESUME:
            return Path(
                settings.RESUME_DIRECTORY
            )

        return Path(
            settings.JD_DIRECTORY
        )

    def save_file(
        self,
        source_path: str | Path,
        document_type: DocumentType,
    ) -> Path:
        """
        Save a PDF into the configured local document
        directory.

        Existing files with the same name are overwritten only
        when they represent the same ingestion operation.
        """

        source = Path(source_path)

        storage_directory = (
            self.get_storage_directory(
                document_type
            )
        )

        storage_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        safe_filename = sanitize_filename(
            source.name
        )

        destination = (
            storage_directory
            / safe_filename
        )

        # Avoid unnecessary copy when the source is already
        # inside the destination directory.
        try:

            if source.resolve() == destination.resolve():
                return destination

        except FileNotFoundError:
            pass

        shutil.copy2(
            source,
            destination,
        )

        logger.info(
            "Document saved | file=%s | destination=%s",
            source.name,
            destination,
        )

        return destination

    # ========================================================
    # CHUNKING
    # ========================================================

    def chunk_documents(
        self,
        documents: list[LangChainDocument],
    ) -> list[LangChainDocument]:
        """
        Split section-aware LangChain Documents into retrieval
        chunks.

        Section metadata is preserved on every resulting chunk.
        """

        if not documents:
            return []

        chunks = (
            self._splitter.split_documents(
                documents
            )
        )

        # ----------------------------------------------------
        # Add deterministic chunk metadata.
        # ----------------------------------------------------

        section_counters: dict[
            str,
            int,
        ] = {}

        for chunk in chunks:

            document_id = chunk.metadata.get(
                "document_id",
                "unknown_document",
            )

            section = chunk.metadata.get(
                "section",
                "general",
            )

            key = (
                f"{document_id}:{section}"
            )

            current_index = (
                section_counters.get(
                    key,
                    0,
                )
            )

            chunk.metadata[
                "chunk_index"
            ] = current_index

            chunk.metadata[
                "chunk_id"
            ] = (
                f"{document_id}_"
                f"{section}_"
                f"{current_index:04d}"
            )

            section_counters[key] = (
                current_index + 1
            )

        logger.info(
            "Document chunking completed | "
            "input_documents=%d | chunks=%d",
            len(documents),
            len(chunks),
        )

        return chunks

    # ========================================================
    # INDEXING
    # ========================================================

    def index_vector_store(
        self,
        chunks: list[LangChainDocument],
    ) -> Any:
        """
        Send chunks to the Pinecone vector-store service.

        The actual embedding and Pinecone logic lives in
        vector_store.py.
        """

        if self.vector_store is None:

            logger.warning(
                "Vector store is not configured. "
                "Skipping Pinecone indexing."
            )

            return None

        logger.info(
            "Starting vector-store indexing | chunks=%d",
            len(chunks),
        )

        return self.vector_store.index_documents(
            chunks
        )

    def index_bm25(
        self,
        chunks: list[LangChainDocument],
    ) -> Any:
        """
        Send chunks to the BM25/hybrid indexing service.

        The actual BM25 implementation lives in
        hybrid_indexer.py.
        """

        if self.hybrid_indexer is None:

            logger.warning(
                "Hybrid indexer is not configured. "
                "Skipping BM25 indexing."
            )

            return None

        logger.info(
            "Starting BM25 indexing | chunks=%d",
            len(chunks),
        )

        return self.hybrid_indexer.add_documents(
            chunks
        )

    # ========================================================
    # REGISTRY
    # ========================================================

    def register_document(
        self,
        parsed_document: ParsedDocument,
    ) -> Any:
        """
        Save successful ingestion metadata to the registry.
        """

        if self.registry is None:
            return None

        return self.registry.save(
            parsed_document
        )

    # ========================================================
    # SINGLE FILE INGESTION
    # ========================================================

    def ingest_file(
        self,
        file_path: str | Path,
        document_type: DocumentType = DocumentType.RESUME,
        save_copy: bool = True,
    ) -> dict[str, Any]:
        """
        Ingest one PDF.

        Pipeline:

            Validate
                ↓
            SHA-256
                ↓
            Duplicate check
                ↓
            Save local copy
                ↓
            Parse PDF
                ↓
            LangChain Documents
                ↓
            Chunk
                ↓
            Pinecone/vector indexing
                ↓
            BM25 indexing
                ↓
            Registry
                ↓
            Result

        Returns:
            Dictionary containing ingestion status and metadata.

        A dictionary is returned here rather than tightly
        constructing a Pydantic response so this service remains
        compatible with the evolving ingestion schemas.
        """

        path = Path(file_path)

        logger.info(
            "Starting ingestion | file=%s | type=%s",
            path.name,
            document_type,
        )

        try:

            # ------------------------------------------------
            # 1. Validate
            # ------------------------------------------------

            self.validate_file(
                path
            )

            # ------------------------------------------------
            # 2. SHA-256
            # ------------------------------------------------

            document_hash = (
                self.calculate_hash(
                    path
                )
            )

            logger.info(
                "Document hash calculated | "
                "file=%s | hash=%s",
                path.name,
                document_hash,
            )

            # ------------------------------------------------
            # 3. Duplicate check
            # ------------------------------------------------

            if self.is_duplicate(
                document_hash
            ):

                logger.info(
                    "Duplicate document skipped | "
                    "file=%s | hash=%s",
                    path.name,
                    document_hash,
                )

                return {
                    "status": (
                        IngestionStatus.SKIPPED
                    ),
                    "source_file": path.name,
                    "document_hash": document_hash,
                    "document_id": None,
                    "candidate_id": None,
                    "candidate_name": None,
                    "chunk_count": 0,
                    "error": None,
                }

            # ------------------------------------------------
            # 4. Save local copy
            # ------------------------------------------------

            stored_path = path

            if save_copy:

                stored_path = (
                    self.save_file(
                        path,
                        document_type,
                    )
                )

            # ------------------------------------------------
            # 5. Parse
            # ------------------------------------------------

            parsed_document = parse_pdf(
                file_path=stored_path,
                document_type=document_type,
                document_hash=document_hash,
            )

            # ------------------------------------------------
            # 6. Convert to LangChain Documents
            # ------------------------------------------------

            documents = (
                to_langchain_documents(
                    parsed_document
                )
            )

            # ------------------------------------------------
            # Log full parsed text at DEBUG only.
            #
            # This is useful during VS Code/backend debugging.
            # It will not be displayed by Streamlit unless the
            # application explicitly surfaces logs.
            # ------------------------------------------------

            logger.debug(
                "FULL PARSED DOCUMENT TEXT | "
                "document_id=%s\n%s",
                parsed_document.document_id,
                parsed_document.text,
            )

            # ------------------------------------------------
            # 7. Chunk
            # ------------------------------------------------

            chunks = self.chunk_documents(
                documents
            )

            if not chunks:

                raise ValueError(
                    "No retrieval chunks were created."
                )

            # ------------------------------------------------
            # 8. Vector indexing
            # ------------------------------------------------

            vector_result = (
                self.index_vector_store(
                    chunks
                )
            )

            # ------------------------------------------------
            # 9. BM25 indexing
            # ------------------------------------------------

            bm25_result = (
                self.index_bm25(
                    chunks
                )
            )

            # ------------------------------------------------
            # 10. Registry
            # ------------------------------------------------

            self.register_document(
                parsed_document
            )

            # ------------------------------------------------
            # Successful result
            # ------------------------------------------------

            result = {
                "status": (
                    IngestionStatus.COMPLETED
                ),
                "source_file": path.name,
                "stored_file": str(
                    stored_path
                ),
                "document_hash": (
                    document_hash
                ),
                "document_id": (
                    parsed_document.document_id
                ),
                "candidate_id": (
                    parsed_document.candidate_id
                ),
                "candidate_name": (
                    parsed_document.candidate_name
                ),
                "chunk_count": len(chunks),
                "page_count": (
                    parsed_document.page_count
                ),
                "sections": list(
                    parsed_document.sections.keys()
                ),
                "vector_result": vector_result,
                "bm25_result": bm25_result,
                "error": None,
            }

            logger.info(
                "Ingestion completed successfully | "
                "file=%s | document_id=%s | "
                "chunks=%d",
                path.name,
                parsed_document.document_id,
                len(chunks),
            )

            return result

        except Exception as exc:

            logger.exception(
                "Ingestion failed | file=%s | "
                "error_type=%s",
                path.name,
                type(exc).__name__,
            )

            return {
                "status": (
                    IngestionStatus.FAILED
                ),
                "source_file": path.name,
                "stored_file": None,
                "document_hash": None,
                "document_id": None,
                "candidate_id": None,
                "candidate_name": None,
                "chunk_count": 0,
                "page_count": 0,
                "sections": [],
                "vector_result": None,
                "bm25_result": None,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

    # ========================================================
    # BATCH INGESTION
    # ========================================================

    def ingest_batch(
        self,
        file_paths: list[str | Path],
        document_type: DocumentType = DocumentType.RESUME,
        save_copy: bool = True,
        progress_callback: Optional[
            Callable[[int, int, dict[str, Any]], None]
        ] = None,
    ) -> dict[str, Any]:
        """
        Ingest multiple files.

        Important behavior:

            One failed document does NOT stop the entire batch.

        This follows the architecture requirement that batch
        ingestion should report individual failures while
        successful files continue. :contentReference[oaicite:1]{index=1}

        Args:
            file_paths:
                Files to ingest.

            document_type:
                RESUME or JD.

            save_copy:
                Whether to save files into data/resumes or
                data/jds.

            progress_callback:
                Optional callback:

                    callback(
                        completed,
                        total,
                        result
                    )
        """

        total = len(
            file_paths
        )

        results: list[
            dict[str, Any]
        ] = []

        completed = 0
        successful = 0
        skipped = 0
        failed = 0
        total_chunks = 0

        logger.info(
            "Starting batch ingestion | total=%d | type=%s",
            total,
            document_type,
        )

        for file_path in file_paths:

            result = self.ingest_file(
                file_path=file_path,
                document_type=document_type,
                save_copy=save_copy,
            )

            results.append(
                result
            )

            completed += 1

            status = result.get(
                "status"
            )

            if status == IngestionStatus.COMPLETED:
                successful += 1

            elif status == IngestionStatus.SKIPPED:
                skipped += 1

            else:
                failed += 1

            total_chunks += int(
                result.get(
                    "chunk_count",
                    0,
                )
                or 0
            )

            if progress_callback:

                try:

                    progress_callback(
                        completed,
                        total,
                        result,
                    )

                except Exception:

                    logger.exception(
                        "Progress callback failed."
                    )

        batch_result = {
            "total_files": total,
            "processed_files": completed,
            "successful_files": successful,
            "skipped_files": skipped,
            "failed_files": failed,
            "total_chunks": total_chunks,
            "results": results,
        }

        logger.info(
            "Batch ingestion completed | "
            "total=%d | successful=%d | "
            "skipped=%d | failed=%d | chunks=%d",
            total,
            successful,
            skipped,
            failed,
            total_chunks,
        )

        return batch_result


# ============================================================
# DEFAULT SERVICE FACTORY
# ============================================================


def create_ingestion_service(
    vector_store: Optional[
        VectorStoreProtocol
    ] = None,
    hybrid_indexer: Optional[
        HybridIndexerProtocol
    ] = None,
    registry: Optional[
        RegistryProtocol
    ] = None,
) -> IngestionService:
    """
    Create an IngestionService.

    Dependencies are optional because the concrete Pinecone,
    BM25 and registry implementations are built in their own
    modules.
    """

    return IngestionService(
        vector_store=vector_store,
        hybrid_indexer=hybrid_indexer,
        registry=registry,
    )


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def ingest_file(
    file_path: str | Path,
    document_type: DocumentType = DocumentType.RESUME,
    vector_store: Optional[
        VectorStoreProtocol
    ] = None,
    hybrid_indexer: Optional[
        HybridIndexerProtocol
    ] = None,
    registry: Optional[
        RegistryProtocol
    ] = None,
) -> dict[str, Any]:
    """
    Convenience wrapper for single-file ingestion.
    """

    service = create_ingestion_service(
        vector_store=vector_store,
        hybrid_indexer=hybrid_indexer,
        registry=registry,
    )

    return service.ingest_file(
        file_path=file_path,
        document_type=document_type,
    )


def ingest_batch(
    file_paths: list[str | Path],
    document_type: DocumentType = DocumentType.RESUME,
    vector_store: Optional[
        VectorStoreProtocol
    ] = None,
    hybrid_indexer: Optional[
        HybridIndexerProtocol
    ] = None,
    registry: Optional[
        RegistryProtocol
    ] = None,
    progress_callback: Optional[
        Callable[[int, int, dict[str, Any]], None]
    ] = None,
) -> dict[str, Any]:
    """
    Convenience wrapper for batch ingestion.
    """

    service = create_ingestion_service(
        vector_store=vector_store,
        hybrid_indexer=hybrid_indexer,
        registry=registry,
    )

    return service.ingest_batch(
        file_paths=file_paths,
        document_type=document_type,
        progress_callback=progress_callback,
    )


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "IngestionService",
    "VectorStoreProtocol",
    "HybridIndexerProtocol",
    "RegistryProtocol",
    "create_ingestion_service",
    "ingest_file",
    "ingest_batch",
]