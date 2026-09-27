"""
Pinecone vector-store implementation for Kokoro AI.

Responsibilities:
- Connect to Pinecone
- Validate the configured index
- Upsert document vectors
- Perform semantic similarity search
- Convert Pinecone results into Kokoro retrieval schemas

This module does not perform:
- BM25 search
- Hybrid score fusion
- Query routing
- Gemini reranking
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from pinecone import Pinecone

from utils.config import config
from utils.logger import logger
from utils.schemas import RetrievedChunk


class PineconeVectorStore:
    """
    Thin application-level wrapper around Pinecone.

    Kokoro uses this class instead of calling Pinecone directly
    from multiple modules.
    """

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
    ) -> None:
        """
        Initialize the Pinecone client and index.

        Parameters
        ----------
        api_key:
            Optional Pinecone API key.

        index_name:
            Optional Pinecone index name.
        """

        self.api_key = (
            api_key
            or config.PINECONE_API_KEY
        )

        self.index_name = (
            index_name
            or config.PINECONE_INDEX_NAME
        )

        if not self.api_key.strip():
            raise ValueError(
                "PINECONE_API_KEY is not configured."
            )

        if not self.index_name.strip():
            raise ValueError(
                "PINECONE_INDEX_NAME cannot be empty."
            )

        # ----------------------------------------------------
        # Create Pinecone client
        # ----------------------------------------------------

        self.client = Pinecone(
            api_key=self.api_key
        )

        # ----------------------------------------------------
        # Connect to index
        # ----------------------------------------------------

        self.index = self.client.Index(
            self.index_name
        )

        logger.info(
            "Pinecone vector store initialized | index=%s",
            self.index_name,
        )

    # ========================================================
    # Index Information
    # ========================================================

    def describe_index(self) -> dict[str, Any]:
        """
        Return Pinecone index information.

        Useful for validating that the actual Pinecone
        configuration matches Kokoro configuration.
        """

        try:
            description = self.client.describe_index(
                self.index_name
            )

            if hasattr(
                description,
                "to_dict",
            ):
                return description.to_dict()

            if isinstance(
                description,
                dict,
            ):
                return description

            return {
                "description": str(description)
            }

        except Exception as exc:
            logger.exception(
                "Failed to describe Pinecone index | index=%s",
                self.index_name,
            )

            raise RuntimeError(
                "Unable to retrieve Pinecone index information."
            ) from exc

    # ========================================================
    # Index Dimension Validation
    # ========================================================

    def validate_dimension(
        self,
        expected_dimension: int | None = None,
    ) -> None:
        """
        Validate the configured Pinecone dimension against
        the actual Pinecone index.

        Parameters
        ----------
        expected_dimension:
            Expected vector dimension.

        Raises
        ------
        ValueError
            If the dimensions do not match.
        """

        expected = (
            expected_dimension
            or config.PINECONE_DIMENSION
        )

        description = self.describe_index()

        # Pinecone API representations can differ slightly
        # between SDK versions, so support the common forms.
        actual_dimension = None

        if isinstance(
            description,
            dict,
        ):
            actual_dimension = description.get(
                "dimension"
            )

            if actual_dimension is None:
                spec = description.get(
                    "spec",
                    {},
                )

                if isinstance(
                    spec,
                    dict,
                ):
                    actual_dimension = spec.get(
                        "dimension"
                    )

        if actual_dimension is None:
            logger.warning(
                "Could not determine Pinecone index dimension "
                "from describe_index response."
            )
            return

        if int(actual_dimension) != int(expected):
            raise ValueError(
                "Pinecone dimension mismatch. "
                f"Configured={expected}, "
                f"Actual={actual_dimension}."
            )

        logger.info(
            "Pinecone dimension validated | dimension=%s",
            actual_dimension,
        )

    # ========================================================
    # Upsert
    # ========================================================

    def upsert_vectors(
        self,
        vectors: list[dict[str, Any]],
        namespace: str = "",
        batch_size: int = 100,
    ) -> None:
        """
        Upsert vectors into Pinecone.

        Each vector must follow the Pinecone structure:

            {
                "id": "...",
                "values": [...],
                "metadata": {...}
            }

        Parameters
        ----------
        vectors:
            Vectors to insert/update.

        namespace:
            Pinecone namespace.

        batch_size:
            Number of vectors sent per request.
        """

        if not vectors:
            logger.warning(
                "No vectors supplied for Pinecone upsert."
            )
            return

        if batch_size < 1:
            raise ValueError(
                "batch_size must be >= 1."
            )

        for start in range(
            0,
            len(vectors),
            batch_size,
        ):
            batch = vectors[
                start : start + batch_size
            ]

            try:
                self.index.upsert(
                    vectors=batch,
                    namespace=namespace,
                )

            except Exception as exc:
                logger.exception(
                    "Pinecone upsert failed | "
                    "batch_start=%d | batch_size=%d",
                    start,
                    len(batch),
                )

                raise RuntimeError(
                    "Failed to upsert vectors into Pinecone."
                ) from exc

        logger.info(
            "Pinecone upsert completed | vectors=%d",
            len(vectors),
        )

    # ========================================================
    # Delete
    # ========================================================

    def delete_document(
        self,
        document_id: str,
        namespace: str = "",
    ) -> None:
        """
        Delete all vectors belonging to a document.

        Requires the vectors to contain:

            metadata.document_id
        """

        if not document_id.strip():
            raise ValueError(
                "document_id cannot be empty."
            )

        try:
            self.index.delete(
                filter={
                    "document_id": {
                        "$eq": document_id
                    }
                },
                namespace=namespace,
            )

        except Exception as exc:
            logger.exception(
                "Pinecone document deletion failed | "
                "document_id=%s",
                document_id,
            )

            raise RuntimeError(
                "Failed to delete document vectors."
            ) from exc

        logger.info(
            "Pinecone document deleted | document_id=%s",
            document_id,
        )

    # ========================================================
    # Semantic Search
    # ========================================================

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        namespace: str = "",
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[RetrievedChunk]:
        """
        Perform semantic vector search.

        Parameters
        ----------
        query_vector:
            Embedded query vector.

        top_k:
            Number of results.

        namespace:
            Pinecone namespace.

        filter:
            Optional Pinecone metadata filter.

        include_metadata:
            Whether metadata should be returned.

        Returns
        -------
        list[RetrievedChunk]
            Validated Kokoro retrieval results.
        """

        if not query_vector:
            raise ValueError(
                "query_vector cannot be empty."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be >= 1."
            )

        try:
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_metadata=include_metadata,
            )

        except Exception as exc:
            logger.exception(
                "Pinecone semantic search failed."
            )

            raise RuntimeError(
                "Pinecone semantic search failed."
            ) from exc

        results: list[RetrievedChunk] = []

        matches = getattr(
            response,
            "matches",
            [],
        )

        for match in matches:

            metadata = getattr(
                match,
                "metadata",
                None,
            ) or {}

            score = getattr(
                match,
                "score",
                None,
            )

            chunk_id = getattr(
                match,
                "id",
                None,
            )

            if not chunk_id:
                continue

            text = str(
                metadata.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                logger.warning(
                    "Pinecone match missing text metadata | "
                    "chunk_id=%s",
                    chunk_id,
                )
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    candidate_id=self._optional_string(
                        metadata.get("candidate_id")
                    ),
                    candidate_name=self._optional_string(
                        metadata.get("candidate_name")
                    ),
                    text=text,
                    semantic_score=(
                        float(score)
                        if score is not None
                        else None
                    ),
                    metadata=dict(metadata),
                )
            )

        logger.info(
            "Pinecone semantic search completed | "
            "requested_top_k=%d | returned=%d",
            top_k,
            len(results),
        )

        return results

    # ========================================================
    # Document Conversion
    # ========================================================

    @staticmethod
    def documents_to_vectors(
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> list[dict[str, Any]]:
        """
        Convert LangChain Documents and embedding vectors
        into Pinecone upsert records.

        Parameters
        ----------
        documents:
            Chunked LangChain Documents.

        embeddings:
            Corresponding embedding vectors.

        Returns
        -------
        list[dict]
            Pinecone-compatible vectors.
        """

        if len(documents) != len(embeddings):
            raise ValueError(
                "documents and embeddings must have the same length."
            )

        vectors: list[dict[str, Any]] = []

        for document, embedding in zip(
            documents,
            embeddings,
        ):
            metadata = dict(
                document.metadata
            )

            chunk_id = metadata.get(
                "chunk_id"
            )

            if not chunk_id:
                raise ValueError(
                    "Every document must contain chunk_id "
                    "before Pinecone indexing."
                )

            # Pinecone metadata must contain serializable
            # primitive values/lists.
            metadata["text"] = document.page_content

            vectors.append(
                {
                    "id": str(chunk_id),
                    "values": embedding,
                    "metadata": metadata,
                }
            )

        return vectors

    # ========================================================
    # Utility
    # ========================================================

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        """
        Safely convert optional metadata to string.
        """

        if value is None:
            return None

        value = str(value).strip()

        return value or None