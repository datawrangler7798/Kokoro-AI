"""
Kokoro AI - Pinecone Vector Store.

Responsibilities
----------------
- Create/validate the Pinecone index.
- Generate embeddings for documents and queries.
- Upsert document chunks into Pinecone.
- Perform semantic similarity search.
- Apply metadata filters.
- Delete vectors by document/candidate/JD.
- Keep Pinecone-specific logic isolated from the rest of the application.

Design principles
-----------------
- No Streamlit dependency.
- No BM25 logic.
- No reranking logic.
- No generation logic.
- Embedding provider and Pinecone client can be replaced independently.
- All configuration comes from utils.config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import get_settings
from utils.logger import get_logger
from utils.schemas import RetrievalMethod, RetrievalResult

logger = get_logger(__name__)
settings = get_settings()


# ============================================================
# Protocols
# ============================================================


class EmbeddingProvider(Protocol):
    """Interface for any embedding implementation."""

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        ...

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        ...


class VectorIndex(Protocol):
    """Minimal interface required from a vector index."""

    def upsert(
        self,
        vectors: Sequence[Mapping[str, Any]],
        namespace: str = "",
    ) -> Any:
        ...

    def query(
        self,
        *,
        vector: Sequence[float] | None = None,
        sparse_vector: Mapping[str, Sequence[Any]] | None = None,
        top_k: int,
        namespace: str = "",
        filter: Mapping[str, Any] | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> Any:
        ...

    def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        filter: Mapping[str, Any] | None = None,
        namespace: str = "",
        delete_all: bool = False,
    ) -> Any:
        ...

    def describe_index_stats(
        self,
        *,
        namespace: str | None = None,
    ) -> Any:
        ...


# ============================================================
# Data Structures
# ============================================================


@dataclass(slots=True)
class VectorRecord:
    """
    Internal representation of a vector before Pinecone upsert.
    """

    vector_id: str
    values: list[float]
    metadata: dict[str, Any]
    sparse_values: dict[str, list[Any]]


@dataclass(slots=True)
class VectorSearchMatch:
    """
    Raw normalized Pinecone match.

    This keeps Pinecone's response format away from the rest
    of the application.
    """

    vector_id: str
    score: float
    metadata: dict[str, Any]


# ============================================================
# Google Embedding Provider
# ============================================================


class GoogleEmbeddingProvider:
    """
    Google Gemini embedding provider.

    Uses the same embedding model for:
        - resume chunks
        - JD chunks
        - recruiter queries

    Configuration:
        model = gemini-embedding-001
        dimension = 768

    Task types:
        - documents/JDs -> RETRIEVAL_DOCUMENT
        - recruiter query -> RETRIEVAL_QUERY

    This ensures document and query vectors are generated
    in the same embedding space.
    """

    DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
    QUERY_TASK_TYPE = "RETRIEVAL_QUERY"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        dimension: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL

        self.dimension = dimension or settings.EMBEDDING_DIMENSION

        self.api_key = (
            api_key or settings.GOOGLE_API_KEY.get_secret_value()
            if hasattr(settings.GOOGLE_API_KEY, "get_secret_value")
            else api_key or str(settings.GOOGLE_API_KEY)
        )

        self._client: Any | None = None
        self._embedding_service: Any | None = None

    def _get_embedding_service(self) -> Any:
        """Return the shared implementation for document and query vectors."""
        if self._embedding_service is None:
            from core.retrieval.embedding import GeminiEmbeddingService

            self._embedding_service = GeminiEmbeddingService(
                model_name=self.model_name,
                dimension=self.dimension,
            )
        return self._embedding_service

    def _get_client(self) -> Any:
        """Create the Google GenAI client lazily."""

        if self._client is not None:
            return self._client

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required for embeddings.")

        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for Google embeddings."
            ) from exc

        self._client = genai.Client(
            api_key=self.api_key,
        )

        return self._client

    def _extract_embedding(
        self,
        response: Any,
    ) -> list[float]:
        """
        Extract a single embedding vector from a GenAI response.

        Handles the response shape without exposing the SDK
        structure to the rest of Kokoro.
        """

        embeddings = getattr(
            response,
            "embeddings",
            None,
        )

        if embeddings is None:
            raise RuntimeError("Embedding API returned no embeddings.")

        if not embeddings:
            raise RuntimeError("Embedding API returned an empty embedding list.")

        first = embeddings[0]

        values = getattr(
            first,
            "values",
            None,
        )

        if values is None and isinstance(first, Mapping):
            values = first.get("values")

        if not values:
            raise RuntimeError("Embedding response does not contain vector values.")

        vector = [float(value) for value in values]

        self._validate_dimension(vector)

        return vector

    def _validate_dimension(
        self,
        vector: Sequence[float],
    ) -> None:
        """
        Validate embedding dimension before sending the vector
        to Pinecone.
        """

        if len(vector) != self.dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.dimension}, "
                f"received {len(vector)}."
            )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for resume/JD document chunks.

        Uses:
            task_type = RETRIEVAL_DOCUMENT
            output_dimensionality = 768
        """

        return self._get_embedding_service().embed_texts(texts)

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a recruiter query.

        Uses:
            task_type = RETRIEVAL_QUERY
            output_dimensionality = 768
        """

        return self._get_embedding_service().embed_query(text)

    def embed_queries(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Batch-embed query chunks in the query task space."""

        return self._get_embedding_service().embed_queries(texts)


# ============================================================
# Pinecone Client Factory
# ============================================================


def _create_pinecone_client() -> Any:
    """Create the Pinecone client lazily."""

    api_key = settings.PINECONE_API_KEY

    if hasattr(
        api_key,
        "get_secret_value",
    ):
        api_key = api_key.get_secret_value()

    if not api_key:
        raise ValueError("PINECONE_API_KEY is required.")

    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise RuntimeError("pinecone package is required.") from exc

    return Pinecone(
        api_key=api_key,
    )


# ============================================================
# Pinecone Vector Store
# ============================================================


class PineconeVectorStore:
    """
    Pinecone-backed semantic vector store.

    Main responsibilities:
        1. Index management
        2. Vector upsert
        3. Semantic search
        4. Metadata filtering
        5. Document/candidate/JD deletion
    """

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        pinecone_client: Any | None = None,
        index: VectorIndex | None = None,
        index_name: str | None = None,
        namespace: str | None = None,
    ) -> None:
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.namespace = (
            namespace
            if namespace is not None
            else getattr(
                settings,
                "PINECONE_NAMESPACE",
                "",
            )
        )

        self.embedding_provider = embedding_provider or GoogleEmbeddingProvider()

        self._pinecone_client = pinecone_client

        self._index = index

    # --------------------------------------------------------
    # Client / Index
    # --------------------------------------------------------

    @property
    def pinecone_client(self) -> Any:
        """Return the lazily created Pinecone client."""

        if self._pinecone_client is None:
            self._pinecone_client = _create_pinecone_client()

        return self._pinecone_client

    @property
    def index(self) -> VectorIndex:
        """Return the Pinecone index."""

        if self._index is None:
            self._index = self.pinecone_client.Index(self.index_name)

        return self._index

    # --------------------------------------------------------
    # Index Management
    # --------------------------------------------------------

    def ensure_index(self) -> None:
        """
        Create the Pinecone index if it does not exist.

        The index dimension must match the embedding dimension.
        """

        existing_indexes = self.pinecone_client.list_indexes()

        names: set[str] = set()

        if hasattr(
            existing_indexes,
            "names",
        ):
            names = set(existing_indexes.names())

        elif isinstance(
            existing_indexes,
            Iterable,
        ):
            for item in existing_indexes:
                if isinstance(
                    item,
                    Mapping,
                ):
                    name = item.get("name")
                else:
                    name = getattr(
                        item,
                        "name",
                        None,
                    )

                if name:
                    names.add(name)

        if self.index_name in names:
            describe_index = getattr(self.pinecone_client, "describe_index", None)
            if callable(describe_index):
                description = describe_index(self.index_name)
                metric = self._get_value(description, "metric")
                dimension = self._get_value(description, "dimension")
                if metric and metric != settings.PINECONE_METRIC:
                    raise RuntimeError(
                        f"Pinecone index {self.index_name!r} uses metric {metric!r}; "
                        "single-index dense+sparse hybrid search requires "
                        f"{settings.PINECONE_METRIC}. "
                        "Delete and recreate it with the updated configuration."
                    )
                if dimension and int(dimension) != settings.PINECONE_DIMENSION:
                    raise RuntimeError(
                        f"Pinecone index {self.index_name!r} has dimension {dimension}; "
                        f"expected {settings.PINECONE_DIMENSION}. Delete and recreate it."
                    )
            self._index = self.pinecone_client.Index(self.index_name)
            logger.info("Pinecone hybrid index already exists: %s", self.index_name)
            return

        create_for_model = getattr(self.pinecone_client, "create_index_for_model", None)
        if not callable(create_for_model):
            raise RuntimeError(
                "The installed pinecone SDK must support create_index_for_model "
                "for the hybrid index's text field map."
            )
        create_for_model(
            name=self.index_name,
            cloud=settings.PINECONE_CLOUD,
            region=settings.PINECONE_REGION,
            embed={
                "model": settings.PINECONE_DENSE_EMBEDDING_MODEL,
                "field_map": {"text": "text"},
                "dimension": settings.PINECONE_DIMENSION,
                "metric": settings.PINECONE_METRIC,
                "write_parameters": {"input_type": "passage"},
                "read_parameters": {"input_type": "query"},
            },
        )
        logger.info(
            "Created Pinecone hybrid index: %s | dense=%s | sparse=%s "
            "| metric=%s | field_map=text:text",
            self.index_name,
            settings.PINECONE_DENSE_EMBEDDING_MODEL,
            settings.PINECONE_SPARSE_EMBEDDING_MODEL,
            settings.PINECONE_METRIC,
        )
        self._index = self.pinecone_client.Index(self.index_name)

    def index_exists(self) -> bool:
        """Check for an index without creating it (used by search and startup)."""
        if self._index is not None:
            return True
        existing_indexes = self.pinecone_client.list_indexes()
        if hasattr(existing_indexes, "names"):
            return self.index_name in set(existing_indexes.names())
        for item in existing_indexes or []:
            name = (
                item.get("name")
                if isinstance(item, Mapping)
                else getattr(item, "name", None)
            )
            if name == self.index_name:
                return True
        return False

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    @staticmethod
    def _to_plain_value(
        value: Any,
    ) -> Any:
        """
        Convert common Python/Pydantic values into Pinecone-safe
        metadata values.
        """

        if value is None:
            return None

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):
            return value

        if isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            return [
                PineconeVectorStore._to_plain_value(item)
                for item in value
                if item is not None
            ]

        if isinstance(
            value,
            Mapping,
        ):
            return {
                str(key): PineconeVectorStore._to_plain_value(item)
                for key, item in value.items()
                if item is not None
            }

        if hasattr(
            value,
            "value",
        ):
            return value.value

        return str(value)

    @classmethod
    def _clean_metadata(
        cls,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Convert metadata into Pinecone-compatible values."""

        if not metadata:
            return {}

        cleaned: dict[str, Any] = {}

        for key, value in metadata.items():
            if value is None:
                continue

            cleaned[str(key)] = cls._to_plain_value(value)

        return cleaned

    # --------------------------------------------------------
    # Chunk Helpers
    # --------------------------------------------------------

    @staticmethod
    def _get_value(
        obj: Any,
        key: str,
        default: Any = None,
    ) -> Any:
        if isinstance(
            obj,
            Mapping,
        ):
            return obj.get(
                key,
                default,
            )

        return getattr(
            obj,
            key,
            default,
        )

    @classmethod
    def _chunk_to_record(
        cls,
        chunk: Any,
        vector: Sequence[float],
        sparse_vector: Mapping[str, Sequence[Any]],
    ) -> VectorRecord:
        """
        Convert a DocumentChunk/Pydantic model/dict into an
        internal vector record.
        """

        metadata = (
            cls._get_value(
                chunk,
                "metadata",
                {},
            )
            or {}
        )

        if hasattr(
            metadata,
            "model_dump",
        ):
            metadata = metadata.model_dump()

        metadata = dict(metadata)

        vector_id = (
            cls._get_value(chunk, "chunk_id")
            or metadata.get("chunk_id")
            or cls._get_value(chunk, "id")
        )
        if not vector_id:
            raise ValueError("Chunk is missing chunk_id metadata.")

        content = (
            cls._get_value(chunk, "content")
            or cls._get_value(chunk, "text")
            or cls._get_value(chunk, "page_content")
            or ""
        )

        # Preserve important retrieval metadata explicitly.
        for key in (
            "chunk_id",
            "document_id",
            "candidate_id",
            "jd_id",
            "document_type",
            "candidate_name",
            "source",
            "page",
            "section",
            "chunk_index",
        ):
            value = cls._get_value(
                chunk,
                key,
            )

            if value is not None:
                metadata.setdefault(
                    key,
                    value,
                )

        metadata.setdefault(
            "text",
            content,
        )

        return VectorRecord(
            vector_id=str(vector_id),
            values=[float(value) for value in vector],
            metadata=cls._clean_metadata(metadata),
            sparse_values={
                "indices": list(sparse_vector.get("indices", [])),
                "values": [float(value) for value in sparse_vector.get("values", [])],
            },
        )

    def _embed_hybrid_texts(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
    ) -> list[tuple[list[float], dict[str, list[Any]]]]:
        """Generate Pinecone-hosted dense and sparse embeddings in matched batches."""
        if not texts:
            return []
        inference = getattr(self.pinecone_client, "inference", None)
        embed = getattr(inference, "embed", None)
        if not callable(embed):
            raise RuntimeError(
                "The installed pinecone SDK must support hosted inference.embed "
                "for dense+sparse hybrid vectors."
            )

        embedded: list[tuple[list[float], dict[str, list[Any]]]] = []
        inference_batch_size = 96
        for start in range(0, len(texts), inference_batch_size):
            batch = list(texts[start : start + inference_batch_size])
            dense_response = embed(
                model=settings.PINECONE_DENSE_EMBEDDING_MODEL,
                inputs=batch,
                parameters={
                    "input_type": input_type,
                    "truncate": "END",
                    "dimension": settings.PINECONE_DIMENSION,
                },
            )
            sparse_response = embed(
                model=settings.PINECONE_SPARSE_EMBEDDING_MODEL,
                inputs=batch,
                parameters={
                    "input_type": input_type,
                    "truncate": "END",
                },
            )
            dense_items = self._get_value(dense_response, "data", []) or []
            sparse_items = self._get_value(sparse_response, "data", []) or []
            if len(dense_items) != len(batch) or len(sparse_items) != len(batch):
                raise RuntimeError(
                    "Pinecone embedding count does not match input count."
                )

            for dense, sparse in zip(dense_items, sparse_items):
                dense_values = self._get_value(dense, "values", []) or []
                sparse_indices = self._get_value(sparse, "sparse_indices", []) or []
                sparse_values = self._get_value(sparse, "sparse_values", []) or []
                if len(dense_values) != settings.PINECONE_DIMENSION:
                    raise ValueError(
                        "Pinecone dense embedding dimension mismatch: "
                        f"expected {settings.PINECONE_DIMENSION}, got {len(dense_values)}."
                    )
                if not sparse_indices or len(sparse_indices) != len(sparse_values):
                    raise ValueError("Pinecone returned an invalid sparse embedding.")
                embedded.append(
                    (
                        [float(value) for value in dense_values],
                        {
                            "indices": [int(value) for value in sparse_indices],
                            "values": [float(value) for value in sparse_values],
                        },
                    )
                )
        return embedded

    # --------------------------------------------------------
    # Upsert
    # --------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: Sequence[Any],
        *,
        batch_size: int = 100,
    ) -> int:
        """
        Embed and upsert document chunks into Pinecone.

        Existing vector IDs are overwritten, which makes ingestion
        idempotent for the same chunk IDs.
        """

        if not chunks:
            return 0

        texts = []

        for chunk in chunks:
            text = (
                self._get_value(
                    chunk,
                    "content",
                )
                or self._get_value(
                    chunk,
                    "text",
                )
                or self._get_value(
                    chunk,
                    "page_content",
                )
                or ""
            )

            if not text.strip():
                raise ValueError("Cannot index an empty chunk.")

            texts.append(text)

        # Create the Pinecone index before the first vector upsert.
        self.ensure_index()

        logger.info(
            "Generating embeddings for %d chunks.",
            len(texts),
        )

        embeddings = self._embed_hybrid_texts(texts, input_type="passage")

        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding count does not match chunk count.")

        records = [
            self._chunk_to_record(
                chunk,
                vector,
                sparse_vector,
            )
            for chunk, (vector, sparse_vector) in zip(
                chunks,
                embeddings,
            )
        ]

        total_upserted = 0

        for start in range(
            0,
            len(records),
            batch_size,
        ):
            batch = records[start : start + batch_size]

            payload = [
                {
                    "id": record.vector_id,
                    "values": record.values,
                    "sparse_values": record.sparse_values,
                    "metadata": record.metadata,
                }
                for record in batch
            ]

            upsert_response = self.index.upsert(
                vectors=payload,
                namespace=self.namespace,
            )

            acknowledged = getattr(upsert_response, "upserted_count", None)
            if acknowledged is None and isinstance(upsert_response, Mapping):
                acknowledged = upsert_response.get("upserted_count")
            if acknowledged is not None and int(acknowledged) != len(batch):
                raise RuntimeError(
                    "Pinecone acknowledged an unexpected vector count: "
                    f"expected={len(batch)} acknowledged={acknowledged}."
                )

            total_upserted += (
                int(acknowledged) if acknowledged is not None else len(batch)
            )

            logger.debug(
                "Upserted Pinecone batch: %d vectors.",
                len(batch),
            )

        logger.info(
            "Pinecone upsert acknowledged: index=%s namespace=%r vectors=%d.",
            self.index_name,
            self.namespace,
            total_upserted,
        )

        return total_upserted

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    def _build_filter(
        self,
        filters: Any | None,
    ) -> dict[str, Any] | None:
        """
        Convert SearchFilters into Pinecone metadata filters.

        Only filters explicitly provided by the caller are sent
        to Pinecone.
        """

        if filters is None:
            return None

        if hasattr(
            filters,
            "model_dump",
        ):
            raw = filters.model_dump(exclude_none=True)
        elif isinstance(
            filters,
            Mapping,
        ):
            raw = dict(filters)
        else:
            raw = {
                key: value for key, value in vars(filters).items() if value is not None
            }

        pinecone_filter: dict[str, Any] = {}

        direct_fields = (
            "document_type",
            "candidate_id",
            "jd_id",
            "location",
            "education",
        )

        for field in direct_fields:
            value = raw.get(field)

            if value is None:
                continue

            if isinstance(
                value,
                (list, tuple, set),
            ):
                # SearchFilters uses empty lists as defaults. Sending an
                # empty $in constraint makes Pinecone reject every record.
                if not value:
                    continue
                pinecone_filter[field] = {"$in": list(value)}
            else:
                if hasattr(
                    value,
                    "value",
                ):
                    value = value.value

                pinecone_filter[field] = value

        candidate_ids = raw.get("candidate_ids")

        if candidate_ids:
            pinecone_filter["candidate_id"] = {"$in": list(candidate_ids)}

        skills = raw.get("skills")

        if skills:
            pinecone_filter["skills"] = {"$in": list(skills)}

        min_experience = raw.get("min_experience")

        if min_experience is not None:
            pinecone_filter["experience_years"] = {"$gte": min_experience}

        max_experience = raw.get("max_experience")

        if max_experience is not None:
            existing = pinecone_filter.get(
                "experience_years",
                {},
            )

            existing["$lte"] = max_experience

            pinecone_filter["experience_years"] = existing

        return pinecone_filter if pinecone_filter else None

    def _parse_matches(
        self,
        response: Any,
    ) -> list[VectorSearchMatch]:
        """Normalize a Pinecone query response."""

        matches = getattr(
            response,
            "matches",
            None,
        )

        if matches is None and isinstance(
            response,
            Mapping,
        ):
            matches = response.get(
                "matches",
                [],
            )

        matches = matches or []

        results: list[VectorSearchMatch] = []

        for match in matches:
            if isinstance(
                match,
                Mapping,
            ):
                vector_id = match.get("id")
                score = match.get(
                    "score",
                    0.0,
                )
                metadata = match.get("metadata") or {}

            else:
                vector_id = getattr(
                    match,
                    "id",
                    None,
                )
                score = getattr(
                    match,
                    "score",
                    0.0,
                )
                metadata = (
                    getattr(
                        match,
                        "metadata",
                        None,
                    )
                    or {}
                )

            if not vector_id:
                continue

            results.append(
                VectorSearchMatch(
                    vector_id=str(vector_id),
                    score=float(score or 0.0),
                    metadata=dict(metadata),
                )
            )

        return results

    # --------------------------------------------------------
    # RetrievalResult Conversion
    # --------------------------------------------------------

    @staticmethod
    def _build_retrieval_result(
        match: VectorSearchMatch,
    ) -> RetrievalResult:
        """
        Convert a Pinecone match into the application's
        RetrievalResult contract.

        model_validate is used so the retrieval layer remains
        compatible with the Pydantic schema without duplicating
        the schema definition here.
        """

        metadata = dict(match.metadata)

        payload: dict[str, Any] = {
            "chunk_id": (metadata.get("chunk_id") or match.vector_id),
            "document_id": metadata.get("document_id") or match.vector_id,
            "document_type": metadata.get("document_type", "resume"),
            "candidate_id": metadata.get("candidate_id"),
            "candidate_name": metadata.get("candidate_name"),
            "section": metadata.get("section"),
            "text": (metadata.get("text") or metadata.get("content") or ""),
            "source_file": metadata.get("source_file") or metadata.get("source"),
            "page_number": metadata.get("page_number") or metadata.get("page"),
            "retrieval_method": RetrievalMethod.DENSE,
            "raw_score": match.score,
            "normalized_score": match.score,
            "rank": 1,
            "metadata": metadata,
        }

        # Pydantic v2.
        if hasattr(
            RetrievalResult,
            "model_validate",
        ):
            return RetrievalResult.model_validate(payload)

        # Defensive compatibility for Pydantic v1.
        return RetrievalResult.parse_obj(payload)

    # --------------------------------------------------------
    # Semantic Search
    # --------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: Any | None = None,
    ) -> list[RetrievalResult]:
        """
        Perform semantic similarity search.

        Split long recruiter inputs into chunks, embed each chunk with
        RETRIEVAL_QUERY, search Pinecone for each vector, then merge
        duplicate chunk matches by their strongest similarity score.
        """

        if not query or not query.strip():
            return []

        k = top_k if top_k is not None else settings.VECTOR_TOP_K

        if k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Searching must not create an empty index; ingestion creates it on demand.
        if not self.index_exists():
            logger.info(
                "Pinecone index is not created yet; skipping search | index=%s",
                self.index_name,
            )
            return []

        query_chunks = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        ).split_text(query.strip())
        if not query_chunks:
            return []

        query_vectors = [
            dense
            for dense, _ in self._embed_hybrid_texts(query_chunks, input_type="query")
        ]

        pinecone_filter = self._build_filter(filters)

        best_matches: dict[str, VectorSearchMatch] = {}
        for chunk_number, vector in enumerate(query_vectors, start=1):
            response = self.index.query(
                vector=vector,
                top_k=k,
                namespace=self.namespace,
                filter=pinecone_filter,
                include_metadata=True,
                include_values=False,
            )
            parsed_matches = self._parse_matches(response)
            logger.info(
                "Pinecone query response | index=%s namespace=%r query_chunk=%d/%d matches=%d",
                self.index_name,
                self.namespace,
                chunk_number,
                len(query_vectors),
                len(parsed_matches),
            )
            for match in parsed_matches:
                current = best_matches.get(match.vector_id)
                if current is None or match.score > current.score:
                    best_matches[match.vector_id] = match

        matches = sorted(
            best_matches.values(),
            key=lambda match: match.score,
            reverse=True,
        )[:k]

        results = [
            self._build_retrieval_result(match).model_copy(update={"rank": rank})
            for rank, match in enumerate(matches, start=1)
        ]

        logger.info(
            "Chunked semantic search completed: chunks=%d results=%d top_k=%d",
            len(query_chunks),
            len(results),
            k,
        )

        return results

    def keyword_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: Any | None = None,
    ) -> list[RetrievalResult]:
        """Search the same index using Pinecone's hosted sparse embedder."""
        if not query or not query.strip():
            return []
        k = top_k if top_k is not None else settings.BM25_TOP_K
        if k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not self.index_exists():
            logger.info(
                "Pinecone index is not created yet; skipping sparse search | index=%s",
                self.index_name,
            )
            return []
        sparse = self._embed_hybrid_texts([query.strip()], input_type="query")[0][1]
        response = self.index.query(
            # Pinecone's hybrid indexes use dense vector_type indexes with
            # sparse vectors attached. The query API requires a dense vector
            # even when this retrieval leg should score only sparse matches.
            # A zero vector contributes 0 to dotproduct while preserving the
            # sparse score for the existing dense/sparse fusion step.
            vector=[0.0] * settings.PINECONE_DIMENSION,
            sparse_vector=sparse,
            top_k=k,
            namespace=self.namespace,
            filter=self._build_filter(filters),
            include_metadata=True,
            include_values=False,
        )
        return [
            self._build_retrieval_result(match).model_copy(
                update={"retrieval_method": RetrievalMethod.SPARSE, "rank": rank}
            )
            for rank, match in enumerate(self._parse_matches(response), start=1)
        ]

    # --------------------------------------------------------
    # Deletion
    # --------------------------------------------------------

    def delete_vectors(
        self,
        vector_ids: Sequence[str],
    ) -> None:
        """Delete vectors by their Pinecone IDs."""

        if not vector_ids:
            return

        self.index.delete(
            ids=list(vector_ids),
            namespace=self.namespace,
        )

        logger.info(
            "Deleted %d Pinecone vectors.",
            len(vector_ids),
        )

    def delete_by_document_id(
        self,
        document_id: str,
    ) -> None:
        """Delete all vectors belonging to a document."""

        if not document_id:
            return

        self.index.delete(
            filter={"document_id": document_id},
            namespace=self.namespace,
        )

        logger.info(
            "Deleted vectors for document_id=%s",
            document_id,
        )

    def delete_by_candidate_id(
        self,
        candidate_id: str,
    ) -> None:
        """Delete all vectors belonging to a candidate."""

        if not candidate_id:
            return

        self.index.delete(
            filter={"candidate_id": candidate_id},
            namespace=self.namespace,
        )

        logger.info(
            "Deleted vectors for candidate_id=%s",
            candidate_id,
        )

    def delete_by_jd_id(
        self,
        jd_id: str,
    ) -> None:
        """Delete all vectors belonging to a JD."""

        if not jd_id:
            return

        self.index.delete(
            filter={"jd_id": jd_id},
            namespace=self.namespace,
        )

        logger.info(
            "Deleted vectors for jd_id=%s",
            jd_id,
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    def stats(self) -> Any:
        """Return Pinecone index statistics."""
        if not self.index_exists():
            return {"namespaces": {}, "total_vector_count": 0}

        # Pinecone's current Index API does not accept namespace here;
        # namespace counts are included in the returned stats mapping.
        return self.index.describe_index_stats()


# ============================================================
# Factory
# ============================================================


def create_vector_store(
    *,
    embedding_provider: EmbeddingProvider | None = None,
    pinecone_client: Any | None = None,
    index: VectorIndex | None = None,
) -> PineconeVectorStore:
    """
    Factory for the application vector store.

    Dependencies can be injected for:
        - unit tests
        - local mocks
        - alternative embedding providers
        - alternative vector databases
    """

    return PineconeVectorStore(
        embedding_provider=embedding_provider,
        pinecone_client=pinecone_client,
        index=index,
    )


__all__ = [
    "EmbeddingProvider",
    "VectorIndex",
    "VectorRecord",
    "VectorSearchMatch",
    "GoogleEmbeddingProvider",
    "PineconeVectorStore",
    "create_vector_store",
]
