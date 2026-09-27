"""
utils/config.py

Centralized application configuration for Kokoro.

Configuration is loaded from environment variables / .env.

Architecture:
    .env
      ↓
    Settings
      ↓
    Ingestion / Retrieval / Reranking / Generation /
    Memory / Guardrails / Evaluation

The configuration is centralized here so that individual modules
do not directly read environment variables.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide configuration.

    Values are loaded from the .env file and can also be overridden
    through environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ============================================================
    # APPLICATION
    # ============================================================

    APP_NAME: str = "Kokoro"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # ============================================================
    # API KEYS
    # ============================================================

    GOOGLE_API_KEY: SecretStr
    PINECONE_API_KEY: SecretStr

    # ============================================================
    # GEMINI LLM
    # ============================================================

    # Gemini 3.6 Flash is used for:
    #   1. Candidate reranking
    #   2. Final response generation

    LLM_MODEL: str = "gemini-3.6-flash"

    LLM_TEMPERATURE: float = 0.0

    LLM_MAX_OUTPUT_TOKENS: int = 2048

    # Maximum LLM requests allowed per minute.
    LLM_REQUESTS_PER_MINUTE: int = 5

    # ============================================================
    # GOOGLE EMBEDDINGS
    # ============================================================

    # The same embedding model is used for:
    #
    #   Resume/document → embedding
    #   User query      → embedding
    #
    # Both must use the same embedding space.

    EMBEDDING_MODEL: str = "gemini-embedding-001"

    # Gemini Embedding 001 supports configurable output
    # dimensionality.
    #
    # Kokoro uses 768 dimensions.
    #
    # This must match the dimension produced by the embedding
    # service and the Pinecone index.

    EMBEDDING_DIMENSION: int = 768

    # ============================================================
    # PINECONE
    # ============================================================

    PINECONE_INDEX_NAME: str = "Kokoro"

    PINECONE_DIMENSION: int = 768

    PINECONE_METRIC: str = "cosine"

    # ============================================================
    # RETRIEVAL
    # ============================================================

    # 1. Semantic retrieval
    # Pinecone returns top 15 chunks.
    VECTOR_TOP_K: int = 15

    # 2. Lexical retrieval
    # BM25 returns top 15 chunks.
    BM25_TOP_K: int = 15

    # 3. Score fusion + normalization + deduplication
    # Keep top 10 hybrid results.
    HYBRID_TOP_K: int = 10

    # 4. Gemini reranking
    # Keep top 5 candidates/evidence units.
    RERANK_TOP_K: int = 5

    # 5. Final evidence sent to generation.
    FINAL_CONTEXT_K: int = 5

    # ============================================================
    # HYBRID SEARCH
    # ============================================================

    ENABLE_HYBRID_SEARCH: bool = True

    # Semantic + lexical retrieval.
    SEMANTIC_WEIGHT: float = 0.6
    LEXICAL_WEIGHT: float = 0.4

    # ============================================================
    # QUERY PROCESSING
    # ============================================================

    ENABLE_MULTI_QUERY: bool = True

    ENABLE_QUERY_DECOMPOSITION: bool = True

    # ============================================================
    # CHUNKING
    # ============================================================

    CHUNK_SIZE: int = 1000

    CHUNK_OVERLAP: int = 150

    # ============================================================
    # INGESTION
    # ============================================================

    MAX_FILE_SIZE_MB: int = 10

    MAX_BATCH_FILES: int = 50

    ALLOWED_FILE_EXTENSIONS: str = ".pdf"

    # ============================================================
    # STORAGE
    # ============================================================

    RESUME_DIRECTORY: str = "data/resumes"

    JD_DIRECTORY: str = "data/jds"

    # BM25 index persistence.
    BM25_DIRECTORY: str = "data/index/bm25"

    # Registry used for incremental/idempotent ingestion.
    DOCUMENT_REGISTRY_PATH: str = (
        "data/index/document_registry.json"
    )

    # ============================================================
    # CACHE
    # ============================================================

    ENABLE_CACHE: bool = True

    CACHE_TTL_SECONDS: int = 3600

    # ============================================================
    # MEMORY
    # ============================================================

    ENABLE_MEMORY: bool = True

    MAX_MEMORY_ITEMS: int = 10

    # ============================================================
    # GUARDRAILS
    # ============================================================

    ENABLE_INPUT_GUARDRAIL: bool = True

    ENABLE_OUTPUT_GUARDRAIL: bool = True

    BLOCK_PROMPT_INJECTION: bool = True

    BLOCK_UNSUPPORTED_CLAIMS: bool = True

    # ============================================================
    # OUTPUT VALIDATION
    # ============================================================

    ENABLE_PYDANTIC_VALIDATION: bool = True

    MAX_RETRY_ATTEMPTS: int = 2

    # ============================================================
    # OBSERVABILITY
    # ============================================================

    ENABLE_OBSERVABILITY: bool = True

    # Do not log unnecessary resume PII.
    LOG_RESUME_PII: bool = False

    LOG_RETRIEVAL_SCORES: bool = True

    LOG_LATENCY: bool = True

    # ============================================================
    # EVALUATION
    # ============================================================

    # Evaluation is exposed through the Streamlit Evaluation tab.

    ENABLE_EVALUATION: bool = True

    ENABLE_RAGAS: bool = True

    # Ground-truth evaluation dataset.
    EVALUATION_DATASET_PATH: str = (
        "evaluation/datasets/evaluation_dataset.json"
    )

    # Evaluation output/history.
    EVALUATION_RESULTS_DIRECTORY: str = (
        "evaluation/results"
    )

    # Precision@K / Recall@K values.
    #
    # Example:
    #   Precision@1
    #   Precision@3
    #   Precision@5
    #   Precision@10
    #
    # Stored as comma-separated values because environment
    # variables are strings.

    EVALUATION_K_VALUES: str = "1,3,5,10"

    # ============================================================
    # UI FILTERS
    # ============================================================

    # Filters remain optional.
    # Streamlit collects filters, but retrieval logic applies them.

    ENABLE_UI_FILTERS: bool = True

    # Available structured filters.
    AVAILABLE_FILTERS: str = (
        "experience,"
        "location,"
        "skills,"
        "education,"
        "certification,"
        "jd,"
        "document_type"
    )

    # ============================================================
    # VALIDATORS
    # ============================================================

    @field_validator(
        "EMBEDDING_DIMENSION",
        "PINECONE_DIMENSION",
    )
    @classmethod
    def validate_dimension(cls, value: int) -> int:
        """Ensure embedding dimensions are positive."""

        if value <= 0:
            raise ValueError(
                "Embedding/Pinecone dimension must be greater than 0."
            )

        return value

    # ------------------------------------------------------------
    # Pinecone dimension must match embedding dimension.
    # ------------------------------------------------------------

    @model_validator(mode="after")
    def validate_dimensions_match(self):
        """Ensure Pinecone and embedding dimensions match."""

        if self.PINECONE_DIMENSION != self.EMBEDDING_DIMENSION:
            raise ValueError(
                "PINECONE_DIMENSION must match "
                "EMBEDDING_DIMENSION."
            )

        return self

    # ------------------------------------------------------------
    # Retrieval top-k validation
    # ------------------------------------------------------------

    @field_validator(
        "VECTOR_TOP_K",
        "BM25_TOP_K",
        "HYBRID_TOP_K",
        "RERANK_TOP_K",
        "FINAL_CONTEXT_K",
    )
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        """Ensure retrieval limits are positive."""

        if value <= 0:
            raise ValueError(
                "Retrieval top-k values must be greater than 0."
            )

        return value

    # ------------------------------------------------------------
    # Retrieval K hierarchy
    # ------------------------------------------------------------

    @model_validator(mode="after")
    def validate_retrieval_hierarchy(self):
        """
        Validate the retrieval pipeline hierarchy.

        Expected flow:

        Pinecone/BM25
            ↓
        Hybrid
            ↓
        Reranker
            ↓
        Final context
        """

        if self.VECTOR_TOP_K < self.HYBRID_TOP_K:
            raise ValueError(
                "VECTOR_TOP_K must be greater than or equal "
                "to HYBRID_TOP_K."
            )

        if self.BM25_TOP_K < self.HYBRID_TOP_K:
            raise ValueError(
                "BM25_TOP_K must be greater than or equal "
                "to HYBRID_TOP_K."
            )

        if self.HYBRID_TOP_K < self.RERANK_TOP_K:
            raise ValueError(
                "HYBRID_TOP_K must be greater than or equal "
                "to RERANK_TOP_K."
            )

        if self.RERANK_TOP_K < self.FINAL_CONTEXT_K:
            raise ValueError(
                "RERANK_TOP_K must be greater than or equal "
                "to FINAL_CONTEXT_K."
            )

        return self

    # ------------------------------------------------------------
    # Hybrid retrieval weights
    # ------------------------------------------------------------

    @field_validator(
        "SEMANTIC_WEIGHT",
        "LEXICAL_WEIGHT",
    )
    @classmethod
    def validate_weight(cls, value: float) -> float:
        """Ensure retrieval weights are between 0 and 1."""

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "Retrieval weights must be between 0 and 1."
            )

        return value

    @model_validator(mode="after")
    def validate_weight_sum(self):
        """
        Ensure semantic and lexical weights sum to 1.
        """

        total = (
            self.SEMANTIC_WEIGHT
            + self.LEXICAL_WEIGHT
        )

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "SEMANTIC_WEIGHT + LEXICAL_WEIGHT "
                "must equal 1.0."
            )

        return self

    # ------------------------------------------------------------
    # Chunk size
    # ------------------------------------------------------------

    @field_validator("CHUNK_SIZE")
    @classmethod
    def validate_chunk_size(cls, value: int) -> int:
        """Ensure chunk size is positive."""

        if value <= 0:
            raise ValueError(
                "CHUNK_SIZE must be greater than 0."
            )

        return value

    # ------------------------------------------------------------
    # Chunk overlap
    # ------------------------------------------------------------

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_chunk_overlap(cls, value: int) -> int:
        """Ensure chunk overlap is non-negative."""

        if value < 0:
            raise ValueError(
                "CHUNK_OVERLAP cannot be negative."
            )

        return value

    @model_validator(mode="after")
    def validate_overlap_against_chunk_size(self):
        """Ensure overlap is smaller than chunk size."""

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
            )

        return self

    # ------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------

    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """Validate Gemini temperature."""

        if not 0.0 <= value <= 2.0:
            raise ValueError(
                "LLM_TEMPERATURE must be between 0 and 2."
            )

        return value

    # ------------------------------------------------------------
    # Positive operational limits
    # ------------------------------------------------------------

    @field_validator(
        "LLM_REQUESTS_PER_MINUTE",
        "MAX_BATCH_FILES",
        "MAX_FILE_SIZE_MB",
        "MAX_MEMORY_ITEMS",
        "MAX_RETRY_ATTEMPTS",
        "CACHE_TTL_SECONDS",
    )
    @classmethod
    def validate_positive_integer(cls, value: int) -> int:
        """Ensure operational limits are positive."""

        if value <= 0:
            raise ValueError(
                "Configuration value must be greater than 0."
            )

        return value

    # ============================================================
    # HELPER FUNCTIONS
    # ============================================================

    def get_evaluation_k_values(self) -> list[int]:
        """
        Convert EVALUATION_K_VALUES into a sorted list of integers.

        Example:
            "1,3,5,10"

        becomes:

            [1, 3, 5, 10]
        """

        try:
            values = [
                int(value.strip())
                for value in self.EVALUATION_K_VALUES.split(",")
                if value.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                "EVALUATION_K_VALUES must contain only integers."
            ) from exc

        if not values:
            raise ValueError(
                "EVALUATION_K_VALUES cannot be empty."
            )

        if any(value <= 0 for value in values):
            raise ValueError(
                "Evaluation K values must be greater than 0."
            )

        return sorted(set(values))

    def get_allowed_extensions(self) -> list[str]:
        """
        Convert comma-separated file extensions into a list.

        Example:
            ".pdf,.PDF"

        becomes:

            [".pdf", ".pdf"]
        """

        extensions = [
            extension.strip().lower()
            for extension in self.ALLOWED_FILE_EXTENSIONS.split(",")
            if extension.strip()
        ]

        return extensions

    def get_available_filters(self) -> list[str]:
        """
        Convert configured UI filters into a list.
        """

        return [
            filter_name.strip()
            for filter_name in self.AVAILABLE_FILTERS.split(",")
            if filter_name.strip()
        ]

    def ensure_directories(self) -> None:
        """
        Create required local directories if they do not exist.

        This supports:

            1. Initial local resume corpus
            2. Streamlit uploads
            3. BM25 persistence
            4. Document registry
            5. Evaluation results
        """

        directories = [
            self.RESUME_DIRECTORY,
            self.JD_DIRECTORY,
            self.BM25_DIRECTORY,
            self.EVALUATION_RESULTS_DIRECTORY,
        ]

        for directory in directories:
            Path(directory).mkdir(
                parents=True,
                exist_ok=True,
            )

        registry_parent = Path(
            self.DOCUMENT_REGISTRY_PATH
        ).parent

        registry_parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset_parent = Path(
            self.EVALUATION_DATASET_PATH
        ).parent

        dataset_parent.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# SETTINGS SINGLETON
# ============================================================


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Create and cache the application settings.

    Using lru_cache ensures that the .env file is parsed only once
    during the application's lifetime.
    """

    settings_instance = Settings()

    # Create required directories during application startup.
    settings_instance.ensure_directories()

    return settings_instance


# Global settings object used throughout the application.
settings = get_settings()