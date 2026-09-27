"""
Central configuration for Kokoro AI.

All environment-based configuration is loaded and validated here.
Other modules should import `config` instead of reading environment
variables directly.
"""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Load values from .env
load_dotenv()


class AppConfig(BaseSettings):
    """
    Application configuration for Kokoro AI.
    """

    # ============================================================
    # Pinecone Configuration
    # ============================================================

    PINECONE_API_KEY: str = Field(default="")

    PINECONE_INDEX_NAME: str = Field(
        default="kokoro",
        min_length=1,
    )

    PINECONE_DIMENSION: int = Field(
        default=768,
        ge=1,
    )

    PINECONE_METRIC: str = Field(
        default="cosine",
    )

    # ============================================================
    # Google Gemini Configuration
    # ============================================================

    GOOGLE_API_KEY: str = Field(default="")

    LLM_MODEL: str = Field(
        default="gemini-3.6-flash",
        min_length=1,
    )

    # ============================================================
    # Application Configuration
    # ============================================================

    LLM_REQUESTS_PER_MINUTE: int = Field(
        default=5,
        ge=1,
    )

    MAX_TEXT_LENGTH: int = Field(
        default=4000,
        ge=100,
    )

    # ============================================================
    # Pydantic Settings Configuration
    # ============================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ============================================================
    # Validators
    # ============================================================

    @field_validator("PINECONE_METRIC")
    @classmethod
    def validate_pinecone_metric(cls, value: str) -> str:
        """
        Validate and normalize the Pinecone similarity metric.
        """

        value = value.strip().lower()

        allowed_metrics = {
            "cosine",
            "euclidean",
            "dotproduct",
        }

        if value not in allowed_metrics:
            raise ValueError(
                f"Unsupported Pinecone metric: {value}. "
                f"Expected one of: {sorted(allowed_metrics)}"
            )

        return value

    @field_validator("PINECONE_INDEX_NAME")
    @classmethod
    def validate_pinecone_index_name(cls, value: str) -> str:
        """
        Validate Pinecone index name.
        """

        value = value.strip()

        if not value:
            raise ValueError(
                "PINECONE_INDEX_NAME cannot be empty."
            )

        return value

    @field_validator("LLM_MODEL")
    @classmethod
    def validate_llm_model(cls, value: str) -> str:
        """
        Validate Gemini model name.
        """

        value = value.strip()

        if not value:
            raise ValueError(
                "LLM_MODEL cannot be empty."
            )

        return value

    # ============================================================
    # Credential Validation
    # ============================================================

    def validate_required_credentials(self) -> None:
        """
        Validate credentials before operations that require
        external services.
        """

        missing = []

        if not self.GOOGLE_API_KEY.strip():
            missing.append("GOOGLE_API_KEY")

        if not self.PINECONE_API_KEY.strip():
            missing.append("PINECONE_API_KEY")

        if missing:
            raise ValueError(
                "Missing required environment variables: "
                + ", ".join(missing)
            )


# ================================================================
# Cached Configuration
# ================================================================

@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Create and cache the application configuration.

    Using a cached configuration ensures that every module
    uses the same configuration object during application
    execution.
    """

    return AppConfig()


# Global configuration object
config = get_config()


# ================================================================
# Convenience Constants
# ================================================================

GOOGLE_API_KEY = config.GOOGLE_API_KEY
LLM_MODEL = config.LLM_MODEL
LLM_REQUESTS_PER_MINUTE = config.LLM_REQUESTS_PER_MINUTE

PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_INDEX_NAME = config.PINECONE_INDEX_NAME
PINECONE_DIMENSION = config.PINECONE_DIMENSION
PINECONE_METRIC = config.PINECONE_METRIC

MAX_TEXT_LENGTH = config.MAX_TEXT_LENGTH