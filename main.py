"""
Kokoro - Application Entry Point.

Responsibilities:
    - Initialize application configuration.
    - Prepare required directories.
    - Validate core services during startup.
    - Provide the application entry point.

The business logic remains inside the core modules.
"""

from __future__ import annotations

import sys

from utils.config import get_settings
from utils.logger import logger


# ============================================================
# Configuration
# ============================================================


def initialize_configuration():
    """
    Load and validate Kokoro configuration.
    """

    settings = get_settings()

    # Create required application directories.
    settings.ensure_directories()

    logger.info(
        "%s configuration initialized.",
        settings.APP_NAME,
    )

    return settings


# ============================================================
# Startup Checks
# ============================================================


def startup_checks(
    settings,
) -> bool:
    """
    Run lightweight startup validation.

    External services are not called automatically here because
    application startup should not unnecessarily consume API
    requests.
    """

    logger.info(
        "Running Kokoro startup checks..."
    )

    # --------------------------------------------------------
    # Google API configuration
    # --------------------------------------------------------

    if not settings.GOOGLE_API_KEY:
        logger.warning(
            "GOOGLE_API_KEY is not configured."
        )

    # --------------------------------------------------------
    # Pinecone configuration
    # --------------------------------------------------------

    if not settings.PINECONE_API_KEY:
        logger.warning(
            "PINECONE_API_KEY is not configured."
        )

    # --------------------------------------------------------
    # Configuration validation
    # --------------------------------------------------------

    try:
        settings.validate_configuration()

    except Exception as exc:

        logger.error(
            "Configuration validation failed: %s",
            exc,
        )

        return False

    logger.info(
        "Kokoro startup checks completed successfully."
    )

    return True


# ============================================================
# Application Initialization
# ============================================================


def initialize_application():
    """
    Initialize Kokoro application services.

    The actual services remain owned by their respective
    modules. This function only initializes the application
    environment.
    """

    settings = initialize_configuration()

    if not startup_checks(settings):

        raise RuntimeError(
            "Kokoro startup checks failed."
        )

    logger.info(
        "%s application initialized.",
        settings.APP_NAME,
    )

    return settings


# ============================================================
# Streamlit Entry Point
# ============================================================


def run_streamlit() -> None:
    """
    Start the Streamlit application.

    Streamlit normally executes app.py directly.
    This function is provided as a convenience entry point.
    """

    try:
        from ui.app import main as streamlit_main

    except ImportError as exc:

        raise RuntimeError(
            "Unable to import the Streamlit application."
        ) from exc

    streamlit_main()


# ============================================================
# Main
# ============================================================


def main() -> int:
    """
    Kokoro application entry point.
    """

    try:

        initialize_application()

        logger.info(
            "Kokoro startup completed."
        )

        return 0

    except Exception as exc:

        logger.exception(
            "Kokoro startup failed: %s",
            exc,
        )

        return 1


# ============================================================
# Script Entry Point
# ============================================================


if __name__ == "__main__":
    sys.exit(
        main()
    )
