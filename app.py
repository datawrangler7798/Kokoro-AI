"""Root Streamlit entry point; the application UI lives in streamlit/app.py."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent / "streamlit" / "app.py"),
    run_name="__main__",
)
