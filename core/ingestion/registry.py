"""Small JSON registry for idempotent local document ingestion."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.config import get_settings


class DocumentRegistry:
    """Persist document hashes so already indexed PDFs are skipped."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or get_settings().DOCUMENT_REGISTRY_PATH)
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            records = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(records, dict):
                self._records = records
        except (OSError, json.JSONDecodeError):
            # A damaged registry should not prevent the app from starting.
            self._records = {}

    def exists(self, document_hash: str) -> bool:
        with self._lock:
            return document_hash in self._records

    def save(self, document: Any) -> None:
        record = {
            "document_id": document.document_id,
            "document_hash": document.document_hash,
            "source_file": document.source_file,
            "document_type": getattr(document.document_type, "value", str(document.document_type)),
            "candidate_id": document.candidate_id,
            "candidate_name": document.candidate_name,
            "indexed": True,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._records[document.document_hash] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(self._records, indent=2), encoding="utf-8")
            temporary.replace(self.path)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            if self.path.exists():
                self.path.unlink()
