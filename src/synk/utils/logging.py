"""Structured logging helpers for SYNK.

Provides a JSON formatter for machine-readable logs and a console formatter
for humans.  Raw clinical note text must never be logged; callers should pass
identifiers only (see ``synk.utils.logging.log_safe``).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key in ("event", "duration_ms", "status_code", "path", "run_id", "model_version"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Compact human-readable console format."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return f"{ts} | {record.levelname:<8} | {record.name:<28} | {record.getMessage()}"


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure the root SYNK logger once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else ConsoleFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # Quiet noisy third-party loggers.
    for name in ("matplotlib", "urllib3", "mlflow", "pytorch_lightning", "lightning", "fontTools", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger; configures logging on first use."""
    configure_logging()
    return logging.getLogger(name)


def log_safe(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Log a message with extra fields, redacting anything that looks like note text.

    Only allowlisted scalar fields are forwarded; everything else is replaced by
    ``<redacted>`` to avoid leaking clinical content into logs.
    """
    allowed = {"patient_id", "encounter_id", "count", "n", "status", "model_version", "run_id", "path"}
    safe_fields = {k: (v if k in allowed else "<redacted>") for k, v in fields.items()}
    suffix = " ".join(f"{k}={v}" for k, v in safe_fields.items())
    logger.log(level, f"{message} {suffix}".strip())
