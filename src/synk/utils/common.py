"""Small shared helpers used across SYNK."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


def utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    """Generate a short readable unique id such as ``pred_ab12cd34``."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@contextmanager
def timer(logger=None, label: str = "") -> Iterator[Any]:
    """Context manager measuring wall-clock duration; logs when a logger is given."""
    start = time.perf_counter()
    holder: dict[str, float] = {}
    try:
        yield holder
    finally:
        elapsed = (time.perf_counter() - start) * 1000.0
        holder["ms"] = elapsed
        if logger is not None:
            logger.info("%s completed in %.1f ms", label, elapsed)


def sha256_of_file(path: Path) -> str:
    """Streaming SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_dict(payload: dict[str, Any]) -> str:
    """Stable checksum of a JSON-serialisable mapping."""
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert to float, returning *default* for None/NaN/inf."""
    try:
        if value is None:
            return default
        out = float(value)
        if out != out or out in (float("inf"), float("-inf")):
            return default
        return out
    except (TypeError, ValueError):
        return default


def format_probability(p: float) -> str:
    return f"{p:.3f}"
