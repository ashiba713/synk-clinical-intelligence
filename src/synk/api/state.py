"""Shared application state for the FastAPI backend.

Holds the loaded dataset (synthetic by default), the inference service and the
database handle.  Initialised once at start-up and shared by all routers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from synk.config.settings import SYNKConfig, load_config
from synk.database.models import Database, build_database

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    config: SYNKConfig
    patients: Optional[pd.DataFrame] = None
    observations: Optional[pd.DataFrame] = None
    notes: Optional[pd.DataFrame] = None
    outcomes: Optional[pd.DataFrame] = None
    service: Any = None  # InferenceService (avoids import cycle at module load)
    db: Optional[Database] = None
    engine_ready: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def load_demo_data(self) -> None:
        from synk.pipeline import load_or_generate_data

        patients, observations, notes, outcomes = load_or_generate_data(self.config)
        self.patients, self.observations, self.notes, self.outcomes = patients, observations, notes, outcomes

    def ensure_engine(self) -> Any:
        """Initialise the inference service once (torch bundle or demo fallback)."""
        if self.service is not None and self.engine_ready:
            return self.service
        from synk.inference.engine import InferenceService

        service = InferenceService(self.config)
        engine_id = service.ensure_ready(self.observations, self.notes, self.outcomes, self.patients)
        logger.info("Inference engine ready: %s", engine_id)
        self.service = service
        self.engine_ready = True
        return service

    def ensure_db(self) -> Database:
        if self.db is None:
            self.db = build_database(self.config.api.default_db_url)
            if self.patients is not None and self.outcomes is not None:
                self.db.seed_from_frames(self.patients, self.outcomes, is_synthetic=True)
        return self.db


_STATE: Optional[AppState] = None


def init_state(config: Optional[SYNKConfig] = None) -> AppState:
    global _STATE
    if _STATE is None:
        cfg = config or load_config()
        state = AppState(config=cfg)
        state.load_demo_data()
        if cfg.api.load_synthetic_on_start:
            state.ensure_engine()
        state.ensure_db()
        _STATE = state
    return _STATE


def get_state() -> AppState:
    assert _STATE is not None, "AppState not initialised"
    return _STATE
