"""Data access layer for the Streamlit dashboard.

Talks to the local inference service directly (default) or to the FastAPI
backend when ``SYNK_API_URL`` is set.  Caches dataframes for responsiveness.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

import pandas as pd
import streamlit as st


@lru_cache(maxsize=1)
def _get_config():
    from synk.config.settings import load_config

    return load_config()


@lru_cache(maxsize=1)
def load_frames():
    """Load (patients, observations, notes, outcomes), generating if needed."""
    from synk.pipeline import load_or_generate_data

    cfg = _get_config()
    return load_or_generate_data(cfg)


@lru_cache(maxsize=1)
def get_service():
    """Initialise the inference service once per process."""
    from synk.inference.engine import InferenceService

    cfg = _get_config()
    patients, observations, notes, outcomes = load_frames()
    service = InferenceService(cfg)
    engine_id = service.ensure_ready(observations, notes, outcomes, patients)
    return service


@st.cache_data(ttl=300, show_spinner="Computing predictions…")
def predictions_frame(_refresh: int = 0) -> pd.DataFrame:
    """All predictions across the cohort (cached)."""
    service = get_service()
    patients, observations, notes, outcomes = load_frames()
    return service.predict_encounters(observations, notes, outcomes, patients)


def sample_set():
    service = get_service()
    patients, observations, notes, outcomes = load_frames()
    return service.featurizer.build(observations, notes, outcomes, patients)[0]


def engine_id() -> str:
    return get_service().engine_id


def model_info() -> dict:
    return get_service().model_info()


def explain(encounter_id: str) -> dict[str, Any]:
    service = get_service()
    ss = sample_set()
    sub = ss.samples[ss.samples["encounter_id"] == encounter_id]
    if sub.empty:
        return {}
    return service.explain_sample(ss, int(sub.index[-1]))


def experiments_frame() -> pd.DataFrame:
    from synk.tracking.registry import ExperimentRegistry
    from pathlib import Path

    cfg = _get_config()
    reg = ExperimentRegistry(Path(cfg.paths.evaluation_dir) / "experiments.json")
    return reg.as_frame()


def latest_eval_report() -> Optional[dict]:
    import json
    from pathlib import Path

    cfg = _get_config()
    eval_dir = Path(cfg.paths.evaluation_dir)
    candidates = sorted(eval_dir.glob("*_test.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return json.loads(candidates[0].read_text(encoding="utf-8"))
