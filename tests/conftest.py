"""Shared fixtures for the SYNK test suite.

All tests run on synthetic data with a fixed seed and tiny dimensions so the
whole suite finishes quickly on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from synk.config.settings import load_config, merge_configs  # noqa: E402


@pytest.fixture(scope="session")
def config():
    cfg = load_config()
    cfg = merge_configs(cfg, {
        "synthetic": {
            "n_patients": 14,
            "min_stay_hours": 36,
            "max_stay_hours": 48,
            "sepsis_prevalence": 0.35,
            "missingness_rate": 0.10,
        },
        "preprocessing": {"observation_window_hours": 8},
        "labels": {"stride_hours": 4, "min_history_hours": 6},
        "model": {
            "temporal": {"hidden_dim": 16, "attention_heads": 2},
            "text": {"encoder_type": "tfidf", "tfidf_max_features": 256, "embedding_dim": 16},
            "fusion": {"hidden_dim": 16},
            "head_hidden": 16,
        },
        "training": {"max_epochs": 2, "batch_size": 16},
        "evaluation": {"bootstrap_n": 0},
    })
    return cfg


@pytest.fixture(scope="session")
def cohort(config):
    from synk.data.synthetic import generate_cohort

    return generate_cohort(config, seed=1234)


@pytest.fixture(scope="session")
def frames(cohort):
    return cohort.patients, cohort.observations, cohort.notes, cohort.outcomes


@pytest.fixture(scope="session")
def prepared(config, frames):
    """Preprocessed sample set + patient-level splits."""
    from synk.data.validation import validate_observations
    from synk.features.dataset import build_samples
    from synk.labels.sepsis_labels import build_label_table
    from synk.pipeline import patient_level_splits_from_labels
    from synk.preprocessing.text import TextPreprocessor
    from synk.preprocessing.vitals import VitalsPreprocessor

    patients, observations, notes, outcomes = frames
    validate_observations(observations, raise_on_error=False)
    notes = TextPreprocessor(config).clean_frame(notes)
    label_table = build_label_table(observations, outcomes, config)
    splits = patient_level_splits_from_labels(label_table, config)
    train_enc = set(splits["train"])
    pre = VitalsPreprocessor(config).fit(observations[observations["encounter_id"].isin(train_enc)])
    cleaned = pre.transform(observations)
    static_cols = [c for c in ("patient_id", "age", "sex", "weight") if c in patients.columns]
    sample_set = build_samples(cleaned, notes, label_table, patients[static_cols], config)
    return sample_set, splits, pre, label_table


@pytest.fixture(scope="session")
def trained_multimodal(config, prepared):
    """Train a tiny multimodal model (2 epochs) for evaluation/explainability tests."""
    from synk.training.trainer import train_model

    sample_set, splits, pre, _ = prepared
    cfg = merge_configs(config, {"model": {"modality": "both"}})
    result = train_model(cfg, sample_set, splits, run_id="test_multimodal")
    return cfg, sample_set, splits, result
