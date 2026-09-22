"""Classical baselines for the research comparison table.

Trains transparent scikit-learn models on the flattened tabular features:

* ``structured``  - logistic regression on the structured summary features
* ``text``        - TF-IDF (word) + logistic regression over clinical notes
* ``fusion``      - late fusion: concatenation of the two probability vectors
                    recalibrated by a small logistic meta-model (early-fusion
                    of flattened features)

These baselines are deliberately simple: they anchor the deep models so the
research question ("does multimodal fusion help?") can be answered against
transparent references, not only against SYNK's own ablations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from synk.features.dataset import SampleSet
from synk.features.tabular import build_tabular_features

logger = logging.getLogger(__name__)


def _prob_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    out: dict[str, float] = {
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else float("nan")
    except ValueError:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        out["auprc"] = float("nan")
    return out


def _aligned_texts(sample_set: SampleSet, encounter_ids: list[str]) -> tuple[pd.Index, list[str], np.ndarray]:
    """Return (indices, texts, y) aligned to the encounter subset."""
    mask = sample_set.samples["encounter_id"].isin(set(encounter_ids))
    subset = sample_set.samples[mask].reset_index(drop=False).rename(columns={"index": "row"})
    texts = subset["text"].astype(str).tolist()
    y = subset["label"].to_numpy().astype(int)
    return subset["row"], texts, y


def train_structured_baseline(
    sample_set: SampleSet,
    train_encounters: list[str],
    val_encounters: list[str],
    seed: int = 42,
) -> dict[str, Any]:
    """Logistic regression on flattened structured features (train split only)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    feats = build_tabular_features(sample_set)
    train_rows, _, y_train = _aligned_texts(sample_set, train_encounters)
    val_rows, _, y_val = _aligned_texts(sample_set, val_encounters)

    X_train = feats.loc[train_rows].drop(columns=["encounter_id", "t"]).to_numpy(dtype=np.float32)
    X_val = feats.loc[val_rows].drop(columns=["encounter_id", "t"]).to_numpy(dtype=np.float32)

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])
    pipe.fit(X_train, y_train)
    val_prob = pipe.predict_proba(X_val)[:, 1]
    return {
        "model": pipe,
        "kind": "structured",
        "feature_names": [c for c in feats.columns if c not in ("encounter_id", "t")],
        "val_prob": val_prob,
        "val_y": y_val,
        "val_metrics": _prob_metrics(y_val, val_prob),
    }


def train_text_baseline(
    sample_set: SampleSet,
    train_encounters: list[str],
    val_encounters: list[str],
    seed: int = 42,
    max_features: int = 4096,
) -> dict[str, Any]:
    """TF-IDF word unigram/bigram + logistic regression over note text."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    _, train_texts, y_train = _aligned_texts(sample_set, train_encounters)
    _, val_texts, y_val = _aligned_texts(sample_set, val_encounters)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])
    filled = [t if t.strip() else "no clinical note available" for t in train_texts]
    pipe.fit(filled, y_train)
    val_prob = pipe.predict_proba([t if t.strip() else "no clinical note available" for t in val_texts])[:, 1]
    return {
        "model": pipe,
        "kind": "text",
        "val_prob": val_prob,
        "val_y": y_val,
        "val_metrics": _prob_metrics(y_val, val_prob),
    }


def train_late_fusion_baseline(
    structured_result: dict[str, Any],
    text_result: dict[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Meta logistic regression over the two unimodal probability vectors.

    Fits on validation-set predictions of the two base models (a standard
    stacking pattern; the meta-learner never sees raw test data).
    """
    from sklearn.linear_model import LogisticRegression

    y_val = structured_result["val_y"]
    X_meta = np.column_stack([
        np.asarray(structured_result["val_prob"], dtype=float),
        np.asarray(text_result["val_prob"], dtype=float),
    ])
    meta = LogisticRegression(C=1.0, random_state=seed)
    meta.fit(X_meta, y_val)
    fused_val = meta.predict_proba(X_meta)[:, 1]
    return {
        "model": meta,
        "kind": "fusion_late",
        "val_prob": fused_val,
        "val_y": y_val,
        "val_metrics": _prob_metrics(y_val, fused_val),
    }


# ---------------------------------------------------------------------------
# Persistence for serving baseline models at inference time
# ---------------------------------------------------------------------------
def save_baselines(payload: dict[str, Any], path: Path) -> None:
    import pickle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    logger.info("Baselines saved: %s", path)


def load_baselines(path: Path) -> dict[str, Any]:
    import pickle

    with open(Path(path), "rb") as fh:
        return pickle.load(fh)
