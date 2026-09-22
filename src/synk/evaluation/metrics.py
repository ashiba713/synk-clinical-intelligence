"""Evaluation metrics for SYNK.

All metric functions operate on plain numpy arrays so they can be unit-tested
independently of any model.  Nothing here fabricates values: every number is
computed from ``(y_true, y_prob)`` pairs produced by an actual model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

RISK_CATEGORIES = ["LOW", "MODERATE", "HIGH", "CRITICAL"]


def risk_category(probability: float, thresholds: list[float]) -> str:
    """Map a probability to a category using configurable thresholds.

    ``thresholds`` holds the three upper bounds [low, moderate, high]; the
    final category (CRITICAL) covers everything above the third bound.
    """
    if len(thresholds) != 3:
        raise ValueError("risk_thresholds must contain exactly three upper bounds")
    p = float(probability)
    if p < thresholds[0]:
        return "LOW"
    if p < thresholds[1]:
        return "MODERATE"
    if p < thresholds[2]:
        return "HIGH"
    return "CRITICAL"


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Threshold-dependent classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
    }


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def auc_with_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "auroc",
    n_boot: int = 200,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Point estimate plus percentile bootstrap CI for AUROC or AUPRC."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    fn = roc_auc_score if metric == "auroc" else average_precision_score
    try:
        point = float(fn(y_true, y_score))
    except ValueError:
        return {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    if n_boot <= 0 or len(set(y_true.tolist())) < 2:
        return {"value": point, "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b, s_b = y_true[idx], y_score[idx]
        if len(set(y_b.tolist())) < 2:
            continue
        stats.append(float(fn(y_b, s_b)))
    if not stats:
        return {"value": point, "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    alpha = (1.0 - ci) / 2.0
    return {
        "value": point,
        "ci_low": float(np.quantile(stats, alpha)),
        "ci_high": float(np.quantile(stats, 1.0 - alpha)),
        "n_boot": len(stats),
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width binned ECE."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_prob)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def calibration_curve_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Reliability-diagram points: mean predicted probability vs observed rate."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    points = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        points.append({
            "bin_low": float(lo),
            "bin_high": float(hi),
            "mean_predicted": float(y_prob[mask].mean()),
            "observed_frequency": float(y_true[mask].mean()),
            "n": int(mask.sum()),
        })
    return points


def roc_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)]


def pr_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return [{"recall": float(r), "precision": float(p)} for r, p in zip(recall, precision)]


def sensitivity_at_operating_points(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: list[float]
) -> list[dict]:
    """Sensitivity/specificity at predefined operating points."""
    out = []
    for thr in thresholds:
        m = classification_metrics(y_true, y_prob, thr)
        out.append({
            "threshold": float(thr),
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "precision": m["precision"],
            "alert_rate": float((np.asarray(y_prob) >= thr).mean()),
        })
    return out


def lead_time_analysis(
    samples: "object",  # pandas DataFrame with encounter_id, t, label
    y_prob: np.ndarray,
    outcomes: "object",  # DataFrame with encounter_id, sepsis_onset_timestamp
    alert_threshold: float,
) -> dict:
    """Clinical early-warning metrics computed from real predictions.

    * detection_rate: fraction of septic encounters with >=1 alert before onset
    * lead time statistics over detected encounters (hours)
    * false alerts per patient over non-septic encounters
    """
    import pandas as pd

    df = samples[["encounter_id", "t"]].copy()
    df["y_prob"] = y_prob
    onset_map = dict(zip(outcomes["encounter_id"], pd.to_datetime(outcomes["sepsis_onset_timestamp"])))
    df["onset"] = df["encounter_id"].map(onset_map)

    septic = df[df["onset"].notna()]
    non_septic = df[df["onset"].isna()]

    leads: list[float] = []
    detected_encounters = 0
    for enc, group in septic.groupby("encounter_id"):
        onset = group["onset"].iloc[0]
        alerts = group[(group["y_prob"] >= alert_threshold) & (group["t"] < onset)]
        if len(alerts):
            detected_encounters += 1
            first_alert = alerts["t"].min()
            leads.append((onset - first_alert).total_seconds() / 3600.0)

    n_septic = septic["encounter_id"].nunique()
    detection_rate = detected_encounters / n_septic if n_septic else float("nan")

    false_alert_encounters = int(
        non_septic.groupby("encounter_id")["y_prob"].apply(lambda s: (s >= alert_threshold).any()).sum()
    )
    n_non_septic = non_septic["encounter_id"].nunique()
    n_false_alerts = int((non_septic["y_prob"] >= alert_threshold).sum())

    return {
        "alert_threshold": float(alert_threshold),
        "n_septic_encounters": int(n_septic),
        "n_non_septic_encounters": int(n_non_septic),
        "detected_encounters": int(detected_encounters),
        "detection_rate": float(detection_rate),
        "lead_time_mean_hours": float(np.mean(leads)) if leads else float("nan"),
        "lead_time_median_hours": float(np.median(leads)) if leads else float("nan"),
        "lead_time_min_hours": float(np.min(leads)) if leads else float("nan"),
        "lead_time_max_hours": float(np.max(leads)) if leads else float("nan"),
        "n_leads": len(leads),
        "false_alert_encounters": false_alert_encounters,
        "false_alert_rate_per_encounter": (
            false_alert_encounters / n_non_septic if n_non_septic else float("nan")
        ),
        "n_false_alerts": n_false_alerts,
    }
