"""Evaluation, calibration and training tests.

Metric functions are verified against hand-computed references so the suite
would catch silent metric fabrication.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def test_classification_metrics_exact():
    from synk.evaluation.metrics import classification_metrics

    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.4, 0.7, 0.8, 0.9, 0.55])
    m = classification_metrics(y, p, threshold=0.5)
    # predictions: [0,0,1,0,1,1,1,1] -> tp=4 fp=1 fn=1 tn=2
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 4}
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(4 / 5)
    assert m["recall"] == pytest.approx(4 / 5)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(0.8)


def test_auroc_auprc_known_values():
    from synk.evaluation.metrics import auc_with_bootstrap_ci

    # perfect separation
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    r = auc_with_bootstrap_ci(y, s, "auroc", n_boot=0)
    assert r["value"] == pytest.approx(1.0)
    r2 = auc_with_bootstrap_ci(y, s, "auprc", n_boot=0)
    assert r2["value"] == pytest.approx(1.0)
    # random guessing
    y = np.array([0, 1, 0, 1])
    s = np.array([0.5, 0.5, 0.5, 0.5])
    r3 = auc_with_bootstrap_ci(y, s, "auroc", n_boot=0)
    assert r3["value"] == pytest.approx(0.5)


def test_brier_and_ece():
    from synk.evaluation.metrics import brier_score, expected_calibration_error

    y = np.array([1, 0])
    p = np.array([1.0, 0.0])
    assert brier_score(y, p) == pytest.approx(0.0)
    assert brier_score(y, 1 - p) == pytest.approx(1.0)
    # perfectly calibrated in one bin -> ECE 0
    y = np.array([1, 0])
    p = np.array([0.75, 0.25])
    # bin [0,0.5]: conf 0.25 acc 0 | bin (0.5,1]: conf 0.75 acc 1 -> ECE 0.25
    assert expected_calibration_error(y, p) == pytest.approx(0.25, abs=1e-6)


def test_risk_categories():
    from synk.evaluation.metrics import risk_category

    thresholds = [0.30, 0.60, 0.80]
    assert risk_category(0.1, thresholds) == "LOW"
    assert risk_category(0.4, thresholds) == "MODERATE"
    assert risk_category(0.7, thresholds) == "HIGH"
    assert risk_category(0.9, thresholds) == "CRITICAL"
    with pytest.raises(ValueError):
        risk_category(0.5, [0.3, 0.6])


def test_calibration_platt_roundtrip(tmp_path):
    from synk.evaluation.calibration import Calibrator

    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    p = np.clip(y * 0.6 + rng.normal(0.5, 0.2, n), 0.01, 0.99)
    cal = Calibrator("platt").fit(p, y)
    out = cal.transform(p)
    assert len(out) == n
    path = tmp_path / "cal.json"
    cal.save(path)
    cal2 = Calibrator.load(path)
    np.testing.assert_allclose(cal2.transform(p), out, atol=1e-6)


def test_train_and_evaluate_multimodal(trained_multimodal, config):
    cfg, sample_set, splits, result = trained_multimodal
    assert result.best_epoch >= 1
    assert result.checkpoint_path.endswith(".pt")

    from pathlib import Path

    from synk.evaluation.evaluator import evaluate_model
    from synk.models.multimodal import load_trained_model

    model, _payload = load_trained_model(result.checkpoint_path, cfg)

    report = evaluate_model(model, sample_set, splits["test"], cfg,
                            outcomes=None, include_ablation=True)
    assert 0.0 <= report["auroc"]["value"] <= 1.0
    assert 0.0 <= report["auprc"]["value"] <= 1.0
    assert 0.0 <= report["brier_score"] <= 1.0
    tm = report["threshold_metrics"]
    assert tm["true_positives"] + tm["false_positives"] + tm["true_negatives"] + tm["false_negatives"] \
        == tm["n_positive"] + tm["n_negative"] == report["n_samples"]
    assert "ablation" in report and "both_active" in report["ablation"]


def test_lead_time_analysis():
    import pandas as pd

    from synk.evaluation.metrics import lead_time_analysis

    onset = pd.Timestamp("2025-01-01 12:00")
    meta = pd.DataFrame({
        "encounter_id": ["a"] * 4 + ["b"] * 2,
        "t": [onset - pd.Timedelta(hours=h) for h in (6, 4, 2, 0)] + [onset - pd.Timedelta(hours=2), onset],
        "label": [0, 0, 0, 0] + [0, 0],
    })
    probs = np.array([0.8, 0.7, 0.6, 0.9, 0.1, 0.1])
    outcomes = pd.DataFrame({
        "encounter_id": ["a"],
        "sepsis_onset_timestamp": [onset],
    })
    res = lead_time_analysis(meta, probs, outcomes, alert_threshold=0.65)
    assert res["n_septic_encounters"] == 1
    assert res["detected_encounters"] == 1
    # first alert at onset-6h => lead 6h
    assert res["lead_time_mean_hours"] == pytest.approx(6.0)
    assert res["n_non_septic_encounters"] == 1
    assert res["n_false_alerts"] == 0
