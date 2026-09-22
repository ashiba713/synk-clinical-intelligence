"""Explainability tests: real attributions, negation handling, uncertainty."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from synk.explainability.text_attribution import extract_text_evidence


def test_evidence_negation_handling():
    positive = extract_text_evidence("Patient has fever and concern for infection.")
    cats = set()
    for s in positive["spans"]:
        if not s["negated"]:
            cats.add(s["category"])
    assert "infection" in cats or "deterioration" in cats

    negated = extract_text_evidence("No evidence of infection. Denies fever or chills.")
    neg_cats = set()
    for s in negated["spans"]:
        if not s["negated"]:
            neg_cats.add(s["category"])
    assert "infection" not in neg_cats
    # and the negated span is marked
    assert any(s["negated"] and s["category"] == "infection" for s in negated["spans"])


def test_structured_attribution_runs(trained_multimodal):
    from synk.explainability.structured_attribution import attribute_structured

    cfg, sample_set, splits, result = trained_multimodal
    from synk.models.multimodal import load_trained_model

    model, _payload = load_trained_model(result.checkpoint_path, cfg)
    model.eval()

    x = torch.from_numpy(sample_set.x[:1])
    m = torch.from_numpy(sample_set.mask[:1])
    s = torch.from_numpy(sample_set.static[:1])
    texts = [str(sample_set.samples.iloc[0]["text"])]
    out = attribute_structured(model, x, m, s, texts, sample_set.variables, n_steps=4)
    assert out["method"] in {"integrated_gradients", "gradient_x_input", "gradient_shap"}
    assert len(out["variables"]) == len(sample_set.variables)
    scores = np.array([v["score"] for v in out["variables"]])
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()  # |attr| aggregation
    assert len(out["attribution_matrix"]) == sample_set.x.shape[1]


def test_explain_prediction_payload(trained_multimodal):
    from synk.features.dataset import collate_fn
    from synk.explainability.explainer import explain_prediction

    cfg, sample_set, splits, result = trained_multimodal
    from synk.models.multimodal import load_trained_model

    model, _payload = load_trained_model(result.checkpoint_path, cfg)
    model.eval()

    row = {
        "x": torch.from_numpy(sample_set.x[1]),
        "mask": torch.from_numpy(sample_set.mask[1]),
        "static": torch.from_numpy(sample_set.static[1]),
        "text": str(sample_set.samples.iloc[1]["text"]),
    }
    batch = collate_fn([row])
    expl = explain_prediction(model, batch, sample_set.variables, cfg, attribution_method="gradient_x_input")
    assert "risk_probability" in expl
    assert "Research prototype" in expl["research_disclaimer"]
    assert "variables" in expl["structured_attribution"]
    assert isinstance(expl["trend_summary"], list)
    assert "uncertainty" in expl
    # uncertainty is genuinely estimated (mc dropout) or explicitly unavailable
    unc = expl["uncertainty"]
    assert unc.get("available") in (True, False)


def test_demo_engine_explanation_is_real(config, frames):
    """The demo fallback must produce real linear-model contributions, not placeholders."""
    from synk.inference.engine import DemoEngine, InferenceService

    patients, observations, notes, outcomes = frames
    service = InferenceService(config)
    # Force the demo path even when a trained bundle exists on disk.
    service.bundle = None
    engine_id = service.ensure_ready(observations, notes, outcomes, patients)
    assert engine_id.startswith("sklearn_demo")
    assert isinstance(service.demo_engine, DemoEngine)
    ss, _ = service.featurizer.build(observations, notes, outcomes, patients)
    expl = service.explain_sample(ss, 0)
    assert expl["engine"] == "sklearn_demo"
    variables = expl["structured_attribution"]["variables"]
    assert len(variables) > 0
    assert all(np.isfinite(v["score"]) for v in variables)
