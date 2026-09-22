"""Reporting, database and CLI tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest


def test_report_build_and_render(config, frames, tmp_path):
    from synk.inference.engine import InferenceService
    from synk.reporting.report import build_patient_report, save_report

    patients, observations, notes, outcomes = frames
    service = InferenceService(config)
    service.ensure_ready(observations, notes, outcomes, patients)
    frame = service.predict_encounters(observations, notes, outcomes, patients)
    assert len(frame) > 0
    latest = frame.sort_values("t").tail(1).iloc[0]
    enc = str(latest["encounter_id"])

    ss, _ = service.featurizer.build(observations, notes, outcomes, patients)
    sub = ss.samples[ss.samples["encounter_id"] == enc]
    expl = service.explain_sample(ss, int(sub.index[-1]))

    report = build_patient_report(str(latest["patient_id"]), enc, latest, expl, config,
                                  observations=observations, notes=notes)
    assert report["patient_id"] == str(latest["patient_id"])
    assert "not for clinical diagnosis" in report["disclaimer"]
    assert report["limitations"], "limitations must be explicit"

    saved = save_report(report, tmp_path, formats=("json", "md", "csv"))
    md = open(saved["md"], encoding="utf-8").read()
    assert "Research Report" in md
    assert "Risk assessment" in md
    payload = json.loads(open(saved["json"], encoding="utf-8").read())
    assert payload["report_id"] == report["report_id"]
    import os
    assert os.path.exists(saved["csv"])


def test_database_roundtrip(config, frames, tmp_path):
    from synk.database.models import build_database

    patients, observations, notes, outcomes = frames
    db = build_database(f"sqlite:///{tmp_path / 'test.db'}")
    db.seed_from_frames(patients, outcomes, is_synthetic=True)

    from synk.inference.engine import InferenceService

    service = InferenceService(config)
    service.ensure_ready(observations, notes, outcomes, patients)
    frame = service.predict_encounters(observations, notes, outcomes, patients).head(20)
    frame = frame.assign(engine="sklearn_demo:v0")
    n = db.store_predictions(frame)
    assert n == 20
    stored = db.latest_predictions(limit=50)
    assert len(stored) == 20
    assert set(stored["risk_category"]) <= {"LOW", "MODERATE", "HIGH", "CRITICAL"}


def test_cli_generate_and_predict(cohort, tmp_path, capsys):
    from scripts.cli import main

    rc = main(["predict", "--encounter", str(cohort.outcomes["encounter_id"].iloc[0]), "--explain"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):])
    assert "prediction" in payload
    assert "Research prototype" in payload["research_disclaimer"]
    assert payload["prediction"]["risk_category"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}


def test_model_registry_roundtrip(config, prepared, tmp_path):
    from synk.tracking.registry import ExperimentRegistry, ModelRegistry

    registry = ModelRegistry(tmp_path / "models")
    experiments = ExperimentRegistry(tmp_path / "experiments.json")
    fake_ckpt = tmp_path / "m.pt"
    import torch

    torch.save({"state_dict": {}}, fake_ckpt)
    bundle = registry.save("synk_both", fake_ckpt, {"a": 1}, {"auroc": 0.5}, dataset_id="synthetic_v1")
    assert registry.latest("synk_both") == bundle
    manifest = registry.load_manifest(bundle)
    assert manifest["model_version"] == "v1"
    assert len(manifest["weights_checksum"]) == 64

    experiments.add({"experiment_id": "e1", "modality": "both", "auroc": 0.5})
    frame = experiments.as_frame()
    assert len(frame) == 1 and frame.iloc[0]["auroc"] == 0.5
