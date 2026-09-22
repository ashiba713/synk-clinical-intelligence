"""FastAPI integration tests.

Exercises the real inference path (demo engine on tiny synthetic data) through
the actual HTTP handlers.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(config, frames):
    from synk.api.main import create_app
    from synk.api.state import AppState

    app = create_app(config)
    # Pre-seed state without the startup event's heavy default load.
    with TestClient(app) as c:  # runs startup (loads demo data + engine)
        yield c


def test_root_and_health(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["disclaimer"].startswith("Research prototype")

    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["is_synthetic_data"] is True


def test_model_info(client):
    r = client.get("/api/v1/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["engine"].startswith("sklearn_demo") or body["engine"].startswith("torch")
    assert "Research prototype" in body["research_disclaimer"]


def test_patients_list_and_filters(client):
    r = client.get("/api/v1/patients")
    assert r.status_code == 200
    body = r.json()
    assert body["n_patients"] >= 1
    first = body["patients"][0]
    assert 0.0 <= first["latest_risk_probability"] <= 1.0
    assert first["latest_risk_category"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}

    r2 = client.get("/api/v1/patients", params={"risk_category": "CRITICAL", "limit": 5})
    assert r2.status_code == 200
    for p in r2.json()["patients"]:
        assert p["latest_risk_category"] == "CRITICAL"


def test_patient_detail_roundtrip(client):
    r = client.get("/api/v1/patients")
    enc = r.json()["patients"][0]["encounter_id"]
    d = client.get(f"/api/v1/patients/{enc}")
    assert d.status_code == 200
    body = d.json()
    assert body["encounter_id"] == enc
    assert len(body["predictions"]) >= 1
    assert "observations" in body and "notes" in body
    assert body["is_synthetic"] is True

    missing = client.get("/api/v1/patients/DOES-NOT-EXIST")
    assert missing.status_code == 404


def test_predict_and_batch(client):
    r = client.get("/api/v1/patients")
    enc = r.json()["patients"][0]["encounter_id"]
    p = client.post("/api/v1/predict", params={"encounter_id": enc})
    assert p.status_code == 200
    body = p.json()
    assert body["n_predictions"] >= 1
    pred = body["predictions"][-1]
    assert 0.0 <= pred["risk_probability"] <= 1.0
    assert pred["risk_category"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
    assert "Research prototype" in body["research_disclaimer"]

    b = client.post("/api/v1/predict/batch", json=[enc])
    assert b.status_code == 200
    assert b.json()["n_predictions"] >= 1

    bad = client.post("/api/v1/predict/batch", json=["NOPE-1"])
    assert bad.status_code == 404


def test_explain_endpoint(client):
    r = client.get("/api/v1/patients")
    enc = r.json()["patients"][0]["encounter_id"]
    e = client.post("/api/v1/explain", params={"encounter_id": enc})
    assert e.status_code == 200
    body = e.json()
    expl = body["explanation"]
    assert "structured_attribution" in expl
    assert "Research prototype" in body["research_disclaimer"]


def test_experiments_empty_ok(client):
    r = client.get("/api/v1/experiments")
    assert r.status_code == 200
    assert "n_experiments" in r.json()


def test_metrics_404_before_training(client):
    # No eval reports exist in the test env (they go to a tmp-free path only if produced).
    r = client.get("/api/v1/metrics")
    assert r.status_code in (200, 404)
