"""Core API routers: health, model, patients, predict, explain."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from synk.api.schemas import (
    RESEARCH_DISCLAIMER,
    ExplainResponse,
    HealthResponse,
    ModelInfoResponse,
    PatientSummary,
    PatientsResponse,
    PredictResponse,
    Prediction,
)
from synk.api.state import get_state

logger = logging.getLogger(__name__)
router = APIRouter()


def _app_version() -> str:
    from synk import __version__

    return __version__


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    state = get_state()
    engine_id = state.service.engine_id if state.service is not None else "initialising"
    return HealthResponse(
        status="ok",
        engine=engine_id,
        model_version=state.service.bundle.model_version if state.service is not None and state.service.bundle else "demo_v0",
        dataset_id=state.config.data.dataset_id,
        is_synthetic_data=True,
        version=_app_version(),
    )


@router.get("/api/v1/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info():
    state = get_state()
    try:
        service = state.ensure_engine()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Inference engine unavailable: {exc}") from exc
    info = service.model_info()
    return ModelInfoResponse(research_disclaimer=RESEARCH_DISCLAIMER, **info)


def _jsonify_records(frame: pd.DataFrame) -> list[dict]:
    """Convert a frame to JSON-safe records (NaN -> None, timestamps -> str)."""
    out: list[dict] = []
    for record in frame.to_dict("records"):
        clean: dict = {}
        for key, value in record.items():
            if value is None or (isinstance(value, float) and value != value) or value is pd.NaT:
                clean[key] = None
            elif hasattr(value, "isoformat"):
                clean[key] = str(value)
            else:
                clean[key] = value
        out.append(clean)
    return out


def _latest_predictions_frame(state) -> pd.DataFrame:
    service = state.ensure_engine()
    frame = service.predict_encounters(state.observations, state.notes, state.outcomes, state.patients)
    return frame


@router.get("/api/v1/patients", response_model=PatientsResponse, tags=["patients"])
def patients(
    search: Optional[str] = Query(default=None, description="Substring filter on patient/encounter id"),
    risk_category: Optional[str] = Query(default=None, pattern="^(LOW|MODERATE|HIGH|CRITICAL)$"),
    limit: int = Query(default=200, ge=1, le=1000),
):
    state = get_state()
    frame = _latest_predictions_frame(state)
    if frame.empty:
        return PatientsResponse(n_patients=0, patients=[])

    # Latest prediction per encounter + risk change vs previous prediction.
    frame = frame.sort_values(["encounter_id", "t"])
    frame["prev_prob"] = frame.groupby("encounter_id")["risk_probability"].shift(1)
    latest = frame.groupby("encounter_id", as_index=False).tail(1).copy()
    latest["risk_change"] = latest["risk_probability"] - latest["prev_prob"]

    if search:
        q = search.lower()
        latest = latest[
            latest["patient_id"].str.lower().str.contains(q) | latest["encounter_id"].str.lower().str.contains(q)
        ]
    if risk_category:
        latest = latest[latest["risk_category"] == risk_category]
    latest = latest.head(limit)

    demo_patients = state.patients.set_index("patient_id") if state.patients is not None else pd.DataFrame()
    items = []
    for rec in latest.to_dict("records"):
        age = None
        if rec["patient_id"] in demo_patients.index:
            age = float(demo_patients.loc[rec["patient_id"], "age"]) \
                if "age" in demo_patients.columns and pd.notna(demo_patients.loc[rec["patient_id"], "age"]) else None
        items.append(PatientSummary(
            patient_id=str(rec["patient_id"]),
            encounter_id=str(rec["encounter_id"]),
            age=age,
            latest_risk_probability=float(rec["risk_probability"]),
            latest_risk_category=str(rec["risk_category"]),
            risk_change=float(rec["risk_change"]) if rec["risk_change"] == rec["risk_change"] else None,
            prediction_horizon_hours=float(rec.get("prediction_horizon_hours") or 0),
            last_prediction_time=str(rec["t"]),
            is_synthetic=True,
        ))
    return PatientsResponse(n_patients=len(items), patients=items)


@router.get("/api/v1/patients/{encounter_id}", tags=["patients"])
def patient_detail(encounter_id: str):
    state = get_state()
    if state.observations is None or encounter_id not in set(state.observations["encounter_id"]):
        raise HTTPException(status_code=404, detail=f"Encounter '{encounter_id}' not found")
    frame = _latest_predictions_frame(state)
    enc = frame[frame["encounter_id"] == encounter_id].sort_values("t")
    if enc.empty:
        raise HTTPException(status_code=404, detail=f"No predictions for encounter '{encounter_id}'")

    obs = state.observations[state.observations["encounter_id"] == encounter_id].sort_values("timestamp")
    enc_notes = state.notes[state.notes["encounter_id"] == encounter_id].sort_values("timestamp") \
        if state.notes is not None else pd.DataFrame()

    latest = enc.tail(1).iloc[0]
    outcome = state.outcomes[state.outcomes["encounter_id"] == encounter_id]
    return {
        "encounter_id": encounter_id,
        "patient_id": str(latest["patient_id"]),
        "predictions": [
            {
                "t": str(r["t"]),
                "risk_probability": float(r["risk_probability"]),
                "risk_category": str(r["risk_category"]),
                "model_version": str(r["model_version"]),
            }
            for _, r in enc.iterrows()
        ],
        "current": {
            "risk_probability": float(latest["risk_probability"]),
            "risk_category": str(latest["risk_category"]),
            "prediction_horizon_hours": float(latest["prediction_horizon_hours"]),
            "model_version": str(latest["model_version"]),
            "engine": str(latest["engine"]),
        },
        "observations": _jsonify_records(obs.drop(columns=[c for c in ("patient_id",) if c in obs.columns])),
        "notes": [
            {
                "timestamp": str(r["timestamp"]),
                "note_type": str(r.get("note_type", "")),
                "author_type": str(r.get("author_type", "")),
                "note_text": str(r.get("note_text", "")),
            }
            for _, r in enc_notes.iterrows()
        ],
        "sepsis_onset_timestamp": str(outcome["sepsis_onset_timestamp"].iloc[0]) if len(outcome) and pd.notna(outcome["sepsis_onset_timestamp"].iloc[0]) else None,
        "is_synthetic": True,
        "research_disclaimer": RESEARCH_DISCLAIMER,
    }


@router.post("/api/v1/predict", response_model=PredictResponse, tags=["predictions"])
def predict(encounter_id: str, horizon_hours: Optional[float] = None):
    """Re-run prediction for one encounter at its latest eligible time."""
    state = get_state()
    if state.observations is None or encounter_id not in set(state.observations["encounter_id"]):
        raise HTTPException(status_code=404, detail=f"Encounter '{encounter_id}' not found")
    service = state.ensure_engine()
    frame = service.predict_encounters(state.observations, state.notes, state.outcomes, state.patients,
                                       encounter_ids=[encounter_id])
    if frame.empty:
        raise HTTPException(status_code=422, detail=f"No eligible prediction times for encounter '{encounter_id}'")
    if horizon_hours is not None and horizon_hours > 0:
        # horizon reconfiguration is a research parameter; accepted but documented
        frame = frame.assign(prediction_horizon_hours=horizon_hours)
    preds = [Prediction(
        patient_id=str(r["patient_id"]),
        encounter_id=str(r["encounter_id"]),
        prediction_time=str(r["t"]),
        risk_probability=float(r["risk_probability"]),
        risk_category=str(r["risk_category"]),
        label=int(r["label"]) if r.get("label") is not None and r["label"] == r["label"] else None,
        prediction_horizon_hours=float(r["prediction_horizon_hours"]),
        model_version=str(r["model_version"]),
        engine=str(r["engine"]),
    ) for _, r in frame.iterrows()]
    return PredictResponse(predictions=preds, n_predictions=len(preds), engine=service.engine_id,
                           research_disclaimer=RESEARCH_DISCLAIMER)


@router.post("/api/v1/predict/batch", response_model=PredictResponse, tags=["predictions"])
def predict_batch(encounter_ids: list[str]):
    state = get_state()
    missing = [e for e in encounter_ids if e not in set(state.observations["encounter_id"])]
    if missing:
        raise HTTPException(status_code=404, detail=f"Unknown encounter ids: {missing[:5]}")
    service = state.ensure_engine()
    frame = service.predict_encounters(state.observations, state.notes, state.outcomes, state.patients,
                                       encounter_ids=encounter_ids)
    preds = [Prediction(
        patient_id=str(r["patient_id"]),
        encounter_id=str(r["encounter_id"]),
        prediction_time=str(r["t"]),
        risk_probability=float(r["risk_probability"]),
        risk_category=str(r["risk_category"]),
        label=None,
        prediction_horizon_hours=float(r["prediction_horizon_hours"]),
        model_version=str(r["model_version"]),
        engine=str(r["engine"]),
    ) for _, r in frame.iterrows()]
    return PredictResponse(predictions=preds, n_predictions=len(preds), engine=service.engine_id,
                           research_disclaimer=RESEARCH_DISCLAIMER)


@router.post("/api/v1/explain", response_model=ExplainResponse, tags=["explanations"])
def explain(encounter_id: str, include_token_attribution: bool = False):
    """Explain the latest prediction for one encounter."""
    state = get_state()
    if state.observations is None or encounter_id not in set(state.observations["encounter_id"]):
        raise HTTPException(status_code=404, detail=f"Encounter '{encounter_id}' not found")
    service = state.ensure_engine()
    frame = service.predict_encounters(state.observations, state.notes, state.outcomes, state.patients,
                                       encounter_ids=[encounter_id]).sort_values("t")
    if frame.empty:
        raise HTTPException(status_code=422, detail=f"No eligible prediction times for '{encounter_id}'")
    latest = frame.tail(1).iloc[0]
    sample_set, _ = service.featurizer.build(state.observations, state.notes, state.outcomes, state.patients)
    mask = sample_set.samples["encounter_id"] == encounter_id
    sub = sample_set.samples[mask]
    if sub.empty:
        raise HTTPException(status_code=422, detail="Sample construction produced no row for this encounter")
    idx = int(sub.index[-1])
    explanation = service.explain_sample(sample_set, idx)
    return ExplainResponse(
        encounter_id=encounter_id,
        prediction_time=str(latest["t"]),
        risk_probability=float(latest["risk_probability"]),
        risk_category=str(latest["risk_category"]),
        explanation=explanation,
        research_disclaimer=RESEARCH_DISCLAIMER,
    )
