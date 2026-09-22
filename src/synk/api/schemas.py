"""Pydantic v1/v2-compatible API schemas.

Request/response models for the SYNK REST API.  Every prediction response
carries the research disclaimer and uncertainty information - the API never
presents a probability as certainty.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    engine: str
    model_version: str
    dataset_id: str
    is_synthetic_data: bool
    version: str


class ModelInfoResponse(BaseModel):
    engine: str
    model_name: str
    model_version: str
    modality: Optional[str] = None
    created_at: Optional[str] = None
    dataset_id: Optional[str] = None
    git_commit: Optional[str] = None
    weights_checksum: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    calibration: Optional[str] = None
    note: Optional[str] = None
    research_disclaimer: str


class PatientSummary(BaseModel):
    patient_id: str
    encounter_id: str
    age: Optional[float] = None
    latest_risk_probability: Optional[float] = None
    latest_risk_category: Optional[str] = None
    risk_change: Optional[float] = None
    prediction_horizon_hours: Optional[float] = None
    last_prediction_time: Optional[str] = None
    is_synthetic: bool = True


class PatientsResponse(BaseModel):
    n_patients: int
    patients: list[PatientSummary]


class Prediction(BaseModel):
    patient_id: str
    encounter_id: str
    prediction_time: str
    risk_probability: float
    risk_category: str
    label: Optional[int] = None
    prediction_horizon_hours: float
    model_version: str
    engine: str


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    n_predictions: int
    engine: str
    research_disclaimer: str


class ExplainResponse(BaseModel):
    encounter_id: str
    prediction_time: str
    risk_probability: float
    risk_category: str
    explanation: dict[str, Any]
    research_disclaimer: str


class ReportResponse(BaseModel):
    report_id: str
    formats: dict[str, str]
    report: dict[str, Any]


class ExperimentsResponse(BaseModel):
    n_experiments: int
    experiments: list[dict[str, Any]]


class MetricsResponse(BaseModel):
    source: str
    metrics: dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


RESEARCH_DISCLAIMER = (
    "Research prototype - not for clinical diagnosis or treatment. "
    "Outputs are model-generated evidence for research interpretation."
)
