"""Canonical SYNK data schemas.

All dataset adapters (synthetic generator, CSV adapter, MIMIC-IV extract
adapter) normalise their output to these schemas so the rest of the pipeline
is dataset-agnostic.  Every schema is intentionally *partial*: adapters may
produce any subset of the clinical columns and missing columns are treated as
unavailable rather than assumed zero.
"""

from __future__ import annotations

import pandas as pd

OBSERVATION_ID_COLUMNS = ["patient_id", "encounter_id", "timestamp"]

# Physiological time-series columns (wide format, one row per timestamp).
OBSERVATION_VALUE_COLUMNS: dict[str, str] = {
    "heart_rate": "bpm",
    "respiratory_rate": "breaths/min",
    "systolic_bp": "mmHg",
    "diastolic_bp": "mmHg",
    "mean_arterial_pressure": "mmHg",
    "temperature": "degC",
    "spo2": "%",
    "oxygen_flow": "L/min",
    "wbc": "10^9/L",
    "platelets": "10^9/L",
    "creatinine": "mg/dL",
    "bilirubin": "mg/dL",
    "lactate": "mmol/L",
    "glucose": "mg/dL",
    "hemoglobin": "g/dL",
    "antibiotics_given": "flag",
    "culture_drawn": "flag",
}

VITAL_VARIABLES = [
    "heart_rate", "respiratory_rate", "systolic_bp", "diastolic_bp",
    "mean_arterial_pressure", "temperature", "spo2", "oxygen_flow",
]

LAB_VARIABLES = [
    "wbc", "platelets", "creatinine", "bilirubin", "lactate", "glucose", "hemoglobin",
]

EVENT_VARIABLES = ["antibiotics_given", "culture_drawn"]

STATIC_VARIABLES = ["age", "sex", "weight"]

# Columns added by preprocessing (never produced by adapters).
DERIVED_VARIABLES = ["heart_rate_variability", "spo2_desaturation_events"]

NOTE_COLUMNS = [
    "patient_id", "encounter_id", "timestamp", "note_type", "author_type", "note_text",
]

OUTCOME_COLUMNS = [
    "patient_id", "encounter_id", "admission_timestamp", "discharge_timestamp",
    "sepsis_onset_timestamp", "scenario",
]

PATIENT_COLUMNS = ["patient_id", "age", "sex", "weight", "is_synthetic"]

ALL_OBSERVATION_COLUMNS = OBSERVATION_ID_COLUMNS + list(OBSERVATION_VALUE_COLUMNS)

# ---------------------------------------------------------------------------
# Reference ranges.
#
# These are *configurable defaults* representing commonly cited adult ICU
# chart reference values.  They are provided for visualisation and for the
# evidence phrasing only; they are NOT clinically validated decision
# thresholds and must never be treated as such.
# ---------------------------------------------------------------------------
REFERENCE_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate": (60.0, 100.0),
    "respiratory_rate": (12.0, 20.0),
    "systolic_bp": (90.0, 140.0),
    "diastolic_bp": (50.0, 90.0),
    "mean_arterial_pressure": (65.0, 100.0),
    "temperature": (36.1, 37.8),
    "spo2": (92.0, 100.0),
    "oxygen_flow": (0.0, 6.0),
    "wbc": (4.0, 11.0),
    "platelets": (150.0, 400.0),
    "creatinine": (0.6, 1.2),
    "bilirubin": (0.2, 1.2),
    "lactate": (0.5, 2.0),
    "glucose": (70.0, 140.0),
    "hemoglobin": (10.0, 16.0),
}

REFERENCE_RANGES_NOTE = (
    "Reference ranges are configurable defaults reflecting commonly cited adult "
    "ICU chart values; they are used for display/evidence phrasing and are not "
    "clinically validated alert thresholds."
)

# Human readable labels used by evidence generation and the UI.
VARIABLE_LABELS: dict[str, str] = {
    "heart_rate": "Heart rate",
    "respiratory_rate": "Respiratory rate",
    "systolic_bp": "Systolic blood pressure",
    "diastolic_bp": "Diastolic blood pressure",
    "mean_arterial_pressure": "Mean arterial pressure",
    "temperature": "Temperature",
    "spo2": "Oxygen saturation",
    "oxygen_flow": "Oxygen flow",
    "wbc": "White blood cell count",
    "platelets": "Platelet count",
    "creatinine": "Creatinine",
    "bilirubin": "Bilirubin",
    "lactate": "Lactate",
    "glucose": "Glucose",
    "hemoglobin": "Hemoglobin",
    "age": "Age",
    "sex": "Sex",
    "weight": "Weight",
}

VARIABLE_UNITS: dict[str, str] = dict(OBSERVATION_VALUE_COLUMNS)


def empty_observations_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in ALL_OBSERVATION_COLUMNS}).astype(
        {c: "object" for c in OBSERVATION_ID_COLUMNS}
    )


def coerce_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw observations frame to the canonical schema.

    - Ensures id columns exist as strings and ``timestamp`` is datetime.
    - Ensures every canonical clinical column exists (NaN when absent).
    - Leaves any additional columns untouched (extensibility).
    """
    out = df.copy()
    for col in OBSERVATION_ID_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"observations frame missing required column '{col}'")
        if col != "timestamp":
            out[col] = out[col].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("observations contain unparseable timestamps")
    for col in OBSERVATION_VALUE_COLUMNS:
        if col not in out.columns:
            out[col] = float("nan")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def coerce_notes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw clinical notes frame to the canonical schema."""
    out = df.copy()
    for col in ["patient_id", "encounter_id", "timestamp", "note_text"]:
        if col not in out.columns:
            raise ValueError(f"notes frame missing required column '{col}'")
    out["patient_id"] = out["patient_id"].astype(str)
    out["encounter_id"] = out["encounter_id"].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("notes contain unparseable timestamps")
    for col in ("note_type", "author_type"):
        if col not in out.columns:
            out[col] = "unknown"
    out["note_text"] = out["note_text"].astype(str)
    return out


def coerce_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw outcomes frame to the canonical schema."""
    out = df.copy()
    for col in ["patient_id", "encounter_id"]:
        if col not in out.columns:
            raise ValueError(f"outcomes frame missing required column '{col}'")
        out[col] = out[col].astype(str)
    for col in ("admission_timestamp", "discharge_timestamp", "sepsis_onset_timestamp"):
        if col not in out.columns:
            out[col] = pd.NaT
        out[col] = pd.to_datetime(out[col], errors="coerce")
    if "scenario" not in out.columns:
        out["scenario"] = ""
    return out


def coerce_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw patients frame to the canonical schema."""
    out = df.copy()
    if "patient_id" not in out.columns:
        raise ValueError("patients frame missing required column 'patient_id'")
    out["patient_id"] = out["patient_id"].astype(str)
    for col in ("age", "weight"):
        if col not in out.columns:
            out[col] = float("nan")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "sex" not in out.columns:
        out["sex"] = float("nan")
    out["sex"] = pd.to_numeric(out["sex"], errors="coerce")
    if "is_synthetic" not in out.columns:
        out["is_synthetic"] = True
    return out
