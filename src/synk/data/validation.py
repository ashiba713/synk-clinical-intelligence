"""Rule-based data validation for SYNK.

Produces a :class:`ValidationReport` describing schema, integrity, and range
checks.  Fatal problems raise :class:`DataValidationError`; warnings are
collected and surfaced to callers/logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from synk.data.schemas import (
    LAB_VARIABLES,
    OBSERVATION_ID_COLUMNS,
    VITAL_VARIABLES,
)

# Physiologically plausible bounds used only to catch *impossible* values
# (unit errors, corrupted rows) - they are deliberately generous.
PLAUSIBLE_BOUNDS: dict[str, tuple[float, float]] = {
    "heart_rate": (20, 260),
    "respiratory_rate": (3, 60),
    "systolic_bp": (40, 260),
    "diastolic_bp": (20, 160),
    "mean_arterial_pressure": (25, 180),
    "temperature": (30, 43),
    "spo2": (50, 100),
    "oxygen_flow": (0, 40),
    "wbc": (0.05, 100),
    "platelets": (1, 1500),
    "creatinine": (0.1, 25),
    "bilirubin": (0.02, 40),
    "lactate": (0.1, 30),
    "glucose": (15, 1200),
    "hemoglobin": (3, 25),
    "age": (0, 120),
    "weight": (2, 350),
}


class DataValidationError(RuntimeError):
    """Raised when data is too broken to process."""


@dataclass
class ValidationReport:
    dataset: str
    n_rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        status = "OK" if self.ok else "FAILED"
        return (f"[{self.dataset}] {status}: {self.n_rows} rows, "
                f"{len(self.errors)} errors, {len(self.warnings)} warnings")


def _check(fn, report: ValidationReport, name: str) -> None:
    report.checks_run.append(name)
    result = fn()
    if isinstance(result, tuple):
        err, warn = result
        if err:
            report.errors.append(err)
        if warn:
            report.warnings.append(warn)
    elif result:
        report.errors.append(result)


def validate_observations(
    df: pd.DataFrame,
    dataset: str = "observations",
    raise_on_error: bool = True,
) -> ValidationReport:
    """Validate the canonical observations frame."""
    report = ValidationReport(dataset=dataset, n_rows=int(len(df)))

    def _schema():
        missing = [c for c in OBSERVATION_ID_COLUMNS if c not in df.columns]
        if missing:
            return f"missing required columns: {missing}", None
        clinical = [c for c in VITAL_VARIABLES + LAB_VARIABLES if c in df.columns]
        if not clinical:
            return "no clinical value columns present", None
        return None, None

    def _ids():
        nulls = int(df[OBSERVATION_ID_COLUMNS].isna().sum().sum())
        return (f"{nulls} null identifier cells", None) if nulls else (None, None)

    def _timestamps():
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        n_bad = int(ts.isna().sum())
        return (f"{n_bad} unparseable timestamps", None) if n_bad else (None, None)

    def _future_timestamps():
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        n_future = int((ts > pd.Timestamp.now(tz=None) + pd.Timedelta(days=1)).sum())
        return (None, f"{n_future} timestamps in the future (synthetic data is expected)") if n_future else (None, None)

    def _duplicates():
        dup = df.duplicated(subset=["encounter_id", "timestamp"], keep=False)
        n_dup = int(dup.sum())
        return (None, f"{n_dup} duplicate encounter/timestamp rows (will be averaged)") if n_dup else (None, None)

    def _ranges():
        issues = []
        for var, (lo, hi) in PLAUSIBLE_BOUNDS.items():
            if var not in df.columns:
                continue
            vals = pd.to_numeric(df[var], errors="coerce")
            n_out = int(((vals < lo) | (vals > hi)).sum())
            if n_out:
                issues.append(f"{var}: {n_out} implausible values outside [{lo}, {hi}]")
        return (("; ".join(issues)) if issues else None,
                None)

    def _monotonic():
        bad = 0
        for _, group in df.groupby("encounter_id"):
            ts = pd.to_datetime(group["timestamp"])
            bad += int((ts.diff().dropna() < pd.Timedelta(0)).sum())
        return (f"{bad} out-of-order timestamps", None) if bad else (None, None)

    def _empty():
        return ("frame is empty", None) if len(df) == 0 else (None, None)

    _check(_empty, report, "non_empty")
    _check(_schema, report, "schema_columns")
    _check(_ids, report, "identifier_completeness")
    _check(_timestamps, report, "timestamp_parse")
    _check(_future_timestamps, report, "timestamp_sanity")
    _check(_duplicates, report, "duplicate_rows")
    _check(_monotonic, report, "temporal_ordering")
    _check(_ranges, report, "physiological_plausibility")

    if raise_on_error and report.errors:
        raise DataValidationError(f"{report.summary()} -> {'; '.join(report.errors[:5])}")
    return report


def validate_notes(
    df: pd.DataFrame,
    dataset: str = "notes",
    raise_on_error: bool = True,
) -> ValidationReport:
    """Validate the canonical clinical notes frame."""
    report = ValidationReport(dataset=dataset, n_rows=int(len(df)))

    def _schema():
        missing = [c for c in ["patient_id", "encounter_id", "timestamp", "note_text"] if c not in df.columns]
        return (f"missing required columns: {missing}", None) if missing else (None, None)

    def _empty_text():
        if len(df) == 0:
            return (None, "notes frame is empty (text modality will be absent)")
        n_blank = int((df["note_text"].astype(str).str.strip() == "").sum())
        return (f"{n_blank} blank note texts", None) if n_blank else (None, None)

    def _dup():
        dup = df.duplicated(subset=["encounter_id", "timestamp", "note_text"])
        return (None, f"{int(dup.sum())} duplicate notes (will be dropped)") if dup.any() else (None, None)

    _check(_schema, report, "schema_columns")
    _check(_empty_text, report, "text_presence")
    _check(_dup, report, "duplicate_notes")

    if raise_on_error and report.errors:
        raise DataValidationError(f"{report.summary()} -> {'; '.join(report.errors[:5])}")
    return report


def validate_outcomes(
    df: pd.DataFrame,
    observations: Optional[pd.DataFrame] = None,
    dataset: str = "outcomes",
    raise_on_error: bool = True,
) -> ValidationReport:
    """Validate the outcomes frame (encounter-level sepsis onset labels)."""
    report = ValidationReport(dataset=dataset, n_rows=int(len(df)))

    def _schema():
        missing = [c for c in ["patient_id", "encounter_id"] if c not in df.columns]
        return (f"missing required columns: {missing}", None) if missing else (None, None)

    def _unique():
        dup = int(df["encounter_id"].duplicated().sum())
        return (f"{dup} duplicated encounter ids", None) if dup else (None, None)

    def _onset_in_stay():
        if "discharge_timestamp" not in df.columns or len(df) == 0:
            return (None, None)
        onset = pd.to_datetime(df["sepsis_onset_timestamp"], errors="coerce")
        discharge = pd.to_datetime(df["discharge_timestamp"], errors="coerce")
        n_bad = int(((onset > discharge) & onset.notna()).sum())
        return (f"{n_bad} sepsis onsets after discharge", None) if n_bad else (None, None)

    def _obs_coverage():
        if observations is None or len(observations) == 0:
            return (None, "no observations provided for cross-check")
        obs_ids = set(observations["encounter_id"].astype(str))
        missing = [e for e in df["encounter_id"].astype(str) if e not in obs_ids]
        return (f"{len(missing)} encounters without observations", None) if missing else (None, None)

    _check(_schema, report, "schema_columns")
    _check(_unique, report, "encounter_uniqueness")
    _check(_onset_in_stay, report, "onset_within_stay")
    _check(_obs_coverage, report, "observation_coverage")

    if raise_on_error and report.errors:
        raise DataValidationError(f"{report.summary()} -> {'; '.join(report.errors[:5])}")
    return report


def missingness_summary(df: pd.DataFrame, variables: Optional[list[str]] = None) -> pd.DataFrame:
    """Per-variable missingness fraction, used by the UI and reports."""
    variables = variables or [c for c in df.columns if c not in OBSERVATION_ID_COLUMNS]
    rows = []
    for var in variables:
        if var in df.columns:
            vals = pd.to_numeric(df[var], errors="coerce")
            rows.append({"variable": var, "missing_fraction": float(vals.isna().mean()), "n": int(len(vals))})
    return pd.DataFrame(rows).sort_values("missing_fraction", ascending=False)


def outlier_mask(values: pd.Series, lower_q: float = 0.001, upper_q: float = 0.999) -> np.ndarray:
    """Boolean mask of values outside training-quantile winsor bounds."""
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.all(np.isnan(vals)):
        return np.zeros(len(vals), dtype=bool)
    lo, hi = np.nanquantile(vals, [lower_q, upper_q])
    return ~np.isnan(vals) & ((vals < lo) | (vals > hi))
