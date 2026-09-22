"""SQLAlchemy persistence layer for SYNK.

Default backend is a local SQLite file (demo mode); PostgreSQL is supported by
pointing ``DATABASE_URL`` at a Postgres DSN.  By design the schema stores
identifiers and *derived* analysis results - raw clinical note text is stored
only for the synthetic demo corpus and never for real data (documented in
docs/privacy.md).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from synk.utils.common import utcnow_iso

logger = logging.getLogger(__name__)

Base = declarative_base()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column(String(64), primary_key=True)
    age = Column(Float, nullable=True)
    sex = Column(Integer, nullable=True)
    weight = Column(Float, nullable=True)
    is_synthetic = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


class Encounter(Base):
    __tablename__ = "encounters"

    encounter_id = Column(String(64), primary_key=True)
    patient_id = Column(String(64), nullable=False, index=True)
    admission_timestamp = Column(DateTime(timezone=True), nullable=True)
    discharge_timestamp = Column(DateTime(timezone=True), nullable=True)
    sepsis_onset_timestamp = Column(DateTime(timezone=True), nullable=True)
    scenario = Column(String(64), nullable=True)
    is_synthetic = Column(Boolean, default=True, nullable=False)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(80), index=True, nullable=False)
    patient_id = Column(String(64), index=True, nullable=False)
    encounter_id = Column(String(64), index=True, nullable=False)
    prediction_time = Column(DateTime(timezone=True), nullable=False)
    risk_probability = Column(Float, nullable=False)
    risk_category = Column(String(16), nullable=False)
    label = Column(Integer, nullable=True)  # ground truth when known (research only)
    prediction_horizon_hours = Column(Float, nullable=False)
    model_version = Column(String(64), nullable=False)
    engine = Column(String(80), nullable=True)
    uncertainty_std = Column(Float, nullable=True)
    is_synthetic = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


class ExplanationRecord(Base):
    __tablename__ = "explanations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(80), index=True, nullable=False)
    payload_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(80), index=True, nullable=False)
    modality = Column(String(32), nullable=False)
    dataset_id = Column(String(64), nullable=True)
    model_kind = Column(String(64), nullable=True)  # synk_deep | sklearn_baseline
    fusion = Column(String(32), nullable=True)
    auroc = Column(Float, nullable=True)
    auprc = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    sensitivity = Column(Float, nullable=True)
    specificity = Column(Float, nullable=True)
    brier = Column(Float, nullable=True)
    detection_rate = Column(Float, nullable=True)
    lead_time_median_hours = Column(Float, nullable=True)
    run_id = Column(String(80), nullable=True)
    model_version = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(64), nullable=False)
    model_version = Column(String(64), nullable=False)
    bundle_dir = Column(String(255), nullable=False)
    metrics_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


class SystemEvent(Base):
    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(64), index=True, nullable=False)
    message = Column(Text, nullable=True)
    payload_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
class Database:
    def __init__(self, db_url: str, echo: bool = False) -> None:
        if db_url.startswith("sqlite:///"):
            # Ensure parent directory exists for file-backed sqlite.
            raw = db_url.replace("sqlite:///", "", 1)
            if raw and raw not in (":memory:",):
                Path(raw).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(db_url, echo=echo, future=True)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return self._session_factory()

    # -- convenience -------------------------------------------------------
    def record_event(self, event_type: str, message: str = "", payload: Optional[dict] = None) -> None:
        with self.session() as s:
            s.add(SystemEvent(
                event_type=event_type,
                message=message,
                payload_json=(pd.io.json.dumps(payload) if payload else None) if hasattr(pd.io, "json") else None,
            ))
            s.commit()

    def seed_from_frames(
        self,
        patients: pd.DataFrame,
        outcomes: pd.DataFrame,
        is_synthetic: bool = True,
    ) -> None:
        """Upsert patients + encounters from canonical frames (ids only)."""
        with self.session() as s:
            for rec in patients.to_dict("records"):
                if s.get(Patient, rec["patient_id"]) is None:
                    s.add(Patient(
                        patient_id=str(rec["patient_id"]),
                        age=_safe_float(rec.get("age")),
                        sex=_safe_int(rec.get("sex")),
                        weight=_safe_float(rec.get("weight")),
                        is_synthetic=is_synthetic,
                    ))
            for rec in outcomes.to_dict("records"):
                if s.get(Encounter, str(rec["encounter_id"])) is None:
                    s.add(Encounter(
                        encounter_id=str(rec["encounter_id"]),
                        patient_id=str(rec["patient_id"]),
                        admission_timestamp=_safe_dt(rec.get("admission_timestamp")),
                        discharge_timestamp=_safe_dt(rec.get("discharge_timestamp")),
                        sepsis_onset_timestamp=_safe_dt(rec.get("sepsis_onset_timestamp")),
                        scenario=str(rec.get("scenario") or ""),
                        is_synthetic=is_synthetic,
                    ))
            s.commit()
        logger.info("Seeded patients/encounters into database")

    def store_predictions(self, frame: pd.DataFrame) -> int:
        """Persist a predictions frame (output of InferenceService)."""
        import json as _json
        import uuid

        count = 0
        with self.session() as s:
            for rec in frame.to_dict("records"):
                pid = f"pred_{uuid.uuid4().hex[:12]}"
                s.add(Prediction(
                    prediction_id=pid,
                    patient_id=str(rec["patient_id"]),
                    encounter_id=str(rec["encounter_id"]),
                    prediction_time=pd.Timestamp(rec["t"]).to_pydatetime(),
                    risk_probability=float(rec["risk_probability"]),
                    risk_category=str(rec["risk_category"]),
                    label=_safe_int(rec.get("label")),
                    prediction_horizon_hours=float(rec.get("prediction_horizon_hours", 0) or 0),
                    model_version=str(rec.get("model_version", "unknown")),
                    engine=str(rec.get("engine", "")),
                    is_synthetic=True,
                ))
                count += 1
            s.commit()
        return count

    def latest_predictions(self, encounter_ids: Optional[list[str]] = None, limit: int = 500) -> pd.DataFrame:
        with self.session() as s:
            stmt = select(Prediction).order_by(Prediction.prediction_time.desc()).limit(limit)
            if encounter_ids:
                stmt = stmt.where(Prediction.encounter_id.in_(encounter_ids))
            rows = s.scalars(stmt).all()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame([{
                "prediction_id": r.prediction_id,
                "patient_id": r.patient_id,
                "encounter_id": r.encounter_id,
                "t": r.prediction_time,
                "risk_probability": r.risk_probability,
                "risk_category": r.risk_category,
                "model_version": r.model_version,
                "engine": r.engine,
                "label": r.label,
                "created_at": r.created_at,
            } for r in rows])


def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _safe_dt(v: Any) -> Optional[datetime]:
    try:
        ts = pd.Timestamp(v)
        return None if pd.isna(ts) else ts.to_pydatetime()
    except (TypeError, ValueError):
        return None


def build_database(db_url: str) -> Database:
    db = Database(db_url)
    db.create_all()
    return db
