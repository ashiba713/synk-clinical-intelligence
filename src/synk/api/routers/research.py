"""Research API routers: experiments, metrics, reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from synk.api.schemas import (
    ExperimentsResponse,
    MetricsResponse,
    ReportResponse,
)
from synk.api.state import get_state
from synk.evaluation.metrics import risk_category
from synk.reporting.report import build_patient_report, save_report
from synk.utils.common import safe_float

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/v1/experiments", response_model=ExperimentsResponse, tags=["research"])
def experiments(limit: int = Query(default=100, ge=1, le=500)):
    """The unimodal-vs-multimodal comparison table (append-only registry)."""
    state = get_state()
    path = Path(state.config.paths.evaluation_dir) / "experiments.json"
    if not path.exists():
        return ExperimentsResponse(n_experiments=0, experiments=[])
    entries = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentsResponse(n_experiments=len(entries), experiments=entries[-limit:])


@router.get("/api/v1/metrics", response_model=MetricsResponse, tags=["research"])
def metrics(source: Optional[str] = Query(default=None, description="Path or experiment id; default = latest multimodal")):
    """Return the most recent evaluation report metrics (real, stored numbers)."""
    state = get_state()
    eval_dir = Path(state.config.paths.evaluation_dir)
    candidates = sorted(eval_dir.glob("*_test.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if source is not None:
        exact = eval_dir / f"{source}.json"
        if exact.exists():
            candidates = [exact]
    if not candidates:
        raise HTTPException(status_code=404, detail="No evaluation reports found; run the training pipeline first")
    report = json.loads(candidates[0].read_text(encoding="utf-8"))
    payload = {
        "report_file": candidates[0].name,
        "auroc": report.get("auroc"),
        "auprc": report.get("auprc"),
        "threshold_metrics": report.get("threshold_metrics"),
        "brier_score": report.get("brier_score"),
        "ece": report.get("ece"),
        "early_warning": report.get("early_warning"),
        "ablation": report.get("ablation"),
    }
    return MetricsResponse(source=candidates[0].name, metrics=payload)


@router.post("/api/v1/reports", response_model=ReportResponse, tags=["reports"])
def create_report(encounter_id: str, formats: Optional[str] = Query(default="json,md")):
    """Generate a research patient report (markdown/json/csv)."""
    state = get_state()
    service = state.ensure_engine()
    frame = service.predict_encounters(state.observations, state.notes, state.outcomes, state.patients,
                                       encounter_ids=[encounter_id]).sort_values("t")
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"No predictions for encounter '{encounter_id}'")
    latest = frame.tail(1).iloc[0]

    sample_set, _ = service.featurizer.build(state.observations, state.notes, state.outcomes, state.patients)
    sub = sample_set.samples[sample_set.samples["encounter_id"] == encounter_id]
    if sub.empty:
        raise HTTPException(status_code=422, detail="No sample row for this encounter")
    explanation = service.explain_sample(sample_set, int(sub.index[-1]))

    report = build_patient_report(
        patient_id=str(latest["patient_id"]),
        encounter_id=encounter_id,
        prediction_row=latest,
        explanation=explanation,
        config=state.config,
        observations=state.observations,
        notes=state.notes,
    )
    fmts = tuple(f.strip() for f in (formats or "json,md").split(",") if f.strip())
    out_dir = Path(state.config.paths.report_dir)
    saved = save_report(report, out_dir, formats=fmts)
    return ReportResponse(report_id=report["report_id"], formats=saved, report=report)


@router.get("/api/v1/reports/{report_id}", tags=["reports"])
def get_report(report_id: str, format: str = Query(default="json", pattern="^(json|md|csv)$")):
    state = get_state()
    report_dir = Path(state.config.paths.report_dir)
    path = report_dir / f"{report_id}.{format}"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' ({format}) not found")
    if format == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/plain")
