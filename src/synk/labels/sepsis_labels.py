"""Label engineering.

SYNK treats sepsis labelling as a *configurable study-design decision*, not a
hard-coded definition.  Two label sources are supported:

``latent_outcomes``
    Uses a per-encounter sepsis onset timestamp supplied by the dataset
    (e.g. produced by the synthetic generator, or by the researcher's own
    Sepsis-3 adjudication script).  This is the default because it keeps the
    prediction task well defined.

``criteria``
    Derives an onset estimate directly from the data using configurable
    criteria: suspected infection (antibiotic + culture within configurable
    windows) combined with an acute organ-dysfunction proxy.  The dysfunction
    proxy is a simplified, configurable derangement count - it is NOT the
    formal SOFA score and this is documented wherever it appears.

For every encounter the engine enumerates candidate prediction times ``t`` on
an hourly grid and assigns:

* ``label = 1``  if the (estimated) onset falls in ``(t, t + horizon]``
* ``label = 0``  if no onset occurs within the horizon
* excluded       if ``t >= onset`` (post-onset, when configured) or ``t`` is
                 earlier than ``min_history_hours``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from synk.utils.logging import get_logger

logger = get_logger(__name__)

# Derangement rules for the simplified organ-dysfunction proxy.  Each entry is
# (variable, comparator, threshold).  Configurable thresholds live in
# LabelCriteriaConfig; these directions follow the Sepsis-3 spirit.
def _derangement_rules(cfg) -> list[tuple[str, str, float]]:
    return [
        ("mean_arterial_pressure", "<", 65.0),
        ("heart_rate", ">", 90.0),
        ("respiratory_rate", ">", 22.0),
        ("temperature", ">", 38.0),
        ("spo2", "<", 90.0),
        ("lactate", ">", 2.0),
        ("creatinine", ">", 1.2),
        ("platelets", "<", 150.0),
        ("bilirubin", ">", 1.2),
    ]


@dataclass
class LabelTable:
    """Per-(encounter, prediction-time) labels plus encounter-level summary."""

    samples: pd.DataFrame          # encounter_id, patient_id, t, timestamp, label
    encounters: pd.DataFrame       # encounter_id, has_sepsis, onset_time, n_samples, n_positive
    label_source: str
    horizon_hours: float

    @property
    def prevalence(self) -> float:
        return float(self.samples["label"].mean()) if len(self.samples) else float("nan")


def _infer_grid_start(observations: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Anchor each encounter's t=0 at its first observation timestamp."""
    starts = observations.groupby("encounter_id")["timestamp"].min().rename("t0")
    return outcomes.merge(starts, on="encounter_id", how="left")


def _criteria_onset(observations: pd.DataFrame, outcomes: pd.DataFrame, cfg) -> pd.DataFrame:
    """Estimate sepsis onset per encounter from configurable criteria."""
    criteria = cfg.labels.criteria
    rows = []
    for enc_id, group in observations.sort_values("timestamp").groupby("encounter_id"):
        ts = group["timestamp"].to_numpy()
        abx = group["antibiotics_given"].fillna(0).to_numpy() > 0 if "antibiotics_given" in group else np.zeros(len(group), bool)
        culture = group["culture_drawn"].fillna(0).to_numpy() > 0 if "culture_drawn" in group else np.zeros(len(group), bool)
        abx_win, cult_win = criteria.antibiotic_window_hours, criteria.culture_window_hours

        # Suspected infection: antibiotic administration with a culture in the
        # surrounding window (either order within the window).
        infection_time = None
        for i in range(len(ts)):
            lo = ts[i] - np.timedelta64(int(abx_win), "h")
            hi = ts[i] + np.timedelta64(int(abx_win), "h")
            if abx[i] and culture[np.argsort(np.abs(ts - ts[i]))[: max(1, int(cult_win))]].any():
                infection_time = ts[i]
                break

        # Organ dysfunction proxy: number of acute derangements at time i
        # versus the encounter baseline (first 6h mean).
        dysfunction_time = None
        base = group.iloc[: max(1, int(6))]
        rules = _derangement_rules(cfg)
        for i in range(len(ts)):
            row = group.iloc[i]
            count = 0
            for var, cmp_op, thr in rules:
                if var not in group.columns:
                    continue
                base_val = pd.to_numeric(base[var], errors="coerce").median()
                val = row.get(var)
                if pd.isna(val) or pd.isna(base_val):
                    continue
                acute = val if cmp_op == ">" else base_val - val
                baseline_shift = (base_val - thr) if cmp_op == ">" else (thr - base_val)
                if (val > thr if cmp_op == ">" else val < thr) and acute - baseline_shift > 0:
                    count += 1
            if count >= criteria.derangement_threshold:
                dysfunction_time = ts[i]
                break

        onset = None
        if infection_time is not None and dysfunction_time is not None:
            onset = max(pd.Timestamp(infection_time), pd.Timestamp(dysfunction_time))
        rows.append({"encounter_id": enc_id, "criteria_onset": onset})
    est = pd.DataFrame(rows)
    merged = outcomes.drop(columns=["sepsis_onset_timestamp"], errors="ignore").merge(est, on="encounter_id", how="left")
    merged = merged.rename(columns={"criteria_onset": "sepsis_onset_timestamp"})
    return merged


def build_label_table(
    observations: pd.DataFrame,
    outcomes: pd.DataFrame,
    cfg,
) -> LabelTable:
    """Enumerate prediction times and assign labels per the configured design."""
    lbl = cfg.labels
    horizon = pd.Timedelta(hours=lbl.horizon_hours)
    stride = pd.Timedelta(hours=lbl.stride_hours)
    min_hist = pd.Timedelta(hours=lbl.min_history_hours)

    observations = observations.copy()
    observations["timestamp"] = pd.to_datetime(observations["timestamp"], errors="coerce")
    outcomes = outcomes.copy()
    for col in ("sepsis_onset_timestamp", "admission_timestamp", "discharge_timestamp"):
        if col in outcomes.columns:
            outcomes[col] = pd.to_datetime(outcomes[col], errors="coerce")

    if lbl.source == "criteria":
        outcomes = _criteria_onset(observations, outcomes, cfg)
        logger.info("Label source=criteria: onset estimated from data-level criteria")

    obs_start = observations.groupby("encounter_id")["timestamp"].agg(["min", "max"])
    outcomes = outcomes.merge(obs_start, left_on="encounter_id", right_index=True, how="left")

    rows: list[dict] = []
    for rec in outcomes.itertuples(index=False):
        enc = rec.encounter_id
        start = rec.min
        end = rec.max
        onset = getattr(rec, "sepsis_onset_timestamp", pd.NaT)
        if pd.isna(start) or pd.isna(end):
            continue
        t = start + min_hist
        while t <= end:
            if not (lbl.exclude_after_onset and pd.notna(onset) and t >= onset):
                if pd.notna(onset) and onset <= t + horizon and onset > t:
                    label = 1
                elif pd.isna(onset) or onset > t + horizon:
                    label = 0
                else:
                    label = 0
                rows.append({
                    "encounter_id": enc,
                    "patient_id": rec.patient_id,
                    "t": t,
                    "label": int(label),
                })
            t = t + stride

    samples = pd.DataFrame(rows)
    enc_summary = (
        samples.groupby("encounter_id")
        .agg(n_samples=("label", "size"), n_positive=("label", "sum"))
        .reset_index()
        .merge(outcomes[["encounter_id", "patient_id", "sepsis_onset_timestamp"]], on="encounter_id", how="left")
    )
    enc_summary["has_sepsis"] = enc_summary["sepsis_onset_timestamp"].notna().astype(int)
    return LabelTable(
        samples=samples,
        encounters=enc_summary,
        label_source=lbl.source,
        horizon_hours=lbl.horizon_hours,
    )
