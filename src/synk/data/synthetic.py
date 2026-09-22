"""Synthetic ICU cohort generator for SYNK.

IMPORTANT: every record produced by this module is synthetic.  It simulates
physiological trajectories with a simple stochastic deterioration process so
that the pipeline can be exercised end-to-end without restricted clinical
datasets.  Synthetic data must NEVER be presented as real clinical evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from synk.data.schemas import (
    LAB_VARIABLES,
    OBSERVATION_VALUE_COLUMNS,
    STATIC_VARIABLES,
    VITAL_VARIABLES,
)
from synk.utils.common import sha256_of_dict, utcnow_iso

GENERATOR_VERSION = "synthetic-v1.2"

BASE_TIMESTAMP = pd.Timestamp("2025-01-01 00:00:00")

DEMO_SCENARIOS = [
    "CASE_A_STABLE",
    "CASE_B_GRADUAL",
    "CASE_C_RAPID",
    "CASE_D_CONFLICTING",
    "CASE_E_PHYSIO_ONLY",
    "CASE_F_TEXT_ONLY",
]


@dataclass
class SyntheticCohort:
    patients: pd.DataFrame
    observations: pd.DataFrame
    notes: pd.DataFrame
    outcomes: pd.DataFrame
    manifest: dict

    def to_disk(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.patients.to_csv(output_dir / "patients.csv", index=False)
        self.observations.to_csv(output_dir / "observations.csv", index=False)
        self.notes.to_csv(output_dir / "notes.csv", index=False)
        self.outcomes.to_csv(output_dir / "outcomes.csv", index=False)
        pd.Series(self.manifest).to_json(output_dir / "manifest.json", indent=2)


@dataclass
class StaySpec:
    patient_idx: int
    stay_idx: int
    patient_id: str
    encounter_id: str
    age: float
    sex: int
    weight: float
    length_hours: int
    septic: bool
    onset_hour: Optional[int]
    prodrome_start: Optional[int]
    prodrome_length: Optional[int]
    abx_hour: Optional[int]
    severity: float
    scenario: str = ""


def _draw_patient(rng: np.random.Generator, idx: int) -> tuple[float, int, float]:
    age = float(np.clip(rng.normal(62, 15), 18, 95))
    sex = int(rng.integers(0, 2))
    weight = float(np.clip(rng.normal(80, 18), 40, 160))
    return age, sex, weight


def _syn_cfg(cfg):
    """Return the synthetic sub-config (supports both nested and flat access)."""
    return getattr(cfg, "synthetic", cfg)


def _manifest_config(cfg) -> dict:
    """JSON-safe snapshot of the generator parameters for the manifest."""
    syn = _syn_cfg(cfg)
    if hasattr(syn, "model_dump"):
        return {k: v for k, v in syn.model_dump().items() if isinstance(v, (int, float, str, bool, list))}
    return {k: (v if isinstance(v, (int, float, str, bool, list)) else str(v))
            for k, v in (asdict(syn) if hasattr(syn, "__dataclass_fields__") else {}).items()}


def _build_stay_specs(cfg, rng: np.random.Generator) -> list[StaySpec]:
    specs: list[StaySpec] = []
    syn = _syn_cfg(cfg)
    patient_idx = 0
    n_scenarios = len(DEMO_SCENARIOS) if syn.include_demo_scenarios else 0
    n_generic = syn.n_patients
    total = n_generic + n_scenarios

    for i in range(total):
        is_demo = i >= n_generic
        scenario = DEMO_SCENARIOS[i - n_generic] if is_demo else ""
        patient_id = f"SYN-P-{patient_idx + 1:05d}"
        encounter_id = f"SYN-E-{i + 1:05d}"
        age, sex, weight = _draw_patient(rng, patient_idx)

        if scenario == "CASE_A_STABLE":
            length, septic = 72, False
        elif scenario == "CASE_B_GRADUAL":
            length, septic = 96, True
        elif scenario == "CASE_C_RAPID":
            length, septic = 60, True
        elif scenario == "CASE_D_CONFLICTING":
            length, septic = 80, True
        elif scenario == "CASE_E_PHYSIO_ONLY":
            length, septic = 84, True
        elif scenario == "CASE_F_TEXT_ONLY":
            length, septic = 84, True
        else:
            length = int(rng.integers(syn.min_stay_hours, syn.max_stay_hours + 1))
            septic = bool(rng.random() < syn.sepsis_prevalence)

        onset: Optional[int] = None
        prodrome_start: Optional[int] = None
        prodrome_length: Optional[int] = None
        abx: Optional[int] = None
        severity = float(np.clip(rng.normal(1.0, 0.25), 0.5, 1.6))

        if septic:
            if scenario == "CASE_B_GRADUAL":
                prodrome_length = 24
            elif scenario == "CASE_C_RAPID":
                prodrome_length = 5
            elif scenario == "CASE_D_CONFLICTING":
                prodrome_length = 18
            elif scenario == "CASE_F_TEXT_ONLY":
                prodrome_length = 20
                severity = 0.45  # deliberately subtle physiology
            else:
                prodrome_length = int(rng.integers(8, 22))
            onset = int(np.clip(rng.integers(int(0.4 * length), int(0.85 * length)), prodrome_length + 2, length))
            prodrome_start = onset - prodrome_length
            abx = int(onset + rng.integers(1, 5))

        specs.append(
            StaySpec(
                patient_idx=patient_idx,
                stay_idx=i,
                patient_id=patient_id,
                encounter_id=encounter_id,
                age=age,
                sex=sex,
                weight=weight,
                length_hours=length,
                septic=septic,
                onset_hour=onset,
                prodrome_start=prodrome_start,
                prodrome_length=prodrome_length,
                abx_hour=abx,
                severity=severity,
                scenario=scenario,
            )
        )
        # Advance to a NEW patient after each stay; ~30% of (non-demo) patients
        # get a second stay via an extra spec sharing the same demographics.
        second_stay = (not is_demo) and rng.random() < 0.30
        if second_stay:
            extra_encounter = f"SYN-E-{i + 1:05d}-B"
            length2 = int(rng.integers(syn.min_stay_hours, syn.max_stay_hours + 1))
            septic2 = bool(rng.random() < syn.sepsis_prevalence * 0.6)
            specs.append(
                StaySpec(
                    patient_idx=patient_idx,
                    stay_idx=i,
                    patient_id=patient_id,
                    encounter_id=extra_encounter,
                    age=age,
                    sex=sex,
                    weight=weight,
                    length_hours=length2,
                    septic=septic2,
                    onset_hour=None,
                    prodrome_start=None,
                    prodrome_length=None,
                    abx_hour=None,
                    severity=float(np.clip(rng.normal(1.0, 0.25), 0.5, 1.6)),
                    scenario="",
                )
            )
        patient_idx += 1
    return specs


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _simulate_physiology(spec: StaySpec, cfg, rng: np.random.Generator) -> pd.DataFrame:
    hours = np.arange(spec.length_hours, dtype=float)
    n = len(hours)
    noise = _syn_cfg(cfg).noise_scale

    def ar1(base: float, sigma: float, phi: float = 0.75) -> np.ndarray:
        """Smooth autoregressive deviation from *base*."""
        out = np.zeros(n)
        for t in range(1, n):
            out[t] = phi * out[t - 1] + rng.normal(0.0, sigma * (1 - phi * phi) ** 0.5)
        return base + out

    values: dict[str, np.ndarray] = {
        "heart_rate": ar1(80 if spec.age > 60 else 76, 3.0 * noise),
        "respiratory_rate": ar1(16.5, 1.0 * noise),
        "systolic_bp": ar1(124, 3.5 * noise),
        "diastolic_bp": ar1(72, 2.5 * noise),
        "temperature": ar1(36.8, 0.12 * noise),
        "spo2": ar1(96.5, 0.8 * noise),
        "oxygen_flow": np.zeros(n),
        "wbc": ar1(7.5, 0.6 * noise),
        "platelets": ar1(255, 12.0 * noise),
        "creatinine": ar1(0.95, 0.06 * noise),
        "bilirubin": ar1(0.7, 0.05 * noise),
        "lactate": ar1(1.3, 0.15 * noise),
        "glucose": ar1(115, 8.0 * noise),
        "hemoglobin": ar1(11.6, 0.25 * noise),
    }

    if spec.septic and spec.onset_hour is not None and spec.prodrome_start is not None:
        prog = _sigmoid((hours - (spec.prodrome_start + 0.55 * spec.prodrome_length)) / (0.18 * spec.prodrome_length))
        s = spec.severity
        leucopenia = rng.random() < 0.25
        values["heart_rate"] += (26 * s) * prog + rng.normal(0, 1.5 * noise, n)
        values["respiratory_rate"] += (8 * s) * prog + rng.normal(0, 0.7 * noise, n)
        values["systolic_bp"] -= (22 * s) * prog + np.maximum(0.0, rng.normal(0, 1.8 * noise, n))
        values["diastolic_bp"] -= (12 * s) * prog
        values["temperature"] += (1.6 * s) * prog + rng.normal(0, 0.1 * noise, n)
        values["spo2"] -= (4.5 * s) * prog + np.maximum(0.0, rng.normal(0, 0.5 * noise, n))
        values["oxygen_flow"] += (4.0 * s) * prog
        values["wbc"] += ((-3.2 if leucopenia else 6.5) * s) * prog
        values["lactate"] += (2.6 * s) * prog + np.maximum(0.0, rng.normal(0, 0.15 * noise, n))
        values["creatinine"] += (0.65 * s) * prog
        values["bilirubin"] += (0.55 * s) * prog
        values["platelets"] -= (85 * s) * prog
        values["hemoglobin"] -= (1.1 * s) * prog
        if spec.abx_hour is not None:
            # After antibiotics are given, 65% of the deterioration signal recovers
            # with an 18h time constant (persistence shrinks from 1.0 toward 0.35).
            recovery = np.exp(-np.maximum(0.0, hours - spec.abx_hour) / 18.0)
            persist = 1.0 - 0.65 * recovery
            for var, effect in (("heart_rate", 26.0), ("temperature", 1.6), ("lactate", 2.6)):
                values[var] -= (effect * s) * prog * (1.0 - persist)
    else:
        # Non-septic confounders: brief fever episode in ~20% of stays.
        if rng.random() < 0.20:
            start = int(rng.integers(6, max(7, n - 14)))
            dur = int(rng.integers(6, 13))
            mask = (hours >= start) & (hours < start + dur)
            values["temperature"] += 1.3 * mask
            values["heart_rate"] += 10 * mask

    values["heart_rate"] = np.clip(values["heart_rate"], 35, 200)
    values["respiratory_rate"] = np.clip(values["respiratory_rate"], 6, 45)
    values["systolic_bp"] = np.clip(values["systolic_bp"], 60, 210)
    values["diastolic_bp"] = np.clip(values["diastolic_bp"], 30, 130)
    values["temperature"] = np.clip(values["temperature"], 34.5, 41.5)
    values["spo2"] = np.clip(values["spo2"], 70, 100)
    values["oxygen_flow"] = np.clip(values["oxygen_flow"], 0, 15)
    values["wbc"] = np.clip(values["wbc"], 0.3, 40)
    values["platelets"] = np.clip(values["platelets"], 15, 600)
    values["creatinine"] = np.clip(values["creatinine"], 0.2, 9)
    values["bilirubin"] = np.clip(values["bilirubin"], 0.1, 15)
    values["lactate"] = np.clip(values["lactate"], 0.3, 15)
    values["glucose"] = np.clip(values["glucose"], 40, 450)
    values["hemoglobin"] = np.clip(values["hemoglobin"], 5, 18)

    frame = pd.DataFrame({"timestamp": BASE_TIMESTAMP + pd.to_timedelta(hours, unit="h")})
    for var, series in values.items():
        frame[var] = series

    # Structured missingness: labs sampled every 6-12h, vitals missing at random.
    lab_interval = float(rng.integers(6, 13))
    for var in LAB_VARIABLES:
        phase = rng.random() * lab_interval
        measured = ((hours - phase) % lab_interval) < 1.0
        frame.loc[~measured, var] = np.nan
    for var in VITAL_VARIABLES:
        drop = rng.random(n) < _syn_cfg(cfg).missingness_rate
        frame.loc[drop, var] = np.nan

    # MAP is deliberately missing in ~30% of rows so preprocessing can derive it.
    map_drop = rng.random(n) < 0.30
    frame["mean_arterial_pressure"] = np.nan
    frame.loc[~map_drop, "mean_arterial_pressure"] = (
        frame.loc[~map_drop, "diastolic_bp"] + (frame.loc[~map_drop, "systolic_bp"] - frame.loc[~map_drop, "diastolic_bp"]) / 3.0
    )

    # Event flags.
    antibiotics = np.zeros(n)
    culture = np.zeros(n)
    if spec.septic and spec.abx_hour is not None:
        antibiotics[hours >= spec.abx_hour] = 1.0
        culture[(hours >= spec.abx_hour - 2) & (hours <= spec.abx_hour + 4)] = 1.0
    elif rng.random() < 0.10:
        # Non-septic stays occasionally receive antibiotics for suspected infection.
        start = int(rng.integers(8, max(9, n - 40)))
        antibiotics[hours >= start] = 1.0
        antibiotics[hours >= start + 48] = 0.0
        culture[(hours >= start - 2) & (hours <= start + 4)] = 1.0
    frame["antibiotics_given"] = antibiotics
    frame["culture_drawn"] = culture

    frame.insert(0, "encounter_id", spec.encounter_id)
    frame.insert(0, "patient_id", spec.patient_id)
    return frame


_NEGATIONS = [
    "Denies fever or chills. No signs of infection at this time.",
    "Afebrile overnight, no new concerns. No evidence of sepsis.",
    "Patient stable, denies shortness of breath. No infection suspected.",
    "Hemodynamically stable on room air. Cultures negative so far.",
]

_STABLE_NOTES = [
    "Patient resting comfortably. Vitals within normal limits. Continue current management.",
    "Overnight unremarkable. Patient ambulating in hallway, tolerating diet.",
    "Plan: continue supportive care, monitor vitals q4h.",
]

_WORRY_PHRASES = [
    "tachycardic with HR {hr:.0f}, increased work of breathing",
    "febrile to {temp:.1f} C, rigors noted",
    "hypotensive, MAP {map:.0f} mmHg, requiring fluid bolus",
    "lactate {lactate:.1f}, concern for tissue hypoperfusion",
    "worsening confusion, family at bedside",
    "urine output decreasing, concern for acute kidney injury",
    "starting empiric broad-spectrum antibiotics after cultures drawn",
    "persistent tachycardia and hypotension, sepsis bundle activated",
]

_REASSURING_PHRASES = [
    "normotensive, MAP {map:.0f} mmHg",
    "afebrile, temp {temp:.1f} C",
    "no respiratory distress, RR {rr:.0f}",
    "labs unremarkable",
]


def _nursing_note(spec: StaySpec, hour: int, row: pd.Series, rng: np.random.Generator) -> str:
    """Build a nursing note grounded in the simulated vitals at *hour*."""
    deterioration = (
        spec.septic and spec.prodrome_start is not None and hour >= spec.prodrome_start
    )
    if spec.scenario == "CASE_E_PHYSIO_ONLY":
        return str(rng.choice(_NEGATIONS + _STABLE_NOTES))
    if deterioration:
        phrases = rng.choice(_WORRY_PHRASES, size=2, replace=False)
        body = "; ".join(
            p.format(hr=row["heart_rate"], temp=row["temperature"], map=row["mean_arterial_pressure"],
                     lactate=row["lactate"], rr=row["respiratory_rate"]) for p in phrases
        )
        return f"Nursing assessment: {body}. Notify provider."
    if spec.scenario == "CASE_D_CONFLICTING" and spec.prodrome_start is not None and hour < spec.prodrome_start:
        return str(rng.choice(_NEGATIONS))
    if rng.random() < 0.5:
        return str(rng.choice(_NEGATIONS))
    return str(rng.choice(_STABLE_NOTES))


def _physician_note(spec: StaySpec, hour: int, row: pd.Series, rng: np.random.Generator) -> str:
    if spec.septic and spec.prodrome_start is not None and hour >= spec.prodrome_start:
        return (
            f"Assessment: rising lactate ({row['lactate']:.1f}) with new hypotension "
            f"(MAP {row['mean_arterial_pressure']:.0f}). Suspect sepsis of pulmonary source; "
            "cultures drawn, empiric antibiotics started, fluid resuscitation ongoing."
        )
    if spec.scenario == "CASE_F_TEXT_ONLY" and 30 <= hour <= 55:
        return (
            "Assessment: spiking temperatures with tachycardia. Concern for sepsis. "
            "Blood cultures drawn, broad-spectrum antibiotics started. Will reassess response."
        )
    return (
        f"Assessment: patient overall stable, afebrile ({row['temperature']:.1f} C), "
        f"hemodynamically stable. Continue current plan. No evidence of infection."
    )


def _generate_notes(spec: StaySpec, physio: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    n = len(physio)
    if spec.scenario == "CASE_E_PHYSIO_ONLY":
        note_hours = [int(n * 0.5)]
    else:
        base_hours = sorted(set(
            list(range(4, n, max(6, int(rng.integers(6, 10)))))
            + ([spec.onset_hour - 2] if spec.septic and spec.onset_hour and spec.onset_hour > 2 else [])
        ))
        note_hours = [h for h in base_hours if 0 <= h < n]
        if spec.septic:
            note_hours += [h for h in range(max(0, (spec.prodrome_start or 0)), n, 4)]
        note_hours = sorted(set(note_hours))

    for hour in note_hours:
        row = physio.iloc[hour]
        is_physician = rng.random() < 0.3
        text = _physician_note(spec, hour, row, rng) if is_physician else _nursing_note(spec, hour, row, rng)
        rows.append({
            "patient_id": spec.patient_id,
            "encounter_id": spec.encounter_id,
            "timestamp": BASE_TIMESTAMP + pd.Timedelta(hours=hour),
            "note_type": "physician_note" if is_physician else "nursing_note",
            "author_type": "MD" if is_physician else "RN",
            "note_text": text,
        })
    return pd.DataFrame(rows)


def generate_cohort(cfg, seed: Optional[int] = None) -> SyntheticCohort:
    """Generate a complete synthetic ICU cohort.

    ``cfg`` must expose the fields of :class:`synk.config.settings.SyntheticConfig`.
    """
    seed = seed if seed is not None else 42
    rng = np.random.default_rng(seed)
    specs = _build_stay_specs(cfg, rng)

    patients: dict[str, dict] = {}
    obs_frames: list[pd.DataFrame] = []
    note_frames: list[pd.DataFrame] = []
    outcome_rows: list[dict] = []

    for spec in specs:
        physio = _simulate_physiology(spec, cfg, rng)
        notes = _generate_notes(spec, physio, rng)
        obs_frames.append(physio)
        if len(notes):
            note_frames.append(notes)
        patients.setdefault(spec.patient_id, {
            "patient_id": spec.patient_id,
            "age": spec.age,
            "sex": spec.sex,
            "weight": spec.weight,
            "is_synthetic": True,
        })
        onset_ts = None
        if spec.septic and spec.onset_hour is not None:
            onset_ts = BASE_TIMESTAMP + pd.Timedelta(hours=int(spec.onset_hour))
        outcome_rows.append({
            "patient_id": spec.patient_id,
            "encounter_id": spec.encounter_id,
            "admission_timestamp": BASE_TIMESTAMP,
            "discharge_timestamp": BASE_TIMESTAMP + pd.Timedelta(hours=int(spec.length_hours)),
            "sepsis_onset_timestamp": onset_ts,
            "scenario": spec.scenario,
        })

    observations = pd.concat(obs_frames, ignore_index=True)
    notes = pd.concat(note_frames, ignore_index=True) if note_frames else pd.DataFrame(
        columns=["patient_id", "encounter_id", "timestamp", "note_type", "author_type", "note_text"])
    outcomes = pd.DataFrame(outcome_rows)
    patients_df = pd.DataFrame(list(patients.values()))

    # Column order + explicit MAP column.
    cols = ["patient_id", "encounter_id", "timestamp"] + list(OBSERVATION_VALUE_COLUMNS)
    observations = observations[cols]

    manifest = {
        "is_synthetic": True,
        "banner": "SYNTHETIC DEMONSTRATION DATA - not real patient data - generated by SYNK",
        "generator_version": GENERATOR_VERSION,
        "generated_at": utcnow_iso(),
        "seed": seed,
        "config": _manifest_config(cfg),
        "n_patients": int(patients_df["patient_id"].nunique()),
        "n_stays": int(len(outcomes)),
        "n_observations": int(len(observations)),
        "n_notes": int(len(notes)),
        "observed_prevalence": float(outcomes["sepsis_onset_timestamp"].notna().mean()),
        "static_variables": list(STATIC_VARIABLES),
        "checksum": "",
    }
    manifest["checksum"] = sha256_of_dict({
        k: manifest[k] for k in ("generator_version", "seed", "n_stays", "n_observations")
    })

    return SyntheticCohort(
        patients=patients_df,
        observations=observations,
        notes=notes,
        outcomes=outcomes,
        manifest=manifest,
    )
