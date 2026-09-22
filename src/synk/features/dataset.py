"""Prediction-sample construction and dataset utilities.

A *sample* is one (encounter, prediction time t) pair produced by the label
engine.  For each sample we materialise:

* ``x``    [T, V] standardised structured values (T = observation window)
* ``mask`` [T, V] 1.0 where a value was genuinely observed (missingness kept)
* ``static`` [S] standardised demographics
* ``text`` concatenated clinical notes within the text lookback window
* ``y``    label from the configured sepsis definition / horizon

Splitting is done at the PATIENT level (never at row level) so no patient
appears in more than one split, and stratified by encounter-level sepsis
status to preserve prevalence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from synk.labels.sepsis_labels import LabelTable
from synk.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sample building
# ---------------------------------------------------------------------------
@dataclass
class SampleSet:
    samples: pd.DataFrame                 # metadata: sample_id, patient_id, encounter_id, t, label, text
    x: np.ndarray                         # [N, T, V] float32 (standardised)
    mask: np.ndarray                      # [N, T, V] float32 observed indicators
    static: np.ndarray                    # [N, S] float32
    variables: list[str] = field(default_factory=list)
    static_variables: list[str] = field(default_factory=list)
    window_hours: int = 12
    label_source: str = "latent_outcomes"
    horizon_hours: float = 6.0
    static_mean: Optional[np.ndarray] = None   # [S] pre-standardisation means
    static_std: Optional[np.ndarray] = None    # [S] pre-standardisation stds

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "arrays.npz", x=self.x, mask=self.mask, static=self.static,
        )
        self.samples.to_csv(directory / "samples.csv", index=False)
        meta = {
            "variables": self.variables,
            "static_variables": self.static_variables,
            "window_hours": self.window_hours,
            "label_source": self.label_source,
            "horizon_hours": self.horizon_hours,
            "n_samples": int(len(self.samples)),
            "static_mean": self.static_mean.tolist() if self.static_mean is not None else None,
            "static_std": self.static_std.tolist() if self.static_std is not None else None,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "SampleSet":
        directory = Path(directory)
        arrays = np.load(directory / "arrays.npz")
        samples = pd.read_csv(directory / "samples.csv", parse_dates=["t"])
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return cls(
            samples=samples,
            x=arrays["x"],
            mask=arrays["mask"],
            static=arrays["static"],
            variables=meta["variables"],
            static_variables=meta["static_variables"],
            window_hours=meta["window_hours"],
            label_source=meta["label_source"],
            horizon_hours=meta["horizon_hours"],
            static_mean=(
                np.asarray(meta["static_mean"], dtype=np.float32)
                if meta.get("static_mean") is not None else None
            ),
            static_std=(
                np.asarray(meta["static_std"], dtype=np.float32)
                if meta.get("static_std") is not None else None
            ),
        )


def _window_matrix(
    grid: pd.DataFrame, value_cols: list[str], obs_cols: list[str], t: pd.Timestamp, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the last *window* hours ending at *t* (front-padded if needed)."""
    if len(grid) == 0:
        return np.zeros((window, len(value_cols)), np.float32), np.zeros((window, len(value_cols)), np.float32)
    hours = grid["hour"].to_numpy()
    end_pos = int(np.searchsorted(hours, np.datetime64(t), side="right"))
    start_pos = max(0, end_pos - window)
    x = grid.iloc[start_pos:end_pos][value_cols].to_numpy(dtype=np.float32)
    m = grid.iloc[start_pos:end_pos][obs_cols].to_numpy(dtype=np.float32)
    if x.shape[0] < window:
        pad = window - x.shape[0]
        x = np.vstack([np.zeros((pad, len(value_cols)), np.float32), x])
        m = np.vstack([np.zeros((pad, len(value_cols)), np.float32), m])
    return x[:window], m[:window]


def build_samples(
    cleaned: pd.DataFrame,
    notes: Optional[pd.DataFrame],
    label_table: LabelTable,
    static_frame: pd.DataFrame,
    cfg,
) -> SampleSet:
    """Materialise prediction samples from the cleaned hourly grid.

    ``cleaned`` is the output of :class:`VitalsPreprocessor.transform` (an
    hourly wide table with ``<var>`` and ``<var>__obs`` columns).  Event-flag
    columns (antibiotics/culture) are intentionally excluded from model
    inputs: they are used by the criteria label engine and would otherwise
    leak treatment information into the features.
    """
    from synk.data.schemas import EVENT_VARIABLES

    pre = cfg.preprocessing
    window = int(pre.observation_window_hours)
    text_cfg = cfg.model.text

    value_cols = [c for c in cleaned.columns if c.endswith("__obs") is False
                  and c not in ("encounter_id", "patient_id", "hour")
                  and not any(c.startswith(ev) for ev in EVENT_VARIABLES)]
    obs_cols = [f"{c}__obs" for c in value_cols]

    cleaned = cleaned.sort_values(["encounter_id", "hour"]).reset_index(drop=True)
    cleaned["hour"] = pd.to_datetime(cleaned["hour"])
    grids: dict[str, pd.DataFrame] = {enc: g for enc, g in cleaned.groupby("encounter_id")}

    if notes is not None and len(notes):
        notes = notes.copy()
        notes["timestamp"] = pd.to_datetime(notes["timestamp"], errors="coerce")
        notes = notes.dropna(subset=["timestamp"]).sort_values("timestamp")
        notes_by_enc = {enc: g for enc, g in notes.groupby("encounter_id")}
    else:
        notes_by_enc = {}

    static_by_patient = static_frame.set_index("patient_id") if len(static_frame) else pd.DataFrame()
    static_vars = [c for c in ("age", "sex", "weight") if len(static_by_patient) == 0 or c in static_by_patient.columns]

    xs, ms, ss, metas = [], [], [], []
    lookback = pd.Timedelta(hours=text_cfg.text_lookback_hours)
    static_mean = np.zeros(len(static_vars), dtype=np.float32)
    static_std = np.ones(len(static_vars), dtype=np.float32)

    for rec in label_table.samples.itertuples(index=False):
        grid = grids.get(rec.encounter_id)
        if grid is None or len(grid) == 0:
            continue
        x, m = _window_matrix(grid, value_cols, obs_cols, rec.t, window)

        enc_notes = notes_by_enc.get(rec.encounter_id)
        if enc_notes is not None and len(enc_notes):
            window_notes = enc_notes[
                (enc_notes["timestamp"] <= rec.t) & (enc_notes["timestamp"] > rec.t - lookback)
            ]
            window_notes = window_notes.tail(text_cfg.max_notes)
            text = " | ".join(str(t_) for t_ in window_notes["note_text"])
        else:
            text = ""

        patient_row = static_by_patient.loc[rec.patient_id] if rec.patient_id in static_by_patient.index else None
        if patient_row is not None:
            static_vals = [float(patient_row.get(v, np.nan)) for v in static_vars]
        else:
            static_vals = [np.nan] * len(static_vars)

        xs.append(x)
        ms.append(m)
        ss.append(static_vals)
        metas.append({
            "sample_id": f"{rec.encounter_id}@{pd.Timestamp(rec.t).isoformat()}",
            "patient_id": rec.patient_id,
            "encounter_id": rec.encounter_id,
            "t": rec.t,
            "label": int(rec.label),
            "text": text,
        })

    if not metas:
        raise ValueError("No samples could be built - check label table and observations overlap")

    x_arr = np.stack(xs).astype(np.float32)
    m_arr = np.stack(ms).astype(np.float32)
    s_arr = np.array(ss, dtype=np.float32)
    samples = pd.DataFrame(metas)

    # Impute + standardise static features using sample-set statistics
    # (computed across all splits here; static demographics contain no
    # outcome information - documented in docs/methodology.md).
    for j, var in enumerate(static_vars):
        col = s_arr[:, j]
        finite = col[np.isfinite(col)]
        med = float(np.median(finite)) if finite.size else 0.0
        col = np.where(np.isfinite(col), col, med)
        mean = float(np.mean(col))
        std = float(np.std(col)) or 1.0
        static_mean[j], static_std[j] = mean, std
        s_arr[:, j] = (col - mean) / std

    return SampleSet(
        samples=samples,
        x=x_arr,
        mask=m_arr,
        static=s_arr,
        variables=value_cols,
        static_variables=list(static_vars),
        window_hours=window,
        label_source=label_table.label_source,
        horizon_hours=label_table.horizon_hours,
        static_mean=static_mean,
        static_std=static_std,
    )


# ---------------------------------------------------------------------------
# Patient-level splitting
# ---------------------------------------------------------------------------
def patient_level_splits(
    sample_set: SampleSet,
    cfg,
) -> dict[str, list[str]]:
    """Assign encounters to train/val/test by PATIENT, stratified by sepsis.

    Returns a mapping split -> list of encounter_ids.  Asserts that no
    patient appears in two splits.
    """
    splits_cfg = cfg.labels.splits
    enc = (
        sample_set.samples.groupby(["encounter_id", "patient_id"])["label"].max().reset_index()
    )
    patient_label = enc.groupby("patient_id")["label"].max().reset_index()
    patient_ids = patient_label["patient_id"].to_numpy()
    labels = patient_label["label"].to_numpy()

    rng_seed = cfg.seed
    frac_train, frac_val, frac_test = splits_cfg.train_frac, splits_cfg.val_frac, splits_cfg.test_frac

    sss = StratifiedShuffleSplit(n_splits=1, train_size=frac_train, random_state=rng_seed)
    train_idx, rest_idx = next(sss.split(patient_ids, labels))
    rest_labels = labels[rest_idx]
    val_share = frac_val / max(frac_val + frac_test, 1e-9)
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_share, random_state=rng_seed + 1)
    val_rel, test_rel = next(sss2.split(np.zeros(len(rest_idx)), rest_labels))
    val_idx = rest_idx[val_rel]
    test_idx = rest_idx[test_rel]

    patients = {
        "train": set(patient_ids[train_idx]),
        "val": set(patient_ids[val_idx]),
        "test": set(patient_ids[test_idx]),
    }
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                overlap = patients[a] & patients[b]
                if overlap:
                    raise AssertionError(f"Patient leakage between {a} and {b}: {overlap}")

    enc_to_patient = dict(zip(enc["encounter_id"], enc["patient_id"]))
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for enc_id, pat in enc_to_patient.items():
        for split in ("train", "val", "test"):
            if pat in patients[split]:
                out[split].append(enc_id)
                break

    logger.info(
        "Patient-level split: train=%d val=%d test=%d encounters",
        len(out["train"]), len(out["val"]), len(out["test"]),
    )
    return out


def save_splits(splits: dict[str, list[str]], directory: Path) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")


def load_splits(directory: Path) -> dict[str, list[str]]:
    payload = json.loads((Path(directory) / "splits.json").read_text(encoding="utf-8"))
    return payload


# ---------------------------------------------------------------------------
# Torch dataset
# ---------------------------------------------------------------------------
def make_torch_dataset(sample_set: SampleSet, encounter_ids: Optional[list[str]] = None, indices: Optional[np.ndarray] = None):
    """Create a torch Dataset over the sample set (optionally filtered)."""
    import torch
    from torch.utils.data import Dataset

    if indices is None:
        if encounter_ids is None:
            indices = np.arange(len(sample_set.samples))
        else:
            indices = np.where(sample_set.samples["encounter_id"].isin(set(encounter_ids)))[0]

    class _SYNKDataset(Dataset):
        def __init__(self, ss: SampleSet, idx: np.ndarray):
            self.ss = ss
            self.idx = idx

        def __len__(self) -> int:
            return len(self.idx)

        def __getitem__(self, item: int) -> dict:
            i = int(self.idx[item])
            row = self.ss.samples.iloc[i]
            return {
                "x": torch.from_numpy(self.ss.x[i]),
                "mask": torch.from_numpy(self.ss.mask[i]),
                "static": torch.from_numpy(self.ss.static[i]),
                "text": str(row["text"]),
                "y": torch.tensor(float(row["label"]), dtype=torch.float32),
                "encounter_id": row["encounter_id"],
                "t": str(row["t"]),
            }

    return _SYNKDataset(sample_set, indices)


def collate_fn(batch: list[dict]) -> dict:
    """Default collate that keeps the raw text list intact.

    Tolerates rows without labels/ids (used for inference-only batches).
    """
    import torch

    out = {
        "x": torch.stack([b["x"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "static": torch.stack([b["static"] for b in batch]),
        "text": [b["text"] for b in batch],
    }
    if all("y" in b for b in batch):
        out["y"] = torch.stack([b["y"] for b in batch])
    if all("encounter_id" in b for b in batch):
        out["encounter_id"] = [b["encounter_id"] for b in batch]
    if all("t" in b for b in batch):
        out["t"] = [b["t"] for b in batch]
    return out
