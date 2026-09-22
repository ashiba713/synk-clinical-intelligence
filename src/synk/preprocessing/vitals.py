"""Structured-vital preprocessing.

The :class:`VitalsPreprocessor` implements a leak-safe fit/transform API:

* ``fit`` computes winsorisation bounds, medians and standardisation statistics
  from the **training** split only.
* ``transform`` applies aggregation to an hourly grid, MAP derivation,
  forward-fill (bounded), median imputation, missingness indicators,
  winsorisation and standardisation.

Fitted parameters are serialised so that validation/test transformation uses
exactly the training-fitted statistics (no leakage).  Missingness is never
hidden: indicator columns record whether a measurement was actually observed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from synk.utils.logging import get_logger

logger = get_logger(__name__)

OBSERVED_SUFFIX = "__obs"


class VitalsPreprocessor:
    """Fit-on-train preprocessing for structured ICU observations."""

    def __init__(self, config) -> None:
        self.config = config
        self.variables: list[str] = []
        self.stats: dict[str, dict[str, float]] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, observations: pd.DataFrame, variables: Optional[list[str]] = None) -> "VitalsPreprocessor":
        """Learn per-variable statistics from (training) observations."""
        q_lo, q_hi = self.config.preprocessing.winsorize_quantiles
        variables = variables or [
            c for c in observations.columns
            if c not in ("patient_id", "encounter_id", "timestamp")
        ]
        self.variables = list(variables)
        self.stats = {}
        for var in self.variables:
            if var not in observations.columns:
                logger.warning("Variable '%s' absent from training data; will remain missing", var)
                self.stats[var] = {"median": 0.0, "mean": 0.0, "std": 1.0, "q_lo": -np.inf, "q_hi": np.inf}
                continue
            vals = pd.to_numeric(observations[var], errors="coerce").to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                self.stats[var] = {"median": 0.0, "mean": 0.0, "std": 1.0, "q_lo": -np.inf, "q_hi": np.inf}
                continue
            q_lo_v, q_hi_v = (float(np.quantile(finite, q_lo)), float(np.quantile(finite, q_hi)))
            if q_hi_v <= q_lo_v:
                q_lo_v, q_hi_v = float(finite.min()), float(finite.max()) + 1e-6
            self.stats[var] = {
                "median": float(np.median(finite)),
                "mean": float(np.mean(finite)),
                "std": float(max(np.std(finite), 1e-6)),
                "q_lo": q_lo_v,
                "q_hi": q_hi_v,
            }
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------
    def transform(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Transform observations to a cleaned hourly wide table.

        Returns a frame indexed by (encounter_id, timestamp) whose columns are
        the standardised variables plus ``<var>__obs`` indicator columns (1 =
        genuinely observed in that hour, before imputation).
        """
        if not self._fitted:
            raise RuntimeError("VitalsPreprocessor.fit must be called before transform")
        cfg = self.config.preprocessing
        df = observations.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.floor(f"{int(cfg.sampling_interval_hours)}h")

        id_cols = ["patient_id", "encounter_id"]
        value_cols = [v for v in self.variables if v in df.columns]

        # Average duplicated measurements within the same encounter/hour.
        agg = df.groupby(id_cols + ["hour"], as_index=False)[value_cols].mean()

        # Derive MAP when missing but SBP/DBP available.
        if cfg.derive_map and "mean_arterial_pressure" in value_cols:
            need = agg["mean_arterial_pressure"].isna()
            have = need & agg["systolic_bp"].notna() & agg["diastolic_bp"].notna()
            agg.loc[have, "mean_arterial_pressure"] = (
                agg.loc[have, "diastolic_bp"] + (agg.loc[have, "systolic_bp"] - agg.loc[have, "diastolic_bp"]) / 3.0
            )

        pieces: list[pd.DataFrame] = []
        step = pd.Timedelta(hours=cfg.sampling_interval_hours)
        limit = int(round(cfg.forward_fill_limit_hours / cfg.sampling_interval_hours))
        for enc_id, group in agg.groupby("encounter_id"):
            group = group.sort_values("hour").set_index("hour")
            full_index = pd.date_range(group.index.min(), group.index.max(), freq=step)
            group = group.reindex(full_index)
            group["encounter_id"] = enc_id
            group["patient_id"] = group["patient_id"].ffill().bfill()
            group.index.name = "hour"

            observed = group[value_cols].notna().astype("float64")
            observed.columns = [f"{c}{OBSERVED_SUFFIX}" for c in value_cols]

            values = group[value_cols].copy()
            if cfg.imputation == "forward_fill_then_median":
                values = values.ffill(limit=limit)
            for var in value_cols:
                stats = self.stats.get(var, {"median": 0.0})
                values[var] = values[var].fillna(stats["median"])

            # Winsorise, then standardise.
            for var in value_cols:
                stats = self.stats[var]
                values[var] = values[var].clip(stats["q_lo"], stats["q_hi"])
                if cfg.standardize == "zscore":
                    values[var] = (values[var] - stats["mean"]) / stats["std"]
                elif cfg.standardize == "minmax":
                    span = max(stats["q_hi"] - stats["q_lo"], 1e-6)
                    values[var] = (values[var] - stats["q_lo"]) / span

            out = pd.concat([values.add_prefix(""), observed], axis=1)
            out["encounter_id"] = enc_id
            out["patient_id"] = group["patient_id"]
            pieces.append(out.reset_index())

        if not pieces:
            empty = pd.DataFrame(columns=["hour", "encounter_id", "patient_id"])
            return empty
        result = pd.concat(pieces, ignore_index=True)
        return result

    # ------------------------------------------------------------------
    # Denormalisation helpers (used by explainability / evidence text)
    # ------------------------------------------------------------------
    def denormalize(self, var: str, standardised: np.ndarray) -> np.ndarray:
        """Map standardised values back to clinical units using fitted stats."""
        stats = self.stats.get(var, {"mean": 0.0, "std": 1.0})
        return np.asarray(standardised) * stats["std"] + stats["mean"]

    def normalize(self, var: str, raw: np.ndarray) -> np.ndarray:
        stats = self.stats.get(var, {"mean": 0.0, "std": 1.0})
        return (np.asarray(raw) - stats["mean"]) / stats["std"]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "variables": self.variables,
            "stats": self.stats,
            "config": self.config.preprocessing.model_dump() if hasattr(self.config.preprocessing, "model_dump") else {},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, config) -> "VitalsPreprocessor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        pre = cls(config)
        pre.variables = payload["variables"]
        pre.stats = payload["stats"]
        pre._fitted = True
        return pre
