"""Flattened tabular features for classical baselines.

Summarises each observation window into per-variable statistics (last value,
mean, min, max, slope, missing fraction) plus static demographics.  Used by
the logistic-regression / random-forest / gradient-boosting baselines and by
the lightweight sklearn demo fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synk.features.dataset import SampleSet

FEATURE_KINDS = ["last", "mean", "min", "max", "slope", "missing_frac"]


def _slope(series: np.ndarray, mask: np.ndarray) -> float:
    """Least-squares slope over observed points; 0 when <2 observations."""
    idx = np.where(mask)[0]
    if idx.size < 2:
        return 0.0
    x = idx.astype(float)
    y = series[idx]
    if np.allclose(y, y[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def build_tabular_features(sample_set: SampleSet) -> pd.DataFrame:
    """Return an [N, F] feature frame aligned with ``sample_set`` rows."""
    n, T, V = sample_set.x.shape
    rows = np.zeros((n, V * len(FEATURE_KINDS)), dtype=np.float32)
    for v, var in enumerate(sample_set.variables):
        block = sample_set.x[:, :, v]
        mask = sample_set.mask[:, :, v]
        for kind_i, kind in enumerate(FEATURE_KINDS):
            col = v * len(FEATURE_KINDS) + kind_i
            if kind == "last":
                idx = np.where(mask.any(axis=1), mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1), 0)
                rows[:, col] = np.where(mask.any(axis=1), block[np.arange(n), idx], 0.0)
            elif kind == "mean":
                denom = np.maximum(mask.sum(axis=1), 1)
                rows[:, col] = np.where(mask.any(axis=1), (block * mask).sum(axis=1) / denom, 0.0)
            elif kind == "min":
                masked = np.where(mask > 0, block, np.inf)
                rows[:, col] = np.where(mask.any(axis=1), masked.min(axis=1), 0.0)
            elif kind == "max":
                masked = np.where(mask > 0, block, -np.inf)
                rows[:, col] = np.where(mask.any(axis=1), masked.max(axis=1), 0.0)
            elif kind == "slope":
                rows[:, col] = [_slope(block[i], mask[i]) for i in range(n)]
            elif kind == "missing_frac":
                rows[:, col] = 1.0 - mask.mean(axis=1)

    static_cols = [f"static_{s}" for s in sample_set.static_variables]
    feature_names = [f"{var}_{kind}" for var in sample_set.variables for kind in FEATURE_KINDS] + static_cols
    frame = pd.DataFrame(
        np.hstack([rows, sample_set.static.astype(np.float32)]),
        columns=feature_names,
    )
    frame.insert(0, "encounter_id", sample_set.samples["encounter_id"].values)
    frame.insert(1, "t", sample_set.samples["t"].values)
    return frame
