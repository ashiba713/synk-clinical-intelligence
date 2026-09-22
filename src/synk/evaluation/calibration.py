"""Probability calibration.

Two post-hoc calibrators fitted **on validation data only** (never on test):

* Platt scaling - logistic regression on logit(probs).
* Isotonic regression - non-parametric monotonic mapping.

The fitted calibrator is serialised to JSON (parameters only) so it ships
inside the model bundle and can be applied at inference time without
re-fitting on evaluation data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class Calibrator:
    """Applies a fitted probability mapping. ``identity`` means no calibration."""

    def __init__(self, method: str = "none") -> None:
        if method not in {"none", "isotonic", "platt"}:
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self._iso = None
        self._platt = None

    # ------------------------------------------------------------------
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "Calibrator":
        y_prob = np.asarray(y_prob, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=int).ravel()
        if self.method == "isotonic":
            # Fall back to identity when the validation set is too small to
            # fit a meaningful isotonic regression.
            if len(y_prob) < 200 or len(set(y_true.tolist())) < 2:
                logger.warning("Isotonic calibration skipped (n=%d < 200); using identity", len(y_prob))
                self.method = "none"
            else:
                self._iso = IsotonicRegression(out_of_bounds="clip")
                self._iso.fit(y_prob, y_true)
        elif self.method == "platt":
            if len(y_prob) < 50 or len(set(y_true.tolist())) < 2:
                logger.warning("Platt calibration skipped (n=%d < 50); using identity", len(y_prob))
                self.method = "none"
            else:
                eps = 1e-6
                clipped = np.clip(y_prob, eps, 1 - eps)
                logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
                self._platt = LogisticRegression(C=1e6, solver="lbfgs")
                self._platt.fit(logit, y_true)
        return self

    # ------------------------------------------------------------------
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        y_prob = np.asarray(y_prob, dtype=float).ravel()
        if self.method == "isotonic" and self._iso is not None:
            try:
                return self._iso.predict(y_prob)
            except AttributeError:
                return self._manual_isotonic(y_prob)
        if self.method == "platt" and self._platt is not None:
            eps = 1e-6
            clipped = np.clip(y_prob, eps, 1 - eps)
            logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            return self._platt.predict_proba(logit)[:, 1]
        return y_prob

    def _manual_isotonic(self, y_prob: np.ndarray) -> np.ndarray:
        """Step interpolation from serialised thresholds (no fitted interp1d needed)."""
        if not hasattr(self._iso, "X_thresholds_"):
            return y_prob
        xs = np.asarray(self._iso.X_thresholds_, dtype=float)
        ys = np.asarray(self._iso.y_thresholds_, dtype=float)
        x_min = float(getattr(self._iso, "X_min_", xs.min()))
        x_max = float(getattr(self._iso, "X_max_", xs.max()))
        clipped = np.clip(y_prob, x_min, x_max)
        return np.interp(clipped, xs, ys)

    # ------------------------------------------------------------------
    def to_json(self) -> dict:
        out: dict = {"method": self.method}
        if self.method == "isotonic" and self._iso is not None:
            out["x_thresholds"] = self._iso.X_thresholds_.tolist()
            out["y_thresholds"] = self._iso.y_thresholds_.tolist()
            out["x_min"] = float(self._iso.X_min_)
            out["x_max"] = float(self._iso.X_max_)
        elif self.method == "platt" and self._platt is not None:
            out["a"] = float(self._platt.coef_[0][0])
            out["b"] = float(self._platt.intercept_[0])
        return out

    @classmethod
    def from_json(cls, data: dict) -> "Calibrator":
        cal = cls(data.get("method", "none"))
        if cal.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.X_thresholds_ = np.asarray(data["x_thresholds"], dtype=float)
            iso.y_thresholds_ = np.asarray(data["y_thresholds"], dtype=float)
            iso.X_min_ = float(data.get("x_min", float(np.min(iso.X_thresholds_))))
            iso.X_max_ = float(data.get("x_max", float(np.max(iso.X_thresholds_))))
            iso.fitted_ = True
            cal._iso = iso
        elif cal.method == "platt":
            lr = LogisticRegression(C=1e6)
            lr.coef_ = np.asarray([[data["a"]]], dtype=float)
            lr.intercept_ = np.asarray([data["b"]], dtype=float)
            lr.classes_ = np.asarray([0, 1])
            lr.n_features_in_ = 1
            cal._platt = lr
        return cal

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
