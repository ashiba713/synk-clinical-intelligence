"""Model evaluation pipeline.

Runs a trained model (torch multimodal/unimodal or sklearn baseline) over the
test split of a :class:`~synk.features.dataset.SampleSet` and produces the
full evaluation report:

* discrimination (AUROC/AUPRC with bootstrap CIs)
* threshold-dependent metrics + confusion matrix counts
* calibration (Brier, ECE, reliability curve)
* early-warning metrics (detection rate, lead time, false alerts)
* modality ablation for the multimodal model (research question support)

Every number is computed from actual model predictions - nothing is hardcoded.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from synk.evaluation.calibration import Calibrator
from synk.evaluation.metrics import (
    auc_with_bootstrap_ci,
    brier_score,
    calibration_curve_data,
    classification_metrics,
    expected_calibration_error,
    lead_time_analysis,
    pr_curve_data,
    roc_curve_data,
    sensitivity_at_operating_points,
)
from synk.features.dataset import SampleSet, collate_fn, make_torch_dataset
from synk.utils.torch_utils import get_device

logger = logging.getLogger(__name__)


def predict_sample_set(
    model: torch.nn.Module,
    sample_set: SampleSet,
    encounter_ids: Optional[list[str]] = None,
    indices: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
    modality_override: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over (part of) a SampleSet; returns (y_prob, y_true)."""
    device = device or get_device()
    model = model.to(device).eval()
    dataset = make_torch_dataset(sample_set, encounter_ids=encounter_ids, indices=indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["x"].to(device),
                batch["mask"].to(device),
                batch["static"].to(device),
                batch["text"],
                modality_override=modality_override,
            )
            probs.append(out["probability"].cpu().numpy())
            labels.append(batch["y"].numpy())
    if not probs:
        raise ValueError("No samples to evaluate - check the encounter filter")
    return np.concatenate(probs).astype(float), np.concatenate(labels).astype(int)


def evaluate_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config,
    sample_meta: Optional[pd.DataFrame] = None,
    outcomes: Optional[pd.DataFrame] = None,
    calibrator: Optional[Calibrator] = None,
) -> dict[str, Any]:
    """Compute the complete metric report for a set of predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob_cal = calibrator.transform(y_prob) if calibrator is not None else y_prob

    threshold = config.evaluation.threshold
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(y_true)),
        "prevalence": float(y_true.mean()) if len(y_true) else float("nan"),
        "auroc": auc_with_bootstrap_ci(
            y_true, y_prob, "auroc", config.evaluation.bootstrap_n, config.evaluation.bootstrap_ci, config.seed
        ),
        "auprc": auc_with_bootstrap_ci(
            y_true, y_prob, "auprc", config.evaluation.bootstrap_n, config.evaluation.bootstrap_ci, config.seed
        ),
        "threshold_metrics": classification_metrics(y_true, y_prob_cal, threshold),
        "brier_score": brier_score(y_true, y_prob_cal),
        "ece": expected_calibration_error(y_true, y_prob_cal),
        "calibration_curve": calibration_curve_data(y_true, y_prob_cal),
        "roc_curve": roc_curve_data(y_true, y_prob),
        "pr_curve": pr_curve_data(y_true, y_prob),
        "operating_points": sensitivity_at_operating_points(
            y_true, y_prob_cal, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    }

    if sample_meta is not None and outcomes is not None and len(sample_meta):
        meta = sample_meta.copy()
        meta["t"] = pd.to_datetime(meta["t"])
        report["early_warning"] = lead_time_analysis(meta, y_prob_cal, outcomes, threshold)
    return report


def _single_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def evaluate_model(
    model: torch.nn.Module,
    sample_set: SampleSet,
    encounter_ids: list[str],
    config,
    outcomes: pd.DataFrame,
    calibrator: Optional[Calibrator] = None,
    include_ablation: bool = False,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Full evaluation of a trained torch model on one encounter subset."""
    y_prob, y_true = predict_sample_set(model, sample_set, encounter_ids, device=device)
    meta = sample_set.samples[sample_set.samples["encounter_id"].isin(set(encounter_ids))].reset_index(drop=True)
    if len(meta) != len(y_true):
        logger.warning("Metadata rows (%d) != prediction rows (%d); early-warning metrics skipped",
                       len(meta), len(y_true))
        meta = None
    report = evaluate_probabilities(y_true, y_prob, config, meta, outcomes, calibrator)

    if include_ablation and getattr(model, "modality", "both") == "both":
        logger.info("Running modality-ablation evaluation (zeroing one modality at a time)")
        struct_prob, _ = predict_sample_set(
            model, sample_set, encounter_ids, device=device, modality_override="structured"
        )
        text_prob, _ = predict_sample_set(
            model, sample_set, encounter_ids, device=device, modality_override="text"
        )
        thr = config.evaluation.threshold
        report["ablation"] = {
            "note": (
                "Ablation runs the SAME multimodal model with one modality's "
                "embedding zeroed before fusion. It measures each modality's "
                "contribution within the fusion model; it is not a substitute "
                "for independently trained unimodal models."
            ),
            "structured_zeroed": {
                "auroc": _single_auc(y_true, struct_prob),
                **{k: v for k, v in classification_metrics(y_true, struct_prob, thr).items()
                   if k in ("f1", "sensitivity", "specificity")},
            },
            "text_zeroed": {
                "auroc": _single_auc(y_true, text_prob),
                **{k: v for k, v in classification_metrics(y_true, text_prob, thr).items()
                   if k in ("f1", "sensitivity", "specificity")},
            },
            "both_active": {
                "auroc": report["auroc"]["value"],
                "f1": report["threshold_metrics"]["f1"],
                "sensitivity": report["threshold_metrics"]["sensitivity"],
                "specificity": report["threshold_metrics"]["specificity"],
            },
        }
    return report


def save_report(report: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Evaluation report written to %s", path)


def load_report(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
