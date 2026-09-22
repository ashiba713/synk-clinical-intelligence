"""End-to-end SYNK pipeline orchestration.

``run_pipeline`` chains the research workflow with leak-safe ordering:

1. generate/load data (synthetic by default)
2. validate
3. preprocess (fit on TRAIN split only)
4. label engineering -> samples
5. patient-level split
6. train structured / text / multimodal deep models
7. fit classical baselines
8. calibrate on validation, evaluate on TEST
9. register models + experiment rows

Each stage persists artifacts under ``artifacts/`` so any stage can be
inspected or re-run in isolation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from synk.config.settings import SYNKConfig, load_config, merge_configs, parse_dotlist
from synk.data.synthetic import generate_cohort
from synk.data.validation import validate_notes, validate_observations, validate_outcomes
from synk.evaluation.calibration import Calibrator
from synk.evaluation.evaluator import evaluate_model, save_report
from synk.features.dataset import (
    SampleSet,
    build_samples,
    patient_level_splits,
    save_splits,
)
from synk.labels.sepsis_labels import build_label_table
from synk.models.baselines import (
    train_late_fusion_baseline,
    train_structured_baseline,
    train_text_baseline,
)
from synk.preprocessing.text import TextPreprocessor
from synk.preprocessing.vitals import VitalsPreprocessor
from synk.tracking.registry import ExperimentRegistry, ModelRegistry, RunTracker
from synk.training.trainer import train_model
from synk.utils.common import new_id, utcnow_iso

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1-2: data
# ---------------------------------------------------------------------------
def load_or_generate_data(cfg: SYNKConfig, generate: Optional[bool] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (patients, observations, notes, outcomes); generates when missing."""
    syn_dir = Path(cfg.synthetic.output_dir)
    patients_p = syn_dir / "patients.csv"
    observations_p = syn_dir / "observations.csv"
    notes_p = syn_dir / "notes.csv"
    outcomes_p = syn_dir / "outcomes.csv"

    should_generate = generate if generate is not None else not observations_p.exists()
    if should_generate:
        logger.info("Generating synthetic cohort -> %s", syn_dir)
        cohort = generate_cohort(cfg, seed=cfg.seed)
        cohort.to_disk(syn_dir)
        logger.info("Synthetic cohort manifest: %s", cohort.manifest)

    patients = pd.read_csv(patients_p)
    observations = pd.read_csv(observations_p)
    notes = pd.read_csv(notes_p) if notes_p.exists() else None
    outcomes = pd.read_csv(outcomes_p)
    return patients, observations, notes, outcomes


# ---------------------------------------------------------------------------
# Stage 3-5: preprocessing + labels + splits
# ---------------------------------------------------------------------------
def prepare_samples(
    cfg: SYNKConfig,
    patients: pd.DataFrame,
    observations: pd.DataFrame,
    notes: Optional[pd.DataFrame],
    outcomes: pd.DataFrame,
) -> tuple[SampleSet, dict[str, list[str]], VitalsPreprocessor]:
    """Leak-safe sample construction: fit preprocessing on train encounters only."""
    validate_observations(observations, raise_on_error=False)
    if notes is not None:
        validate_notes(notes, raise_on_error=False)
    validate_outcomes(outcomes, observations, raise_on_error=False)

    notes = TextPreprocessor(cfg).clean_frame(notes)

    # Preliminary label table is needed to derive the patient split.
    label_table = build_label_table(observations, outcomes, cfg)
    if len(label_table.samples) == 0:
        raise ValueError("Label engine produced zero samples - check horizons vs observation coverage")

    # Two-pass split: split patients by encounter sepsis status, then fit the
    # preprocessor on the training encounters only.
    tmp_cfg = cfg
    splits = patient_level_splits_from_labels(label_table, cfg)

    train_enc = set(splits["train"])
    pre = VitalsPreprocessor(cfg).fit(observations[observations["encounter_id"].isin(train_enc)])

    cleaned = pre.transform(observations)
    static_cols = [c for c in ("patient_id", "age", "sex", "weight") if c in patients.columns]
    static_frame = patients[static_cols].copy() if len(patients) else pd.DataFrame(columns=["patient_id"])
    sample_set = build_samples(cleaned, notes, label_table, static_frame, cfg)

    interim = Path(cfg.data.processed_dir)
    interim.mkdir(parents=True, exist_ok=True)
    sample_set.save(interim / "samples")
    save_splits(splits, interim / "samples")
    pre.save(interim / "samples" / "preprocessor.json")
    return sample_set, splits, pre


def patient_level_splits_from_labels(label_table, cfg) -> dict[str, list[str]]:
    """Patient-level split from the label table (no dependency on SampleSet)."""
    from sklearn.model_selection import StratifiedShuffleSplit

    lbl = cfg.labels
    enc = label_table.encounters[["encounter_id", "patient_id", "has_sepsis"]]
    patient_label = enc.groupby("patient_id")["has_sepsis"].max().reset_index()
    patient_ids = patient_label["patient_id"].to_numpy()
    labels = patient_label["has_sepsis"].to_numpy()

    def _stratified():
        sss = StratifiedShuffleSplit(n_splits=1, train_size=lbl.splits.train_frac, random_state=cfg.seed)
        train_idx, rest_idx = next(sss.split(patient_ids, labels))
        rest_labels = labels[rest_idx]
        val_share = lbl.splits.val_frac / max(lbl.splits.val_frac + lbl.splits.test_frac, 1e-9)
        sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_share, random_state=cfg.seed + 1)
        val_rel, test_rel = next(sss2.split(np.zeros(len(rest_idx)), rest_labels))
        val_idx = rest_idx[val_rel]
        test_idx = rest_idx[test_rel]
        return train_idx, val_idx, test_idx

    def _fallback():
        # Tiny / degenerate cohorts: deterministic interleaved assignment that
        # still keeps every patient in exactly one split (documented fallback).
        n = len(patient_ids)
        order = np.argsort(patient_ids, kind="stable")
        train_idx = order[::3]
        val_idx = order[1::3]
        test_idx = order[2::3]
        return train_idx, val_idx, test_idx

    try:
        train_idx, val_idx, test_idx = _stratified()
    except ValueError:
        logger.warning("Stratified patient split impossible (tiny cohort); using deterministic interleaved fallback")
        train_idx, val_idx, test_idx = _fallback()

    patients = {
        "train": set(patient_ids[train_idx]),
        "val": set(patient_ids[val_idx]),
        "test": set(patient_ids[test_idx]),
    }
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b and (patients[a] & patients[b]):
                raise AssertionError(f"Patient leakage between {a} and {b}")
    enc_to_patient = dict(zip(enc["encounter_id"], enc["patient_id"]))
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for enc_id, pat in enc_to_patient.items():
        for split in ("train", "val", "test"):
            if pat in patients[split]:
                out[split].append(enc_id)
                break
    return out


# ---------------------------------------------------------------------------
# Stage 6-9: train, evaluate, register
# ---------------------------------------------------------------------------
def _run_one_modality(
    cfg: SYNKConfig,
    sample_set: SampleSet,
    splits: dict[str, list[str]],
    modality: str,
    outcomes: pd.DataFrame,
    registry: ModelRegistry,
    experiments: ExperimentRegistry,
    dataset_id: str,
) -> dict[str, Any]:
    run_id = f"{modality}_{new_id('run').split('_', 1)[1]}"
    run_cfg = merge_configs(cfg, {"model": {"modality": modality}})
    tracker = RunTracker(run_cfg, run_id)
    result = train_model(run_cfg, sample_set, splits, run_id, tracking=tracker)

    # Rebuild the model from the checkpoint for evaluation.
    from synk.models.multimodal import load_trained_model

    model, _payload = load_trained_model(result.checkpoint_path, run_cfg)

    # Calibrate on VALIDATION only.
    from synk.evaluation.evaluator import predict_sample_set

    val_prob, val_y = predict_sample_set(model, sample_set, splits["val"])
    calibrator = Calibrator(cfg.calibration.method).fit(val_prob, val_y)

    report = evaluate_model(
        model, sample_set, splits["test"], run_cfg, outcomes,
        calibrator=calibrator, include_ablation=(modality == "both"),
    )
    report["run_id"] = run_id
    report["modality"] = modality
    report["train_result"] = {
        "best_epoch": result.best_epoch,
        "best_val_auroc": result.best_val_auroc,
        "best_val_auprc": result.best_val_auprc,
        "duration_seconds": result.duration_seconds,
        "device": result.device,
    }

    eval_dir = Path(cfg.paths.evaluation_dir)
    save_report(report, eval_dir / f"{run_id}_test.json")

    metrics = {
        "auroc": report["auroc"]["value"],
        "auprc": report["auprc"]["value"],
        "f1": report["threshold_metrics"]["f1"],
        "sensitivity": report["threshold_metrics"]["sensitivity"],
        "specificity": report["threshold_metrics"]["specificity"],
        "brier": report["brier_score"],
    }
    early = report.get("early_warning") or {}
    exp_row = {
        "experiment_id": run_id,
        "modality": modality,
        "dataset_id": dataset_id,
        "model_kind": "synk_deep",
        "fusion": run_cfg.model.fusion.type if modality == "both" else None,
        **{k: v for k, v in metrics.items() if v == v},
        "detection_rate": early.get("detection_rate"),
        "lead_time_median_hours": early.get("lead_time_median_hours"),
        "run_id": run_id,
    }
    experiments.add(exp_row)
    tracker.finalize({**metrics, "stage": "train_and_evaluate"})

    bundle_dir = registry.save(
        name=f"synk_{modality}",
        checkpoint_path=Path(result.checkpoint_path),
        config_dict=run_cfg.model_dump_dict(),
        metrics=metrics,
        extra_files={
            "text_backend.pkl": Path(result.checkpoint_path).with_suffix(".text.pkl"),
        },
        dataset_id=dataset_id,
    )
    # Persist the calibrator (validation-fitted) and the fitted preprocessor
    # inside the bundle so inference reproduces the exact training-time mapping.
    calibrator.save(Path(bundle_dir) / "calibrator.json")
    pre_path = Path(cfg.data.processed_dir) / "samples" / "preprocessor.json"
    if pre_path.exists():
        import shutil

        shutil.copy2(pre_path, Path(bundle_dir) / "preprocessor.json")
    return {"report": report, "bundle_dir": bundle_dir, "metrics": metrics, "run_id": run_id}


def run_pipeline(
    config_path: Optional[Path] = None,
    overrides: Optional[list[str]] = None,
    skip_training: bool = False,
) -> dict[str, Any]:
    """Run the full research pipeline; returns the summary payload."""
    cfg = load_config(override_paths=[config_path] if config_path else None)
    if overrides:
        cfg = merge_configs(cfg, parse_dotlist(overrides))
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))

    dataset_id = cfg.data.dataset_id
    run_id = new_id("pipe")
    logger.info("SYNK pipeline %s | dataset=%s | seed=%d", run_id, dataset_id, cfg.seed)

    patients, observations, notes, outcomes = load_or_generate_data(cfg)
    sample_set, splits, preprocessor = prepare_samples(cfg, patients, observations, notes, outcomes)
    logger.info(
        "Samples: n=%d prevalence=%.3f variables=%d",
        len(sample_set.samples), float(sample_set.samples["label"].mean()), len(sample_set.variables),
    )

    registry = ModelRegistry(Path(cfg.paths.model_registry_dir))
    experiments = ExperimentRegistry(Path(cfg.paths.evaluation_dir) / "experiments.json")

    results: dict[str, Any] = {"run_id": run_id, "dataset_id": dataset_id, "n_samples": int(len(sample_set.samples))}

    # Classical baselines (fast, transparent).
    struct_base = train_structured_baseline(sample_set, splits["train"], splits["val"], cfg.seed)
    text_base = train_text_baseline(sample_set, splits["train"], splits["val"], cfg.seed)
    fusion_base = train_late_fusion_baseline(struct_base, text_base, cfg.seed)
    experiments.add({
        "experiment_id": f"baseline_structured_{run_id}",
        "modality": "structured", "dataset_id": dataset_id, "model_kind": "sklearn_baseline",
        **{k: v for k, v in struct_base["val_metrics"].items() if v == v},
        "run_id": run_id,
    })
    experiments.add({
        "experiment_id": f"baseline_text_{run_id}",
        "modality": "text", "dataset_id": dataset_id, "model_kind": "sklearn_baseline",
        **{k: v for k, v in text_base["val_metrics"].items() if v == v},
        "run_id": run_id,
    })
    experiments.add({
        "experiment_id": f"baseline_fusion_{run_id}",
        "modality": "both", "dataset_id": dataset_id, "model_kind": "sklearn_baseline",
        "fusion": "late",
        **{k: v for k, v in fusion_base["val_metrics"].items() if v == v},
        "run_id": run_id,
    })

    if not skip_training:
        modalities = ["structured", "text", "both"]
        for modality in modalities:
            logger.info("=== Training modality: %s ===", modality)
            out = _run_one_modality(cfg, sample_set, splits, modality, outcomes, registry, experiments, dataset_id)
            results[modality] = out["metrics"]
            results[f"{modality}_bundle"] = str(out["bundle_dir"])
            if modality == "both":
                results["multimodal_report"] = out["report"]

    results["completed_at"] = utcnow_iso()
    results["disclaimer"] = (
        "All metrics are computed from the trained models on the held-out test split of the "
        "configured dataset. Results on synthetic data demonstrate pipeline correctness, not clinical performance."
    )
    return results
