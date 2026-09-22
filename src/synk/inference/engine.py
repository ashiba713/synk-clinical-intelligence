"""Inference engine.

Loads a model bundle from the registry (torch weights + config + text-backend
state + calibrator) and turns raw encounters into prediction payloads with
explanations and uncertainty.  A fully offline *demo baseline* (logistic
regression on structured + TF-IDF text features) is supported so the API and
dashboard work before any training run has happened; demo outputs are clearly
labelled with ``engine: sklearn_demo``.

All outputs are model evidence for research interpretation - never clinical
instructions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from synk.config.settings import SYNKConfig, load_config
from synk.evaluation.calibration import Calibrator
from synk.evaluation.metrics import risk_category
from synk.features.dataset import SampleSet, build_samples
from synk.labels.sepsis_labels import build_label_table
from synk.models.multimodal import SYNKModel
from synk.preprocessing.vitals import VitalsPreprocessor
from synk.utils.torch_utils import get_device

logger = logging.getLogger(__name__)

RESEARCH_DISCLAIMER = (
    "Research prototype - not for clinical diagnosis or treatment. "
    "Outputs are model-generated evidence for research interpretation."
)


# ---------------------------------------------------------------------------
# Bundle dataclass
# ---------------------------------------------------------------------------
@dataclass
class ModelBundle:
    model: Optional[SYNKModel]
    config: SYNKConfig
    calibrator: Optional[Calibrator] = None
    text_backend_path: Optional[Path] = None
    manifest: dict[str, Any] = field(default_factory=dict)
    bundle_dir: Optional[Path] = None

    @property
    def model_version(self) -> str:
        return str(self.manifest.get("model_version", "unregistered"))

    @property
    def model_name(self) -> str:
        return str(self.manifest.get("model_name", "synk_multimodal"))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_bundle(
    model_dir: Optional[Path] = None,
    config: Optional[SYNKConfig] = None,
) -> Optional[ModelBundle]:
    """Load the latest (or given) registered model bundle; None if absent."""
    cfg = config or load_config()
    if model_dir is None:
        from synk.tracking.registry import ModelRegistry

        registry_dir = Path(cfg.paths.model_registry_dir)
        if not registry_dir.exists():
            return None
        registry = ModelRegistry(registry_dir)
        latest = registry.latest("synk_multimodal") or registry.latest("synk_both")
        if latest is None:
            return None
        model_dir = latest

    model_dir = Path(model_dir)
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        logger.warning("No manifest.json in %s; cannot load bundle", model_dir)
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    saved = manifest.get("config", {})
    bundle_cfg = SYNKConfig.model_validate({**cfg.model_dump(mode="python"), **saved}) \
        if saved else cfg.model_copy(deep=True)

    weights = model_dir / "model.pt"
    if not weights.exists():
        logger.warning("No model.pt in %s; cannot load torch model", model_dir)
        return None

    from synk.models.multimodal import load_trained_model

    model, _payload = load_trained_model(weights, bundle_cfg, text_backend_path=model_dir / "text_backend.pkl")

    calibrator = None
    calib_path = model_dir / "calibrator.json"
    if calib_path.exists():
        calibrator = Calibrator.load(calib_path)

    text_backend = model_dir / "text_backend.pkl"
    if text_backend.exists() and bundle_cfg.model.text.encoder_type == "tfidf":
        from synk.models.nlp.text_encoder import TfidfTextBackend

        model.text_encoder.backend = TfidfTextBackend.load(text_backend)
        model.text_encoder._output_dim = model.text_encoder.backend.output_dim()

    return ModelBundle(
        model=model,
        config=bundle_cfg,
        calibrator=calibrator,
        manifest=manifest,
        bundle_dir=model_dir,
    )


# ---------------------------------------------------------------------------
# Encounter -> samples (shared by torch and demo engines)
# ---------------------------------------------------------------------------
class EncounterFeaturizer:
    """Transforms raw frames for a set of encounters into prediction samples.

    Reuses the exact preprocessing stack from training (fitted statistics are
    loaded from the bundle directory when present) so training/inference never
    drift.
    """

    def __init__(self, config: SYNKConfig, preprocessor: Optional[VitalsPreprocessor] = None) -> None:
        self.config = config
        self.preprocessor = preprocessor

    @classmethod
    def from_bundle(cls, bundle: Optional[ModelBundle], config: SYNKConfig) -> "EncounterFeaturizer":
        pre: Optional[VitalsPreprocessor] = None
        if bundle is not None and bundle.bundle_dir is not None:
            pp_path = bundle.bundle_dir / "preprocessor.json"
            if pp_path.exists():
                pre = VitalsPreprocessor.load(pp_path, config)
        return cls(config, pre)

    def fit_if_needed(self, observations: pd.DataFrame) -> None:
        if self.preprocessor is None:
            self.preprocessor = VitalsPreprocessor(self.config).fit(observations)

    def build(self, observations, notes, outcomes, patients) -> tuple[SampleSet, Any]:
        """Return (sample_set, label_table) for the supplied raw frames."""
        assert self.preprocessor is not None, "featurizer not fitted"
        cleaned = self.preprocessor.transform(observations)
        label_table = build_label_table(observations, outcomes, self.config)
        static_frame = patients[[c for c in ("patient_id", "age", "sex", "weight") if c in patients.columns]] \
            if patients is not None and len(patients) else pd.DataFrame(columns=["patient_id"])
        sample_set = build_samples(cleaned, notes, label_table, static_frame, self.config)
        return sample_set, label_table


# ---------------------------------------------------------------------------
# Torch engine
# ---------------------------------------------------------------------------
class TorchEngine:
    """Prediction service over a trained SYNK torch model."""

    def __init__(self, bundle: ModelBundle, featurizer: EncounterFeaturizer, device: Optional[torch.device] = None) -> None:
        self.bundle = bundle
        self.config = bundle.config
        self.featurizer = featurizer
        self.device = device or get_device()
        self.model = bundle.model.to(self.device).eval() if bundle.model is not None else None

    @property
    def engine_id(self) -> str:
        return f"torch:{self.bundle.model_name}:{self.bundle.model_version}"

    def predict_samples(
        self,
        sample_set: SampleSet,
        calibrate: bool = True,
        modality_override: Optional[str] = None,
    ) -> pd.DataFrame:
        from torch.utils.data import DataLoader

        from synk.features.dataset import collate_fn, make_torch_dataset
        from synk.utils.torch_utils import get_device as _gd

        device = _gd()
        model = self.model
        assert model is not None
        dataset = make_torch_dataset(sample_set)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        probs: list[float] = []
        with torch.no_grad():
            for batch in loader:
                out = model(
                    batch["x"].to(device), batch["mask"].to(device),
                    batch["static"].to(device), batch["text"],
                    modality_override=modality_override,
                )
                probs.extend(out["probability"].cpu().numpy().tolist())
        probs_arr = np.asarray(probs, dtype=float)
        if calibrate and self.bundle.calibrator is not None:
            probs_arr = self.bundle.calibrator.transform(probs_arr)
        frame = sample_set.samples.copy()
        frame["risk_probability"] = probs_arr
        frame["risk_category"] = [
            risk_category(p, self.config.evaluation.risk_thresholds) for p in probs_arr
        ]
        frame["prediction_horizon_hours"] = float(self.config.labels.horizon_hours)
        frame["model_version"] = self.bundle.model_version
        return frame

    def explain_one(self, sample_set: SampleSet, index: int, include_token_attribution: bool = False) -> dict:
        from synk.explainability.explainer import explain_prediction
        from synk.features.dataset import collate_fn

        assert self.model is not None
        row = {
            "x": torch.from_numpy(sample_set.x[index]),          # (T, V) - collate adds batch dim
            "mask": torch.from_numpy(sample_set.mask[index]),
            "static": torch.from_numpy(sample_set.static[index]),
            "text": str(sample_set.samples.iloc[index]["text"]),  # collate wraps text into a list
        }
        batch = collate_fn([row])
        return explain_prediction(
            self.model, batch, sample_set.variables, self.config,
            device=self.device, include_token_attribution=include_token_attribution,
        )


# ---------------------------------------------------------------------------
# Demo (sklearn) engine - transparent fallback, clearly labelled
# ---------------------------------------------------------------------------
class DemoEngine:
    """Offline logistic-regression fallback used before any training run.

    It is a real model (fitted on synthetic data at API start-up when no
    trained bundle exists), not a stub: it produces genuine probabilities from
    the same features.  Every payload it produces is labelled
    ``engine: sklearn_demo``.
    """

    def __init__(self, config: SYNKConfig, structured_model, text_model, meta_model) -> None:
        self.config = config
        self.structured = structured_model
        self.text_model = text_model
        self.meta = meta_model

    @property
    def engine_id(self) -> str:
        return "sklearn_demo:v0"

    def predict_samples(self, sample_set: SampleSet, calibrate: bool = True, **_) -> pd.DataFrame:
        from synk.features.tabular import build_tabular_features

        feats = build_tabular_features(sample_set)
        X = feats.drop(columns=["encounter_id", "t"]).to_numpy(dtype=np.float32)
        p_struct = self.structured.predict_proba(X)[:, 1]
        texts = [t if str(t).strip() else "no clinical note available" for t in sample_set.samples["text"]]
        p_text = self.text_model.predict_proba(texts)[:, 1]
        p = self.meta.predict_proba(np.column_stack([p_struct, p_text]))[:, 1]
        frame = sample_set.samples.copy()
        frame["risk_probability"] = p
        frame["risk_category"] = [risk_category(x, self.config.evaluation.risk_thresholds) for x in p]
        frame["prediction_horizon_hours"] = float(self.config.labels.horizon_hours)
        frame["model_version"] = "demo_v0"
        return frame

    def explain_one(self, sample_set: SampleSet, index: int, include_token_attribution: bool = False) -> dict:
        """Deterministic, transparent explanation from the demo linear models."""
        from synk.explainability.text_attribution import extract_text_evidence
        from synk.features.tabular import build_tabular_features

        feats = build_tabular_features(sample_set)
        row = feats.iloc[index]
        X = feats.drop(columns=["encounter_id", "t"]).to_numpy(dtype=np.float32)[index: index + 1]
        p = float(self.meta.predict_proba(
            np.column_stack([
                self.structured.predict_proba(X)[:, 1],
                self.text_model.predict_proba([sample_set.samples.iloc[index]["text"] or "no clinical note available"])[:, 1],
            ])
        )[0, 1])

        names = [c for c in feats.columns if c not in ("encounter_id", "t")]
        coefs = self.structured.named_steps["clf"].coef_[0]
        contributions = sorted(zip(names, (coefs * X[0]).tolist()), key=lambda kv: abs(kv[1]), reverse=True)[:8]
        text = str(sample_set.samples.iloc[index]["text"] or "")
        evidence = extract_text_evidence(text)
        return {
            "engine": "sklearn_demo",
            "risk_probability": p,
            "structured_attribution": {
                "method": "linear_coefficient_x_value",
                "variables": [{"name": n, "score": float(s)} for n, s in contributions],
            },
            "text_evidence": [{"text": text, "evidence": evidence}] if text else [],
            "uncertainty": {
                "method": "none", "available": False,
                "note": "The demo logistic model has no uncertainty estimate; train the torch model to enable MC dropout.",
            },
            "research_disclaimer": RESEARCH_DISCLAIMER,
        }


# ---------------------------------------------------------------------------
# Unified facade
# ---------------------------------------------------------------------------
class InferenceService:
    """Chooses the best available engine and exposes a stable API."""

    def __init__(self, config: Optional[SYNKConfig] = None, bundle: Optional[ModelBundle] = None) -> None:
        self.config = config or load_config()
        self.bundle = bundle if bundle is not None else load_bundle(config=self.config)
        self.torch_engine: Optional[TorchEngine] = None
        self.demo_engine: Optional[DemoEngine] = None

    # -- lifecycle ---------------------------------------------------------
    def ensure_ready(self, observations, notes, outcomes, patients) -> str:
        """Fit the demo fallback (if no trained bundle) and return engine id."""
        if self.bundle is not None and self.bundle.model is not None:
            featurizer = EncounterFeaturizer.from_bundle(self.bundle, self.config)
            featurizer.fit_if_needed(observations)
            self.torch_engine = TorchEngine(self.bundle, featurizer)
            self.featurizer = featurizer
            return self.torch_engine.engine_id

        # Demo path: fit transparent sklearn models on the supplied data.
        from synk.models.baselines import (
            train_late_fusion_baseline,
            train_structured_baseline,
            train_text_baseline,
        )

        self.featurizer = EncounterFeaturizer(self.config)
        self.featurizer.fit_if_needed(observations)
        sample_set, _ = self.featurizer.build(observations, notes, outcomes, patients)
        splits = self._demo_splits(sample_set)
        try:
            struct = train_structured_baseline(sample_set, splits["train"], splits["val"], self.config.seed)
            text = train_text_baseline(sample_set, splits["train"], splits["val"], self.config.seed)
            meta = train_late_fusion_baseline(struct, text, self.config.seed)
        except ValueError as exc:
            # Degenerate cohort (e.g. one split empty): fit on ALL data. This is
            # the documented demo-only fallback - the research pipeline never
            # takes this path because it enforces strict splits.
            logger.warning("Demo baseline split training failed (%s); fitting on all data", exc)
            all_ids = sample_set.samples["encounter_id"].unique().tolist()
            struct = train_structured_baseline(sample_set, all_ids, all_ids, self.config.seed)
            text = train_text_baseline(sample_set, all_ids, all_ids, self.config.seed)
            meta = train_late_fusion_baseline(struct, text, self.config.seed)
        self.demo_engine = DemoEngine(self.config, struct["model"], text["model"], meta["model"])
        return self.demo_engine.engine_id

    @staticmethod
    def _demo_splits(sample_set: SampleSet) -> dict[str, list[str]]:
        """Deterministic patient-level split for the demo fallback."""
        enc = sample_set.samples.groupby(["encounter_id", "patient_id"]).size().reset_index()
        patients = sorted(enc["patient_id"].unique().tolist())
        n = len(patients)
        n_train = max(1, int(n * 0.6))
        n_val = max(1, int(n * 0.2)) if n > 2 else 0
        # Guarantee all three splits are non-empty when the cohort allows it.
        n_val = min(n_val, n - n_train) if n >= 3 else n_val
        train_p = set(patients[:n_train])
        val_p = set(patients[n_train:n_train + n_val])
        enc_to_pat = dict(zip(enc["encounter_id"], enc["patient_id"]))
        out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        for enc_id, pat in enc_to_pat.items():
            key = "train" if pat in train_p else "val" if pat in val_p else "test"
            out[key].append(enc_id)
        return out

    # -- prediction ---------------------------------------------------------
    @property
    def engine_id(self) -> str:
        if self.torch_engine is not None:
            return self.torch_engine.engine_id
        if self.demo_engine is not None:
            return self.demo_engine.engine_id
        return "uninitialised"

    @property
    def is_torch(self) -> bool:
        return self.torch_engine is not None

    def predict_encounters(self, observations, notes, outcomes, patients, encounter_ids=None):
        sample_set, _ = self.featurizer.build(observations, notes, outcomes, patients)
        if encounter_ids:
            mask = sample_set.samples["encounter_id"].isin(set(encounter_ids))
            sub = sample_set.samples[mask].reset_index(drop=True)
            idx = np.asarray(sub["row_index"]) if "row_index" in sub else None
        else:
            idx = None
        engine = self.torch_engine or self.demo_engine
        assert engine is not None, "call ensure_ready() first"
        frame = engine.predict_samples(sample_set)
        if encounter_ids:
            frame = frame[frame["encounter_id"].isin(set(encounter_ids))].reset_index(drop=True)
        frame["engine"] = self.engine_id
        frame["research_disclaimer"] = RESEARCH_DISCLAIMER
        return frame

    def explain_sample(self, sample_set: SampleSet, index: int) -> dict:
        engine = self.torch_engine or self.demo_engine
        assert engine is not None
        return engine.explain_one(sample_set, index)

    def model_info(self) -> dict[str, Any]:
        if self.bundle is not None:
            m = self.bundle.manifest
            return {
                "engine": self.engine_id,
                "model_name": self.bundle.model_name,
                "model_version": self.bundle.model_version,
                "modality": m.get("config", {}).get("model", {}).get("modality", "unknown"),
                "created_at": m.get("created_at"),
                "dataset_id": m.get("dataset_id"),
                "git_commit": m.get("git_commit"),
                "weights_checksum": m.get("weights_checksum"),
                "metrics": m.get("metrics", {}),
                "calibration": self.bundle.calibrator.method if self.bundle.calibrator else "none",
            }
        return {
            "engine": self.engine_id,
            "model_name": "demo_baseline",
            "model_version": "demo_v0",
            "note": "No trained SYNK bundle found; serving the transparent sklearn demo baseline.",
        }
