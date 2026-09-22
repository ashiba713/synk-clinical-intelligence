# SYNK Architecture

Research prototype — not for clinical diagnosis or treatment.

## Data flow

```
DATA SOURCES
    ├── Synthetic generator  (data/synthetic/*.csv; manifest labels it synthetic)
    ├── CSV adapter          (user-provided column-mapped exports)
    └── MIMIC-IV adapter     (interface for authorised local extracts only)
        │
        ▼
DATA VALIDATION          synk/data/validation.py
        │                schema, timestamps, duplicates, plausibility bounds
        ▼
PREPROCESSING            synk/preprocessing/vitals.py  (fit on TRAIN split only)
        │                hourly grid, MAP derivation, bounded ffill, median
        │                imputation, missingness indicators, winsorise, z-score
        ├──────────────────────────────┐
        ▼                              ▼
TEMPORAL ENCODER                CLINICAL NLP ENCODER
synk/models/temporal/           synk/models/nlp/
TFT-style: VSN + LSTM +         configurable HF transformer or
static gating + masked attn     offline TF-IDF backend
        │                              │
        └──────────► MULTIMODAL FUSION ◄┘
                     synk/models/fusion/
                     gated | concat | cross_attention
                              │
                    RISK PREDICTION HEAD
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
     EXPLAINABILITY                       UNCERTAINTY
     synk/explainability/                 calibration (validation-fit)
     IG/GradientShap, text evidence,      MC dropout
     fusion gates, trend summary
            └─────────────────┬─────────────────┘
                              ▼
                      CLINICAL REPORT       synk/reporting/
                              ▼
                  SYNK API (FastAPI)  ·  SYNK UI (Streamlit)
```

## Module reference

| Module | Responsibility |
| --- | --- |
| `synk/config/settings.py` | Typed pydantic settings; YAML merge + `SYNK_SECTION__KEY` env overrides; fail-fast validation |
| `synk/data/schemas.py` | Canonical observation/note/outcome/patient schemas; coercion helpers; configurable reference ranges (labelled as such) |
| `synk/data/synthetic.py` | Stochastic ICU cohort generator with six demo scenarios; manifest marked `is_synthetic` |
| `synk/data/validation.py` | Rule-based checks producing `ValidationReport` (errors vs warnings) |
| `synk/data/adapters/` | `csv_adapter.py` (column mapping) and `mimic_iv.py` (interface for authorised extracts) |
| `synk/labels/sepsis_labels.py` | Configurable label engine (`latent_outcomes` or `criteria`); hourly prediction-time enumeration with exclusions |
| `synk/preprocessing/vitals.py` | `VitalsPreprocessor` fit/transform; serialisable statistics; `__obs` indicators preserve missingness |
| `synk/preprocessing/text.py` | Note cleaning, optional de-identification, transparent NegEx-style negation scoping |
| `synk/features/dataset.py` | `SampleSet` construction + persistence; patient-level splitting; torch dataset/collate |
| `synk/features/tabular.py` | Flattened window features (last/mean/min/max/slope/missing-frac) for baselines |
| `synk/models/temporal/` | GRN, Variable Selection Network, temporal encoder (returns inspectable weights) |
| `synk/models/nlp/` | `TransformerTextBackend` / `TfidfTextBackend` behind one interface |
| `synk/models/fusion/` | `GatedFusion` (default), `ConcatFusion`, `CrossAttentionFusion` |
| `synk/models/losses.py` | Data-derived `pos_weight` BCE, focal loss, plain BCE |
| `synk/models/multimodal.py` | `SYNKModel` wiring + modality overrides (ablation) + `load_trained_model` |
| `synk/models/baselines.py` | sklearn structured/text/late-fusion baselines |
| `synk/training/trainer.py` | Seeded training loop: early stopping, scheduling, clipping, checkpoints (weights + text backend sidecar) |
| `synk/evaluation/metrics.py` | Discrimination, threshold metrics, calibration data, early-warning metrics |
| `synk/evaluation/calibration.py` | Platt / isotonic fitted on validation only; JSON-serialisable |
| `synk/evaluation/evaluator.py` | Full test-split reports + modality ablation |
| `synk/explainability/` | Structured attribution (Captum IG/GradientShap → gradient×input fallback), text token attribution, deterministic evidence extractor, MC-dropout uncertainty, trend summariser |
| `synk/inference/engine.py` | Bundle loading, `TorchEngine`, `DemoEngine` (transparent sklearn fallback), `InferenceService` facade |
| `synk/api/` | FastAPI app factory, routers, pydantic schemas, shared `AppState` |
| `synk/database/models.py` | SQLAlchemy entities + `Database` (SQLite demo / PostgreSQL-compatible) |
| `synk/reporting/report.py` | Research report structure + markdown/json/csv rendering |
| `synk/tracking/registry.py` | `RunTracker` (local manifest, optional MLflow), `ModelRegistry`, `ExperimentRegistry` |
| `synk/pipeline.py` | End-to-end orchestration with leak-safe ordering |
| `app/` | Streamlit design system (`theme.py`), cached data service, dashboard pages |

## Key invariants

1. **No leakage** — preprocessing statistics, loss weights, TF-IDF vocabularies and calibrators are
   fitted on training/validation data only; splits are patient-level and asserted disjoint.
2. **Nothing fabricated** — every displayed metric/probability/attributions comes from a real model
   run; missing capabilities are reported as unavailable with an explanation.
3. **Missingness is information** — observed-indicator channels flow through the whole temporal stack.
4. **Uncertainty ≠ probability** — MC-dropout intervals, calibration status and disclaimers accompany
   predictions; the UI never presents a probability as certainty.
5. **Synthetic is labelled** — data, API payloads and the UI all state when data is synthetic.
