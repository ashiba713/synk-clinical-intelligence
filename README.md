# SYNK

**Clinical Intelligence Before Critical Moments.**

SYNK is an **explainable multimodal AI platform for early sepsis risk stratification** built on ICU
physiological time-series and clinical documentation. It is a *research prototype* for studying
multimodal fusion — not a medical device.

> ⚠ **Research prototype — not for clinical diagnosis or treatment.**
> All demo/serving data shipped with this repository is synthetic. Never present synthetic data as
> clinical evidence.

---

## Research question

> Can multimodal fusion of ICU physiological time-series data and clinical text improve early sepsis
> risk stratification compared with unimodal approaches?

**Hypothesis (under study, not proven):** a multimodal model combining temporal physiological
information with contextual clinical language may capture complementary evidence and provide improved
early risk stratification compared with either modality alone.

The repository operationalises this as a controlled experiment: **structured-only**, **text-only** and
**multimodal** models trained and evaluated on identical, leak-safe splits, with an append-only
experiment registry comparing them.

## What is implemented

| Area | Implementation |
| --- | --- |
| Temporal model | TFT-style encoder: per-variable embeddings, Variable Selection Network, LSTM, static gating, masked self-attention, attention pooling (PyTorch) |
| Text model | Configurable HF transformer (default `emilyalsentzer/Bio_ClinicalBERT`) **or** offline TF-IDF backend |
| Fusion | `gated` (default), `concat`, `cross_attention` — all independently testable |
| Labels | Configurable sepsis definition: latent onset timestamps or data-derived criteria; configurable horizon (default 6 h), stride, exclusions |
| Splitting | Patient-level, stratified, leakage-asserted (train/val/test) |
| Preprocessing | Fit-on-train-only: hourly aggregation, MAP derivation, bounded forward-fill, median imputation, missingness indicators, winsorisation, standardisation |
| Training | Seeded, checkpointed, early stopping, LR scheduling, gradient clipping, class-imbalance-aware loss (`bce_weighted` / `focal`), CPU/GPU auto |
| Evaluation | AUROC/AUPRC + bootstrap CIs, precision/recall/F1/sensitivity/specificity, confusion matrix, Brier, ECE, reliability curves, ROC/PR curves, sensitivity at operating points |
| Early warning | Detection rate, lead-time distribution, false alerts per encounter — computed from real predictions only |
| Calibration | Platt / isotonic fitted on **validation only**, shipped inside the model bundle |
| Explainability | Captum Integrated Gradients / GradientShap (gradient×input fallback), token attribution for transformer text encoders, deterministic negation-aware clinical text evidence, fusion-gate contribution, MC-dropout uncertainty |
| Inference | Versioned model registry + bundle loader (weights, config, text backend, calibrator, preprocessor); transparent sklearn demo fallback before first training |
| API | FastAPI: health, model info, patients, predict, predict/batch, explain, experiments, metrics, reports |
| Dashboard | Streamlit: Overview, Patients, Patient Analysis (timeline, charts, notes, evidence), Model Performance, Experiments, Explainability, Reports, System Status, About |
| Tracking | Local run manifests always; MLflow optional; W&B left as an extension point |
| Persistence | SQLAlchemy: patients, encounters, predictions, explanations, experiments, models, system_events (SQLite demo / PostgreSQL-compatible) |
| CI | GitHub Actions: tests on py3.10/3.11, pipeline smoke, API smoke, Docker build |

## Architecture

```
DATA SOURCES (synthetic generator | CSV adapter | MIMIC-IV extract*)
        │
        ▼
DATA INGESTION ──► DATA VALIDATION ──► PREPROCESSING (fit on train only)
        │                                     │
        ▼                                     ├──────────────┐
LABEL ENGINEERING                             ▼              ▼
(hourly prediction times)          TEMPORAL ENCODER   CLINICAL NLP ENCODER
        │                                     │              │
        └──────────────────► MULTIMODAL FUSION ◄─────────────┘
                                      │
                          RISK PREDICTION HEAD
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
             EXPLAINABILITY                       UNCERTAINTY
     (IG attribution, text evidence,        (MC dropout, calibration,
      fusion gates, trends)                  reliability curves)
                    └─────────────────┬─────────────────┘
                                      ▼
                            CLINICAL REPORT
                                      ▼
                          SYNK API  ·  SYNK STREAMLIT UI
```

See [`docs/architecture.md`](docs/architecture.md) for module-level detail and
[`docs/methodology.md`](docs/methodology.md) for the study design.

## Installation

Requires Python **3.10 / 3.11** (developed on 3.10; CPU-only torch works out of the box).

```bash
# 1. Create an environment (any manager)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install CPU torch first (smaller wheels), then everything else
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Install SYNK itself (editable)
pip install -e . --no-deps
```

Environment configuration lives in `.env.example` — copy to `.env` and adjust. Secrets are read from
environment variables only; nothing sensitive is ever committed.

## Quickstart

```bash
# Generate the synthetic demo cohort (clearly labelled synthetic)
python -m scripts.cli generate-data --n-patients 40

# Full research pipeline: baselines + structured/text/multimodal training + evaluation
python -m scripts.cli train --set training.max_epochs=8

# Predict one encounter with a full explanation payload
python -m scripts.cli predict --encounter SYN-E-00001 --explain
```

### Training command

```bash
python -m scripts.cli train                          # defaults from configs/base.yaml
python -m scripts.cli train --set training.lr=0.0005 --set model.fusion.type=cross_attention
python -m scripts.cli train --skip-training          # classical baselines only (fast)
```

Any config key can be overridden: `--set section.key=value`.

### Evaluation command

```bash
python -m scripts.cli evaluate --model-dir artifacts/models/synk_both/v1 --ablation
```

### Inference command

```bash
python -m scripts.cli predict --encounter SYN-E-00007 --explain
```

### API startup

```bash
python -m scripts.cli serve                          # http://localhost:8000  (docs at /docs)
# or
uvicorn synk.api.main:create_app --factory --port 8000
```

### Streamlit startup

```bash
python -m scripts.cli dashboard                      # http://localhost:8501
```

### Docker

```bash
docker compose up          # API on :8000, dashboard on :8501
docker compose run api python -m scripts.cli train --set training.max_epochs=3
```

### Tests

```bash
pytest tests/ -q            # 41 tests: config, data, pipeline, evaluation, explainability, API, reporting
```

## The multimodal model

Three components, each independently testable:

1. **Temporal encoder** (`models/temporal/temporal_encoder.py`) — each timestep encodes
   `[standardised value, observed indicator]` per variable; a Variable Selection Network produces soft
   per-timestep variable weights (inspectable — this is what global feature-importance views use); an
   LSTM encodes sequence dynamics; static covariates (age/sex/weight) enrich the sequence through a
   learned gate; masked multi-head self-attention and attention pooling produce the fixed-length
   structured representation. Missingness is never hidden: the observed-indicator channel is a
   first-class input.
2. **Text encoder** (`models/nlp/text_encoder.py`) — notes within a configurable lookback window
   (default 24 h) are joined and encoded either by a configurable Hugging Face clinical transformer
   (mean-pooled, token-level gradients exposed for attribution) or by an offline TF-IDF + learned
   projection backend for CPU/demo mode. The demo backend is a genuine encoder and is always labelled
   as such.
3. **Fusion** (`models/fusion/fusion.py`) — default *gated fusion*: each modality is projected and
   scaled by a sigmoid gate conditioned on both embeddings, so the model can learn when language
   corroborates or contradicts physiology. The gate values are exposed for explanation. `concat` and
   `cross_attention` (text queries the temporal sequence) are configurable alternatives.

The prediction head outputs a logit → probability; thresholds map to LOW / MODERATE / HIGH / CRITICAL
categories with **configurable** cut-points (defaults 0.30/0.60/0.80) that are study parameters, not
validated clinical thresholds.

## Research methodology (summary)

- **Task**: predict sepsis onset within a configurable horizon (default 6 h) from an observation
  window (default 12 h) of hourly-aggregated structured data plus notes in a 24 h text lookback.
- **Labels**: encounter-level sepsis onset timestamps (synthetic generator or researcher-supplied
  adjudication); an optional `criteria` source derives onset from suspected-infection + organ-dysfunction
  proxies with configurable thresholds. The dysfunction proxy is explicitly **not** a validated SOFA score.
- **Splitting**: patient-level stratified splits (70/15/15); any patient overlap between splits raises
  an assertion. Preprocessing statistics are fitted on the training encounters only.
- **Class imbalance**: `bce_weighted` (data-derived `pos_weight`) by default; focal loss available;
  PR-space metrics emphasised over accuracy.
- **Comparison protocol**: identical splits/features for structured-only, text-only and multimodal
  models; sklearn baselines included as transparent anchors; ablation (zeroing one modality inside the
  fusion model) reported alongside — and explicitly distinguished from — independently trained unimodals.

Full detail: [`docs/methodology.md`](docs/methodology.md). Evaluation definitions:
[`docs/evaluation.md`](docs/evaluation.md).

## Results on synthetic data (example run, CPU)

These are the numbers produced by an actual run of
`python -m scripts.cli train --set synthetic.n_patients=40 --set training.max_epochs=6` (seed 42) on
this machine, and are included only to demonstrate that the pipeline computes real metrics. They say
nothing about clinical performance; the synthetic deterioration process is intentionally learnable.

| Model | AUROC | AUPRC | F1 | Notes |
| --- | --- | --- | --- | --- |
| sklearn structured baseline | 0.991 | 0.768 | — | logistic regression on flattened features |
| sklearn text baseline | 0.981 | 0.497 | — | TF-IDF + logistic regression |
| sklearn late fusion | 0.990 | 0.707 | — | meta-LR over the two unimodal probabilities |
| SYNK structured | 0.991 | 0.882 | 0.813 | temporal encoder only |
| SYNK text | 0.954 | 0.456 | 0.516 | TF-IDF text encoder only |
| SYNK multimodal | 0.978 (CI 0.940–1.000) | 0.865 | 0.800 | gated fusion |

Ablation of the fusion model on the same test split: text zeroed → AUROC 0.887, structured zeroed →
0.905, both active → 0.978. Re-run the pipeline to regenerate your own table — nothing here is
hardcoded.

## Limitations

- **Research prototype.** No regulatory clearance, no clinical validation, no deployment approval.
- **Synthetic data by default.** Results characterise software correctness, never clinical utility.
- Labels (especially the `criteria` source) are simplified proxies; a real study requires
  researcher-supplied, documented sepsis adjudication (e.g. Sepsis-3) on authorised data.
- The TF-IDF text backend is a deliberately lightweight stand-in; transformer mode requires
  downloading the configured model weights.
- Token-level text attribution is only available for the transformer backend; the deterministic
  evidence extractor is rule-based and auditable but shallow.
- MIMIC-IV support is an adapter interface for user-provided extracts — SYNK never downloads or
  redistributes restricted data.
- Uncertainty is MC-dropout-based (approximate) and calibration is post-hoc; both are study tools,
  not guarantees.

## Repository layout

```
├── app/                    Streamlit dashboard (design system + pages)
├── configs/                base / training / inference YAML
├── docs/                   architecture, methodology, evaluation, model card, privacy…
├── scripts/                CLI entrypoints (generate-data, train, evaluate, predict, serve, dashboard)
├── src/synk/
│   ├── api/                FastAPI app, routers, schemas, state
│   ├── config/             typed settings + env overrides
│   ├── data/               schemas, validation, synthetic generator, CSV/MIMIC adapters
│   ├── database/           SQLAlchemy models + session helpers
│   ├── evaluation/         metrics, calibration, evaluator
│   ├── explainability/     structured attribution, text evidence/token attribution, explainer
│   ├── features/           sample construction, dataset/collate, tabular features
│   ├── inference/          bundle loading, torch + demo engines, service facade
│   ├── labels/             configurable sepsis label engine
│   ├── models/             temporal encoder, text encoder, fusion, losses, baselines
│   ├── preprocessing/      vitals (fit-on-train) + text preprocessing
│   ├── reporting/          research report builder (md/json/csv)
│   ├── tracking/           run manifests, model registry, experiment registry
│   ├── training/           unified trainer for all modalities
│   └── utils/              logging, seeding, device helpers
├── tests/                  pytest suite (41 tests)
├── Dockerfile / docker-compose.yml / .github/workflows/ci.yml
└── pyproject.toml / requirements.txt
```

## Documentation map

- [`docs/architecture.md`](docs/architecture.md) — module reference and data flow
- [`docs/methodology.md`](docs/methodology.md) — study design, leakage prevention, label definitions
- [`docs/evaluation.md`](docs/evaluation.md) — metric definitions and early-warning metrics
- [`docs/model_card.md`](docs/model_card.md) — model card (intended use, out-of-scope use, caveats)
- [`docs/privacy.md`](docs/privacy.md) — privacy and data-handling requirements
- [`docs/deployment.md`](docs/deployment.md) — Docker/MLflow operational notes

## Citation / status

Final-year engineering research project. The implementation is the foundation for a future paper;
all experimental claims must be backed by runs of the pipeline in this repository.
