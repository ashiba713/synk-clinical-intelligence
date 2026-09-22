# SYNK Research Methodology

Research prototype — not for clinical diagnosis or treatment.

This document defines the study design implemented by the pipeline. It is deliberately explicit so
that a future publication can state exactly what was done.

## 1. Prediction task

For each encounter and each eligible prediction time `t` (hourly grid, stride configurable, default
2 h), SYNK predicts:

> the probability that sepsis onset occurs in the interval `(t, t + H]`, where `H` is the prediction
> horizon (default 6 h, configurable; the conceptual target is 6–12 h early warning).

- **Observation window**: the `W` hours ending at `t` (default 12 h) of structured data.
- **Text lookback**: notes in the `L` hours ending at `t` (default 24 h, latest `max_notes` joined).
- **Exclusion**: when `exclude_after_onset` is set (default), prediction times at or after a known
  onset are excluded, so the model never trains on post-onset physiology as a negative.
- **Minimum history**: `t` must be at least `min_history_hours` (default 8 h) after stay start.

## 2. Label definition (configurable by design)

SYNK deliberately does **not** invent a clinical gold standard. Two configurable label sources exist:

| Source | Meaning | Caveat |
| --- | --- | --- |
| `latent_outcomes` (default) | Uses per-encounter `sepsis_onset_timestamp` supplied by the dataset (synthetic generator, or the researcher's own adjudication script, e.g. Sepsis-3). | Quality depends entirely on the supplied onsets. |
| `criteria` | Derives an onset estimate: suspected infection (antibiotic + culture within configurable windows) **and** an acute organ-dysfunction proxy (configurable derangement count vs the encounter baseline). | The dysfunction proxy is **not** a validated SOFA score; results using it must be framed accordingly. |

Documented per run in the manifest: source, horizon, stride, exclusion rule, thresholds.

## 3. Data splitting and leakage prevention

- **Patient-level splits** (70/15/15 by default, stratified by encounter-level sepsis status). Any
  patient appearing in two splits raises an `AssertionError` — the invariant is tested, not assumed.
- **Preprocessing leakage**: `VitalsPreprocessor.fit` sees training encounters only; winsorisation
  bounds, medians and z-score statistics are frozen and applied to val/test. The fitted state ships
  inside the model bundle so inference reproduces training-time mapping exactly.
- **Text leakage**: TF-IDF vocabularies are fitted on training notes only (both deep and sklearn
  text models).
- **Loss weighting leakage**: `pos_weight` is computed from training labels only.
- **Calibration leakage**: calibrators (Platt/isotonic) are fitted on *validation* predictions and
  stored in the bundle; the test split is never used for any fitting.
- **Temporal leakage**: event flags (antibiotics/culture) used by the `criteria` label engine are
  excluded from model inputs; every sample's features end at `t`.

## 4. Missing data

- Vitals missing at random + labs on realistic sparse schedules (synthetic generator mirrors ICU
  measurement patterns).
- Hourly aggregation averages duplicate measurements; bounded forward-fill (configurable limit)
  then median imputation; every value carries an observed-indicator channel consumed by the model,
  so imputation never fabricates certainty.
- Impossibility filters (e.g. heart rate 9999) are caught by validation, not silently dropped.

## 5. Class imbalance

- Default loss `bce_weighted` with data-derived `pos_weight = n_neg/n_pos` (clipped 0.2–20).
- `focal` loss (documented gamma) and unweighted `bce` available as controls.
- Threshold-dependent metrics are reported across operating points; precision–recall space is
  treated as primary for imbalanced evaluation. Accuracy alone is never used to claim performance.

## 6. Comparison protocol (the research experiment)

All models share identical splits, sample construction and evaluation code:

1. **Deep SYNK** — `model.modality ∈ {structured, text, both}` trained with identical settings.
2. **Sklearn baselines** — structured LR, text TF-IDF LR, late fusion (meta-LR on validation
   predictions of the two unimodal models; the meta-learner never sees test data).
3. **Ablation** — the multimodal model re-evaluated with each modality's embedding zeroed before
   fusion. This measures within-model contribution and is explicitly **not** a substitute for
   independently trained unimodals (the evaluator attaches this caveat to every ablation result).

Reported per model: AUROC (+bootstrap CI), AUPRC (+CI), F1, sensitivity, specificity, Brier,
detection rate and median lead time where outcomes support them. All rows land in the append-only
experiment registry consumed by the API/UI.

## 7. Statistical practice

- Bootstrap percentile CIs (configurable `n`, default 200; CI 95%) for AUROC/AUPRC.
- Calibration: reliability curves, Brier, ECE (equal-width bins); Platt/isotonic as post-hoc
  calibrators fitted on validation only.
- Uncertainty: MC-dropout with configurable sample counts; reported as an approximate interval,
  never as certainty.
