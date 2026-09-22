# SYNK Evaluation Definitions

Research prototype — not for clinical diagnosis or treatment.

All metrics are computed by `synk/evaluation/metrics.py` from `(y_true, y_prob)` pairs produced by an
actual model run. Nothing is hardcoded or simulated.

## Discrimination

| Metric | Definition | Notes |
| --- | --- | --- |
| AUROC | Area under the receiver-operating-characteristic curve | Percentile bootstrap CI (configurable n, default 200; 95%) |
| AUPRC | Average precision (area under precision–recall curve) | Primary metric under class imbalance |

## Threshold metrics

Computed at the configured threshold (`evaluation.threshold`, default 0.5) and at every operating
point in `sensitivity_at_operating_points`:

- Accuracy, Precision, Recall/Sensitivity, Specificity, F1
- Confusion counts: TP / FP / TN / FN (also exposed as a matrix)
- Alert rate (fraction of samples above threshold)

## Calibration

| Metric | Definition |
| --- | --- |
| Brier score | Mean squared difference between predicted probability and outcome |
| ECE | Equal-width 10-bin expected calibration error, Σ (n_b/N)·|conf_b − acc_b| |
| Reliability curve | Per-bin mean predicted probability vs observed frequency |

Post-hoc calibration: Platt scaling (logistic on the logit) or isotonic regression, fitted on
**validation** predictions only and shipped inside the model bundle.

## Early-warning metrics

Computed only when encounter-level outcomes are available (detection/lead-time values are never
invented):

- **Detection rate**: fraction of septic encounters with ≥1 alert before onset.
- **Lead time**: onset minus first pre-onset alert over detected encounters (mean/median/min/max).
- **False alerts per encounter**: alert-bearing non-septic encounters / non-septic encounters.
- Reported at the configured alert threshold, alongside `n_septic` / `n_non_septic` denominators.

## Modality ablation

The multimodal model is re-evaluated with one modality's embedding zeroed before fusion
(`structured_zeroed`, `text_zeroed`) vs `both_active`. The report attaches the caveat that this
measures contribution within the fusion model and does not replace independently trained unimodal
models.
