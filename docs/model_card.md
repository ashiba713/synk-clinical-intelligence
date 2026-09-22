# SYNK Model Card

Research prototype — not for clinical diagnosis or treatment.

## Model details

- **Model**: SYNK multimodal risk model — TFT-style temporal encoder + configurable clinical-text
  encoder (HF transformer or TF-IDF backend) + gated fusion + risk head.
- **Versions**: registered under `artifacts/models/<synk_structured|synk_text|synk_both>/vN` with a
  manifest containing the resolved config, metrics, git commit and weights checksum.
- **Inputs**
  - Structured: `W`-hour (default 12 h) hourly matrix of standardised vitals/labs with
    observed-indicator channels; static age/sex/weight.
  - Text: notes within the lookback window (default 24 h), concatenated.
- **Output**: logit → probability of sepsis onset within the prediction horizon (default 6 h);
  category LOW/MODERATE/HIGH/CRITICAL via configurable thresholds (0.30/0.60/0.80); MC-dropout
  interval where the model has dropout.

## Intended use

- **Primary**: research into multimodal early-warning modelling; educational demonstration of an
  explainable clinical-AI pipeline.
- **Settings**: local research environments on synthetic or authorised, de-identified datasets.

## Out-of-scope use

- Any clinical decision-making, diagnosis, triage or treatment selection.
- Deployment against real patients or prospective data without full regulatory and ethical review.
- Treating thresholds, reference ranges or evidence phrases as validated clinical guidance.

## Training data

- Default: the SYNK synthetic generator (`synthetic-v1.2`) — stochastic physiological trajectories
  with a documented deterioration process and six labelled demo scenarios. Manifest carries
  `is_synthetic: true` and a banner string.
- Real datasets: supported through adapters for **user-provided, authorised** extracts (CSV mapping
  or MIMIC-IV-shaped tables). SYNK never downloads or redistributes restricted data.

## Performance

- Reported exclusively from stored evaluation reports (`artifacts/evaluation/*_test.json`) and the
  experiment registry. See the README table for an example synthetic-data run; these numbers
  demonstrate pipeline correctness only.

## Caveats

- Attributions are model evidence (what moved the output), not clinical recommendations or causes.
- The `criteria` label source uses a simplified dysfunction proxy, not a validated SOFA score.
- MC-dropout uncertainty is approximate; calibration is post-hoc on validation data.
- The TF-IDF text backend is a lightweight stand-in for the transformer encoder and is always
  labelled as the demo option.
