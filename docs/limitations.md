# SYNK Known Limitations & Future Work

Research prototype — not for clinical diagnosis or treatment.

## Implementation limitations (transparent fallbacks, not hidden gaps)

| Area | Limitation | Mitigation |
| --- | --- | --- |
| Label source `criteria` | Organ-dysfunction proxy is a configurable derangement count, **not** a validated SOFA score | Documented at every use site; `latent_outcomes` is the default |
| Text encoder (demo) | TF-IDF backend is a lightweight stand-in; no token-level attribution | Clearly labelled in UI/API/model card; transformer mode provides token attribution |
| Token attribution | Gradient × embedding-norm at the encoder output — measures token influence on the text representation, not on the fused logit | Stated in the payload `note` field |
| Uncertainty | MC dropout is approximate; unavailable for the sklearn demo engine (reported as unavailable, never faked) | Calibrator + calibration curves provided as complementary view |
| Ablation | Zeroing an embedding inside the fusion model ≠ independently trained unimodal | Explicit caveat attached to every ablation payload |
| Attention | Attention/gate values are contribution proxies, not explanations by themselves | Combined with gradient attribution and deterministic evidence |
| Reference ranges | Display bands are configurable charting defaults, not validated alert thresholds | Labelled in schema constants and UI captions |
| PDF export | Reports export markdown/JSON/CSV; PDF is left to the environment (browser print of the markdown) | Documented in `reporting/report.py` |

## Scientific limitations

- All shipped results are on **synthetic data**; they demonstrate pipeline correctness and nothing
  about clinical performance. The synthetic deterioration process is intentionally learnable, so
  absolute metric values are optimistic.
- A real study requires: authorised data access, researcher-supplied sepsis adjudication
  (e.g. Sepsis-3 with documented antibiotics/culture/SOFA logic), much larger cohorts, and
  external validation.
- The current evaluation uses a single split per run; repeated seeds and confidence intervals over
  seeds are future work (the seeding and registry infrastructure already supports it).
- Lead-time metrics depend on the label definition; comparing lead times across different label
  sources is invalid.

## Known issues

- First API/dashboard start fits the demo engine (a few seconds on the default cohort); subsequent
  starts reuse the trained bundle when present.
- `transformers`-based text encoders require downloading HF weights on first use — offline machines
  should stay on the `tfidf` backend (default).
- Windows console output uses CP1252 in some shells; Unicode glyphs in the dashboard degrade
  gracefully there.

## Future work

1. **Research**: multi-seed experiment orchestration, DeLong/bootstrap comparisons between
   multimodal and unimodal AUROCs, calibration-aware threshold selection on validation.
2. **Models**: native transformer temporal backbone, windowed cross-attention over per-note
   embeddings (currently notes are concatenated), conformal prediction intervals.
3. **Data**: a fully specified MIMIC-IV extraction cookbook (users provide credentialed extracts),
   chart-events level sampling frequencies.
4. **Product**: user-managed operating-point profiles, cohort-level drift monitoring on the
   System Status page, scheduled re-evaluation jobs.
