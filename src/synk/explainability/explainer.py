"""SYNK explainer: turns model internals into a clinician-readable explanation.

Combines:
* structured-variable attribution (Integrated Gradients / gradient x input)
* temporal trend summarisation of the observation window
* deterministic clinical-text evidence extraction
* optional token-level attribution for transformer text encoders
* MC-Dropout uncertainty estimation
* fusion-gate inspection (how much each modality contributed)

Every element is derived from the actual model run on the actual patient
window - no fabricated evidence, no templated recommendations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch

from synk.explainability.structured_attribution import attribute_structured
from synk.explainability.text_attribution import attribute_text_tokens, extract_text_evidence

logger = logging.getLogger(__name__)


def mc_dropout_uncertainty(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    static: torch.Tensor,
    texts: list[str],
    n_samples: int = 30,
    min_samples: int = 5,
    device: Optional[torch.device] = None,
) -> dict:
    """Monte-Carlo Dropout uncertainty: enable dropout at inference, sample."""
    from synk.utils.torch_utils import get_device

    device = device or get_device()
    if not model.has_dropout():
        return {
            "method": "mc_dropout",
            "available": False,
            "note": "Model contains no dropout layers; uncertainty not estimated.",
        }
    if n_samples < min_samples:
        return {
            "method": "mc_dropout",
            "available": False,
            "note": f"MC dropout disabled: {n_samples} < {min_samples} samples configured.",
        }

    was_training = model.training
    model.train()  # enable dropout; BN stats unaffected (no BatchNorm in model)
    model.eval()  # base state
    model.train()
    # Note: model.train() re-enables dropout; LayerNorm is training-invariant.
    probs: list[float] = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x.to(device), mask.to(device), static.to(device), texts)
            probs.append(float(out["probability"].item()))
    if was_training:
        model.train()
    else:
        model.eval()

    arr = np.asarray(probs)
    return {
        "method": "mc_dropout",
        "available": True,
        "n_samples": int(n_samples),
        "mean_probability": float(arr.mean()),
        "std_deviation": float(arr.std()),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
        "samples": [float(p) for p in probs],
    }


def summarise_trends(
    series: np.ndarray,
    mask: np.ndarray,
    variable_names: list[str],
    timestamps: Optional[list] = None,
) -> list[dict]:
    """Human-readable trend summaries per variable over the window."""
    out: list[dict] = []
    T, V = series.shape
    for j, name in enumerate(variable_names):
        vals = series[:, j]
        observed = mask[:, j] if mask.ndim == 2 else mask
        observed = np.asarray(observed).astype(bool)
        if observed.sum() < 2:
            out.append({
                "variable": name,
                "status": "insufficient_data",
                "n_observed": int(observed.sum()),
            })
            continue
        v = vals[observed]
        first, last = float(v[0]), float(v[-1])
        delta = last - first
        if len(v) >= 3:
            slope = float(np.polyfit(np.arange(len(v)), v, 1)[0])
        else:
            slope = 0.0
        direction = "increasing" if slope > 0.05 else "decreasing" if slope < -0.05 else "stable"
        out.append({
            "variable": name,
            "status": "ok",
            "n_observed": int(observed.sum()),
            "first": first,
            "last": last,
            "delta": delta,
            "slope_per_hour": slope,
            "direction": direction,
        })
    return out


def explain_prediction(
    model: torch.nn.Module,
    batch: dict,
    variable_names: list[str],
    config,
    device: Optional[torch.device] = None,
    attribution_method: str = "integrated_gradients",
    include_token_attribution: bool = False,
    tokenizer=None,  # unused; kept for API compatibility
) -> dict:
    """Full explanation payload for one (1, ...) batch."""
    from synk.utils.torch_utils import get_device

    device = device or get_device()
    x, mask = batch["x"].to(device), batch["mask"].to(device)
    static, texts = batch["static"].to(device), batch["text"]

    out = model(x, mask, static, texts)
    prob = float(out["probability"].item())
    aux = out.get("aux", {})

    explanation: dict[str, Any] = {
        "risk_probability": prob,
        "research_disclaimer": (
            "Research prototype - not for clinical diagnosis or treatment. "
            "Attributions are model evidence, not clinical recommendations."
        ),
    }

    # ---- structured attribution -----------------------------------------
    explanation["structured_attribution"] = attribute_structured(
        model, x, mask, static, texts, variable_names,
        method=attribution_method,
    )

    # ---- temporal trend summary ------------------------------------------
    x_np = x.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    if x_np.ndim == 3:      # (B, T, V) -> first sample (T, V)
        x_np = x_np[0]
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    if mask_np.ndim == 1:   # (T,) -> repeat across variables
        mask_np = np.repeat(mask_np[:, None], x_np.shape[-1], axis=1)
    explanation["trend_summary"] = summarise_trends(x_np, mask_np, variable_names)

    # ---- text evidence ----------------------------------------------------
    note_evidence = []
    for note_text in texts:
        if note_text and note_text.strip():
            note_evidence.append({
                "text": note_text,
                "evidence": extract_text_evidence(note_text),
            })
    explanation["text_evidence"] = note_evidence
    if include_token_attribution and texts:
        explanation["token_attribution"] = attribute_text_tokens(model, texts[0], device=device)

    # ---- fusion gate ------------------------------------------------------
    gate = aux.get("gates") if aux else None
    if gate is not None and torch.is_tensor(gate) and gate.dim() >= 2 and gate.shape[-1] >= 2:
        g = gate[..., 0].mean().item(), gate[..., 1].mean().item()
        total = float(g[0] + g[1]) or 1.0
        explanation["fusion_contribution"] = {
            "structured": float(g[0] / total),
            "text": float(g[1] / total),
            "note": "Mean gated-fusion weights for this sample; contribution proxy, not causation.",
        }

    # ---- uncertainty -------------------------------------------------------
    explanation["uncertainty"] = mc_dropout_uncertainty(
        model, x, mask, static, texts,
        n_samples=config.uncertainty.mc_dropout_samples,
        min_samples=config.uncertainty.mc_dropout_min_samples,
        device=device,
    )

    return explanation
