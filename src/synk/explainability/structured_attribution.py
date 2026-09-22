"""Structured-modality attribution.

Computes per-variable, per-timestep attributions for the temporal branch using
*gradient-based* methods (Captum Integrated Gradients and GradientShap when
available, falling back to a plain gradient-times-input implementation).  The
result is aggregated to per-variable importance over the observation window.

These attributions are model evidence, not clinical recommendations: they say
which inputs moved the output, not what the clinician should do.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from captum.attr import GradientShap, IntegratedGradients

    CAPTUM_AVAILABLE = True
except Exception:  # pragma: no cover
    CAPTUM_AVAILABLE = False


def _struct_forward(struct_model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor, static: torch.Tensor, texts: list[str]):
    """Wrap a SYNKModel forward so Captum sees one differentiable input."""

    def fwd(x_in: torch.Tensor) -> torch.Tensor:
        out = struct_model(x_in, mask, static, texts)
        return out["logit"]

    return fwd


def _tfidf_text_embedding(model: torch.nn.Module, texts: list[str], device: torch.device) -> torch.Tensor:
    """Non-differentiable text embedding for tfidf-backed models (attribution only)."""
    encoder = getattr(model, "text_encoder", None)
    return encoder.transform(texts, device)  # type: ignore[attr-defined]


def attribute_structured(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    static: torch.Tensor,
    texts: list[str],
    variable_names: list[str],
    method: str = "integrated_gradients",
    n_steps: int = 16,
) -> dict:
    """Return per-timestep and per-variable attributions for one sample.

    x: (1, T, V) standardised input tensor; mask: (1, T) float; static: (1, S).
    Returns dict with:
      per_timestep: list of {t_index, score} (sum over variables)
      variables:    list of {name, score} sorted by |score| desc
      method:       attribution method actually used
    """
    device = x.device
    model = model.to(device).eval()
    # The tfidf backend's projection participates in the forward; its input
    # (the TF-IDF vector) must stay fixed for the text branch during x-only
    # attribution, which the standard forward already guarantees.
    fwd = _struct_forward(model, x, mask, static, texts)

    method = method.lower()
    attrs: np.ndarray | None = None
    used = method

    if CAPTUM_AVAILABLE and method == "integrated_gradients":
        try:
            ig = IntegratedGradients(fwd)
            att = ig.attribute(x, n_steps=n_steps)
            attrs = att.detach().cpu().numpy()[0]  # (T, V)
        except Exception as exc:  # pragma: no cover
            logger.warning("Captum IG attribution failed (%s); falling back to gradient x input", exc)
            attrs = None
            used = "gradient_x_input"
    elif CAPTUM_AVAILABLE and method == "gradient_shap":
        try:
            gs = GradientShap(fwd)
            baseline = torch.zeros_like(x)
            att = gs.attribute(x, baselines=baseline, n_samples=8, stdevs=0.01)
            attrs = att.detach().cpu().numpy()[0]
        except Exception as exc:  # pragma: no cover
            logger.warning("Captum GradientShap failed (%s); falling back to gradient x input", exc)
            attrs = None
            used = "gradient_x_input"

    if attrs is None:
        x_in = x.clone().requires_grad_(True)
        logit = fwd(x_in)
        grads = torch.autograd.grad(logit.sum(), x_in)[0]
        attrs = (grads * x_in).detach().cpu().numpy()[0]  # gradient x input
        used = "gradient_x_input"

    per_timestep = [
        {"t_index": int(i), "score": float(np.sum(attrs[i]))} for i in range(attrs.shape[0])
    ]
    var_scores = np.sum(np.abs(attrs), axis=0)  # (V,)
    variables = [
        {"name": variable_names[j], "score": float(var_scores[j])}
        for j in range(len(variable_names))
    ]
    variables.sort(key=lambda d: d["score"], reverse=True)
    return {
        "method": used,
        "per_timestep": per_timestep,
        "variables": variables,
        "attribution_matrix": attrs.tolist(),  # (T, V) raw, for UI heatmap
    }
