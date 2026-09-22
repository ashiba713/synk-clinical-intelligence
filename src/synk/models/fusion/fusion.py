"""Multimodal fusion modules.

Three configurable fusion mechanisms with a common interface:

``concat``            project the concatenated embeddings (simplest ablation)
``gated``             learned per-modality gates conditioned on both inputs
``cross_attention``   the text embedding queries the temporal sequence so the
                      model can attend to the hours corroborated by language

Each module returns the fused representation plus an ``aux`` dict with
inspectable quantities (gate values / attention weights) for explainability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(self, struct_dim: int, text_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(struct_dim + text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(self, s: torch.Tensor, t: torch.Tensor, seq: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        fused = self.proj(torch.cat([s, t], dim=-1))
        return fused, {}


class GatedFusion(nn.Module):
    """z = [g_s * W_s(s), g_t * W_t(t)] with g_* = sigmoid(W([s; t]))."""

    def __init__(self, struct_dim: int, text_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.s_proj = nn.Linear(struct_dim, hidden_dim)
        self.t_proj = nn.Linear(text_dim, hidden_dim)
        self.s_gate = nn.Linear(struct_dim + text_dim, hidden_dim)
        self.t_gate = nn.Linear(struct_dim + text_dim, hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(self, s: torch.Tensor, t: torch.Tensor, seq: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        gs = torch.sigmoid(self.s_gate(torch.cat([s, t], dim=-1)))
        gt = torch.sigmoid(self.t_gate(torch.cat([s, t], dim=-1)))
        zs = gs * self.s_proj(s)
        zt = gt * self.t_proj(t)
        fused = self.out_proj(torch.cat([zs, zt], dim=-1))
        return fused, {"gate_structured": gs.mean().detach(), "gate_text": gt.mean().detach(),
                       "gates": torch.stack([gs, gt], dim=-1)}


class CrossAttentionFusion(nn.Module):
    """Text queries the temporal sequence; both pooled vectors are appended."""

    def __init__(self, struct_dim: int, text_dim: int, hidden_dim: int, dropout: float, heads: int = 4) -> None:
        super().__init__()
        self.s_proj = nn.Linear(struct_dim, hidden_dim)
        self.t_proj = nn.Linear(text_dim, hidden_dim)
        self.query_proj = nn.Linear(text_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(self, s: torch.Tensor, t: torch.Tensor, seq: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        if seq is None:
            # Fall back to attending over the single pooled structured vector.
            seq = s.unsqueeze(1)
        query = self.query_proj(t).unsqueeze(1)              # [B, 1, H]
        attn_out, attn_weights = self.attn(query, seq, seq, average_attn_weights=True)
        context = self.norm(attn_out.squeeze(1))             # [B, H]
        fused = self.out_proj(torch.cat([self.s_proj(s), self.t_proj(t), context], dim=-1))
        return fused, {"temporal_attention": attn_weights.squeeze(1).detach()}  # [B, T]


def build_fusion(config, struct_dim: int, text_dim: int) -> nn.Module:
    fcfg = config.model.fusion
    kinds = {
        "concat": ConcatFusion,
        "gated": GatedFusion,
        "cross_attention": CrossAttentionFusion,
    }
    cls = kinds[fcfg.type]
    kwargs = {}
    if fcfg.type == "cross_attention":
        kwargs["heads"] = config.model.temporal.attention_heads
    return cls(struct_dim, text_dim, fcfg.hidden_dim, fcfg.dropout, **kwargs)
