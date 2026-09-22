"""Temporal encoder for structured ICU time-series.

A Temporal Fusion Transformer (TFT)-style encoder, implemented directly on
PyTorch so that variable selection weights and attention remain inspectable
for explainability.  Components:

* per-variable input embeddings (value + observed indicator),
* a Variable Selection Network (soft per-timestep feature weighting),
* an LSTM sequence encoder,
* static-covariate enrichment via gating,
* masked multi-head temporal self-attention,
* attention pooling into a fixed-length representation.

Rationale for not using ``pytorch_forecing.TimeSeriesDataSet``: its fixed
dataset format is oriented to single-target forecasting with its own
imputation/scaling, whereas SYNK needs variable-length multimodal batches and
leak-safe preprocessing fitted outside the model.  The architecture here
follows TFT principles (Lim et al., 2021) while keeping every stage
independently testable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.proj(x))


class GatedResidualNetwork(nn.Module):
    """GRN(x) = LayerNorm(x_proj + GLU(Linear2(ELU(Linear1(x + ctx)))))"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float, context_dim: int = 0) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.context_proj = nn.Linear(context_dim, output_dim) if context_dim > 0 else None
        self.fc1 = nn.Linear(output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.glu = GatedLinearUnit(output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.input_proj(x)
        enriched = residual
        if self.context_proj is not None and context is not None:
            enriched = enriched + self.context_proj(context)
        hidden = F.elu(self.fc1(enriched))
        out = self.dropout(self.glu(self.fc2(hidden)))
        return self.layer_norm(residual + out)


class VariableSelectionNetwork(nn.Module):
    """Soft variable selection over per-timestep variable embeddings."""

    def __init__(self, n_vars: int, var_input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.var_grns = nn.ModuleList(
            [GatedResidualNetwork(var_input_dim, hidden_dim, hidden_dim, dropout) for _ in range(n_vars)]
        )
        self.weight_grn = GatedResidualNetwork(n_vars * hidden_dim, hidden_dim, n_vars, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, V, var_dim] -> (selected [B, T, H], weights [B, T, V])."""
        var_outputs = [grn(x[:, :, v]) for v, grn in enumerate(self.var_grns)]
        stacked = torch.stack(var_outputs, dim=-2)              # [B, T, V, H]
        flat = stacked.flatten(start_dim=-2)                    # [B, T, V*H]
        weights = torch.softmax(self.weight_grn(flat), dim=-1)  # [B, T, V]
        selected = (stacked * weights.unsqueeze(-1)).sum(dim=-2)  # [B, T, H]
        return selected, weights


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        n_variables: int,
        n_static: int,
        hidden_dim: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.15,
        attention_heads: int = 4,
        use_static: bool = True,
    ) -> None:
        super().__init__()
        self.n_variables = n_variables
        self.use_static = use_static
        var_input_dim = 2  # standardised value + observed indicator
        self.vsn = VariableSelectionNetwork(n_variables, var_input_dim, hidden_dim, dropout)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=0.0)
        if use_static and n_static > 0:
            self.static_grn = GatedResidualNetwork(n_static, hidden_dim, hidden_dim, dropout)
            self.static_gate = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.static_grn = None
            self.static_gate = None
        self.self_attn = nn.MultiheadAttention(hidden_dim, attention_heads, batch_first=True, dropout=dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.out_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.pool_scores = nn.Linear(hidden_dim, 1)
        self.output_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,        # [B, T, V] standardised values (0 where missing)
        mask: torch.Tensor,     # [B, T, V] observed indicators
        static: torch.Tensor,   # [B, S]
    ) -> dict:
        batch, steps, _ = x.shape
        # Per-variable input embedding: [value, indicator] -> hidden.
        var_input = torch.stack([x, mask], dim=-1)              # [B, T, V, 2]
        selected, vsn_weights = self.vsn(var_input)             # [B, T, H], [B, T, V]

        seq_out, _ = self.lstm(selected)                        # [B, T, H]

        static_ctx = None
        if self.static_grn is not None:
            static_ctx = self.static_grn(static)                # [B, H]
            gate = torch.sigmoid(self.static_gate(static_ctx)).unsqueeze(1)  # [B, 1, H]
            seq_out = seq_out * (1.0 + gate)                    # static enrichment by gating

        attn_out, _ = self.self_attn(seq_out, seq_out, seq_out, need_weights=False)
        seq_out = self.attn_norm(seq_out + attn_out)
        seq_out = self.out_grn(seq_out)

        # Masked attention pooling over time steps. Steps that are entirely
        # unobserved (all indicators zero) receive zero weight.
        step_valid = (mask.sum(dim=-1) > 0).float()             # [B, T]
        scores = self.pool_scores(seq_out).squeeze(-1)          # [B, T]
        scores = scores.masked_fill(step_valid == 0, float("-inf"))
        pool_weights = torch.softmax(scores, dim=-1)
        pooled = (seq_out * pool_weights.unsqueeze(-1)).sum(dim=1)  # [B, H]

        # All-masked edge case (empty window): fall back to zero vector.
        degenerate = (step_valid.sum(dim=1, keepdim=True) == 0)
        pooled = torch.where(degenerate, torch.zeros_like(pooled), pooled)

        return {
            "sequence": seq_out,           # [B, T, H]
            "pooled": pooled,              # [B, H]
            "vsn_weights": vsn_weights,    # [B, T, V]
            "pool_weights": pool_weights,  # [B, T]
            "static_context": static_ctx,
        }
