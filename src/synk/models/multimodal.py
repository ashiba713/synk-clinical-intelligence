"""The SYNK multimodal model.

Wires together the temporal encoder, text encoder, fusion layer and risk
prediction head.  ``model.modality`` selects the trained configuration:

* ``both``       - full multimodal model
* ``structured`` - structured-only ablation (text branch bypassed)
* ``text``       - text-only ablation (temporal branch bypassed)

At inference time a *multimodal* model can additionally be run with
``modality_override`` to quantify each modality's contribution by ablation
(zeroing one embedding before fusion).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from synk.models.fusion.fusion import build_fusion
from synk.models.nlp.text_encoder import TextEncoder
from synk.models.temporal.temporal_encoder import TemporalEncoder


def load_trained_model(
    checkpoint_path: str | Path,
    config,
    text_backend_path: str | Path | None = None,
) -> tuple["SYNKModel", dict]:
    """Rebuild a SYNKModel from a checkpoint and load its weights.

    For TF-IDF text encoders the saved vectoriser/projection is restored first
    so the state dict keys match.  Returns (model, checkpoint_payload).
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variables = payload.get("variables", [])
    model = SYNKModel(config, n_variables=len(variables), n_static=int(payload.get("n_static", 0)))

    if config.model.text.encoder_type == "tfidf":
        sidecar = Path(text_backend_path) if text_backend_path else Path(checkpoint_path).with_suffix(".text.pkl")
        if sidecar.exists():
            from synk.models.nlp.text_encoder import TfidfTextBackend

            model.text_encoder.backend = TfidfTextBackend.load(sidecar)
            model.text_encoder._output_dim = model.text_encoder.backend.output_dim()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _projection_block(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class RiskHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logit


class SYNKModel(nn.Module):
    def __init__(self, config, n_variables: int, n_static: int) -> None:
        super().__init__()
        self.config = config
        self.modality = config.model.modality
        tcfg = config.model.temporal
        fcfg = config.model.fusion

        self.text_encoder = TextEncoder(config)
        text_dim = self.text_encoder.output_dim

        if self.modality in ("both", "structured"):
            self.temporal: TemporalEncoder | None = TemporalEncoder(
                n_variables=n_variables,
                n_static=n_static,
                hidden_dim=tcfg.hidden_dim,
                lstm_layers=tcfg.lstm_layers,
                dropout=tcfg.dropout,
                attention_heads=tcfg.attention_heads,
                use_static=tcfg.use_static,
            )
            struct_dim = self.temporal.output_dim
        else:
            self.temporal = None
            struct_dim = 0

        if self.modality == "both":
            self.fusion = build_fusion(config, struct_dim, text_dim)
            fused_dim = self.fusion.output_dim
            self.struct_direct = None
            self.text_direct = None
        elif self.modality == "structured":
            self.fusion = None
            self.struct_direct = _projection_block(struct_dim, fcfg.hidden_dim, fcfg.dropout)
            self.text_direct = None
            fused_dim = fcfg.hidden_dim
        else:  # text-only
            self.fusion = None
            self.struct_direct = None
            self.text_direct = _projection_block(text_dim, fcfg.hidden_dim, fcfg.dropout)
            fused_dim = fcfg.hidden_dim

        self.head = RiskHead(fused_dim, config.model.head_hidden, fcfg.dropout)

    # ------------------------------------------------------------------
    def encode_structured(self, x: torch.Tensor, mask: torch.Tensor, static: torch.Tensor) -> dict:
        assert self.temporal is not None, "Model has no temporal branch (text-only modality)"
        # Broadcast mask/static when x was batch-expanded (Captum attribution).
        if x.shape[0] != mask.shape[0]:
            mask = mask.expand(x.shape[0], *mask.shape[1:])
        if x.shape[0] != static.shape[0]:
            static = static.expand(x.shape[0], *static.shape[1:])
        return self.temporal(x, mask, static)

    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        return self.text_encoder.forward(texts, device)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor,
        texts: list[str],
        modality_override: str | None = None,
        structured_sequence: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass.

        ``modality_override`` lets a *multimodal* model be evaluated with one
        modality zeroed out (contribution analysis).  ``text_embedding`` may
        be precomputed (and made differentiable) for text attribution.
        """
        device = x.device
        mode = modality_override or self.modality
        aux: dict = {}

        # Attribution libraries (Captum) expand the perturbed input to an
        # internal batch while the remaining inputs keep batch size 1.  A
        # single list[str] must therefore be fanned out to match the input
        # batch, and mask/static are broadcast in the temporal encoder call.
        expanded_texts = texts
        if texts is not None and len(texts) != x.shape[0]:
            expanded_texts = [texts[0]] * x.shape[0]

        if mode == "both":
            assert self.temporal is not None and self.fusion is not None
            s_out = self.encode_structured(x, mask, static)
            struct_pooled = s_out["pooled"]
            struct_seq = structured_sequence if structured_sequence is not None else s_out["sequence"]
            if text_embedding is None:
                text_embedding = self.encode_text(expanded_texts, device)
            if text_embedding.shape[0] != x.shape[0] and text_embedding.shape[0] == 1:
                text_embedding = text_embedding.expand(x.shape[0], -1)
            fused, faux = self.fusion(struct_pooled, text_embedding, struct_seq)
            aux.update(faux)
            aux.update({"vsn_weights": s_out["vsn_weights"], "pool_weights": s_out["pool_weights"]})
        elif mode in ("structured", "text") and self.fusion is not None:
            # Ablation path for the multimodal model: zero the disabled
            # modality's embedding BEFORE fusion (same weights, same head).
            assert self.temporal is not None
            s_out = self.encode_structured(x, mask, static)
            struct_pooled = s_out["pooled"]
            struct_seq = structured_sequence if structured_sequence is not None else s_out["sequence"]
            if text_embedding is None:
                text_embedding = self.encode_text(expanded_texts, device)
            if text_embedding.shape[0] != x.shape[0] and text_embedding.shape[0] == 1:
                text_embedding = text_embedding.expand(x.shape[0], -1)
            if mode == "structured":
                text_embedding = torch.zeros_like(text_embedding)
            else:
                struct_pooled = torch.zeros_like(struct_pooled)
                struct_seq = torch.zeros_like(struct_seq)
            fused, faux = self.fusion(struct_pooled, text_embedding, struct_seq)
            aux.update(faux)
            aux.update({"vsn_weights": s_out["vsn_weights"], "pool_weights": s_out["pool_weights"]})
            aux["ablation_zeroed"] = mode
        elif mode == "structured":
            assert self.temporal is not None and self.struct_direct is not None
            s_out = self.encode_structured(x, mask, static)
            fused = self.struct_direct(s_out["pooled"])
            aux.update({"vsn_weights": s_out["vsn_weights"], "pool_weights": s_out["pool_weights"]})
        elif mode == "text":
            assert self.text_direct is not None
            if text_embedding is None:
                text_embedding = self.encode_text(expanded_texts, device)
            fused = self.text_direct(text_embedding)
        else:
            raise ValueError(f"Unknown modality override: {mode}")

        logit = self.head(fused)
        return {"logit": logit, "probability": torch.sigmoid(logit), "aux": aux}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, batch: dict, device: torch.device | str = "cpu") -> torch.Tensor:
        self.eval()
        out = self.forward(
            batch["x"].to(device),
            batch["mask"].to(device),
            batch["static"].to(device),
            batch["text"],
        )
        return out["probability"]

    def has_dropout(self) -> bool:
        return any(isinstance(m, nn.Dropout) for m in self.modules())
