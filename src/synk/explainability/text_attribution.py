"""Text-modality attribution and evidence extraction.

Two complementary mechanisms:

1. *Gradient attribution* over token embeddings: the gradient of the encoder's
   pooled output w.r.t. input embeddings (gradient x embedding-norm).  This is
   computed at the encoder output - a documented limitation: it measures token
   influence on the clinical-text representation, not on the final fused logit.
2. *Deterministic clinical-evidence extraction*: scans note text for curated
   infection / organ-dysfunction / deterioration language with negation
   handling ("no evidence of infection" is NOT positive evidence).

The extractor is transparent, deterministic and auditable; the gradient
attribution is model-specific evidence.  The UI shows both and labels them.
"""

from __future__ import annotations

import logging
import re

import torch

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Deterministic clinical evidence extractor
# ----------------------------------------------------------------------------

NEGATION_PATTERNS = [
    r"\b(no|without|denies|denied|not|negative for|free of|ruled out|absent|afebrile)\b",
    r"\b(no signs? of)\b",
]
NEGATION_RE = re.compile("|".join(NEGATION_PATTERNS), re.IGNORECASE)
NEG_WINDOW = 48  # characters before a match to scan for negation

EVIDENCE_LEXICON: dict[str, list[str]] = {
    "infection": [
        r"\bsepsis\b",
        r"\bseptic\b",
        r"\binfection\b",
        r"\binfected\b",
        r"\bpneumonia\b",
        r"\burosepsis\b",
        r"\bbacteremia\b",
        r"\bcellulitis\b",
        r"\babscess\b",
        r"\bfebrile\b",
        r"\brigors?\b",
    ],
    "organ_dysfunction": [
        r"\bhypoxia\b",
        r"\bhypoxemic\b",
        r"\bhypotension\b",
        r"\bhypotensive\b",
        r"\boliguria\b",
        r"\bacute kidney injury\b",
        r"\bAKI\b",
        r"\bencephalopathy\b",
        r"\bconfusion\b",
        r"\blethargic\b",
        r"\bacidosis\b",
        r"\blactate\b",
        r"\bhypoperfusion\b",
        r"\bincreased work of breathing\b",
    ],
    "deterioration": [
        r"\bdeteriorat\w*\b",
        r"\bworsening\b",
        r"\bunstable\b",
        r"\bcrashing\b",
        r"\btachycardic\b",
        r"\btachycardia\b",
        r"\btachypneic\b",
        r"\bfever\b",
        r"\bchills\b",
        r"\bdecreasing\b",
    ],
    "intervention": [
        r"\bcultures? drawn\b",
        r"\bblood cultures?\b",
        r"\bcultures? sent\b",
        r"\bbroad-spectrum\b",
        r"\bantibiotics?\b",
        r"\bvancomycin\b",
        r"\bpiperacillin\b",
        r"\bfluid (?:bolus|resuscitation|challenge)\b",
        r"\bvasopressor\w*\b",
        r"\bnorepinephrine\b",
        r"\blevophed\b",
        r"\boxygen\b",
        r"\bsepsis bundle\b",
        r"\bnotify provider\b",
        r"\bICU transfer\b",
        r"\brapid response\b",
    ],
}


def extract_text_evidence(text: str) -> dict:
    """Deterministically extract evidence spans from one note.

    Returns dict with ``spans`` [{start, end, text, category, negated}] and
    ``positive_categories`` (categories with >=1 non-negated match).
    """
    spans: list[dict] = []
    positive: set[str] = set()
    for category, patterns in EVIDENCE_LEXICON.items():
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                start, end = m.start(), m.end()
                window = text[max(0, start - NEG_WINDOW): start]
                negated = bool(NEGATION_RE.search(window))
                spans.append({
                    "start": start,
                    "end": end,
                    "text": m.group(0),
                    "category": category,
                    "negated": negated,
                })
                if not negated:
                    positive.add(category)
    spans.sort(key=lambda s: s["start"])
    return {"spans": spans, "positive_categories": sorted(positive)}


# ----------------------------------------------------------------------------
# Gradient-based token attribution (transformer backend only)
# ----------------------------------------------------------------------------

def attribute_text_tokens(model: torch.nn.Module, text: str, max_length: int = 128,
                          device: torch.device | None = None) -> dict:
    """Per-token attribution via gradient x embedding-norm at the encoder."""
    encoder = getattr(model, "text_encoder", None)
    if encoder is None or not getattr(encoder, "is_transformer", False):
        return {
            "tokens": [],
            "scores": [],
            "available": False,
            "note": "Token attribution requires the transformer text encoder "
                    "(model.text.encoder_type=transformer); the TF-IDF demo "
                    "encoder does not expose token embeddings.",
        }
    if not text or not text.strip():
        return {"tokens": [], "scores": [], "available": False, "note": "Empty note."}

    device = device or next(model.parameters()).device
    backend = encoder.backend  # TransformerTextBackend
    enc = backend.tokenize([text], device)
    embeddings_layer = backend.model.get_input_embeddings()
    emb = embeddings_layer(enc["input_ids"]).detach().clone().requires_grad_(True)

    out = backend.model(inputs_embeds=emb, attention_mask=enc["attention_mask"])
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    grads = torch.autograd.grad(pooled.sum(), emb)[0]
    scores = grads.norm(dim=-1).squeeze(0).detach().cpu().numpy()  # (L,)

    keep = enc["attention_mask"].squeeze(0).bool().cpu().numpy()
    tokens = backend.tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0).tolist())
    tokens = [t for t, k in zip(tokens, keep) if k]
    scores = scores[: len(tokens)]

    return {
        "tokens": tokens,
        "scores": [float(s) for s in scores],
        "available": True,
        "note": "Gradient x embedding-norm attribution at the clinical-text "
                "encoder output (token influence on the text representation).",
    }
