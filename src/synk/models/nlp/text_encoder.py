"""Clinical text encoder.

Two interchangeable backends selected by configuration
(``model.text.encoder_type``):

``transformer``
    A Hugging Face transformer (configurable ``model_name``, e.g. a
    BioClinicalBERT-family checkpoint).  Documents are encoded with mean
    pooling over token embeddings.  Token-level gradients are exposed for
    text attribution.

``tfidf``
    A lightweight, fully offline TF-IDF + learned linear projection backend.
    It is a genuine (if simple) text encoder used for the CPU demo mode and
    as a text-pipeline sanity baseline; it is clearly identified as such in
    every model card and the UI.  Fitting happens on TRAINING notes only.

Both backends expose the same interface: ``fit`` (train split only), ``embed``
(differentiable input construction for attribution), ``decode`` (embedding ->
fixed-size representation), and ``output_dim``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from synk.utils.logging import get_logger

logger = get_logger(__name__)


class TfidfTextBackend(nn.Module):
    def __init__(self, max_features: int = 4096, embedding_dim: int = 128) -> None:
        super().__init__()
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        self.max_features = max_features
        self.projection: Optional[nn.Linear] = None
        self.embedding_dim = embedding_dim
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            # Degenerate corpus: still create a usable 1-dim vector space.
            self.vectorizer.fit(["empty placeholder note"])
        else:
            self.vectorizer.fit(texts)
        n_features = len(self.vectorizer.vocabulary_)
        self.projection = nn.Linear(n_features, self.embedding_dim)
        self._fitted = True
        logger.info("TF-IDF text backend fitted: %d features -> %d dims", n_features, self.embedding_dim)

    @property
    def feature_names(self) -> list[str]:
        names = [""] * len(self.vectorizer.vocabulary_)
        for token, idx in self.vectorizer.vocabulary_.items():
            names[idx] = token
        return names

    def transform(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Texts -> dense TF-IDF tensor [B, F]."""
        if not self._fitted:
            raise RuntimeError("TfidfTextBackend.fit must be called before transform")
        vec = self.vectorizer.transform([t if t and t.strip() else "empty placeholder note" for t in texts])
        arr = np.asarray(vec.todense(), dtype=np.float32)
        return torch.from_numpy(arr).to(device)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        assert self.projection is not None
        return torch.relu(self.projection(features))

    def output_dim(self) -> int:
        return self.embedding_dim

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({"vectorizer": self.vectorizer, "max_features": self.max_features,
                         "embedding_dim": self.embedding_dim}, handle)

    @classmethod
    def load(cls, path: Path) -> "TfidfTextBackend":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        backend = cls(payload["max_features"], payload["embedding_dim"])
        backend.vectorizer = payload["vectorizer"]
        n_features = len(backend.vectorizer.vocabulary_)
        backend.projection = nn.Linear(n_features, backend.embedding_dim)
        backend._fitted = True
        return backend


class TransformerTextBackend(nn.Module):
    def __init__(self, model_name: str, max_length: int = 128) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - transformers is a core dep
            raise RuntimeError("transformers is required for encoder_type=transformer") from exc
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        hidden = self.model.config.hidden_size
        self._output_dim = hidden
        logger.info("Transformer text backend loaded: %s (hidden=%d)", model_name, hidden)

    def tokenize(self, texts: list[str], device: torch.device) -> dict:
        return self.tokenizer(
            [t if t and t.strip() else "no clinical note available" for t in texts],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

    def embed(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, dict]:
        """Return input embeddings (differentiable leaf) + tokenisation metadata."""
        enc = self.tokenize(texts, device)
        embeddings = self.model.get_input_embeddings()(enc["input_ids"]).detach().requires_grad_(True)
        return embeddings, enc

    def decode(self, embeddings: torch.Tensor, enc: dict) -> torch.Tensor:
        out = self.model(inputs_embeds=embeddings, attention_mask=enc["attention_mask"])
        last_hidden = out.last_hidden_state                       # [B, L, H]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    def output_dim(self) -> int:
        return self._output_dim


class TextEncoder(nn.Module):
    """Unified text encoder wrapping either backend."""

    def __init__(self, config) -> None:
        super().__init__()
        tcfg = config.model.text
        self.encoder_type = tcfg.encoder_type
        if self.encoder_type == "tfidf":
            self.backend: nn.Module = TfidfTextBackend(tcfg.tfidf_max_features, tcfg.embedding_dim)
        else:
            self.backend = TransformerTextBackend(tcfg.model_name, tcfg.max_length)
        self._output_dim = self.backend.output_dim()

    # Delegated API -----------------------------------------------------
    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def is_transformer(self) -> bool:
        return self.encoder_type == "transformer"

    def fit(self, texts: list[str]) -> None:
        if self.encoder_type == "tfidf":
            self.backend.fit(texts)

    @property
    def fitted(self) -> bool:
        if self.encoder_type == "tfidf":
            return bool(getattr(self.backend, "_fitted", False))
        return True

    def transform(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Non-differentiable convenience transform (tfidf backend)."""
        return self.backend.transform(texts, device)  # type: ignore[attr-defined]

    def embed(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, Optional[dict]]:
        """Build a differentiable input representation for attribution."""
        if self.encoder_type == "tfidf":
            features = self.backend.transform(texts, device)
            return features, None
        return self.backend.embed(texts, device)  # type: ignore[attr-defined]

    def decode(self, embeddings: torch.Tensor, enc: Optional[dict] = None) -> torch.Tensor:
        if self.encoder_type == "tfidf":
            return self.backend.decode(embeddings)
        assert enc is not None
        return self.backend.decode(embeddings, enc)  # type: ignore[attr-defined]

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        embeddings, enc = self.embed(texts, device)
        return self.decode(embeddings, enc)

    # Serialisation ------------------------------------------------------
    def save_state(self, path: Path) -> None:
        if self.encoder_type == "tfidf":
            self.backend.save(path)  # type: ignore[attr-defined]
        # Transformer backends are re-created from model_name; no local copy.

    def load_state_file(self, path: Path) -> None:
        if self.encoder_type == "tfidf":
            loaded = TfidfTextBackend.load(path)
            self.backend = loaded.to(next(self.parameters()).device if len(list(self.parameters())) else "cpu")
            self._output_dim = loaded.output_dim()

    def token_metadata(self, texts: list[str], device: torch.device) -> Optional[dict]:
        if self.is_transformer:
            return self.backend.tokenize(texts, device)  # type: ignore[attr-defined]
        return None
