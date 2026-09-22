"""Feature/sample construction, leakage prevention and model tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch


def test_build_samples_shapes(prepared):
    sample_set, splits, pre, label_table = prepared
    n = len(sample_set.samples)
    T = config_window(pre)
    assert sample_set.x.shape == (n, T, len(sample_set.variables))
    assert sample_set.mask.shape == sample_set.x.shape
    assert sample_set.static.shape[0] == n
    assert np.isfinite(sample_set.x).all()
    assert np.isfinite(sample_set.static).all()
    # labels align with the label table
    assert len(sample_set.samples) <= len(label_table.samples)


def config_window(pre) -> int:
    return int(pre.config.preprocessing.observation_window_hours)


def test_patient_level_split_no_leakage(prepared):
    sample_set, splits, *_ = prepared
    enc_to_pat = dict(zip(sample_set.samples["encounter_id"], sample_set.samples["patient_id"]))
    sets = {k: {enc_to_pat[e] for e in v} for k, v in splits.items()}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])
    assert set(splits) == {"train", "val", "test"}
    assert all(len(v) > 0 for v in splits.values())


def test_missingness_preserved(prepared):
    sample_set, *_ = prepared
    # there must be genuine missingness (mask has zeros), never silently filled
    assert (sample_set.mask == 0).mean() > 0.0


def test_torch_dataset_roundtrip(prepared):
    from synk.features.dataset import collate_fn, make_torch_dataset

    sample_set, splits, *_ = prepared
    ds = make_torch_dataset(sample_set, encounter_ids=splits["train"][:2])
    assert len(ds) > 0
    item = ds[0]
    assert item["x"].dtype == torch.float32
    assert item["y"].dtype == torch.float32
    batch = collate_fn([ds[i] for i in range(min(4, len(ds)))])
    assert batch["x"].dim() == 3
    assert isinstance(batch["text"], list)


def test_model_forward_all_modalities(config, prepared):
    from synk.config.settings import merge_configs
    from synk.models.multimodal import SYNKModel

    sample_set, splits, *_ = prepared
    batch_size = 4
    x = torch.from_numpy(sample_set.x[:batch_size])
    m = torch.from_numpy(sample_set.mask[:batch_size])
    s = torch.from_numpy(sample_set.static[:batch_size])
    texts = sample_set.samples["text"].head(batch_size).tolist()

    def _fitted_model(modality: str):
        cfg = merge_configs(config, {"model": {"modality": modality}})
        model = SYNKModel(cfg, n_variables=len(sample_set.variables), n_static=sample_set.static.shape[1])
        model.text_encoder.fit([t if t.strip() else "no note" for t in texts])
        return model

    for modality in ("both", "structured", "text"):
        model = _fitted_model(modality)
        out = model(x, m, s, texts)
        assert out["logit"].shape == (batch_size,)
        assert out["probability"].shape == (batch_size,)
        assert ((out["probability"] >= 0) & (out["probability"] <= 1)).all()

    # ablation override on the multimodal model keeps fusion active
    model = _fitted_model("both")
    out_s = model(x, m, s, texts, modality_override="structured")
    out_t = model(x, m, s, texts, modality_override="text")
    assert out_s["probability"].shape == out_t["probability"].shape == (batch_size,)


def test_loss_functions(config):
    from synk.models.losses import build_loss, compute_pos_weight, focal_loss_with_logits

    y = np.array([0, 0, 0, 1])
    pw = compute_pos_weight(y)
    assert pw == pytest.approx(3.0)
    loss_fn, meta = build_loss(config, y)
    logits = torch.tensor([0.1, -0.2, 0.4, 1.2])
    yv = torch.tensor([0.0, 0.0, 0.0, 1.0])
    val = loss_fn(logits, yv)
    assert torch.isfinite(val)
    assert "pos_weight" in meta or "focal_gamma" in meta
    fl = focal_loss_with_logits(logits, yv)
    assert torch.isfinite(fl)


def test_fusion_modules(config, prepared):
    from synk.config.settings import merge_configs
    from synk.models.fusion.fusion import build_fusion

    sample_set, *_ = prepared
    s = torch.randn(3, 16)
    t = torch.randn(3, 16)
    seq = torch.randn(3, 8, 16)
    for kind in ("concat", "gated", "cross_attention"):
        cfg = merge_configs(config, {"model": {"fusion": {"type": kind}}})
        fusion = build_fusion(cfg, 16, 16)
        out, aux = fusion(s, t, seq)
        assert out.shape[0] == 3
        assert fusion.output_dim == cfg.model.fusion.hidden_dim
        if kind == "gated":
            assert "gates" in aux
