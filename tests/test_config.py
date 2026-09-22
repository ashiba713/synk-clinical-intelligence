"""Configuration system tests."""

from __future__ import annotations

import pytest

from synk.config.settings import SYNKConfig, load_config, merge_configs, parse_dotlist


def test_load_defaults():
    cfg = load_config()
    assert isinstance(cfg, SYNKConfig)
    assert cfg.seed == 42
    assert cfg.model.modality in {"both", "structured", "text"}
    assert cfg.labels.horizon_hours > 0


def test_invalid_modality_rejected():
    with pytest.raises(Exception):
        SYNKConfig.model_validate({"model": {"modality": "multimodal"}})


def test_invalid_fusion_rejected():
    with pytest.raises(Exception):
        SYNKConfig.model_validate({"model": {"fusion": {"type": "magic"}}})


def test_merge_and_dotlist():
    cfg = load_config()
    override = parse_dotlist(["training.lr=0.01", "model.modality=structured", "labels.horizon_hours=12"])
    merged = merge_configs(cfg, override)
    assert merged.training.lr == 0.01
    assert merged.model.modality == "structured"
    assert merged.labels.horizon_hours == 12
    # original untouched
    assert cfg.model.modality in {"both", "structured", "text"}


def test_env_override(monkeypatch):
    monkeypatch.setenv("SYNK_MODEL__MODALITY", "text")
    monkeypatch.setenv("SYNK_RANDOM_SEED", "7")
    cfg = load_config()
    assert cfg.model.modality == "text"
    assert cfg.seed == 7
