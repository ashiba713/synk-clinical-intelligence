"""Typed configuration for SYNK.

Configuration is loaded from YAML files (configs/base.yaml plus optional
overrides) with environment-variable overrides applied last.  Every key in the
YAML files maps to a typed field below so that invalid configuration fails
fast with a clear message instead of silently misbehaving at run time.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class PathsConfig(BaseModel):
    data_dir: str = "data"
    artifact_dir: str = "artifacts"
    checkpoint_dir: str = "artifacts/checkpoints"
    model_registry_dir: str = "artifacts/models"
    run_dir: str = "artifacts/runs"
    report_dir: str = "artifacts/reports"
    evaluation_dir: str = "artifacts/evaluation"


class DataConfig(BaseModel):
    dataset_id: str = "synthetic_v1"
    raw_dir: str = "data/raw"
    synthetic_dir: str = "data/synthetic"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"


class SyntheticConfig(BaseModel):
    n_patients: int = Field(default=120, ge=1)
    stays_per_patient: list[int] = Field(default_factory=lambda: [1, 2])
    min_stay_hours: int = Field(default=36, ge=12)
    max_stay_hours: int = Field(default=96, ge=24)
    sepsis_prevalence: float = Field(default=0.30, gt=0.0, lt=1.0)
    missingness_rate: float = Field(default=0.10, ge=0.0, lt=1.0)
    noise_scale: float = Field(default=1.0, ge=0.0)
    include_demo_scenarios: bool = True
    output_dir: str = "data/synthetic"


class PreprocessingConfig(BaseModel):
    observation_window_hours: int = Field(default=12, ge=1)
    sampling_interval_hours: float = Field(default=1.0, gt=0.0)
    imputation: str = "forward_fill_then_median"
    forward_fill_limit_hours: float = Field(default=6.0, gt=0.0)
    winsorize_quantiles: list[float] = Field(default_factory=lambda: [0.001, 0.999])
    add_missingness_indicators: bool = True
    derive_map: bool = True
    standardize: str = "zscore"
    text_deid: bool = True

    @field_validator("imputation")
    @classmethod
    def _check_imputation(cls, v: str) -> str:
        allowed = {"forward_fill_then_median", "median", "mean", "zero"}
        if v not in allowed:
            raise ValueError(f"imputation must be one of {sorted(allowed)}, got '{v}'")
        return v

    @field_validator("standardize")
    @classmethod
    def _check_standardize(cls, v: str) -> str:
        allowed = {"zscore", "minmax", "none"}
        if v not in allowed:
            raise ValueError(f"standardize must be one of {sorted(allowed)}, got '{v}'")
        return v


class LabelCriteriaConfig(BaseModel):
    antibiotic_window_hours: float = Field(default=24, gt=0)
    culture_window_hours: float = Field(default=24, gt=0)
    derangement_threshold: int = Field(default=2, ge=1)


class LabelSplitsConfig(BaseModel):
    method: str = "stratified_group"
    train_frac: float = Field(default=0.7, gt=0, lt=1)
    val_frac: float = Field(default=0.15, ge=0, lt=1)
    test_frac: float = Field(default=0.15, gt=0, lt=1)

    @field_validator("method")
    @classmethod
    def _check_method(cls, v: str) -> str:
        allowed = {"stratified_group", "group", "random"}
        if v not in allowed:
            raise ValueError(f"splits.method must be one of {sorted(allowed)}, got '{v}'")
        return v


class LabelsConfig(BaseModel):
    source: str = "latent_outcomes"
    horizon_hours: float = Field(default=6, gt=0)
    stride_hours: float = Field(default=2, gt=0)
    min_history_hours: float = Field(default=8, gt=0)
    exclude_after_onset: bool = True
    criteria: LabelCriteriaConfig = Field(default_factory=LabelCriteriaConfig)
    splits: LabelSplitsConfig = Field(default_factory=LabelSplitsConfig)

    @field_validator("source")
    @classmethod
    def _check_source(cls, v: str) -> str:
        allowed = {"latent_outcomes", "criteria"}
        if v not in allowed:
            raise ValueError(f"labels.source must be one of {sorted(allowed)}, got '{v}'")
        return v


class TemporalModelConfig(BaseModel):
    hidden_dim: int = Field(default=64, ge=8)
    lstm_layers: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.15, ge=0, lt=1)
    attention_heads: int = Field(default=4, ge=1)
    use_static: bool = True


class TextModelConfig(BaseModel):
    encoder_type: str = "tfidf"
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_length: int = Field(default=128, ge=16)
    max_notes: int = Field(default=3, ge=1)
    text_lookback_hours: float = Field(default=24, gt=0)
    tfidf_max_features: int = Field(default=4096, ge=64)
    embedding_dim: int = Field(default=128, ge=16)

    @field_validator("encoder_type")
    @classmethod
    def _check_encoder(cls, v: str) -> str:
        allowed = {"transformer", "tfidf"}
        if v not in allowed:
            raise ValueError(f"text.encoder_type must be one of {sorted(allowed)}, got '{v}'")
        return v


class FusionConfig(BaseModel):
    type: str = "gated"
    hidden_dim: int = Field(default=128, ge=8)
    dropout: float = Field(default=0.2, ge=0, lt=1)

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        allowed = {"concat", "gated", "cross_attention"}
        if v not in allowed:
            raise ValueError(f"fusion.type must be one of {sorted(allowed)}, got '{v}'")
        return v


class ModelConfig(BaseModel):
    modality: str = "both"
    temporal: TemporalModelConfig = Field(default_factory=TemporalModelConfig)
    text: TextModelConfig = Field(default_factory=TextModelConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    head_hidden: int = Field(default=64, ge=8)

    @field_validator("modality")
    @classmethod
    def _check_modality(cls, v: str) -> str:
        allowed = {"both", "structured", "text"}
        if v not in allowed:
            raise ValueError(f"model.modality must be one of {sorted(allowed)}, got '{v}'")
        return v


class LossConfig(BaseModel):
    type: str = "bce_weighted"
    focal_gamma: float = Field(default=2.0, gt=0)

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        allowed = {"bce_weighted", "focal", "bce"}
        if v not in allowed:
            raise ValueError(f"loss.type must be one of {sorted(allowed)}, got '{v}'")
        return v


class TrainingConfig(BaseModel):
    batch_size: int = Field(default=64, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    max_epochs: int = Field(default=30, ge=1)
    gradient_clip_val: float = Field(default=1.0, ge=0)
    scheduler: str = "cosine"
    early_stopping_patience: int = Field(default=6, ge=1)
    num_workers: int = Field(default=0, ge=0)
    accelerator: str = "auto"
    precision: int = 32
    fast_dev_run: bool = False

    @field_validator("scheduler")
    @classmethod
    def _check_scheduler(cls, v: str) -> str:
        allowed = {"cosine", "plateau", "none"}
        if v not in allowed:
            raise ValueError(f"training.scheduler must be one of {sorted(allowed)}, got '{v}'")
        return v


class EvaluationConfig(BaseModel):
    threshold: float = Field(default=0.5, gt=0, lt=1)
    risk_thresholds: list[float] = Field(default_factory=lambda: [0.30, 0.60, 0.80])
    bootstrap_n: int = Field(default=200, ge=0)
    bootstrap_ci: float = Field(default=0.95, gt=0, lt=1)


class CalibrationConfig(BaseModel):
    method: str = "isotonic"

    @field_validator("method")
    @classmethod
    def _check_method(cls, v: str) -> str:
        allowed = {"isotonic", "platt", "none"}
        if v not in allowed:
            raise ValueError(f"calibration.method must be one of {sorted(allowed)}, got '{v}'")
        return v


class UncertaintyConfig(BaseModel):
    mc_dropout_samples: int = Field(default=30, ge=0)
    mc_dropout_min_samples: int = Field(default=5, ge=1)


class InferenceConfig(BaseModel):
    engine: str = "auto"
    baseline_path: str = ""

    @field_validator("engine")
    @classmethod
    def _check_engine(cls, v: str) -> str:
        allowed = {"auto", "torch", "sklearn"}
        if v not in allowed:
            raise ValueError(f"inference.engine must be one of {sorted(allowed)}, got '{v}'")
        return v


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    load_synthetic_on_start: bool = True
    default_db_url: str = "sqlite:///data/synk_demo.db"


class TrackingConfig(BaseModel):
    backend: str = "none"
    mlflow_tracking_uri: str = ""
    experiment_name: str = "synk"
    wandb_project: str = ""

    @field_validator("backend")
    @classmethod
    def _check_backend(cls, v: str) -> str:
        allowed = {"mlflow", "none"}
        if v not in allowed:
            raise ValueError(f"tracking.backend must be one of {sorted(allowed)}, got '{v}'")
        return v


class SYNKConfig(BaseModel):
    """Root configuration object for SYNK."""

    seed: int = 42
    env: str = "development"
    log_level: str = "INFO"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    model_config = {"extra": "forbid"}

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def resolve(self, repo_root: Optional[Path] = None) -> "SYNKConfig":
        """Return a copy with path fields anchored at *repo_root*."""
        cfg = self.model_copy(deep=True)
        root = Path(repo_root) if repo_root else find_repo_root()
        for section in ("paths", "data", "synthetic"):
            obj = getattr(cfg, section)
            for field_name in obj.model_fields:
                value = getattr(obj, field_name)
                if isinstance(value, str) and not Path(value).is_absolute():
                    setattr(obj, field_name, str(root / value))
        if not Path(cfg.api.default_db_url).is_absolute() and cfg.api.default_db_url.startswith("sqlite:///"):
            db_path = cfg.api.default_db_url.replace("sqlite:///", "", 1)
            cfg.api.default_db_url = f"sqlite:///{root / db_path}"
        return cfg

    def model_dump_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")




ENV_PREFIX = "SYNK_"


def _apply_env_overrides(cfg: SYNKConfig) -> SYNKConfig:
    """Apply SYNK_SECTION__KEY=value environment overrides onto *cfg*."""
    data = cfg.model_dump(mode="python")
    for raw_key, raw_value in os.environ.items():
        if not raw_key.startswith(ENV_PREFIX) or raw_key in {
            "SYNK_ENV", "SYNK_LOG_LEVEL", "SYNK_RANDOM_SEED", "SYNK_TEXT_ENCODER_TYPE",
            "SYNK_TEXT_MODEL_NAME", "SYNK_API_URL",
        }:
            # Simple top-level aliases handled below.
            continue
        parts = raw_key[len(ENV_PREFIX):].lower().split("__")
        node: Any = data
        ok = True
        for part in parts[:-1]:
            if isinstance(node, dict) and part in node and isinstance(node[part], dict):
                node = node[part]
            else:
                ok = False
                break
        if ok and isinstance(node, dict) and parts[-1] in node:
            current = node[parts[-1]]
            try:
                node[parts[-1]] = _coerce(raw_value, current)
            except (ValueError, TypeError):
                node[parts[-1]] = raw_value
    cfg = SYNKConfig.model_validate(data)

    # Simple aliases.
    if os.environ.get("SYNK_LOG_LEVEL"):
        cfg.log_level = os.environ["SYNK_LOG_LEVEL"].upper()
    if os.environ.get("SYNK_RANDOM_SEED"):
        try:
            cfg.seed = int(os.environ["SYNK_RANDOM_SEED"])
        except ValueError:
            pass
    if os.environ.get("SYNK_TEXT_ENCODER_TYPE"):
        cfg.model.text.encoder_type = os.environ["SYNK_TEXT_ENCODER_TYPE"]
    if os.environ.get("SYNK_TEXT_MODEL_NAME"):
        cfg.model.text.model_name = os.environ["SYNK_TEXT_MODEL_NAME"]
    if os.environ.get("MLFLOW_TRACKING_URI"):
        cfg.tracking.mlflow_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    if os.environ.get("MLFLOW_TRACKING_URI") == "":
        cfg.tracking.mlflow_tracking_uri = ""
    if os.environ.get("MLFLOW_EXPERIMENT_NAME"):
        cfg.tracking.experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
    if os.environ.get("DATABASE_URL"):
        cfg.api.default_db_url = os.environ["DATABASE_URL"]
    return cfg


def _coerce(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        items = [item.strip() for item in raw.split(",") if item.strip()]
        if current and isinstance(current[0], float):
            return [float(item) for item in items]
        if current and isinstance(current[0], int):
            return [int(item) for item in items]
        return items
    return raw


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk upwards until a directory containing ``configs/base.yaml`` is found."""
    current = Path(start or os.getcwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs" / "base.yaml").exists():
            return candidate
    return current


def load_config(
    base_path: Optional[Path] = None,
    override_paths: Optional[list[Path]] = None,
    repo_root: Optional[Path] = None,
) -> SYNKConfig:
    """Load the SYNK configuration.

    Order of precedence (last wins):
        1. Built-in defaults
        2. ``configs/base.yaml``
        3. Any *override_paths* (e.g. ``configs/training.yaml``)
        4. Environment variables (``SYNK_SECTION__KEY=value``)
    """
    root = Path(repo_root) if repo_root else find_repo_root()
    base_path = Path(base_path) if base_path else root / "configs" / "base.yaml"

    merged: dict[str, Any] = {}
    files: list[Path] = []
    if base_path.exists():
        files.append(base_path)
    for override in override_paths or []:
        override = Path(override)
        if override.exists():
            files.append(override)

    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        _deep_merge(merged, loaded)

    cfg = SYNKConfig.model_validate(merged) if merged else SYNKConfig()
    cfg = _apply_env_overrides(cfg)
    return cfg.resolve(repo_root=root)


def merge_configs(base: SYNKConfig, override: dict[str, Any]) -> SYNKConfig:
    """Merge a raw override mapping (e.g. CLI ``--set a.b=1``) into *base*."""
    data = base.model_dump(mode="python")
    _deep_merge(data, override)
    return SYNKConfig.model_validate(data).resolve()


def parse_dotlist(items: Optional[list[str]]) -> dict[str, Any]:
    """Parse ``['a.b=1', 'c=2']`` style CLI overrides into a nested mapping."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, value = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        leaf = parts[-1]
        # Coerce scalars.
        coerced: Any
        if value.lower() in {"true", "false"}:
            coerced = value.lower() == "true"
        else:
            try:
                coerced = int(value)
            except ValueError:
                try:
                    coerced = float(value)
                except ValueError:
                    coerced = value
        node[leaf] = coerced
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
