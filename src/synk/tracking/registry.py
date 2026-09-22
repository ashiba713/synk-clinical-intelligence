"""Experiment tracking and model registry.

``RunTracker`` wraps MLflow (optional; enabled via ``tracking.backend``) and
always writes a local JSON run manifest so the platform remains fully usable
offline and without any external service credentials.

``ModelRegistry`` versions trained models under ``artifacts/models/<name>/<version>``
with a manifest containing the resolved configuration, metrics, git commit and
a SHA-256 checksum of the weights.

``ExperimentRegistry`` maintains the flat comparison table shown in the UI
(structured / text / multimodal / baselines).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from synk.utils.common import sha256_of_dict, sha256_of_file, utcnow_iso
from synk.utils.logging import get_logger

logger = get_logger(__name__)


def get_git_commit(repo_root: Optional[Path] = None) -> str:
    """Best-effort git commit hash; empty string when unavailable."""
    import subprocess

    try:
        root = repo_root or Path.cwd()
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - git is optional metadata
        return ""


class RunTracker:
    """Tracks a single training/evaluation run."""

    def __init__(self, cfg, run_id: str) -> None:
        self.cfg = cfg
        self.run_id = run_id
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.artifacts: list[str] = []
        self._mlflow_run = None
        self.run_dir = Path(cfg.paths.run_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if cfg.tracking.backend == "mlflow":
            try:
                import mlflow

                if cfg.tracking.mlflow_tracking_uri:
                    mlflow.set_tracking_uri(cfg.tracking.mlflow_tracking_uri)
                mlflow.set_experiment(cfg.tracking.experiment_name)
                self._mlflow_run = mlflow.start_run(run_name=run_id)
                logger.info("MLflow run started: %s", self._mlflow_run.info.run_id)
            except Exception as exc:  # noqa: BLE001 - tracking must never crash training
                logger.warning("MLflow unavailable (%s); continuing with local tracking only", exc)
                self._mlflow_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)
        if self._mlflow_run is not None:
            import mlflow

            mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))})

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        self.metrics.update(metrics)
        if self._mlflow_run is not None:
            import mlflow

            mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v},
                step=step,
            )

    def log_artifact(self, path: str | Path) -> None:
        self.artifacts.append(str(path))
        if self._mlflow_run is not None:
            import mlflow

            try:
                mlflow.log_artifact(str(path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("MLflow artifact logging failed: %s", exc)

    def finalize(self, final_metrics: Optional[dict[str, Any]] = None) -> Path:
        if final_metrics:
            self.log_metrics(final_metrics)
        if self._mlflow_run is not None:
            import mlflow

            mlflow.end_run()
        manifest = {
            "run_id": self.run_id,
            "created_at": utcnow_iso(),
            "git_commit": get_git_commit(),
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "seed": self.cfg.seed,
            "dataset_id": self.cfg.data.dataset_id,
        }
        out = self.run_dir / "manifest.json"
        out.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        logger.info("Run manifest written: %s", out)
        return out


class ModelRegistry:
    """Saves and loads versioned model bundles."""

    def __init__(self, registry_dir: Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def _next_version(self, name: str) -> str:
        base = self.registry_dir / name
        existing = [int(p.name[1:]) for p in base.glob("v*") if p.name[1:].isdigit()]
        return f"v{max(existing, default=0) + 1}"

    def save(
        self,
        name: str,
        checkpoint_path: Path,
        config_dict: dict[str, Any],
        metrics: dict[str, Any],
        extra_files: Optional[dict[str, Path]] = None,
        dataset_id: str = "",
    ) -> Path:
        version = self._next_version(name)
        model_dir = self.registry_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        weights = model_dir / "model.pt"
        shutil.copy2(checkpoint_path, weights)

        manifest = {
            "model_name": name,
            "model_version": version,
            "created_at": utcnow_iso(),
            "git_commit": get_git_commit(),
            "dataset_id": dataset_id,
            "config": config_dict,
            "metrics": metrics,
            "weights_checksum": sha256_of_file(weights),
            "config_checksum": sha256_of_dict(config_dict),
            "weights_path": str(weights),
        }
        (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        for fname, src in (extra_files or {}).items():
            if src is not None and Path(src).exists():
                shutil.copy2(src, model_dir / fname)
        logger.info("Registered model %s %s at %s", name, version, model_dir)
        return model_dir

    def latest(self, name: str = "synk_multimodal") -> Optional[Path]:
        base = self.registry_dir / name
        if not base.exists():
            return None
        versions = sorted([p for p in base.glob("v*") if p.is_dir()])
        return versions[-1] if versions else None

    def load_manifest(self, model_dir: Path) -> dict[str, Any]:
        return json.loads((Path(model_dir) / "manifest.json").read_text(encoding="utf-8"))


class ExperimentRegistry:
    """Flat append-only table of experiment results used across the UI."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.entries: list[dict[str, Any]] = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.entries = []

    def add(self, entry: dict[str, Any]) -> None:
        entry = dict(entry)
        entry.setdefault("created_at", utcnow_iso())
        entry.setdefault("git_commit", get_git_commit())
        self.entries.append(entry)
        self.path.write_text(json.dumps(self.entries, indent=2, default=str), encoding="utf-8")
        logger.info("Experiment registered: %s (%s)", entry.get("experiment_id"), entry.get("modality"))

    def as_frame(self):
        import pandas as pd

        if not self.entries:
            return pd.DataFrame()
        return pd.DataFrame(self.entries)
