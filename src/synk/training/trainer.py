"""Unified training loop for SYNK models.

One trainer covers all three model variants (structured / text / multimodal).
Features: seeded splits, weighted loss, gradient clipping, LR scheduling,
early stopping on validation AUROC/AUPRC, checkpointing, metric logging via
the tracking backend, and automatic CPU/GPU selection.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from synk.features.dataset import SampleSet, collate_fn, make_torch_dataset
from synk.models.losses import build_loss
from synk.models.multimodal import SYNKModel
from synk.utils.logging import get_logger
from synk.utils.torch_utils import get_device, set_seed

logger = get_logger(__name__)


@dataclass
class TrainResult:
    run_id: str
    modality: str
    best_epoch: int
    best_val_auroc: float
    best_val_auprc: float
    best_val_brier: float
    train_history: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0
    checkpoint_path: str = ""
    device: str = "cpu"


def _make_loader(sample_set: SampleSet, encounter_ids: list[str], cfg, shuffle: bool) -> DataLoader:
    dataset = make_torch_dataset(sample_set, encounter_ids=encounter_ids)
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=cfg.training.num_workers,
        drop_last=False,
    )


def train_model(
    cfg,
    sample_set: SampleSet,
    splits: dict[str, list[str]],
    run_id: str,
    tracking=None,
    seed: Optional[int] = None,
) -> TrainResult:
    """Train a SYNK model for the configured modality."""
    seed = seed if seed is not None else cfg.seed
    set_seed(seed)
    device = get_device(prefer_gpu=cfg.training.accelerator != "cpu")
    logger.info("Training device: %s", device)

    train_ids, val_ids = splits["train"], splits["val"]
    train_loader = _make_loader(sample_set, train_ids, cfg, shuffle=True)
    val_loader = _make_loader(sample_set, val_ids, cfg, shuffle=False)

    model = SYNKModel(cfg, n_variables=len(sample_set.variables), n_static=sample_set.static.shape[1]).to(device)

    # Fit text encoder on training data only.
    if cfg.model.text.encoder_type == "tfidf":
        train_texts = (
            sample_set.samples.loc[sample_set.samples["encounter_id"].isin(train_ids), "text"]
            .astype(str)
            .tolist()
        )
        model.text_encoder.fit(train_texts)
        model.text_encoder.to(device)

    # Data-derived loss weighting from the TRAINING split only.
    y_train = sample_set.samples.loc[sample_set.samples["encounter_id"].isin(train_ids), "label"].to_numpy()
    loss_fn, loss_meta = build_loss(cfg, y_train)
    logger.info("Loss: %s", loss_meta)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = None
    if cfg.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.max_epochs)
    elif cfg.training.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    if tracking is not None:
        tracking.log_params({
            "modality": cfg.model.modality,
            "loss": loss_meta,
            "n_variables": len(sample_set.variables),
            "n_train_samples": int(len(train_loader.dataset)),
            "n_val_samples": int(len(val_loader.dataset)),
            "fusion": cfg.model.fusion.type,
            "text_encoder": cfg.model.text.encoder_type,
            "device": str(device),
        })

    history: list[dict] = []
    best = {"auroc": -1.0, "epoch": -1, "state": None, "auprc": 0.0, "brier": 1.0}
    patience_left = cfg.training.early_stopping_patience
    started = time.time()

    # ---- epoch loop -----------------------------------------------------
    for epoch in range(1, cfg.training.max_epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(
                batch["x"].to(device),
                batch["mask"].to(device),
                batch["static"].to(device),
                batch["text"],
            )
            y = batch["y"].to(device)
            loss = loss_fn(out["logit"], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)
            optimizer.step()
            epoch_loss += float(loss.detach())
            n_batches += 1

        val_metrics = _validate(model, val_loader, device)
        if scheduler is not None:
            if cfg.training.scheduler == "plateau":
                scheduler.step(val_metrics["auroc"])
            else:
                scheduler.step()

        entry = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(n_batches, 1),
            **val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(entry)
        logger.info("epoch %d | loss %.4f | val AUROC %.4f | val AUPRC %.4f",
                    epoch, entry["train_loss"], val_metrics["auroc"], val_metrics["auprc"])
        if tracking is not None:
            tracking.log_metrics(entry, step=epoch)

        if val_metrics["auroc"] > best["auroc"] + 1e-5:
            best = {
                "auroc": val_metrics["auroc"],
                "auprc": val_metrics["auprc"],
                "brier": val_metrics["brier"],
                "epoch": epoch,
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
            patience_left = cfg.training.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping at epoch %d (best epoch %d)", epoch, best["epoch"])
                break

    duration = time.time() - started
    ckpt_path = _save_checkpoint(cfg, run_id, best["state"], sample_set, seed)
    if cfg.model.text.encoder_type == "tfidf":
        _save_text_backend(model, ckpt_path.with_suffix(".text.pkl"))
    if tracking is not None:
        tracking.log_artifact(ckpt_path)

    return TrainResult(
        run_id=run_id,
        modality=cfg.model.modality,
        best_epoch=best["epoch"],
        best_val_auroc=float(best["auroc"]) if best["auroc"] > 0 else float("nan"),
        best_val_auprc=float(best["auprc"]),
        best_val_brier=float(best["brier"]),
        train_history=history,
        duration_seconds=duration,
        checkpoint_path=str(ckpt_path),
        device=str(device),
    )


@torch.no_grad()
def _validate(model: SYNKModel, loader: DataLoader, device: torch.device) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.eval()
    ys, ps = [], []
    for batch in loader:
        out = model(batch["x"].to(device), batch["mask"].to(device), batch["static"].to(device), batch["text"])
        ps.extend(out["probability"].cpu().numpy().tolist())
        ys.extend(batch["y"].numpy().tolist())
    y = np.array(ys)
    p = np.array(ps)
    metrics: dict = {}
    try:
        metrics["auroc"] = float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")
    except ValueError:
        metrics["auroc"] = float("nan")
    try:
        metrics["auprc"] = float(average_precision_score(y, p))
    except ValueError:
        metrics["auprc"] = float("nan")
    metrics["brier"] = float(np.mean((p - y) ** 2))
    return metrics


def _save_checkpoint(cfg, run_id: str, state: Optional[dict], sample_set: SampleSet, seed: int) -> Path:
    if state is None:
        raise RuntimeError("No model state to save - training produced no validation improvement")
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{run_id}.pt"
    payload = {
        "state_dict": state,
        "config": cfg.model_dump_dict(),
        "variables": sample_set.variables,
        "n_static": int(sample_set.static.shape[1]),
        "modality": cfg.model.modality,
        "seed": seed,
    }
    torch.save(payload, path)
    logger.info("Checkpoint saved: %s", path)
    return path


def _save_text_backend(model: SYNKModel, path: Path) -> None:
    """Persist the fitted text-backend state (tfidf vectoriser / projection)."""
    try:
        model.text_encoder.save_state(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save text backend state: %s", exc)
