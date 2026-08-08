"""Training loop for chest X-ray classification.

Design notes
------------
* Model selection uses **validation** macro ROC-AUC (or val loss), never test.
* The best checkpoint is kept separately from the last one, so an over-fitted
  final epoch cannot silently become the reported model.
* Mixed precision is enabled on CUDA - roughly 2x faster with no accuracy cost
  at this scale.
* Everything a report needs (per-epoch history CSV, best checkpoint, config
  snapshot, environment metadata) is written under ``outputs/classification/<run>/``.

Nothing here fabricates results: metrics are computed only from real forward
passes over real data you provide.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..common.io_utils import history_to_csv, save_json, save_run_metadata
from ..common.paths import run_dirs
from .metrics import evaluate_all_labels


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    name = str(cfg.get("train.optimizer", "adamw")).lower()
    lr = float(cfg.get("train.lr", 1e-4))
    weight_decay = float(cfg.get("train.weight_decay", 1e-4))
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=float(cfg.get("train.momentum", 0.9)),
                               weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unknown train.optimizer={name!r}. Use 'adamw', 'adam' or 'sgd'.")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg, steps_per_epoch: int = 1):
    name = str(cfg.get("train.scheduler", "plateau")).lower()
    epochs = int(cfg.get("train.epochs", 10))

    if name in {"none", "off"}:
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=int(cfg.get("train.scheduler_patience", 2)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=float(cfg.get("train.lr", 1e-4)),
            epochs=epochs, steps_per_epoch=max(steps_per_epoch, 1),
        )
    raise ValueError(f"Unknown train.scheduler={name!r}.")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device, criterion: nn.Module | None = None,
            return_paths: bool = False):
    """Run the model over a loader and return ``(y_true, y_prob, mean_loss[, paths])``.

    Probabilities are sigmoid(logits), matching the BCE-with-logits training
    objective. Used for validation during training and for final evaluation.
    """
    model.eval()
    all_true, all_prob, all_paths = [], [], []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        images, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            total_loss += float(criterion(logits, targets).item())
            n_batches += 1
        all_true.append(targets.detach().cpu().numpy())
        all_prob.append(torch.sigmoid(logits).detach().cpu().numpy())
        if return_paths and len(batch) > 2:
            all_paths.extend(list(batch[2]))

    y_true = np.concatenate(all_true) if all_true else np.empty((0, 0))
    y_prob = np.concatenate(all_prob) if all_prob else np.empty((0, 0))
    mean_loss = total_loss / max(n_batches, 1) if criterion is not None else float("nan")

    if return_paths:
        return y_true, y_prob, mean_loss, all_paths
    return y_true, y_prob, mean_loss


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device, scaler=None,
                    scheduler=None, grad_clip: float | None = None,
                    progress: bool = True) -> float:
    """One pass over the training data. Returns the mean training loss."""
    model.train()
    total_loss, n_batches = 0.0, 0

    iterator = loader
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(loader, desc="train", leave=False)
        except ImportError:
            pass

    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images), targets)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # OneCycle steps every batch; the others step once per epoch.
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        total_loss += float(loss.item())
        n_batches += 1
        if progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    return total_loss / max(n_batches, 1)


def train_model(model: nn.Module, loaders: dict[str, DataLoader], criterion: nn.Module,
                cfg, device, run_name: str | None = None) -> dict:
    """Full training run with validation, early stopping and checkpointing.

    Returns a dict with the trained model (best weights loaded), the per-epoch
    history DataFrame, and the paths of everything written to disk.
    """
    labels = list(cfg.data.labels)
    epochs = int(cfg.get("train.epochs", 10))
    patience = int(cfg.get("train.early_stopping_patience", 5))
    monitor = str(cfg.get("train.monitor", "val_roc_auc"))
    grad_clip = cfg.get("train.grad_clip")
    run_name = run_name or str(cfg.get("run_name", "run"))

    dirs = run_dirs(cfg.get("output.root", "outputs/classification"), run_name)
    cfg.save(dirs["root"] / "config_used.yaml")

    if "train" not in loaders:
        raise ValueError("loaders must contain a 'train' entry.")
    has_val = "val" in loaders
    if not has_val:
        print("[warn] no validation loader: early stopping and threshold selection are disabled.")

    model = model.to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(loaders["train"]))
    use_amp = bool(cfg.get("train.amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    higher_is_better = not monitor.endswith("loss")
    best_score = -np.inf if higher_is_better else np.inf
    best_epoch, epochs_without_improvement = -1, 0
    best_path = dirs["models"] / f"{run_name}_best.pt"
    last_path = dirs["models"] / f"{run_name}_last.pt"
    history: list[dict] = []

    print(f"\n[train] run='{run_name}' device={device} epochs={epochs} amp={use_amp} monitor={monitor}")
    print(f"[train] outputs -> {dirs['root']}\n")
    start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device,
                                     scaler=scaler, scheduler=scheduler, grad_clip=grad_clip,
                                     progress=bool(cfg.get("train.progress", True)))

        row = {"epoch": epoch, "train_loss": train_loss,
               "lr": optimizer.param_groups[0]["lr"]}

        if has_val:
            y_true, y_prob, val_loss = predict(model, loaders["val"], device, criterion)
            val_table = evaluate_all_labels(y_true, y_prob, labels, thresholds=0.5)
            macro = val_table.iloc[-1] if len(labels) > 1 else val_table.iloc[0]
            row.update({
                "val_loss": val_loss,
                "val_roc_auc": float(macro["roc_auc"]),
                "val_pr_auc": float(macro["pr_auc"]),
                "val_f1": float(macro["f1"]),
                "val_recall": float(macro["recall_sensitivity"]),
                "val_specificity": float(macro["specificity"]),
                "val_accuracy": float(macro["accuracy"]),
            })

        row["seconds"] = round(time.time() - epoch_start, 1)
        history.append(row)

        msg = f"epoch {epoch:>3d}/{epochs}  train_loss={train_loss:.4f}"
        if has_val:
            msg += (f"  val_loss={row['val_loss']:.4f}  val_roc_auc={row['val_roc_auc']:.4f}"
                    f"  val_f1={row['val_f1']:.4f}")
        msg += f"  ({row['seconds']}s)"

        # Model selection + early stopping on validation only.
        if has_val and monitor in row:
            score = row[monitor]
            improved = (score > best_score) if higher_is_better else (score < best_score)
            if np.isfinite(score) and improved:
                best_score, best_epoch, epochs_without_improvement = score, epoch, 0
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "monitor": monitor, "score": float(score),
                            "labels": labels, "config": cfg.to_dict()}, best_path)
                msg += "  <- best"
            else:
                epochs_without_improvement += 1
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
        elif scheduler is not None and not isinstance(
            scheduler, (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.ReduceLROnPlateau)
        ):
            scheduler.step()

        print(msg)

        if has_val and patience > 0 and epochs_without_improvement >= patience:
            print(f"[train] early stopping: no {monitor} improvement for {patience} epochs.")
            break

    torch.save({"model_state": model.state_dict(), "epoch": len(history),
                "labels": labels, "config": cfg.to_dict()}, last_path)

    elapsed = time.time() - start
    if best_epoch > 0 and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
        print(f"[train] restored best checkpoint from epoch {best_epoch} ({monitor}={best_score:.4f})")
    else:
        print("[train] no validation improvement recorded; keeping the final-epoch weights.")

    history_df = history_to_csv(history, dirs["metrics"] / "training_history.csv")
    save_run_metadata(
        dirs["root"] / "run_metadata.json",
        config=cfg.to_dict(),
        extra={"run_name": run_name, "best_epoch": best_epoch, "monitor": monitor,
               "best_score": None if not np.isfinite(best_score) else float(best_score),
               "epochs_completed": len(history), "total_seconds": round(elapsed, 1),
               "best_checkpoint": str(best_path if best_path.exists() else last_path)},
    )
    print(f"[train] finished in {elapsed / 60:.1f} min")

    return {
        "model": model,
        "history": history_df,
        "best_epoch": best_epoch,
        "best_score": float(best_score) if np.isfinite(best_score) else None,
        "dirs": dirs,
        "best_checkpoint": best_path if best_path.exists() else last_path,
    }


def load_checkpoint(path: str | Path, model: nn.Module, device) -> nn.Module:
    """Load weights saved by :func:`train_model` into an already-built model."""
    ckpt = torch.load(Path(path), map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print(f"[model] loaded checkpoint from epoch {ckpt['epoch']} ({Path(path).name})")
    return model
