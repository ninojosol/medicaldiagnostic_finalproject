"""Training loop for U-Net brain-tumour segmentation.

Mirrors the classification trainer: validation-based model selection, early
stopping, mixed precision, and every artefact written under
``outputs/segmentation/<run>/``. Model selection uses **validation Dice**, the
metric the project actually reports.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..common.io_utils import history_to_csv, save_run_metadata
from ..common.paths import run_dirs
from .metrics import dice_coefficient, iou_score


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    name = str(cfg.get("train.optimizer", "adamw")).lower()
    lr = float(cfg.get("train.lr", 1e-3))
    weight_decay = float(cfg.get("train.weight_decay", 1e-5))
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unknown train.optimizer={name!r}.")


def build_scheduler(optimizer, cfg):
    name = str(cfg.get("train.scheduler", "plateau")).lower()
    if name in {"none", "off"}:
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=int(cfg.get("train.scheduler_patience", 3)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg.get("train.epochs", 30))
        )
    raise ValueError(f"Unknown train.scheduler={name!r}.")


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device,
             threshold: float = 0.5) -> dict:
    """Mean loss, Dice and IoU over a loader (batch-size-weighted)."""
    model.eval()
    total_loss, dice_sum, iou_sum, n_images, n_batches = 0.0, 0.0, 0.0, 0, 0

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        masks = batch[1].to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(criterion(logits, masks).item())
        n_batches += 1

        dice_sum += float(dice_coefficient(logits, masks, threshold).sum())
        iou_sum += float(iou_score(logits, masks, threshold).sum())
        n_images += images.shape[0]

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": dice_sum / max(n_images, 1),
        "iou": iou_sum / max(n_images, 1),
    }


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None,
                    grad_clip: float | None = None, threshold: float = 0.5,
                    progress: bool = True) -> dict:
    """One training pass. Returns mean loss and (train-time) Dice."""
    model.train()
    total_loss, dice_sum, n_images, n_batches = 0.0, 0.0, 0, 0

    iterator = loader
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(loader, desc="train", leave=False)
        except ImportError:
            pass

    for images, masks in iterator:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        dice_sum += float(dice_coefficient(logits.detach().float(), masks, threshold).sum())
        n_images += images.shape[0]
        if progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{total_loss / n_batches:.4f}",
                                 dice=f"{dice_sum / max(n_images, 1):.4f}")

    return {"loss": total_loss / max(n_batches, 1), "dice": dice_sum / max(n_images, 1)}


def train_segmentation(model: nn.Module, loaders: dict[str, DataLoader], criterion: nn.Module,
                       cfg, device, run_name: str | None = None) -> dict:
    """Full segmentation training run. Returns the model, history and output paths."""
    epochs = int(cfg.get("train.epochs", 30))
    patience = int(cfg.get("train.early_stopping_patience", 8))
    threshold = float(cfg.get("eval.threshold", 0.5))
    grad_clip = cfg.get("train.grad_clip")
    run_name = run_name or str(cfg.get("run_name", "unet"))

    dirs = run_dirs(cfg.get("output.root", "outputs/segmentation"), run_name)
    cfg.save(dirs["root"] / "config_used.yaml")

    if "train" not in loaders:
        raise ValueError("loaders must contain a 'train' entry.")
    has_val = "val" in loaders

    model = model.to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    use_amp = bool(cfg.get("train.amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_dice, best_epoch, stale = -np.inf, -1, 0
    best_path = dirs["models"] / f"{run_name}_best.pt"
    last_path = dirs["models"] / f"{run_name}_last.pt"
    history: list[dict] = []

    print(f"\n[train] run='{run_name}' device={device} epochs={epochs} amp={use_amp}")
    print(f"[train] outputs -> {dirs['root']}\n")
    start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_stats = train_one_epoch(model, loaders["train"], criterion, optimizer, device,
                                      scaler=scaler, grad_clip=grad_clip, threshold=threshold,
                                      progress=bool(cfg.get("train.progress", True)))
        row = {"epoch": epoch, "train_loss": train_stats["loss"], "train_dice": train_stats["dice"],
               "lr": optimizer.param_groups[0]["lr"]}

        if has_val:
            val_stats = validate(model, loaders["val"], criterion, device, threshold)
            row.update({"val_loss": val_stats["loss"], "val_dice": val_stats["dice"],
                        "val_iou": val_stats["iou"]})

        row["seconds"] = round(time.time() - epoch_start, 1)
        history.append(row)

        msg = (f"epoch {epoch:>3d}/{epochs}  train_loss={row['train_loss']:.4f} "
               f"train_dice={row['train_dice']:.4f}")
        if has_val:
            msg += f"  val_loss={row['val_loss']:.4f} val_dice={row['val_dice']:.4f} val_iou={row['val_iou']:.4f}"
        msg += f"  ({row['seconds']}s)"

        if has_val:
            if row["val_dice"] > best_dice:
                best_dice, best_epoch, stale = row["val_dice"], epoch, 0
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "val_dice": float(best_dice), "config": cfg.to_dict()}, best_path)
                msg += "  <- best"
            else:
                stale += 1
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(row["val_dice"])
                else:
                    scheduler.step()
        elif scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        print(msg)

        if has_val and patience > 0 and stale >= patience:
            print(f"[train] early stopping: val_dice has not improved for {patience} epochs.")
            break

    torch.save({"model_state": model.state_dict(), "epoch": len(history),
                "config": cfg.to_dict()}, last_path)

    elapsed = time.time() - start
    if best_epoch > 0 and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
        print(f"[train] restored best checkpoint from epoch {best_epoch} (val_dice={best_dice:.4f})")
    else:
        print("[train] no validation improvement recorded; keeping the final-epoch weights.")

    history_df = history_to_csv(history, dirs["metrics"] / "training_history.csv")
    save_run_metadata(
        dirs["root"] / "run_metadata.json",
        config=cfg.to_dict(),
        extra={"run_name": run_name, "best_epoch": best_epoch,
               "best_val_dice": None if not np.isfinite(best_dice) else float(best_dice),
               "epochs_completed": len(history), "total_seconds": round(elapsed, 1),
               "best_checkpoint": str(best_path if best_path.exists() else last_path)},
    )
    print(f"[train] finished in {elapsed / 60:.1f} min")

    return {"model": model, "history": history_df, "best_epoch": best_epoch,
            "best_val_dice": float(best_dice) if np.isfinite(best_dice) else None,
            "dirs": dirs, "best_checkpoint": best_path if best_path.exists() else last_path}


def load_checkpoint(path: str | Path, model: nn.Module, device) -> nn.Module:
    """Load weights saved by :func:`train_segmentation`."""
    ckpt = torch.load(Path(path), map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print(f"[model] loaded checkpoint from epoch {ckpt['epoch']} ({Path(path).name})")
    return model
