"""Training loop for the 2D slice-wise MRI whole-tumour U-Net.

Model selection rule (enforced here, not by convention):
  * the only quantity that drives checkpointing, LR scheduling and early stopping
    is **validation** Dice;
  * the held-out internal test split is never loaded by this module — it has no
    code path to it at all.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import ConfusionCounts, batch_confusion, batch_mean_slice_dice


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_slice_dice: float
    valid_loss: float
    valid_slice_dice: float
    valid_micro_dice: float
    valid_micro_iou: float
    lr: float
    epoch_seconds: float
    is_best: bool = False


@dataclass
class TrainState:
    history: list[EpochRecord] = field(default_factory=list)
    best_metric: float = -1.0
    best_epoch: int = -1
    epochs_without_improvement: int = 0


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: "torch.amp.GradScaler | None" = None,
    amp: bool = False,
    grad_clip: float | None = 1.0,
    threshold: float = 0.5,
    progress_every: int = 100,
    tag: str = "train",
) -> dict:
    """One pass. `optimizer=None` -> evaluation mode (no grad, no augmentation)."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_slices = 0
    dice_sum = 0.0
    dice_batches = 0
    counts = ConfusionCounts()
    n_batches = len(loader)
    started = time.time()

    for step, batch in enumerate(loader, 1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
            loss = loss_fn(logits, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        bs = images.size(0)
        total_loss += float(loss.detach()) * bs
        total_slices += bs
        dice_sum += batch_mean_slice_dice(logits.detach(), masks, threshold=threshold)
        dice_batches += 1
        counts = counts + batch_confusion(logits.detach(), masks, threshold=threshold)

        if progress_every and (step % progress_every == 0 or step == n_batches):
            elapsed = time.time() - started
            print(
                f"    [{tag}] {step}/{n_batches} loss={total_loss / max(total_slices, 1):.4f} "
                f"slice_dice={dice_sum / max(dice_batches, 1):.4f} ({elapsed:.0f}s)",
                flush=True,
            )

    return {
        "loss": total_loss / max(total_slices, 1),
        "slice_dice": dice_sum / max(dice_batches, 1),
        "micro_dice": counts.dice,
        "micro_iou": counts.iou,
        "counts": counts,
        "seconds": time.time() - started,
        "n_slices": total_slices,
    }


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    loss_fn: torch.nn.Module,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool = True,
    grad_clip: float | None = 1.0,
    threshold: float = 0.5,
    early_stopping_patience: int = 8,
    scheduler_patience: int = 3,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
    checkpoint_dir: Path,
    run_name: str,
    history_csv: Path,
    train_dataset=None,
) -> TrainState:
    """Train, selecting the checkpoint on validation Dice only."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_csv.parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=scheduler_factor,
        patience=scheduler_patience, min_lr=min_lr,
    )
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    state = TrainState()
    best_path = checkpoint_dir / f"{run_name}_best.pt"
    last_path = checkpoint_dir / f"{run_name}_last.pt"

    import csv

    with history_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["epoch", "train_loss", "train_slice_dice", "valid_loss", "valid_slice_dice",
             "valid_micro_dice", "valid_micro_iou", "lr", "epoch_seconds", "is_best"]
        )

    for epoch in range(1, epochs + 1):
        if train_dataset is not None and hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)

        print(f"\n[epoch {epoch}/{epochs}]", flush=True)
        tr = run_epoch(
            model, train_loader, loss_fn, device=device, optimizer=optimizer,
            scaler=scaler, amp=use_amp, grad_clip=grad_clip, threshold=threshold, tag="train",
        )
        va = run_epoch(
            model, valid_loader, loss_fn, device=device, optimizer=None,
            amp=use_amp, threshold=threshold, tag="valid",
        )

        # ---- model selection: validation Dice, nothing else --------------
        selection_metric = va["micro_dice"]
        scheduler.step(selection_metric)
        current_lr = float(optimizer.param_groups[0]["lr"])

        is_best = selection_metric > state.best_metric
        if is_best:
            state.best_metric = selection_metric
            state.best_epoch = epoch
            state.epochs_without_improvement = 0
        else:
            state.epochs_without_improvement += 1

        rec = EpochRecord(
            epoch=epoch,
            train_loss=tr["loss"],
            train_slice_dice=tr["slice_dice"],
            valid_loss=va["loss"],
            valid_slice_dice=va["slice_dice"],
            valid_micro_dice=va["micro_dice"],
            valid_micro_iou=va["micro_iou"],
            lr=current_lr,
            epoch_seconds=tr["seconds"] + va["seconds"],
            is_best=is_best,
        )
        state.history.append(rec)

        with history_csv.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                [rec.epoch, f"{rec.train_loss:.6f}", f"{rec.train_slice_dice:.6f}",
                 f"{rec.valid_loss:.6f}", f"{rec.valid_slice_dice:.6f}",
                 f"{rec.valid_micro_dice:.6f}", f"{rec.valid_micro_iou:.6f}",
                 f"{rec.lr:.8f}", f"{rec.epoch_seconds:.1f}", int(rec.is_best)]
            )

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "valid_micro_dice": va["micro_dice"],
            "valid_micro_iou": va["micro_iou"],
            "valid_loss": va["loss"],
            "selection_metric": "validation micro Dice",
            "run_name": run_name,
        }
        torch.save(payload, last_path)
        if is_best:
            torch.save(payload, best_path)

        flag = "  <-- best" if is_best else ""
        print(
            f"  epoch {epoch}: train_loss={rec.train_loss:.4f} valid_loss={rec.valid_loss:.4f} "
            f"valid_Dice={rec.valid_micro_dice:.4f} valid_IoU={rec.valid_micro_iou:.4f} "
            f"lr={current_lr:.2e} ({rec.epoch_seconds:.0f}s){flag}",
            flush=True,
        )

        if state.epochs_without_improvement >= early_stopping_patience:
            print(
                f"\n[early stopping] no validation Dice improvement for "
                f"{early_stopping_patience} epochs (best epoch {state.best_epoch}, "
                f"Dice {state.best_metric:.4f})",
                flush=True,
            )
            break

    return state


def load_checkpoint(path: Path, model: torch.nn.Module, *, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return ckpt
