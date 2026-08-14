"""GPU-memory + throughput smoke test to choose the U-Net input size.

Measures, for 160x160 and 192x192 at several batch sizes: peak allocated/reserved
VRAM and steady-state step time for a real forward+backward pass on synthetic
4-channel input. Writes the measurements to the run audit directory so the choice
of input size recorded in run_metadata.json is evidence-backed, not asserted.

Usage::

    python scripts/mri_gpu_smoke_test.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.segmentation.mri.losses import DiceBCELoss  # noqa: E402
from src.segmentation.mri.unet import build_mri_unet, count_parameters  # noqa: E402

SIZES = (160, 192)
BATCHES = (8, 16, 24, 32)
FEATURES = (32, 64, 128, 256)
WARMUP, MEASURE = 3, 8


def bench(size: int, batch: int, device: torch.device, amp: bool) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build_mri_unet(in_channels=4, out_channels=1, features=FEATURES).to(device)
    loss_fn = DiceBCELoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    images = torch.randn(batch, 4, size, size, device=device)
    masks = (torch.rand(batch, 1, size, size, device=device) > 0.97).float()

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp):
            logits = model(images)
        loss = loss_fn(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    try:
        for _ in range(WARMUP):
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(MEASURE):
            step()
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / MEASURE

        result = {
            "input_size": size,
            "batch_size": batch,
            "amp": amp,
            "ok": True,
            "step_seconds": round(elapsed, 4),
            "slices_per_second": round(batch / elapsed, 1),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        }
    except torch.cuda.OutOfMemoryError:
        result = {"input_size": size, "batch_size": batch, "amp": amp, "ok": False,
                  "error": "CUDA out of memory"}
    finally:
        del model, loss_fn, opt, images, masks
        torch.cuda.empty_cache()
    return result


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available — smoke test requires a GPU.")
        return 1

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    model = build_mri_unet(in_channels=4, out_channels=1, features=FEATURES)
    n_params = count_parameters(model)
    del model

    print(f"GPU        : {props.name}")
    print(f"total VRAM : {props.total_memory / 1e9:.2f} GB")
    print(f"U-Net params: {n_params:,} (features={FEATURES})\n")

    results = []
    for size in SIZES:
        for batch in BATCHES:
            r = bench(size, batch, device, amp=True)
            results.append(r)
            if r["ok"]:
                print(
                    f"  {size}x{size} batch={batch:<3} "
                    f"peak_alloc={r['peak_allocated_gb']:.2f}GB "
                    f"reserved={r['peak_reserved_gb']:.2f}GB "
                    f"step={r['step_seconds'] * 1000:.0f}ms "
                    f"{r['slices_per_second']:.0f} slices/s"
                )
            else:
                print(f"  {size}x{size} batch={batch:<3} OOM")

    payload = {
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": props.name,
        "total_vram_gb": round(props.total_memory / 1e9, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "unet_features": list(FEATURES),
        "unet_parameters": n_params,
        "amp": True,
        "results": results,
    }
    out_dir = REPO / "outputs" / "segmentation" / "_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "gpu_smoke_test.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten -> {out.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
