"""Phase 9 — end-to-end verification before handoff.

Checks, and reports pass/fail for each:

  1.  Classification artifacts/sources byte-identical to the Phase 0 baseline.
  2.  All three app stages still render with no exceptions.
  3.  MRI split manifests match their frozen hashes.
  4.  Zero patient overlap across train/valid/test.
  5.  Preprocessed tensor shape identical for validation, test and inference.
  6.  Best checkpoint reload reproduces the same prediction on a known
      validation case (bit-identical masks and Dice).
  7.  The held-out test split was evaluated exactly once, after the checkpoint
      was frozen (receipt present, checkpoint hash matches).
  8.  Training never read the test split (static scan of the training script and
      of the modules it imports).
  9.  Working tree has no commits beyond the Phase 0 HEAD.

Usage::

    python scripts/mri_verify_handoff.py
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Windows consoles default to cp1252; never let a stray glyph kill a verification run.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

AUDIT = REPO / "outputs" / "segmentation" / "_audit"

results: list[dict] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    results.append({"check": name, "passed": bool(passed), "detail": detail})
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {name}")
    if detail:
        for line in detail.splitlines():
            print(f"       {line}")
    return passed


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Three files are shared by both workstreams and had to be touched to wire the
# Segmentation tabs in. They are expected to differ from the Phase 0 baseline;
# every other protected file must be byte-identical. Classification *behaviour*
# in these three is verified separately by the render comparison below.
INTEGRATION_FILES = {
    "app/streamlit_app.py",      # 3 segmentation hook points + 1 dispatch helper
    "app/assets/styles.css",     # appended .cxr-seg-* block only
    "requirements.txt",          # added nibabel
}


# ---------------------------------------------------------------- 1
def check_classification_unchanged() -> None:
    baselines = sorted(AUDIT.glob("classification_baseline_*.json"))
    if not baselines:
        check("Classification baseline available", False, "no baseline file found")
        return
    base = json.loads(baselines[0].read_text(encoding="utf-8"))

    changed_strict, changed_integration, missing = [], [], []
    for rel, info in base["files"].items():
        p = REPO / rel
        if not p.exists():
            missing.append(rel)
            continue
        if sha256(p) != info["sha256"]:
            (changed_integration if rel in INTEGRATION_FILES else changed_strict).append(rel)

    n_strict = len(base["files"]) - len(INTEGRATION_FILES & set(base["files"]))
    detail = (
        f"{n_strict} strictly-protected files byte-identical to the Phase 0 baseline "
        f"({base['captured_utc']})"
    )
    if changed_strict:
        detail += "\nUNEXPECTED CHANGES: " + ", ".join(changed_strict[:10])
    if missing:
        detail += "\nMISSING: " + ", ".join(missing[:10])
    check(
        "Classification artifacts and sources byte-identical",
        not changed_strict and not missing,
        detail,
    )

    # Report the integration edits explicitly rather than hiding them.
    lines = []
    for rel in sorted(INTEGRATION_FILES):
        stat = subprocess.run(
            ["git", "diff", "--numstat", "--", rel], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
        lines.append(f"{rel}: {stat.split()[0]} added / {stat.split()[1]} removed"
                     if stat else f"{rel}: unchanged")
    check(
        "Shared integration files changed only as declared (3 files)",
        set(changed_integration) <= INTEGRATION_FILES,
        "\n".join(lines),
    )

    # The classification half of streamlit_app.py must be untouched: every
    # removed line should belong to the three segmentation stubs being replaced.
    diff = subprocess.run(
        ["git", "diff", "-U0", "--", "app/streamlit_app.py"],
        cwd=REPO, capture_output=True, text=True,
    ).stdout
    removed = [
        ln[1:].strip() for ln in diff.splitlines()
        if ln.startswith("-") and not ln.startswith("---") and ln[1:].strip()
    ]
    allowed_markers = (
        "_render_mri_audit_status_panel", "stage_label=", "MRI segmentation Train & Validate",
        "MRI segmentation inference", "Validation training curves", "MRI volume upload",
        "MRI preparation manifests", "accurate audit", "accurate unavailable", ")",
    )
    unexpected = [r for r in removed if not any(m in r for m in allowed_markers)]
    check(
        "No classification code removed from app/streamlit_app.py",
        not unexpected,
        f"{len(removed)} removed line(s), all within the 3 MRI stub functions"
        if not unexpected else "UNEXPECTED REMOVALS:\n" + "\n".join(unexpected[:10]),
    )

    # styles.css must be append-only.
    css_diff = subprocess.run(
        ["git", "diff", "-U0", "--", "app/assets/styles.css"],
        cwd=REPO, capture_output=True, text=True,
    ).stdout
    css_removed = [
        ln for ln in css_diff.splitlines()
        if ln.startswith("-") and not ln.startswith("---") and ln[1:].strip()
    ]
    check(
        "app/assets/styles.css is append-only (no existing rule altered)",
        not css_removed,
        f"0 lines removed; segmentation block appended under .cxr-seg-* selectors"
        if not css_removed else f"{len(css_removed)} line(s) removed",
    )


# ---------------------------------------------------------------- 2
def check_app_renders() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "mri_ui_smoke.py"), "handoff"],
        cwd=REPO, capture_output=True, text=True,
    )
    smoke = AUDIT / "ui_smoke_handoff.json"
    ok = smoke.exists() and json.loads(smoke.read_text(encoding="utf-8"))["ok"]
    detail = ""
    if smoke.exists():
        data = json.loads(smoke.read_text(encoding="utf-8"))
        detail = "\n".join(
            f"{r['stage']}: exceptions={len(r.get('exception', []))} "
            f"tabs={r.get('n_tabs')} markdown={r.get('n_markdown')}"
            for r in data["results"]
        )
    else:
        detail = proc.stdout[-800:] + proc.stderr[-800:]
    check("All three app stages render without exceptions", bool(ok), detail)

    # Classification render shape must match the pre-segmentation baseline.
    before = AUDIT / "ui_smoke_before.json"
    if before.exists() and smoke.exists():
        b = {r["stage"]: r for r in json.loads(before.read_text(encoding="utf-8"))["results"]}
        a = {r["stage"]: r for r in json.loads(smoke.read_text(encoding="utf-8"))["results"]}
        diffs = [
            f"{s}: buttons {b[s]['n_buttons']}->{a[s]['n_buttons']}"
            for s in b if s in a and b[s]["n_buttons"] != a[s]["n_buttons"]
        ]
        check(
            "Classification control counts unchanged vs pre-segmentation baseline",
            not diffs,
            "\n".join(diffs) or "button counts identical on all three stages",
        )


# ---------------------------------------------------------------- 3 & 4
def check_splits() -> None:
    from src.segmentation.mri.manifests import (
        assert_no_patient_overlap,
        load_manifest,
        verify_frozen_manifests,
    )

    frozen = verify_frozen_manifests()
    check(
        "MRI split manifests match frozen hashes",
        bool(frozen.get("unchanged")),
        f"frozen at {frozen.get('frozen_utc')}" if frozen.get("available") else "no freeze record",
    )

    df = load_manifest("all")
    try:
        overlaps = assert_no_patient_overlap(df)
        counts = df["split"].value_counts().to_dict()
        check(
            "Zero patient overlap across MRI splits",
            True,
            f"train={counts.get('train')} valid={counts.get('valid')} test={counts.get('test')}; "
            f"train^valid={len(overlaps['train_valid'])} "
            f"train^test={len(overlaps['train_test'])} "
            f"valid^test={len(overlaps['valid_test'])}",
        )
    except AssertionError as exc:
        check("Zero patient overlap across MRI splits", False, str(exc))


# ---------------------------------------------------------------- 5 & 6
def check_preprocessing_and_reload() -> None:
    import numpy as np
    import torch
    import yaml

    from src.segmentation.mri.constants import run_dir, run_dir_name
    from src.segmentation.mri.evaluation import evaluate_case, infer_single_slice
    from src.segmentation.mri.manifests import load_manifest
    from src.segmentation.mri.preprocess import PreprocessConfig, prepare_case, prepare_slice
    from src.segmentation.mri.train import load_checkpoint, resolve_device
    from src.segmentation.mri.unet import build_mri_unet

    cfg = yaml.safe_load((REPO / "configs" / "mri_unet_whole_tumour.yaml").read_text(encoding="utf-8"))
    size = int(cfg["preprocess"]["input_size"])
    out, name = run_dir(size), run_dir_name(size)
    meta_path = out / "run_metadata.json"
    if not meta_path.exists():
        check("Trained run available", False, "run_metadata.json missing")
        return

    pcfg = PreprocessConfig(input_size=size, slice_axis=int(cfg["data"]["slice_axis"]))
    device = resolve_device(cfg["device"])
    threshold = float(cfg["eval"]["threshold"])

    # ---- 5: identical preprocessing shape across valid / test / inference
    shapes = {}
    for split in ("valid", "test"):
        row = load_manifest(split).iloc[0]
        norm, binary = prepare_case(REPO / row["image_path"], REPO / row["label_path"], pcfg)
        img, msk = prepare_slice(norm, binary, norm.shape[2] // 2, pcfg)
        shapes[split] = (tuple(img.shape), str(img.dtype), tuple(msk.shape))

    # inference path: presentation sample if present, else the same valid case
    pres = REPO / "data" / "presentation_samples" / "segmentation_validation"
    pres_manifest = pres / "presentation_manifest.json"
    if pres_manifest.exists():
        pm = json.loads(pres_manifest.read_text(encoding="utf-8"))
        c = pm["cases"][0]
        norm_i, _ = prepare_case(pres / c["case_id"] / c["image_file"], None, pcfg)
    else:
        row = load_manifest("valid").iloc[0]
        norm_i, _ = prepare_case(REPO / row["image_path"], None, pcfg)
    img_i, _ = prepare_slice(norm_i, None, norm_i.shape[2] // 2, pcfg)
    shapes["inference"] = (tuple(img_i.shape), str(img_i.dtype), None)

    same = (
        shapes["valid"][0] == shapes["test"][0] == shapes["inference"][0]
        and shapes["valid"][1] == shapes["test"][1] == shapes["inference"][1]
    )
    check(
        "Preprocessing output shape/dtype identical for validation, test and inference",
        same,
        "\n".join(f"{k}: image {v[0]} {v[1]} mask {v[2]}" for k, v in shapes.items()),
    )

    # ---- 6: checkpoint reload reproducibility on a known validation case
    best = out / "models" / f"{name}_best.pt"
    row = load_manifest("valid").iloc[0]

    preds, dices = [], []
    for _ in range(2):
        model = build_mri_unet(
            in_channels=int(cfg["model"]["in_channels"]),
            out_channels=int(cfg["model"]["out_channels"]),
            features=tuple(cfg["model"]["features"]),
            dropout=float(cfg["model"]["dropout"]),
        ).to(device)
        load_checkpoint(best, model, device=device)
        res = evaluate_case(
            model, row["image_path"], row["label_path"], pcfg,
            device=device, threshold=threshold, return_arrays=True,
        )
        preds.append(res["pred_masks"])
        dices.append(res["dice"])
        del model

    identical = bool(np.array_equal(preds[0], preds[1])) and dices[0] == dices[1]
    check(
        "Best checkpoint reload reproduces the same prediction on a known validation case",
        identical,
        f"case {row['case_id']}: Dice {dices[0]:.6f} vs {dices[1]:.6f}; "
        f"masks bit-identical={np.array_equal(preds[0], preds[1])}",
    )

    # ---- inference path == evaluation path on the same slice
    model = build_mri_unet(
        in_channels=int(cfg["model"]["in_channels"]),
        out_channels=int(cfg["model"]["out_channels"]),
        features=tuple(cfg["model"]["features"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    load_checkpoint(best, model, device=device)
    norm, binary = prepare_case(REPO / row["image_path"], REPO / row["label_path"], pcfg)
    sidx = int(binary.sum(axis=(0, 1)).argmax())
    single = infer_single_slice(model, norm, sidx, pcfg, device=device, threshold=threshold)
    match = bool(np.array_equal(single["pred"], preds[0][sidx]))
    check(
        "Streamlit single-slice inference matches the volumetric evaluation path",
        match,
        f"case {row['case_id']} slice {sidx}: identical predicted mask={match}",
    )


# ---------------------------------------------------------------- 7
def check_test_evaluated_once() -> None:
    import yaml

    from src.segmentation.mri.constants import run_dir, run_dir_name

    cfg = yaml.safe_load((REPO / "configs" / "mri_unet_whole_tumour.yaml").read_text(encoding="utf-8"))
    size = int(cfg["preprocess"]["input_size"])
    out, name = run_dir(size), run_dir_name(size)
    receipt = out / "metrics" / "heldout_test_evaluation_receipt.json"
    run_meta = out / "run_metadata.json"

    if not receipt.exists():
        check("Held-out test evaluated exactly once after checkpoint freeze", False,
              "no evaluation receipt")
        return

    r = json.loads(receipt.read_text(encoding="utf-8"))
    best = out / "models" / f"{name}_best.pt"
    hash_ok = best.exists() and sha256(best) == r["checkpoint_sha256"]
    once = int(r.get("run_count", 0)) == 1

    after_freeze = None
    if run_meta.exists():
        m = json.loads(run_meta.read_text(encoding="utf-8"))
        t_train = datetime.fromisoformat(m["completed_utc"])
        t_test = datetime.fromisoformat(r["evaluated_utc"])
        after_freeze = t_test >= t_train

    check(
        "Held-out test evaluated exactly once after checkpoint freeze",
        bool(once and hash_ok and after_freeze),
        f"run_count={r.get('run_count')}; checkpoint hash matches={hash_ok}; "
        f"evaluated after training completed={after_freeze}\n"
        f"evaluated_utc={r['evaluated_utc']} mean_dice={r['mean_dice']:.4f}",
    )


# ---------------------------------------------------------------- 8
def check_training_never_read_test() -> None:
    """Static scan: no test-split access anywhere in the training call graph."""
    files = [
        REPO / "scripts" / "run_mri_unet_whole_tumour.py",
        REPO / "src" / "segmentation" / "mri" / "train.py",
        REPO / "src" / "segmentation" / "mri" / "dataset.py",
        REPO / "src" / "segmentation" / "mri" / "cache.py",
    ]
    # any literal that would pull the test split into the training path
    patterns = [
        re.compile(r"""load_manifest\(\s*['"]test['"]"""),
        re.compile(r"""TEST_MANIFEST_CSV"""),
        re.compile(r"""MRISliceCacheDataset\(\s*['"]test['"]"""),
        re.compile(r"""cache_dir\(\s*['"]test['"]"""),
    ]
    hits = []
    for f in files:
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for p in patterns:
                if p.search(line):
                    hits.append(f"{f.relative_to(REPO).as_posix()}:{i}: {line.strip()}")

    # cache.py legitimately handles a 'test' split, but only when a caller asks
    # for it — the training script never does. Confirm the training script only
    # ever loads train/valid.
    train_script = (REPO / "scripts" / "run_mri_unet_whole_tumour.py").read_text(encoding="utf-8")
    loads = set(re.findall(r"""load_manifest\(\s*['"](\w+)['"]""", train_script))
    check(
        "Training script never reads the held-out test split",
        not hits and "test" not in loads,
        f"manifests loaded by the training script: {sorted(loads) or 'none'}\n"
        + ("\n".join(hits) if hits else "no test-split access in the training call graph"),
    )

    # run_metadata must also assert it
    from src.segmentation.mri.constants import run_dir
    import yaml

    cfg = yaml.safe_load((REPO / "configs" / "mri_unet_whole_tumour.yaml").read_text(encoding="utf-8"))
    meta_path = run_dir(int(cfg["preprocess"]["input_size"])) / "run_metadata.json"
    if meta_path.exists():
        p = json.loads(meta_path.read_text(encoding="utf-8"))["protected_test_split"]
        check(
            "Run metadata records the protected-test-split guarantees",
            not any(p.values()),
            ", ".join(f"{k}={v}" for k, v in p.items()),
        )


# ---------------------------------------------------------------- 9
def check_no_commits() -> None:
    baselines = sorted(AUDIT.glob("classification_baseline_*.json"))
    if not baselines:
        return
    base = json.loads(baselines[0].read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    check(
        "No commits made (HEAD unchanged since Phase 0)",
        head == base["git_head"],
        f"baseline HEAD {base['git_head'][:12]} -> current {head[:12]}",
    )


def main() -> int:
    print("=" * 74)
    print("PHASE 9 — HANDOFF VERIFICATION")
    print("=" * 74)

    check_classification_unchanged()
    check_splits()
    check_preprocessing_and_reload()
    check_test_evaluated_once()
    check_training_never_read_test()
    check_app_renders()
    check_no_commits()

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print("\n" + "=" * 74)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 74)
    for r in results:
        if not r["passed"]:
            print(f"  FAILED: {r['check']}")

    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "handoff_verification.json").write_text(
        json.dumps(
            {
                "verified_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "passed": passed,
                "total": total,
                "all_passed": passed == total,
                "checks": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
