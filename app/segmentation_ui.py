"""Streamlit UI for the MRI whole-tumour Segmentation workstream.

Everything rendered by this module belongs to the Segmentation workstream only.
It reads exclusively from:

    data/processed/mri/**
    outputs/segmentation/mri_unet_whole_tumour_2d_<size>/**
    data/presentation_samples/segmentation_{validation,test}/**

It never reads a classification artifact and never writes anything except a
Streamlit cache entry. `app/streamlit_app.py` calls only the three
`render_segmentation_*` entry points at the bottom of this file.

Model shown throughout: "2D slice-wise U-Net using four MRI sequences".
"""

from __future__ import annotations

import html
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# constants mirrored from src.segmentation.mri.constants (imported lazily so the
# UI still renders a clean "not available" state if torch is missing)
# --------------------------------------------------------------------------
MODEL_LABEL = "2D slice-wise U-Net using four MRI sequences"
MODALITY_NAMES = ("FLAIR", "T1w", "T1gd", "T2w")
PROCESSED_MRI = REPO / "data" / "processed" / "mri"
SEG_OUTPUT_ROOT = REPO / "outputs" / "segmentation"
PRESENTATION_VALID = REPO / "data" / "presentation_samples" / "segmentation_validation"
PRESENTATION_TEST = REPO / "data" / "presentation_samples" / "segmentation_test"

GT_HEX = "#ffb020"
PRED_HEX = "#00d0ff"
OVERLAP_HEX = "#7cf0a8"


# ==========================================================================
# artifact discovery
# ==========================================================================
def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(show_spinner=False)
def find_run_dir() -> str | None:
    """Newest completed MRI segmentation run directory, if any."""
    if not SEG_OUTPUT_ROOT.is_dir():
        return None
    candidates = [
        d for d in SEG_OUTPUT_ROOT.glob("mri_unet_whole_tumour_2d_*")
        if (d / "run_metadata.json").exists()
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda d: (d / "run_metadata.json").stat().st_mtime)
    return newest.as_posix()


@st.cache_data(show_spinner=False)
def load_artifacts() -> dict[str, Any]:
    """Every segmentation artifact the UI needs, loaded once."""
    run = find_run_dir()
    run_path = Path(run) if run else None

    art: dict[str, Any] = {
        "run_dir": run_path,
        "run_name": run_path.name if run_path else None,
        "split_summary": _read_json(PROCESSED_MRI / "split_summary.json"),
        "inventory": _read_json(PROCESSED_MRI / "dataset_inventory.json"),
        "manifest": _read_csv(PROCESSED_MRI / "mri_manifest.csv"),
        "run_metadata": None,
        "history": None,
        "val_per_case": None,
        "val_aggregate": None,
        "test_per_case": None,
        "test_aggregate": None,
        "visual_validation": None,
        "val_saved_index": None,
        "test_saved_index": None,
        "presentation_valid": _read_json(PRESENTATION_VALID / "presentation_manifest.json"),
        "presentation_test": _read_json(PRESENTATION_TEST / "presentation_manifest.json"),
    }
    if run_path:
        m = run_path / "metrics"
        art.update(
            {
                "run_metadata": _read_json(run_path / "run_metadata.json"),
                "history": _read_csv(m / "training_history.csv"),
                "val_per_case": _read_csv(m / "validation_per_case_metrics.csv"),
                "val_aggregate": _read_json(m / "validation_aggregate_metrics.json"),
                "test_per_case": _read_csv(m / "heldout_test_per_case_metrics.csv"),
                "test_aggregate": _read_json(m / "heldout_test_aggregate_metrics.json"),
                "visual_validation": _read_json(m / "visual_validation.json"),
                "val_saved_index": _read_json(run_path / "predictions" / "validation_saved_index.json"),
                "test_saved_index": _read_json(run_path / "predictions" / "heldout_test_saved_index.json"),
            }
        )
    return art


def _has_data_prep() -> bool:
    a = load_artifacts()
    return a["split_summary"] is not None


def _has_run() -> bool:
    a = load_artifacts()
    return a["run_metadata"] is not None


# ==========================================================================
# small HTML builders (mirror the Classification visual system)
# ==========================================================================
def _pane_title(number: str, title: str) -> None:
    st.markdown(
        f'<div class="cxr-eval-pane__head">'
        f'<span class="cxr-eval-pane__num">{html.escape(number)}</span>'
        f'<p class="cxr-eval-pane__title">{html.escape(title)}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _metric_cards(items: list[tuple[str, str]], *, columns: int = 4, tone: str = "") -> str:
    cards = "".join(
        f'<div class="cxr-metric-card cxr-seg-metric{tone}">'
        f'<p class="cxr-metric-card__value">{html.escape(value)}</p>'
        f'<p class="cxr-metric-card__label">{html.escape(label)}</p>'
        f"</div>"
        for value, label in items
    )
    return (
        f'<div class="cxr-metric-grid cxr-seg-metric-grid" '
        f'style="grid-template-columns:repeat({columns},minmax(0,1fr));">{cards}</div>'
    )


def _table(headers: list[str], rows: list[list[str]], numeric_cols: set[int] | None = None) -> str:
    numeric = numeric_cols or set()
    head = "".join(
        f'<th class="cxr-num">{html.escape(h)}</th>' if i in numeric else f"<th>{html.escape(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="cxr-num">{html.escape(str(c))}</td>' if i in numeric
            else f"<td>{html.escape(str(c))}</td>"
            for i, c in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="cxr-table-wrap"><table class="cxr-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _pipeline(steps: list[str]) -> str:
    parts = []
    for i, s in enumerate(steps):
        if i:
            parts.append('<span class="cxr-pipeline__arrow">&rarr;</span>')
        parts.append(f'<span class="cxr-pipeline__step">{html.escape(s)}</span>')
    return f'<div class="cxr-pipeline">{"".join(parts)}</div>'


def _note(text: str, *, kind: str = "info") -> str:
    return f'<p class="cxr-seg-note cxr-seg-note--{kind}">{text}</p>'


def _legend() -> str:
    return (
        '<div class="cxr-seg-legend">'
        f'<span><i style="background:{GT_HEX}"></i>Ground truth</span>'
        f'<span><i style="background:{PRED_HEX}"></i>Prediction</span>'
        f'<span><i style="background:{OVERLAP_HEX}"></i>Agreement</span>'
        "</div>"
    )


def _pending(stage: str) -> None:
    st.markdown(
        f"""
        <div class="cxr-empty-state cxr-empty-state--compact">
          <p class="cxr-empty-state__title">MRI segmentation artifacts not available</p>
          <p class="cxr-empty-state__text">
            {html.escape(stage)} needs the presentation asset bundle
            (checkpoint + small samples) and/or committed run metrics.
            Install with <code>scripts/install_presentation_assets.ps1</code>
            — see <code>docs/TEAM_SETUP.md</code>.
            The full Decathlon MRI archive is <strong>not</strong> included in GitHub.
          </p>
          <span class="cxr-badge cxr-badge--planned">Assets required</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


# ==========================================================================
# model loading for inference
# ==========================================================================
@st.cache_resource(show_spinner=False)
def load_segmentation_model() -> dict[str, Any]:
    """Load the frozen best checkpoint. Cached across reruns."""
    import sys

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import torch

    from src.segmentation.mri.preprocess import PreprocessConfig
    from src.segmentation.mri.train import load_checkpoint, resolve_device
    from src.segmentation.mri.unet import build_mri_unet, count_parameters

    art = load_artifacts()
    run_path = art["run_dir"]
    meta = art["run_metadata"]
    asset_hint = (
        "Install the presentation asset bundle "
        "(release v1.0-presentation-demo or docs/TEAM_ASSETS.md), then restart the app."
    )
    if run_path is None or meta is None:
        raise FileNotFoundError(
            f"No trained MRI segmentation run found under outputs/segmentation/. {asset_hint}"
        )

    ckpt_path = run_path / meta["artifacts"]["best_checkpoint"]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint missing: {ckpt_path}. {asset_hint}"
        )

    device = resolve_device("auto")
    model = build_mri_unet(
        in_channels=meta["model"]["in_channels"],
        out_channels=meta["model"]["out_channels"],
        features=tuple(meta["model"]["features"]),
        dropout=meta["model"].get("dropout", 0.0) or 0.0,
    )
    ckpt = load_checkpoint(ckpt_path, model, device=device)
    cfg = PreprocessConfig(
        input_size=int(meta["preprocessing"]["input_size"]),
        slice_axis=int(meta["preprocessing"]["slice_axis"]),
    )
    return {
        "model": model,
        "device": device,
        "cfg": cfg,
        "epoch": int(ckpt["epoch"]),
        "checkpoint_name": ckpt_path.name,
        "threshold": float(meta["evaluation"]["threshold"]),
        "parameters": count_parameters(model),
        "torch": torch.__version__,
    }


@st.cache_data(show_spinner=False, max_entries=6)
def prepare_presentation_case(image_path: str, label_path: str | None, input_size: int):
    """Load + normalize a presentation case (cached; identical to eval preprocessing)."""
    import sys

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.segmentation.mri.preprocess import PreprocessConfig, prepare_case

    cfg = PreprocessConfig(input_size=input_size)
    norm, binary = prepare_case(image_path, label_path, cfg)
    return norm, binary


# ==========================================================================
# PHASE 6 — Data Preparation -> Segmentation
# ==========================================================================
def render_segmentation_data_prep() -> None:
    art = load_artifacts()
    summary = art["split_summary"]

    if summary is None:
        _pending("MRI dataset overview, patient-safe split summary and preprocessing flow")
        return

    src = summary["source"]
    task = summary["task"]
    splits = summary["splits"]
    leak = summary["leakage_check"]
    inv = summary["inventory"]
    total_cases = summary["totals"]["cases"]

    # ---- 01 dataset overview ------------------------------------------
    with st.container(border=True, key="dp_seg_pane_01"):
        _pane_title("01", "Dataset overview")
        st.markdown(
            _metric_cards(
                [
                    (f"{total_cases}", "Labelled MRI cases"),
                    ("4", "MRI sequences per case"),
                    ("Binary", "Whole tumour vs background"),
                    (f"{summary['totals']['total_slices']:,}", "Axial slices available"),
                ]
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            _table(
                ["Property", "Value"],
                [
                    ["Source dataset", src["dataset_name"]],
                    ["Download date (UTC)", src["download_date"]],
                    ["Licence", src["licence"]],
                    ["Modalities", " · ".join(task["input_modalities"])],
                    ["Volume shape", ", ".join(inv["image_shape_histogram"].keys())],
                    ["Source labels", ", ".join(
                        f"{k} = {v}" for k, v in task["source_label_mapping"].items()
                    )],
                    ["Project target", f"binary whole tumour — {task['rule']}"],
                    ["Slice axis", task["slice_axis_name"]],
                    ["Matched image/label pairs", f"{inv['n_matched_pairs']}"],
                    ["Unmatched volumes", f"{inv['n_images_without_label'] + inv['n_labels_without_image']}"],
                    ["Failed integrity checks", f"{inv['n_failed']}"],
                ],
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            _note(
                "Whole tumour merges every documented non-background source label "
                "(1 edema, 2 non-enhancing tumor, 3 enhancing tumour) into a single "
                "foreground class. This is <strong>not</strong> multi-class subregion segmentation."
            ),
            unsafe_allow_html=True,
        )

    # ---- 02 patient-safe split ----------------------------------------
    with st.container(border=True, key="dp_seg_pane_02"):
        _pane_title("02", "Patient-safe split")
        st.markdown(
            _metric_cards(
                [
                    (f"{splits['train']['cases']}", "Train cases"),
                    (f"{splits['valid']['cases']}", "Validation cases"),
                    (f"{splits['test']['cases']}", "Held-out internal test cases"),
                ],
                columns=3,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            _table(
                ["Split", "Cases", "% of cases", "Slices", "Tumour slices", "Mean tumour fraction"],
                [
                    [
                        {"train": "Train", "valid": "Validation",
                         "test": "Held-out internal test"}[s],
                        f"{splits[s]['cases']}",
                        f"{splits[s]['percent_of_cases']:.1f}%",
                        f"{splits[s]['total_slices']:,}",
                        f"{splits[s]['tumour_slices']:,}",
                        f"{splits[s]['mean_tumour_fraction']:.5f}",
                    ]
                    for s in ("train", "valid", "test")
                ],
                numeric_cols={1, 2, 3, 4, 5},
            ),
            unsafe_allow_html=True,
        )
        overlap_ok = leak["zero_patient_overlap"]
        st.markdown(
            f'<div class="cxr-seg-check cxr-seg-check--{"ok" if overlap_ok else "bad"}">'
            f'<span class="cxr-seg-check__mark"></span>'
            f'<span><strong>Zero patient overlap {"confirmed" if overlap_ok else "FAILED"}</strong> — '
            f"train∩validation = {len(leak['train_valid_overlap'])}, "
            f"train∩test = {len(leak['train_test_overlap'])}, "
            f"validation∩test = {len(leak['valid_test_overlap'])}. "
            f"Splitting is at patient/case level with seed {summary['split_policy']['seed']}; "
            "no slice or patch crosses a split boundary.</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            _note(
                "The test split is a <strong>project-created internal held-out split</strong> "
                "drawn from the same public training archive. It is not an external or "
                "clinical test set. It was not used for architecture choice, hyperparameter "
                "tuning, early stopping or threshold selection.",
                kind="warn",
            ),
            unsafe_allow_html=True,
        )

    # ---- 03 preprocessing flow ----------------------------------------
    with st.container(border=True, key="dp_seg_pane_03"):
        _pane_title("03", "Preprocessing flow")
        st.markdown(
            _pipeline(
                [
                    "4 MRI sequences",
                    "nonzero-voxel normalization",
                    "2D slice preparation",
                    "binary whole-tumour mask",
                ]
            ),
            unsafe_allow_html=True,
        )
        meta = art["run_metadata"]
        size = meta["preprocessing"]["input_size"] if meta else "—"
        st.markdown(
            _table(
                ["Step", "Definition"],
                [
                    ["Load", "NIfTI .nii.gz read with nibabel; image/label affines verified equal"],
                    ["Channels", f"4 sequences stacked as input channels — {' · '.join(MODALITY_NAMES)}"],
                    ["Normalization", "per volume, per modality z-score over nonzero brain voxels only; "
                                      "background written back as exactly 0"],
                    ["Binary target", "mask = (label > 0) — every non-background label merged"],
                    ["Slicing", "axial axis (index axis 2 of the 240×240×155 volume)"],
                    ["Resize", f"{size}×{size} — bilinear for the image, nearest-neighbour for masks"],
                    ["Sample shape", f"image (4, {size}, {size}) · mask (1, {size}, {size})"],
                ],
            ),
            unsafe_allow_html=True,
        )
        vv = art["visual_validation"]
        if vv:
            st.markdown(
                f'<div class="cxr-seg-check cxr-seg-check--{"ok" if vv["all_passed"] else "bad"}">'
                f'<span class="cxr-seg-check__mark"></span>'
                f'<span><strong>Mask alignment verified</strong> on {vv["n_cases_checked"]} random '
                "training cases — tumour voxels sit inside brain tissue and on FLAIR-hyperintense "
                "signal, and the tumour centroid falls inside the brain bounding box.</span></div>",
                unsafe_allow_html=True,
            )

    # ---- 04 real visual example ---------------------------------------
    with st.container(border=True, key="dp_seg_pane_04"):
        _pane_title("04", "Real example — MRI slice, ground-truth mask, overlay")
        _render_data_prep_example()


def _render_data_prep_example() -> None:
    """One real training-split slice rendered live from the raw NIfTI."""
    art = load_artifacts()
    manifest = art["manifest"]
    meta = art["run_metadata"]
    if manifest is None:
        _pending("A real MRI example")
        return

    size = int(meta["preprocessing"]["input_size"]) if meta else 192
    train_rows = manifest[(manifest["split"] == "train") & (manifest["tumour_voxels"] > 0)]
    if train_rows.empty:
        _pending("A real MRI example")
        return

    ids = train_rows["case_id"].tolist()
    case_id = st.selectbox(
        "Training case", ids, index=0, key="dp_seg_example_case",
        help="Example drawn from the training split only.",
    )
    row = train_rows[train_rows["case_id"] == case_id].iloc[0]

    try:
        norm, binary = prepare_presentation_case(
            str(REPO / row["image_path"]), str(REPO / row["label_path"]), size
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load {case_id}: {exc}")
        return

    import sys

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.segmentation.mri.preprocess import PreprocessConfig, prepare_slice, tumour_slice_indices
    from src.segmentation.mri.viz import overlay_masks, to_uint8_rgb

    cfg = PreprocessConfig(input_size=size)
    tidx = tumour_slice_indices(binary)
    default = int(tidx[len(tidx) // 2]) if tidx.size else binary.shape[2] // 2
    slice_index = st.slider(
        "Axial slice", 0, int(binary.shape[2]) - 1, default, key="dp_seg_example_slice"
    )

    image, mask = prepare_slice(norm, binary, slice_index, cfg)
    img = image.numpy()
    m = mask.squeeze(0).numpy()

    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.image(to_uint8_rgb(img[0]), caption=f"{MODALITY_NAMES[0]} — slice {slice_index}",
                 use_container_width=True)
    with c2:
        st.image((np.stack([m] * 3, -1) * 255).astype(np.uint8),
                 caption="Ground-truth whole-tumour mask", use_container_width=True)
    with c3:
        st.image(overlay_masks(img[0], gt=m), caption="Overlay", use_container_width=True)

    tumour_px = int(m.sum())
    st.markdown(
        _table(
            ["Case", "Slice", "Tumour pixels", "Slice tumour area", "Modalities", "Split"],
            [[
                case_id, f"{slice_index}", f"{tumour_px:,}",
                f"{100.0 * tumour_px / m.size:.2f}%",
                " · ".join(MODALITY_NAMES), "train",
            ]],
            numeric_cols={1, 2, 3},
        ),
        unsafe_allow_html=True,
    )


# ==========================================================================
# PHASE 7 — Train & Validate -> Segmentation
# ==========================================================================
def render_segmentation_train_validate() -> None:
    art = load_artifacts()
    if not _has_run():
        _pending("Validation Dice/IoU, training curves and qualitative validation examples")
        return

    st.markdown('<div class="cxr-tv-panes-marker"></div>', unsafe_allow_html=True)

    with st.container(border=True, key="tv_seg_pane_01"):
        _pane_title("01", "Segmentation validation performance")
        _panel_validation_performance(art)

    with st.container(border=True, key="tv_seg_pane_02"):
        _pane_title("02", "Training history")
        _panel_training_history(art)

    with st.container(border=True, key="tv_seg_pane_03"):
        _pane_title("03", "Qualitative validation examples")
        _panel_qualitative_validation(art)

    with st.container(border=True, key="tv_seg_pane_04"):
        _pane_title("04", "Per-case validation metrics")
        _panel_per_case_metrics(art)

    with st.container(border=True, key="tv_seg_pane_05"):
        _pane_title("05", "Model and preprocessing")
        _panel_model_and_preprocessing(art)

    with st.container(border=True, key="tv_seg_pane_06"):
        _pane_title("06", "Saved artifact summary")
        _panel_artifacts(art)


def _panel_validation_performance(art: dict) -> None:
    meta, agg = art["run_metadata"], art["val_aggregate"]
    if agg is None:
        _pending("Validation metrics")
        return

    st.markdown(
        '<p class="cxr-seg-scope cxr-seg-scope--val">Validation split — used for model selection</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        _metric_cards(
            [
                (_fmt(agg["mean_dice"]), "Validation Dice (mean per case)"),
                (_fmt(agg["mean_iou"]), "Validation IoU (mean per case)"),
                (_fmt(_final_valid_loss(art)), "Validation loss (best epoch)"),
                (f"{meta['training']['best_epoch']}", "Best epoch"),
            ]
        ),
        unsafe_allow_html=True,
    )
    ci_d, ci_i = agg.get("dice_bootstrap_ci") or {}, agg.get("iou_bootstrap_ci") or {}
    st.markdown(
        _table(
            ["Validation statistic", "Dice", "IoU"],
            [
                ["Mean (per case)", _fmt(agg["mean_dice"]), _fmt(agg["mean_iou"])],
                ["Median", _fmt(agg["dice"]["median"]), _fmt(agg["iou"]["median"])],
                ["IQR", f"{_fmt(agg['dice']['p25'])} – {_fmt(agg['dice']['p75'])}",
                 f"{_fmt(agg['iou']['p25'])} – {_fmt(agg['iou']['p75'])}"],
                ["95% CI (bootstrap)",
                 f"{_fmt(ci_d.get('lo'))} – {_fmt(ci_d.get('hi'))}",
                 f"{_fmt(ci_i.get('lo'))} – {_fmt(ci_i.get('hi'))}"],
                ["Micro (voxel-pooled)", _fmt(agg["micro"]["dice"]), _fmt(agg["micro"]["iou"])],
            ],
            numeric_cols={1, 2},
        ),
        unsafe_allow_html=True,
    )

    test_agg = art["test_aggregate"]
    st.markdown('<div class="cxr-seg-divider"></div>', unsafe_allow_html=True)
    if test_agg:
        st.markdown(
            '<p class="cxr-seg-scope cxr-seg-scope--test">Held-out internal test — '
            "separate one-time generalization check, not used for any selection</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="cxr-seg-testcard">'
            f'<div class="cxr-seg-testcard__row">'
            f'<div><p class="cxr-metric-card__value">{_fmt(test_agg["mean_dice"])}</p>'
            f'<p class="cxr-metric-card__label">Held-out internal test Dice</p></div>'
            f'<div><p class="cxr-metric-card__value">{_fmt(test_agg["mean_iou"])}</p>'
            f'<p class="cxr-metric-card__label">Held-out internal test IoU</p></div>'
            f'<div><p class="cxr-metric-card__value">{test_agg["n_cases"]}</p>'
            f'<p class="cxr-metric-card__label">Test cases (evaluated once)</p></div>'
            f"</div>"
            f'<p class="cxr-seg-testcard__note">Evaluated once after the '
            f"validation-selected checkpoint was frozen. Project-created internal split — "
            f"not external or clinical validation.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            _note("Held-out internal test evaluation has not been run yet.", kind="muted"),
            unsafe_allow_html=True,
        )


def _final_valid_loss(art: dict) -> float | None:
    h, meta = art["history"], art["run_metadata"]
    if h is None or meta is None:
        return None
    best = meta["training"]["best_epoch"]
    row = h[h["epoch"] == best]
    return float(row["valid_loss"].iloc[0]) if len(row) else None


def _panel_training_history(art: dict) -> None:
    import altair as alt

    h = art["history"]
    if h is None or h.empty:
        _pending("Training curves")
        return

    loss = h.melt(
        id_vars="epoch", value_vars=["train_loss", "valid_loss"],
        var_name="series", value_name="value",
    )
    loss["series"] = loss["series"].map(
        {"train_loss": "Train loss", "valid_loss": "Validation loss"}
    )
    dice = h[["epoch", "valid_micro_dice"]].rename(columns={"valid_micro_dice": "value"})
    dice["series"] = "Validation Dice"

    base_cfg = dict(width="container", height=300)
    loss_chart = (
        alt.Chart(loss, **base_cfg)
        .mark_line(point=alt.OverlayMarkDef(size=28), strokeWidth=2.4)
        .encode(
            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1, labelColor="#41505f",
                                                            titleColor="#1f2733", grid=False)),
            y=alt.Y("value:Q", title="Dice + BCE loss",
                    axis=alt.Axis(labelColor="#41505f", titleColor="#1f2733", gridColor="#eef1f6")),
            color=alt.Color(
                "series:N", title=None,
                scale=alt.Scale(domain=["Train loss", "Validation loss"],
                                range=["#2f6fed", "#8b5cf6"]),
                legend=alt.Legend(orient="top", labelColor="#1f2733", symbolStrokeWidth=3),
            ),
            tooltip=["epoch:Q", "series:N", alt.Tooltip("value:Q", format=".4f")],
        )
    )
    dice_chart = (
        alt.Chart(dice, **base_cfg)
        .mark_line(point=alt.OverlayMarkDef(size=28), strokeWidth=2.6)
        .encode(
            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1, labelColor="#41505f",
                                                            titleColor="#1f2733", grid=False)),
            y=alt.Y("value:Q", title="Validation Dice", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(labelColor="#12a594", titleColor="#12a594", gridColor="#eef1f6")),
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(domain=["Validation Dice"], range=["#12a594"]),
                            legend=alt.Legend(orient="top", labelColor="#1f2733", symbolStrokeWidth=3)),
            tooltip=["epoch:Q", alt.Tooltip("value:Q", format=".4f")],
        )
    )
    best = art["run_metadata"]["training"]["best_epoch"]
    rule = (
        alt.Chart(pd.DataFrame({"epoch": [best]}))
        .mark_rule(color="#f59e0b", strokeDash=[4, 3], strokeWidth=1.6)
        .encode(x="epoch:Q")
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.altair_chart(
            (loss_chart + rule).configure_view(fill="#ffffff", stroke="#d7dce5")
            .configure(background="#ffffff"),
            use_container_width=True,
        )
    with c2:
        st.altair_chart(
            (dice_chart + rule).configure_view(fill="#ffffff", stroke="#d7dce5")
            .configure(background="#ffffff"),
            use_container_width=True,
        )
    st.markdown(
        _note(
            f"Amber dashed line marks the best epoch ({best}), selected on validation Dice only. "
            f"{len(h)} epoch(s) completed.",
            kind="muted",
        ),
        unsafe_allow_html=True,
    )


def _panel_qualitative_validation(art: dict) -> None:
    saved = art["val_saved_index"]
    run_dir = art["run_dir"]
    if not saved or run_dir is None:
        _pending("Qualitative validation examples")
        return

    st.markdown(
        '<p class="cxr-seg-scope cxr-seg-scope--val">Validation samples</p>',
        unsafe_allow_html=True,
    )
    labels = [f"{s['case_id']} — Dice {s['dice']:.3f}" for s in saved]
    choice = st.selectbox("Validation case", labels, index=0, key="tv_seg_qual_case")
    entry = saved[labels.index(choice)]

    npz_path = run_dir / entry["file"]
    if not npz_path.exists():
        st.warning(f"Saved prediction missing: {entry['file']}")
        return
    data = np.load(npz_path)

    import sys

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.segmentation.mri.viz import overlay_masks, to_uint8_rgb

    flair, gt, pred = data["flair"], data["gt"], data["pred"]
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        st.image(to_uint8_rgb(flair), caption=f"FLAIR — slice {int(data['slice_index'])}",
                 use_container_width=True)
    with c2:
        st.image((np.stack([gt] * 3, -1) * 255).astype(np.uint8),
                 caption="Ground truth", use_container_width=True)
    with c3:
        st.image((np.stack([pred] * 3, -1) * 255).astype(np.uint8),
                 caption="Predicted mask", use_container_width=True)
    with c4:
        st.image(overlay_masks(flair, gt=gt, pred=pred), caption="Overlay",
                 use_container_width=True)

    st.markdown(_legend(), unsafe_allow_html=True)
    st.markdown(
        f'<p class="cxr-seg-samplecap"><strong>Validation sample</strong> · {entry["case_id"]} · '
        f"slice {int(data['slice_index'])} · case Dice {entry['dice']:.4f} · "
        f"case IoU {entry['iou']:.4f}</p>",
        unsafe_allow_html=True,
    )

    fig = run_dir / "figures" / "validation_qualitative_examples.png"
    if fig.exists():
        with st.expander("All saved validation overlays (figure)", expanded=False):
            st.image(str(fig), use_container_width=True)


def _panel_per_case_metrics(art: dict) -> None:
    df = art["val_per_case"]
    if df is None or df.empty:
        _pending("Per-case validation metrics")
        return

    view = pd.DataFrame(
        {
            "Case ID": df["case_id"],
            "Dice": df["dice"].round(4),
            "IoU": df["iou"].round(4),
            "Tumour area (voxels)": df["gt_tumour_voxels"].astype(int),
            "Predicted area (voxels)": df["pred_tumour_voxels"].astype(int),
            "Tumour area (%)": (df["gt_tumour_fraction"] * 100).round(3),
            "Predicted area (%)": (df["pred_tumour_fraction"] * 100).round(3),
        }
    ).sort_values("Dice", ascending=False)

    st.markdown(
        '<p class="cxr-seg-scope cxr-seg-scope--val">Validation split — '
        f"{len(view)} cases, sorted by Dice</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="cxr-seg-table-wrap">', unsafe_allow_html=True)
    with st.expander(f"Per-case validation metrics ({len(view)} cases)", expanded=True):
        st.dataframe(
            view, use_container_width=True, hide_index=True, height=340,
            column_config={
                "Dice": st.column_config.NumberColumn(format="%.4f"),
                "IoU": st.column_config.NumberColumn(format="%.4f"),
                "Tumour area (voxels)": st.column_config.NumberColumn(format="%d"),
                "Predicted area (voxels)": st.column_config.NumberColumn(format="%d"),
                "Tumour area (%)": st.column_config.NumberColumn(format="%.3f"),
                "Predicted area (%)": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        _note(
            "Dice and IoU are per-case volumetric: confusion counts are accumulated over "
            "every axial slice of the case, then computed once from the case totals.",
            kind="muted",
        ),
        unsafe_allow_html=True,
    )


def _panel_model_and_preprocessing(art: dict) -> None:
    meta = art["run_metadata"]
    if meta is None:
        _pending("Model and preprocessing summary")
        return
    m, t, p, d = meta["model"], meta["training"], meta["preprocessing"], meta["dataset"]

    st.markdown(
        _table(
            ["Setting", "Value"],
            [
                ["Model", m["label"]],
                ["Architecture", f"{m['name']} · encoder features {m['features']} · "
                                 f"{m['parameters']:,} parameters"],
                ["Input channels", f"{m['in_channels']} — {' · '.join(d['modalities'])}"],
                ["Output", f"{m['out_channels']} binary channel · {m['output']}"],
                ["Input size", f"{p['input_size']} × {p['input_size']}"],
                ["Normalization", "per-modality z-score over nonzero brain voxels"],
                ["Target", "binary whole tumour — mask = (label > 0)"],
                ["Loss", t["loss"]],
                ["Optimizer", f"{t['optimizer']} · lr {t['lr']} · weight decay {t['weight_decay']}"],
                ["Scheduler", t["scheduler"]],
                ["Batch size", f"{t['batch_size']}"],
                ["Epochs completed / best", f"{t['epochs_completed']} / {t['best_epoch']}"],
                ["Selection metric", t["selection_metric"]],
                ["Early stopping", f"patience {t['early_stopping_patience']} on validation Dice"],
                ["Mixed precision", "enabled" if t["amp"] else "disabled"],
                ["Seed / deterministic", f"{t['seed']} / {t['deterministic']}"],
                ["Decision threshold", f"{meta['evaluation']['threshold']} (fixed, not tuned)"],
                ["Split counts", f"train {d['train_cases']} · validation {d['valid_cases']} · "
                                 f"held-out internal test {d['test_cases']} cases"],
                ["Training slices", f"{d['train_slices']:,} "
                                    f"({d['train_tumour_slices']:,} tumour · "
                                    f"{d['train_empty_slices']:,} empty)"],
                ["Validation slices", f"{d['valid_slices']:,}"],
                ["Training duration", f"{t['training_minutes']:.1f} min on "
                                      f"{meta['hardware']['gpu'] or meta['hardware']['device']}"],
            ],
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        _note(
            "Slice-wise 2D U-Net: every convolution is 2D and every sample is one axial "
            "slice with four sequence channels. This is not a 3D U-Net."
        ),
        unsafe_allow_html=True,
    )


def _panel_artifacts(art: dict) -> None:
    meta, run_dir = art["run_metadata"], art["run_dir"]
    if meta is None or run_dir is None:
        _pending("Saved artifacts")
        return

    entries = [
        ("Best checkpoint", meta["artifacts"]["best_checkpoint"]),
        ("Last checkpoint", meta["artifacts"]["last_checkpoint"]),
        ("Training history", meta["artifacts"]["training_history"]),
        ("Validation per-case metrics", meta["artifacts"]["validation_per_case"]),
        ("Validation aggregate metrics", meta["artifacts"]["validation_aggregate"]),
        ("Validation overlays (figure)", meta["artifacts"]["validation_overlays"]),
        ("Training history (figure)", meta["artifacts"]["training_history_figure"]),
        ("Saved validation predictions", "predictions/validation/"),
        ("Config used", meta["artifacts"]["config_used"]),
        ("Run metadata", "run_metadata.json"),
        ("Split summary", meta["artifacts"]["split_summary"]),
    ]
    if art["test_aggregate"]:
        entries += [
            ("Held-out test per-case metrics", "metrics/heldout_test_per_case_metrics.csv"),
            ("Held-out test aggregate metrics", "metrics/heldout_test_aggregate_metrics.json"),
            ("Held-out test overlays (figure)", "figures/heldout_test_qualitative_examples.png"),
            ("Saved held-out test predictions", "predictions/heldout_test/"),
        ]

    rows = []
    for label, rel in entries:
        p = run_dir / rel
        if p.is_dir():
            files = list(p.glob("*"))
            size = sum(f.stat().st_size for f in files if f.is_file())
            rows.append([label, rel, f"{len(files)} files", _size(size)])
        elif p.exists():
            rows.append([label, rel, "present", _size(p.stat().st_size)])
        else:
            rows.append([label, rel, "missing", "—"])

    st.markdown(
        _note(
            f"Run directory: <code>outputs/segmentation/{meta['run_name']}/</code> "
            "(paths below are relative to it).",
            kind="muted",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        _table(["Artifact", "Relative path", "Status", "Size"], rows, numeric_cols={3}),
        unsafe_allow_html=True,
    )


def _size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


# ==========================================================================
# PHASE 8 — Inference Demo -> Segmentation
# ==========================================================================
def _presentation_cases() -> list[dict]:
    art = load_artifacts()
    cases: list[dict] = []
    for key, root, label in (
        ("presentation_valid", PRESENTATION_VALID, "Validation"),
        ("presentation_test", PRESENTATION_TEST, "Held-out internal test"),
    ):
        man = art[key]
        if not man:
            continue
        for c in man["cases"]:
            cases.append({**c, "root": root / c["case_id"], "group": label})
    return cases


def render_segmentation_inference() -> None:
    art = load_artifacts()
    if not _has_run():
        _pending("MRI slice selection, predicted tumour mask and overlay")
        return

    try:
        bundle = load_segmentation_model()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Segmentation model unavailable: {exc}")
        return

    meta = art["run_metadata"]
    st.markdown(
        f'<div class="cxr-seg-modelbar">'
        f'<span class="cxr-seg-modelbar__k">Model</span>'
        f'<span class="cxr-seg-modelbar__v">{html.escape(MODEL_LABEL)}</span>'
        f'<span class="cxr-seg-modelbar__k">Checkpoint</span>'
        f'<span class="cxr-seg-modelbar__v">{html.escape(bundle["checkpoint_name"])} '
        f'(epoch {bundle["epoch"]})</span>'
        f'<span class="cxr-seg-modelbar__k">Input</span>'
        f'<span class="cxr-seg-modelbar__v">4 × {bundle["cfg"].input_size}×{bundle["cfg"].input_size}</span>'
        f'<span class="cxr-seg-modelbar__k">Threshold</span>'
        f'<span class="cxr-seg-modelbar__v">{bundle["threshold"]}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    cases = _presentation_cases()
    tab_sample, tab_upload = st.tabs(
        ["Presentation sample (known case)", "Upload NIfTI volume (no ground truth)"]
    )
    with tab_sample:
        if not cases:
            _pending("Presentation samples")
        else:
            _render_sample_inference(cases, bundle, meta)
    with tab_upload:
        _render_upload_inference(bundle)

    st.markdown(
        _note(
            "Single-slice inference only. This demo never triggers a batch evaluation of the "
            "held-out test split — those metrics come from the single frozen evaluation run "
            "offline.",
            kind="muted",
        ),
        unsafe_allow_html=True,
    )


def _render_sample_inference(cases: list[dict], bundle: dict, meta: dict) -> None:
    labels = [f"{c['group']} · {c['case_id']} (case Dice {c['case_dice']:.3f})" for c in cases]
    choice = st.selectbox("Presentation case", labels, index=0, key="inf_seg_case")
    case = cases[labels.index(choice)]

    is_test = case["group"].startswith("Held-out")
    role = (
        "one-time generalization check, not used for model selection"
        if is_test
        else "this split selected the checkpoint"
    )
    st.markdown(
        f'<p class="cxr-seg-scope cxr-seg-scope--{"test" if is_test else "val"}">'
        f'{html.escape(case["group"])} sample — {html.escape(role)}</p>',
        unsafe_allow_html=True,
    )

    img_path = case["root"] / case["image_file"]
    lab_path = case["root"] / case["label_file"]
    if not img_path.exists():
        st.warning(f"Sample file missing: {case['image_file']}")
        return

    try:
        norm, binary = prepare_presentation_case(
            str(img_path), str(lab_path) if lab_path.exists() else None,
            bundle["cfg"].input_size,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load sample: {exc}")
        return

    import sys

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.segmentation.mri.evaluation import (
        ground_truth_masks,
        infer_single_slice,
        slice_metrics,
    )
    from src.segmentation.mri.preprocess import tumour_slice_indices

    n_slices = norm.shape[bundle["cfg"].slice_axis]
    default = int(n_slices // 2)
    if binary is not None:
        tidx = tumour_slice_indices(binary)
        if tidx.size:
            default = int(tidx[len(tidx) // 2])

    col_a, col_b = st.columns([2, 1], gap="medium")
    with col_a:
        slice_index = st.slider("Axial slice", 0, n_slices - 1, default, key="inf_seg_slice")
    with col_b:
        modality = st.selectbox("Modality view", MODALITY_NAMES, index=0, key="inf_seg_modality")

    result = infer_single_slice(
        bundle["model"], norm, slice_index, bundle["cfg"],
        device=bundle["device"], threshold=bundle["threshold"],
    )
    gt_slice = None
    if binary is not None:
        gt_slice = ground_truth_masks(binary, bundle["cfg"])[slice_index]

    _render_inference_panels(result, gt_slice, modality)

    pred_px = result["pred_pixels"]
    total_px = result["pred"].size
    rows = [
        ["Predicted tumour pixels (slice)", f"{pred_px:,}"],
        ["Predicted tumour area (slice)", f"{100.0 * pred_px / total_px:.2f}%"],
        ["Case-level predicted tumour voxels", f"{case['pred_tumour_voxels']:,}"],
        ["Model", MODEL_LABEL],
        ["Checkpoint", f"{bundle['checkpoint_name']} (epoch {bundle['epoch']})"],
        ["Decision threshold", f"{bundle['threshold']}"],
    ]
    st.markdown(_table(["Prediction", "Value"], rows), unsafe_allow_html=True)

    if gt_slice is not None:
        sm = slice_metrics(result["pred"], gt_slice)
        st.markdown(
            f'<div class="cxr-seg-gtcard cxr-seg-gtcard--{"test" if is_test else "val"}">'
            f'<p class="cxr-seg-gtcard__title">Ground truth available — '
            f'{html.escape(case["group"])} case</p>'
            + _table(
                ["Single-slice metric", "Value"],
                [
                    ["Slice Dice", _fmt(sm["dice"])],
                    ["Slice IoU", _fmt(sm["iou"])],
                    ["Slice precision / recall", f"{_fmt(sm['precision'])} / {_fmt(sm['recall'])}"],
                    ["Ground-truth pixels", f"{sm['gt_pixels']:,}"],
                    ["Predicted pixels", f"{sm['pred_pixels']:,}"],
                    ["Whole-case Dice / IoU",
                     f"{case['case_dice']:.4f} / {case['case_iou']:.4f}"],
                ],
            )
            + "</div>",
            unsafe_allow_html=True,
        )


def _render_inference_panels(result: dict, gt_slice, modality: str) -> None:
    import sys

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.segmentation.mri.viz import overlay_masks, to_uint8_rgb

    ch = MODALITY_NAMES.index(modality)
    base = result["image"][ch]
    pred = result["pred"]

    n_cols = 4 if gt_slice is not None else 3
    cols = st.columns(n_cols, gap="small")
    with cols[0]:
        st.image(to_uint8_rgb(base), caption=f"{modality} — slice {result['slice_index']}",
                 use_container_width=True)
    idx = 1
    if gt_slice is not None:
        with cols[idx]:
            st.image((np.stack([gt_slice] * 3, -1) * 255).astype(np.uint8),
                     caption="Ground-truth mask", use_container_width=True)
        idx += 1
    with cols[idx]:
        st.image((np.stack([pred] * 3, -1) * 255).astype(np.uint8),
                 caption="Predicted tumour mask", use_container_width=True)
    with cols[idx + 1]:
        st.image(overlay_masks(base, gt=gt_slice, pred=pred), caption="Overlay",
                 use_container_width=True)
    st.markdown(_legend() if gt_slice is not None else
                f'<div class="cxr-seg-legend"><span><i style="background:{PRED_HEX}"></i>'
                "Prediction</span></div>", unsafe_allow_html=True)


def _render_upload_inference(bundle: dict) -> None:
    st.markdown(
        _note(
            "Upload a 4-modality brain MRI volume as NIfTI (<code>.nii</code> / "
            "<code>.nii.gz</code>) shaped <code>(H, W, D, 4)</code> in the Decathlon channel "
            "order FLAIR · T1w · T1gd · T2w. A single 2D image cannot be used: this model "
            "requires four co-registered MRI sequences as input channels.",
        ),
        unsafe_allow_html=True,
    )

    upload = st.file_uploader(
        "NIfTI volume", type=["nii", "gz"], key="inf_seg_upload",
        help="4-channel MRI volume. Ground truth is not available for uploads.",
    )
    if upload is None:
        return

    import tempfile

    import nibabel as nib

    suffix = ".nii.gz" if upload.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(upload.getvalue())
        tmp_path = Path(tmp.name)

    try:
        vol = np.asanyarray(nib.load(tmp_path).dataobj).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the NIfTI file: {exc}")
        tmp_path.unlink(missing_ok=True)
        return

    if vol.ndim != 4 or vol.shape[3] != 4:
        st.error(
            f"Expected a 4D volume with 4 modality channels, got shape {tuple(vol.shape)}. "
            "This model needs all four MRI sequences — it cannot run on a single image."
        )
        tmp_path.unlink(missing_ok=True)
        return

    import sys

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.segmentation.mri.evaluation import infer_single_slice
    from src.segmentation.mri.preprocess import normalize_volume

    norm = normalize_volume(vol)
    tmp_path.unlink(missing_ok=True)

    n_slices = norm.shape[bundle["cfg"].slice_axis]
    col_a, col_b = st.columns([2, 1], gap="medium")
    with col_a:
        slice_index = st.slider("Axial slice", 0, n_slices - 1, n_slices // 2,
                                key="inf_seg_upload_slice")
    with col_b:
        modality = st.selectbox("Modality view", MODALITY_NAMES, index=0,
                                key="inf_seg_upload_modality")

    result = infer_single_slice(
        bundle["model"], norm, slice_index, bundle["cfg"],
        device=bundle["device"], threshold=bundle["threshold"],
    )
    _render_inference_panels(result, None, modality)

    st.markdown(
        f'<div class="cxr-seg-gtcard cxr-seg-gtcard--none">'
        f'<p class="cxr-seg-gtcard__title">Ground truth unavailable for uploaded volumes</p>'
        f'<p class="cxr-seg-gtcard__body">No Dice or IoU can be reported for this input — '
        f"only the predicted mask and overlay are shown.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        _table(
            ["Prediction", "Value"],
            [
                ["File", upload.name],
                ["Volume shape", f"{vol.shape[0]} × {vol.shape[1]} × {vol.shape[2]} × {vol.shape[3]}"],
                ["Predicted tumour pixels (slice)", f"{result['pred_pixels']:,}"],
                ["Predicted tumour area (slice)", f"{100.0 * result['pred_fraction']:.2f}%"],
                ["Model", MODEL_LABEL],
                ["Checkpoint", f"{bundle['checkpoint_name']} (epoch {bundle['epoch']})"],
                ["Decision threshold", f"{bundle['threshold']}"],
            ],
        ),
        unsafe_allow_html=True,
    )
