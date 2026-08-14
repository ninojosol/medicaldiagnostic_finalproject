"""
Medical Imaging AI Research Console — Streamlit UI.

Academic decision-support prototype. Not for clinical diagnosis or treatment.
Shared AI-engineering lifecycle with Chest X-ray Classification and MRI Segmentation tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import io
import json
import sys
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from PIL import Image
except ImportError:  # pragma: no cover - reported at runtime if missing
    Image = None  # type: ignore[assignment, misc]


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STYLES_PATH = APP_DIR / "assets" / "styles.css"

XRAY_TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "xray" / "train_clean.csv"
XRAY_VALID_CSV = PROJECT_ROOT / "data" / "processed" / "xray" / "valid_clean.csv"
XRAY_TEST_CSV = PROJECT_ROOT / "data" / "raw" / "xray" / "nih" / "test.csv"
XRAY_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "xray" / "nih" / "images-small"

XRAY_DISEASE_COLUMNS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
)
XRAY_REQUIRED_COLUMNS = ("Image", "PatientId") + XRAY_DISEASE_COLUMNS
XRAY_EXPECTED_ROW_COUNTS = {"train": 795, "valid": 207, "test": 420}

# NIH 14 labels as they appear in the saved validation artifacts.
# These match the label tokens inside metrics_validation.csv.
XRAY_NIH14_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
)

# Saved validation-only artifact roots (never touch the protected official test set).
XRAY_BASELINE_ARTIFACT_DIR = (
    PROJECT_ROOT / "outputs" / "classification" / "xray_baseline_cnn_from_scratch_multilabel_320"
)
XRAY_DENSENET_ARTIFACT_DIR = (
    PROJECT_ROOT / "outputs" / "classification" / "xray_finetuned_densenet_multilabel_320"
)
XRAY_EFFICIENTNET_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "xray_finetuned_efficientnet_b0_multilabel_320"
)
XRAY_BASELINE_CHECKPOINT = (
    XRAY_BASELINE_ARTIFACT_DIR
    / "models"
    / "xray_baseline_cnn_from_scratch_multilabel_320_best.pt"
)
XRAY_BASELINE_THRESHOLDS = XRAY_BASELINE_ARTIFACT_DIR / "metrics" / "thresholds.json"
XRAY_BASELINE_RUN_META = XRAY_BASELINE_ARTIFACT_DIR / "run_metadata.json"
XRAY_DENSENET_CHECKPOINT = (
    XRAY_DENSENET_ARTIFACT_DIR
    / "models"
    / "xray_finetuned_densenet_multilabel_320_best.pt"
)
XRAY_DENSENET_THRESHOLDS = XRAY_DENSENET_ARTIFACT_DIR / "metrics" / "thresholds.json"
XRAY_DENSENET_RUN_META = XRAY_DENSENET_ARTIFACT_DIR / "run_metadata.json"
XRAY_EFFICIENTNET_CHECKPOINT = (
    XRAY_EFFICIENTNET_ARTIFACT_DIR
    / "models"
    / "xray_finetuned_efficientnet_b0_multilabel_320_best.pt"
)
XRAY_EFFICIENTNET_THRESHOLDS = XRAY_EFFICIENTNET_ARTIFACT_DIR / "metrics" / "thresholds.json"
XRAY_EFFICIENTNET_RUN_META = XRAY_EFFICIENTNET_ARTIFACT_DIR / "run_metadata.json"
XRAY_VIT_ARTIFACT_DIR = (
    PROJECT_ROOT / "outputs" / "classification" / "xray_finetuned_vit_b16_multilabel_224"
)
XRAY_VIT_CHECKPOINT = (
    XRAY_VIT_ARTIFACT_DIR / "models" / "xray_finetuned_vit_b16_multilabel_224_best.pt"
)
XRAY_VIT_THRESHOLDS = XRAY_VIT_ARTIFACT_DIR / "metrics" / "thresholds.json"
XRAY_VIT_RUN_META = XRAY_VIT_ARTIFACT_DIR / "run_metadata.json"
# Frozen three-model record. Read-only here; never written by the app.
XRAY_MODEL_COMPARISON_JSON = (
    PROJECT_ROOT / "outputs" / "classification" / "model_comparison.json"
)
# Derived four-model record: the frozen three entries copied verbatim plus the
# experimental ViT entry. This is the source for the four-model comparison UI.
XRAY_MODEL_COMPARISON_WITH_VIT_JSON = (
    PROJECT_ROOT / "outputs" / "classification" / "model_comparison_with_vit_experimental.json"
)
XRAY_VIT_DISPLAY_NAME = "Fine-tuned ViT-B/16"
# ViT leads on validation macro ROC-AUC but is an experimental comparator, not the
# frozen selection. Never shorten this to "best" or "selected".
XRAY_VIT_STATUS = "Validation-leading experimental comparator"
XRAY_INFERENCE_ACCEPTED_SUFFIXES = {".png", ".jpg", ".jpeg"}
XRAY_INFERENCE_DISCLAIMER = (
    "Academic demonstration only. These model outputs are research results, not a medical "
    "diagnosis, treatment recommendation, or clinical decision."
)

# Four-model comparison specs for Inference Demo (no ensemble, no retraining).
# The first three are the frozen runs and are unchanged; ViT-B/16 is the experimental
# comparator and loads strictly from its own checkpoint, config and thresholds.
XRAY_COMPARISON_MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "baseline_cnn",
        "display_name": "Baseline CNN",
        "artifact_dir": XRAY_BASELINE_ARTIFACT_DIR,
        "checkpoint": XRAY_BASELINE_CHECKPOINT,
        "thresholds": XRAY_BASELINE_THRESHOLDS,
        "run_meta": XRAY_BASELINE_RUN_META,
        "comparison_json_key": "baseline_cnn",
    },
    {
        "key": "fine_tuned_densenet",
        "display_name": "Fine-tuned DenseNet",
        "artifact_dir": XRAY_DENSENET_ARTIFACT_DIR,
        "checkpoint": XRAY_DENSENET_CHECKPOINT,
        "thresholds": XRAY_DENSENET_THRESHOLDS,
        "run_meta": XRAY_DENSENET_RUN_META,
        "comparison_json_key": "fine_tuned_densenet",
    },
    {
        "key": "fine_tuned_efficientnet_b0",
        "display_name": "Fine-tuned EfficientNet-B0",
        "artifact_dir": XRAY_EFFICIENTNET_ARTIFACT_DIR,
        "checkpoint": XRAY_EFFICIENTNET_CHECKPOINT,
        "thresholds": XRAY_EFFICIENTNET_THRESHOLDS,
        "run_meta": XRAY_EFFICIENTNET_RUN_META,
        "comparison_json_key": "fine_tuned_efficientnet_b0",
    },
    {
        "key": "fine_tuned_vit_b16",
        "display_name": XRAY_VIT_DISPLAY_NAME,
        "artifact_dir": XRAY_VIT_ARTIFACT_DIR,
        "checkpoint": XRAY_VIT_CHECKPOINT,
        "thresholds": XRAY_VIT_THRESHOLDS,
        "run_meta": XRAY_VIT_RUN_META,
        "comparison_json_key": "fine_tuned_vit_b16",
    },
)
XRAY_COMPARISON_DEFAULT_CHART_KEY = "fine_tuned_efficientnet_b0"

# (stage number, display label used for state + content routing)
PIPELINE_STAGES = (
    ("01", "Data Preparation"),
    ("02", "Train & Validate"),
    ("03", "Inference Demo"),
)

PREP_ACTIVITIES = (
    "01 Dataset Overview",
    "02 Quality Checks",
    "03 Safe Train / Validation / Test Split",
    "04 Preprocessing",
)

TRAIN_ACTIVITIES = (
    "01. Training Objective",
    "02. Model & Input Design",
    "03. Loss & Optimization",
    "04. Validation Monitoring",
    "05. Performance Evaluation",
    "06. Model Selection Gate",
)

# Shared Classification / Segmentation workstream tab labels (Markdown + Material icons).
# First tab is the default selected workstream on every workflow page.
WORKSTREAM_TAB_CLASSIFICATION = (
    ":material/radiology: **Classification**  \n"
    ":gray[Chest X-ray · 14 findings]"
)
WORKSTREAM_TAB_SEGMENTATION = (
    ":material/neurology: **Segmentation**  \n"
    ":gray[Brain MRI · tumor mask]"
)
WORKSTREAM_CONTEXT_CLASSIFICATION = "Chest X-ray Classification"
WORKSTREAM_CONTEXT_SEGMENTATION = "Brain MRI Segmentation"


def load_styles() -> None:
    """Inject external CSS while keeping presentation out of Python logic."""
    if STYLES_PATH.is_file():
        css = STYLES_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _val_macro_roc_auc_from_artifacts(
    *,
    run_meta_path: Path,
    artifact_dir: Path,
    comparison_json_key: str,
) -> float | None:
    """Load validation macro ROC-AUC from saved artifacts only (never recompute)."""
    if run_meta_path.is_file():
        try:
            meta = _read_json_file(run_meta_path)
            score = meta.get("best_score")
            monitor = str(meta.get("monitor") or "")
            if score is not None and monitor in {"", "val_roc_auc"}:
                return float(score)
        except Exception:  # noqa: BLE001
            pass

    metrics_csv = artifact_dir / "metrics" / "metrics_validation.csv"
    if metrics_csv.is_file():
        try:
            df = pd.read_csv(metrics_csv)
            macro = df.loc[df["label"].astype(str) == "MACRO_AVG"]
            if not macro.empty and "roc_auc" in macro.columns:
                return float(macro.iloc[0]["roc_auc"])
        except Exception:  # noqa: BLE001
            pass

    # The four-model record is checked first because it is the only one carrying the
    # ViT entry; both files are opened read-only.
    for source in (XRAY_MODEL_COMPARISON_WITH_VIT_JSON, XRAY_MODEL_COMPARISON_JSON):
        if not source.is_file():
            continue
        try:
            comparison = _read_json_file(source)
            macro = (comparison.get(comparison_json_key) or {}).get("val_macro_metrics") or {}
            if macro.get("roc_auc") is not None:
                return float(macro["roc_auc"])
        except Exception:  # noqa: BLE001
            pass
    return None


def _build_xray_architecture(
    model_cfg: dict[str, Any],
    *,
    num_labels: int,
):
    """Instantiate the exact saved architecture; checkpoint weights replace pretrained init."""
    from src.classification.models import SimpleCNN, build_transfer_model

    name = str(model_cfg.get("name") or "").lower()
    dropout = float(model_cfg.get("dropout", 0.2))
    if name in {"simple_cnn", "baseline", "simplecnn"}:
        return SimpleCNN(
            num_labels=num_labels,
            base_width=int(model_cfg.get("base_width", 32)),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )
    if name in {"densenet121", "efficientnet_b0", "efficientnet_b3", "resnet18", "resnet50",
                "vit_b_16", "vit_b_32"}:
        # pretrained=False avoids an ImageNet download; saved weights replace the backbone.
        # build_transfer_model swaps the ViT `heads` module for Dropout+Linear(768, 14),
        # exactly as the saved ViT run did, so the checkpoint loads with strict=True.
        return build_transfer_model(
            name=name,
            num_labels=num_labels,
            pretrained=False,
            freeze_backbone=False,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported saved model architecture: {name!r}")


def _load_frozen_xray_model_bundle(spec: dict[str, Any]) -> dict[str, Any]:
    """Load one frozen checkpoint with its matching preprocessing and label order.

    Raises if the checkpoint or matching configuration artifacts are missing/mismatched.
    Does not train, evaluate the protected test set, or alter thresholds/metrics.
    """
    from src.classification.train import load_checkpoint
    from src.classification.transforms import build_transforms
    from src.common.device import get_device

    checkpoint: Path = spec["checkpoint"]
    thresholds_path: Path = spec["thresholds"]
    run_meta_path: Path = spec["run_meta"]
    display_name = str(spec["display_name"])

    asset_hint = (
        "Install the presentation asset bundle (see docs/TEAM_SETUP.md), then restart the app."
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing {display_name} checkpoint: {checkpoint}. {asset_hint}"
        )
    if not run_meta_path.is_file():
        raise FileNotFoundError(
            f"Missing {display_name} run metadata / preprocessing config: "
            f"{run_meta_path}. {asset_hint}"
        )
    if not thresholds_path.is_file():
        raise FileNotFoundError(
            f"Missing {display_name} validation thresholds: {thresholds_path}. {asset_hint}"
        )

    meta = _read_json_file(run_meta_path)
    cfg = meta.get("config") or {}
    data_cfg = cfg.get("data") or {}
    model_cfg = cfg.get("model") or {}

    labels = list(data_cfg.get("labels") or [])
    if not labels:
        raise ValueError(f"{display_name}: saved label order is missing from run metadata.")
    if labels != list(XRAY_NIH14_LABELS):
        raise ValueError(
            f"{display_name}: saved label order does not match the expected NIH-14 labels."
        )

    thresholds_raw = _read_json_file(thresholds_path)
    thresholds = {label: float(thresholds_raw[label]) for label in labels}

    image_size = int(data_cfg.get("image_size", 320))
    # Match the saved run's preprocessing: pretrained flag selects ImageNet vs neutral norm.
    transform_pretrained = bool(model_cfg.get("pretrained", True))
    transform = build_transforms(
        image_size=image_size,
        train=False,
        pretrained=transform_pretrained,
        aug=None,
    )

    model = _build_xray_architecture(model_cfg, num_labels=len(labels))
    device = get_device("auto")
    model = load_checkpoint(checkpoint, model, device)
    model.eval()

    val_macro_roc_auc = _val_macro_roc_auc_from_artifacts(
        run_meta_path=run_meta_path,
        artifact_dir=spec["artifact_dir"],
        comparison_json_key=str(spec["comparison_json_key"]),
    )

    return {
        "available": True,
        "key": spec["key"],
        "display_name": display_name,
        "model": model,
        "device": device,
        "labels": labels,
        "thresholds": thresholds,
        "transform": transform,
        "image_size": image_size,
        "checkpoint_path": str(checkpoint.resolve()),
        "run_name": str(meta.get("run_name") or spec["artifact_dir"].name),
        "val_macro_roc_auc": val_macro_roc_auc,
        "model_name": str(model_cfg.get("name") or ""),
        "error": None,
    }


def _unavailable_xray_model_bundle(spec: dict[str, Any], error: str) -> dict[str, Any]:
    """Stub for a model that cannot be loaded from its own saved artifacts."""
    val_macro_roc_auc = _val_macro_roc_auc_from_artifacts(
        run_meta_path=spec["run_meta"],
        artifact_dir=spec["artifact_dir"],
        comparison_json_key=str(spec["comparison_json_key"]),
    )
    return {
        "available": False,
        "key": spec["key"],
        "display_name": str(spec["display_name"]),
        "model": None,
        "device": None,
        "labels": list(XRAY_NIH14_LABELS),
        "thresholds": {},
        "transform": None,
        "image_size": None,
        "checkpoint_path": str(spec["checkpoint"]),
        "run_name": spec["artifact_dir"].name,
        "val_macro_roc_auc": val_macro_roc_auc,
        "model_name": None,
        "error": error,
    }


@st.cache_resource(show_spinner="Loading four-model comparison checkpoints…")
def load_xray_comparison_models() -> dict[str, dict[str, Any]]:
    """Load all four comparators, each from its own saved artifacts.

    Baseline CNN, DenseNet and EfficientNet-B0 load exactly as before (320 x 320);
    ViT-B/16 loads from its own checkpoint, its own ``run_metadata.json`` (which is what
    sets its 224 x 224 input) and its own thresholds. Every model's preprocessing and
    thresholds come from its own run — nothing is shared or substituted.

    Missing checkpoint/config marks that model unavailable — never reuse another
    checkpoint or invent probabilities. No training; protected test set untouched.
    """
    bundles: dict[str, dict[str, Any]] = {}
    for spec in XRAY_COMPARISON_MODEL_SPECS:
        try:
            bundles[str(spec["key"])] = _load_frozen_xray_model_bundle(spec)
        except Exception as exc:  # noqa: BLE001 - keep other models usable
            bundles[str(spec["key"])] = _unavailable_xray_model_bundle(spec, str(exc))
    return bundles


def _init_state() -> None:
    if "active_stage" not in st.session_state:
        st.session_state.active_stage = PIPELINE_STAGES[0][1]
    if "train_activity" not in st.session_state:
        st.session_state.train_activity = TRAIN_ACTIVITIES[0]
    if "train_activity_pills" not in st.session_state:
        st.session_state.train_activity_pills = TRAIN_ACTIVITIES[0]
    if "xray_inference_results" not in st.session_state:
        st.session_state.xray_inference_results = None
    if "xray_uploader_nonce" not in st.session_state:
        st.session_state.xray_uploader_nonce = 0
    if "xray_processed_signature" not in st.session_state:
        st.session_state.xray_processed_signature = None
    if "xray_preview" not in st.session_state:
        st.session_state.xray_preview = None
    if "xray_filename" not in st.session_state:
        st.session_state.xray_filename = None
    if "xray_upload_meta" not in st.session_state:
        st.session_state.xray_upload_meta = None


def render_header() -> None:
    st.markdown(
        """
        <div class="cxr-header">
          <span class="cxr-header__title">Medical Imaging AI Research Console</span>
          <span class="cxr-header__subtitle">Academic decision-support prototype · Not for clinical diagnosis or treatment</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_workstream_tabs(key: str) -> tuple[Any, Any]:
    """Large equal-width Classification / Segmentation tabs (Classification default)."""
    st.markdown('<div class="cxr-workstream-tabs-marker"></div>', unsafe_allow_html=True)
    return st.tabs(
        [WORKSTREAM_TAB_CLASSIFICATION, WORKSTREAM_TAB_SEGMENTATION],
        key=key,
    )


def _render_workstream_context(kind: str) -> None:
    """Compact active-workstream context pill under the tab bar."""
    label = (
        WORKSTREAM_CONTEXT_CLASSIFICATION
        if kind == "classification"
        else WORKSTREAM_CONTEXT_SEGMENTATION
    )
    st.markdown(
        f'<div class="cxr-workstream-context" role="status">'
        f'<span class="cxr-workstream-context__pill">{html.escape(label)}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_mri_audit_status_panel(*, stage_label: str) -> None:
    """Accurate compact MRI status — no invented counts or metrics."""
    st.markdown(
        f"""
        <div class="cxr-empty-state cxr-empty-state--compact">
          <p class="cxr-empty-state__title">MRI segmentation pipeline: audit in progress</p>
          <p class="cxr-empty-state__text">
            {html.escape(stage_label)} will appear here after MRI preparation, patient-safe
            splitting, and evaluation artifacts are connected. No segmentation results are
            shown until those outputs exist.
          </p>
          <span class="cxr-badge cxr-badge--planned">Audit in progress</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _seg_ui_call(func_name: str, *, fallback_stage: str) -> None:
    """Dispatch into app/segmentation_ui.py (MRI workstream only).

    Segmentation rendering lives in its own module so the Classification code in
    this file is not touched by it. If the module or its dependencies are
    unavailable, the honest audit panel is shown instead of a broken tab.
    """
    try:
        import app.segmentation_ui as seg_ui
    except Exception:  # noqa: BLE001
        try:
            import segmentation_ui as seg_ui  # running with app/ on sys.path
        except Exception as exc:  # noqa: BLE001
            _render_mri_audit_status_panel(stage_label=fallback_stage)
            st.caption(f"Segmentation UI module unavailable: {exc}")
            return
    try:
        getattr(seg_ui, func_name)()
    except Exception as exc:  # noqa: BLE001
        _render_mri_audit_status_panel(stage_label=fallback_stage)
        st.error(f"Segmentation panel error: {type(exc).__name__}: {exc}")


def _render_segmentation_data_prep() -> None:
    """MRI segmentation Data Preparation — real dataset, split and preprocessing."""
    _seg_ui_call(
        "render_segmentation_data_prep",
        fallback_stage="MRI preparation manifests, patient-safe splits, and preprocessing flow",
    )


def render_stage_navigation() -> str:
    """Full-width three-stage pipeline navigation — one active stage at a time."""
    st.markdown('<div class="cxr-stage-nav-marker"></div>', unsafe_allow_html=True)

    cols = st.columns(3, gap="medium")
    for col, (number, stage) in zip(cols, PIPELINE_STAGES):
        is_active = st.session_state.active_stage == stage
        with col:
            clicked = st.button(
                f"{number}\n{stage}",
                key=f"pipeline_stage_{stage}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            )
            if clicked and not is_active:
                st.session_state.active_stage = stage
                st.rerun()

    st.markdown('<div class="cxr-stage-nav-spacer"></div>', unsafe_allow_html=True)
    return st.session_state.active_stage


def _track_card(
    label: str,
    title: str,
    rows: list[tuple[str, str]],
    accent: str = "accent",
    badges: list[tuple[str, str]] | None = None,
    extra_html: str = "",
) -> str:
    list_html = ""
    if rows:
        items = "".join(
            f'<li><span class="cxr-k">{key}</span><span class="cxr-v">{value}</span></li>'
            for key, value in rows
        )
        list_html = f'<ul class="cxr-card__list">{items}</ul>'
    badge_html = ""
    if badges:
        parts = "".join(
            f'<span class="cxr-badge cxr-badge--{kind}">{text}</span>'
            for text, kind in badges
        )
        badge_html = f'<div class="cxr-card__badges">{parts}</div>'
    return f"""
    <div class="cxr-card cxr-card--{accent}">
      <p class="cxr-card__label">{label}</p>
      <p class="cxr-card__title">{title}</p>
      {badge_html}
      {list_html}
      {extra_html}
    </div>
    """


def _checklist(items: list[str]) -> str:
    rows = "".join(
        f'<li class="cxr-check__item"><span class="cxr-check__box" aria-hidden="true"></span>'
        f"<span>{item}</span></li>"
        for item in items
    )
    return f'<ul class="cxr-check">{rows}</ul>'


def _process_pipeline(steps: list[str]) -> str:
    parts: list[str] = []
    for index, step in enumerate(steps):
        if index:
            parts.append('<span class="cxr-pipeline__arrow">→</span>')
        parts.append(f'<span class="cxr-pipeline__step">{step}</span>')
    return f'<div class="cxr-pipeline">{"".join(parts)}</div>'


def _data_table(
    headers: list[str],
    rows: list[list[str]],
    numeric_cols: set[int] | None = None,
    total_row: bool = False,
) -> str:
    """Plain academic data table. Values are passed in literally — nothing computed."""
    numeric = numeric_cols or set()
    head = "".join(
        f'<th class="cxr-num">{h}</th>' if i in numeric else f"<th>{h}</th>"
        for i, h in enumerate(headers)
    )
    body: list[str] = []
    for row_index, row in enumerate(rows):
        is_total = total_row and row_index == len(rows) - 1
        row_class = ' class="cxr-table__total"' if is_total else ""
        cells = "".join(
            f'<td class="cxr-num">{c}</td>' if i in numeric else f"<td>{c}</td>"
            for i, c in enumerate(row)
        )
        body.append(f"<tr{row_class}>{cells}</tr>")
    return (
        '<div class="cxr-table-wrap"><table class="cxr-table">'
        f"<thead><tr>{head}</tr></thead>"
        f'<tbody>{"".join(body)}</tbody>'
        "</table></div>"
    )


def _confirmations(items: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="cxr-confirm"><span class="cxr-confirm__mark" aria-hidden="true"></span>'
        f'<span class="cxr-confirm__text"><span class="cxr-confirm__k">{key}</span>'
        f'<span class="cxr-confirm__v">{value}</span></span></div>'
        for key, value in items
    )
    return f'<div class="cxr-confirm-grid">{cells}</div>'


def _microcopy(text: str) -> str:
    return f'<p class="cxr-microcopy">{text}</p>'


@dataclass(frozen=True)
class XrayReadinessResult:
    all_passed: bool
    failures: tuple[str, ...]


@st.cache_data(show_spinner=False)
def _verify_xray_data_readiness_cached() -> tuple[bool, tuple[str, ...]]:
    """Read-only filesystem verification (pickle-safe return for st.cache_data)."""
    failures: list[str] = []

    csv_paths = {
        "train": XRAY_TRAIN_CSV,
        "valid": XRAY_VALID_CSV,
        "test": XRAY_TEST_CSV,
    }
    for split, path in csv_paths.items():
        if not path.is_file():
            failures.append(f"Missing required CSV: {path.relative_to(PROJECT_ROOT).as_posix()}")

    if not XRAY_IMAGES_DIR.is_dir():
        failures.append(
            f"Missing image directory: {XRAY_IMAGES_DIR.relative_to(PROJECT_ROOT).as_posix()}/"
        )

    if failures:
        return False, tuple(failures)

    frames: dict[str, pd.DataFrame] = {}
    for split, path in csv_paths.items():
        try:
            frames[split] = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 - surface read errors in UI
            failures.append(f"Could not read {path.name}: {exc}")
            return False, tuple(failures)

        expected = XRAY_EXPECTED_ROW_COUNTS[split]
        actual = len(frames[split])
        if actual != expected:
            failures.append(
                f"{split.title()} CSV row count is {actual:,}; expected {expected:,}."
            )

        missing_cols = [col for col in XRAY_REQUIRED_COLUMNS if col not in frames[split].columns]
        if missing_cols:
            failures.append(
                f"{split.title()} CSV is missing required columns: {', '.join(missing_cols)}."
            )

        duplicate_images = int(frames[split]["Image"].duplicated().sum())
        if duplicate_images:
            failures.append(
                f"{split.title()} CSV contains {duplicate_images} duplicate Image value(s)."
            )

    if failures:
        return False, tuple(failures)

    train_images = set(frames["train"]["Image"].astype(str))
    valid_images = set(frames["valid"]["Image"].astype(str))
    test_images = set(frames["test"]["Image"].astype(str))

    for left_name, left_set, right_name, right_set in (
        ("train", train_images, "valid", valid_images),
        ("train", train_images, "test", test_images),
        ("valid", valid_images, "test", test_images),
    ):
        overlap = len(left_set & right_set)
        if overlap:
            failures.append(
                f"{overlap} Image value(s) overlap between {left_name} and {right_name}."
            )

    train_patients = set(frames["train"]["PatientId"].astype(str))
    valid_patients = set(frames["valid"]["PatientId"].astype(str))
    test_patients = set(frames["test"]["PatientId"].astype(str))

    for left_name, left_set, right_name, right_set in (
        ("train", train_patients, "valid", valid_patients),
        ("train", train_patients, "test", test_patients),
        ("valid", valid_patients, "test", test_patients),
    ):
        overlap = len(left_set & right_set)
        if overlap:
            failures.append(
                f"{overlap} PatientId value(s) overlap between {left_name} and {right_name}."
            )

    image_files = {
        path.name
        for path in XRAY_IMAGES_DIR.iterdir()
        if path.is_file()
    }
    referenced_images = train_images | valid_images | test_images
    missing_images = sorted(referenced_images - image_files)
    if missing_images:
        preview = ", ".join(missing_images[:5])
        suffix = f" (+{len(missing_images) - 5} more)" if len(missing_images) > 5 else ""
        failures.append(
            f"{len(missing_images)} referenced image file(s) missing from images-small/: "
            f"{preview}{suffix}"
        )

    return (not failures), tuple(failures)


def verify_xray_data_readiness() -> XrayReadinessResult:
    """Read-only filesystem verification for the X-ray classification dataset."""
    all_passed, failures = _verify_xray_data_readiness_cached()
    return XrayReadinessResult(all_passed=all_passed, failures=failures)


def clear_xray_data_readiness_cache() -> None:
    _verify_xray_data_readiness_cached.clear()


def _render_xray_readiness_status(readiness: XrayReadinessResult) -> None:
    if readiness.all_passed:
        st.markdown(
            """
            <div class="cxr-readiness cxr-readiness--pass">
              <p class="cxr-readiness__eyebrow">X-ray data readiness</p>
              <p class="cxr-readiness__status">DATA PREPARATION COMPLETE</p>
              <p class="cxr-readiness__text">
                Chest X-ray files, labels, duplicate controls, patient-safe splits, and
                preprocessing specification are ready for model training.
              </p>
              <p class="cxr-readiness__next">
                Next step: Proceed to baseline training using train_clean.csv and valid_clean.csv.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    failure_items = "".join(f"<li>{item}</li>" for item in readiness.failures)
    st.markdown(
        f"""
        <div class="cxr-readiness cxr-readiness--blocked">
          <p class="cxr-readiness__eyebrow">X-ray data readiness</p>
          <p class="cxr-readiness__status">DATA PREPARATION BLOCKED</p>
          <ul class="cxr-readiness__failures">{failure_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _prep_dataset_overview() -> None:
    """01 · Dataset Overview — scope, target, and verified dataset totals."""
    overview = _track_card(
        "Study definition",
        "Chest X-ray Classification Task",
        [
            ("Task", "Multi-label chest X-ray classification"),
            ("Input", "Non-identifiable frontal chest X-ray image"),
            ("Target", "14 binary disease-finding labels per image"),
            ("Image source", "1,422 grayscale PNG images, each 1024 × 1024"),
            ("Dataset role", "Academic AI engineering demonstration"),
            (
                "Restriction",
                "Not for clinical diagnosis, treatment recommendation, or patient-care use",
            ),
        ],
        accent="accent",
    )

    totals = _data_table(
        ["Scope", "Images", "Patients"],
        [
            ["Development data", "1,002", "930"],
            ["Protected official test data", "420", "389"],
            ["Total", "1,422", "1,319"],
        ],
        numeric_cols={1, 2},
        total_row=True,
    )
    totals_card = (
        '<div class="cxr-card cxr-card--accent">'
        '<p class="cxr-card__label">Verified totals</p>'
        '<p class="cxr-card__title">Dataset Scope</p>'
        f"{totals}"
        + _microcopy(
            "Counts reflect unique images after correcting duplicated records in "
            "the original course development files."
        )
        + "</div>"
    )

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(overview, unsafe_allow_html=True)
    with right:
        st.markdown(totals_card, unsafe_allow_html=True)


def _prep_quality_checks() -> None:
    """02 · Quality Checks — factual quality evidence only."""
    structure_card = _track_card(
        "Audit evidence",
        "Metadata and File Integrity",
        [],
        accent="accent",
        extra_html=_checklist(
            [
                "Original course development metadata contained 198 duplicate image records.",
                "The duplicate records were exact copies with no conflicting labels.",
                "Duplicates were removed before splitting.",
                "All 1,422 referenced image files are present, readable grayscale PNG files.",
                "All images have dimensions of 1024 × 1024.",
            ]
        ),
    )

    balance_card = _track_card(
        "Audit evidence",
        "Label Coverage and Balance",
        [],
        accent="muted",
        extra_html=_checklist(
            [
                "The dataset is imbalanced; rare findings require cautious metric interpretation.",
                "Every one of the 14 findings retains at least one positive example in validation.",
            ]
        ),
    )

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(structure_card, unsafe_allow_html=True)
    with right:
        st.markdown(balance_card, unsafe_allow_html=True)


def _prep_safe_split() -> None:
    """03 · Safe Train / Validation / Test Split — leakage-control centrepiece."""
    split_table = _data_table(
        ["Split", "Unique images", "Unique patients", "Purpose"],
        [
            ["Training", "795", "744", "Model learning"],
            ["Validation", "207", "186", "Model selection and tuning"],
            ["Official test", "420", "389", "One final evaluation only"],
        ],
        numeric_cols={1, 2},
    )

    confirmations = _confirmations(
        [
            ("Image overlap", "0 image overlap across all splits"),
            ("Patient overlap", "0 patient overlap across all splits"),
            (
                "Official test set",
                "Official test set protected from training and validation decisions",
            ),
        ]
    )

    flow = _process_pipeline(
        [
            "Verify duplicate records",
            "Deduplicate by Image",
            "Split development data by PatientId",
            "Verify split isolation",
            "Freeze official test set",
        ]
    )

    st.markdown(
        '<div class="cxr-emphasis">'
        '<div class="cxr-card cxr-card--accent">'
        '<p class="cxr-card__label">Leakage control</p>'
        '<p class="cxr-card__title">Safe Train / Validation / Test Split</p>'
        '<p class="cxr-card__body">'
        "Development and official test partitions are isolated at both image and patient level."
        "</p>"
        f"{split_table}"
        f"{confirmations}"
        '<p class="cxr-flow-label">Preparation sequence</p>'
        f"{flow}"
        '<div class="cxr-note">'
        "Development data uses <code>GroupShuffleSplit</code> with "
        "<code>PatientId</code> as the grouping variable "
        "(<code>test_size=0.20</code>, <code>random_state=42</code>)."
        "</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _prep_preprocessing() -> None:
    """04 · Preprocessing — model-ready specification, not a finalized decision."""
    flow = _process_pipeline(
        [
            "Validate image",
            "Convert grayscale to 3 channels",
            "Resize to 320 × 320",
            "Apply ImageNet mean/std normalization",
            "Send to model",
        ]
    )

    st.markdown(
        '<div class="cxr-card cxr-card--accent">'
        '<p class="cxr-card__label">Model-ready specification</p>'
        '<p class="cxr-card__title">Preprocessing Path</p>'
        f"{flow}"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="cxr-status-board">
          <p class="cxr-status-board__label">Preprocessing governance</p>
          <p class="cxr-status-board__title">Consistency Requirements</p>
          <div class="cxr-status-row">
            <span class="cxr-status-row__k">Training and validation</span>
            <span class="cxr-status-row__v">Identical preprocessing pipeline</span>
          </div>
          <div class="cxr-status-row">
            <span class="cxr-status-row__k">Protected official test set</span>
            <span class="cxr-status-row__v">Identical pipeline only after model selection</span>
          </div>
          <div class="cxr-status-row">
            <span class="cxr-status-row__k">Future inference</span>
            <span class="cxr-status-row__v">Same frozen preprocessing specification</span>
          </div>
          <div class="cxr-status-row">
            <span class="cxr-status-row__k">Version control</span>
            <span class="cxr-status-row__v">Model and preprocessing versions frozen after validation-based selection</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_prep_governance_panel() -> None:
    """Full-width panel shared by all four preparation tabs."""
    st.markdown(
        """
        <div class="cxr-govpanel">
          <p class="cxr-govpanel__eyebrow">Shared standard</p>
          <h3 class="cxr-govpanel__title">Data Governance and Reproducibility</h3>
          <ul class="cxr-govpanel__list">
            <li>Only approved, non-identifiable academic data may be used.</li>
            <li>Duplicate checks and patient-level isolation protect evaluation validity.</li>
            <li>Validation data supports model selection; the official test set remains protected.</li>
            <li>Final preprocessing and model versions must be documented before inference is connected.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_preparation() -> None:
    st.markdown('<div class="cxr-prep-workspace">', unsafe_allow_html=True)
    st.markdown('<div class="cxr-prep-tabs-marker"></div>', unsafe_allow_html=True)

    if "data_prep_activity" not in st.session_state:
        st.session_state["data_prep_activity"] = PREP_ACTIVITIES[0]
    elif st.session_state["data_prep_activity"] not in PREP_ACTIVITIES:
        legacy_map = {
            "Task & Dataset": PREP_ACTIVITIES[0],
            "Quality & Exploratory Review": PREP_ACTIVITIES[1],
            "Patient-Safe Split": PREP_ACTIVITIES[2],
            "Standardized Preprocessing": PREP_ACTIVITIES[3],
        }
        st.session_state["data_prep_activity"] = legacy_map.get(
            st.session_state["data_prep_activity"],
            PREP_ACTIVITIES[0],
        )

    tab_classification, tab_segmentation = _render_workstream_tabs("dp_workstream_tabs")

    with tab_classification:
        _render_workstream_context("classification")

        selected = st.session_state["data_prep_activity"]
        col1, col2, col3, col4 = st.columns(4, gap="small")
        with col1:
            if st.button(
                PREP_ACTIVITIES[0],
                key="dp_task",
                type="primary" if selected == PREP_ACTIVITIES[0] else "secondary",
                use_container_width=True,
            ):
                st.session_state["data_prep_activity"] = PREP_ACTIVITIES[0]
        with col2:
            if st.button(
                PREP_ACTIVITIES[1],
                key="dp_quality",
                type="primary" if selected == PREP_ACTIVITIES[1] else "secondary",
                use_container_width=True,
            ):
                st.session_state["data_prep_activity"] = PREP_ACTIVITIES[1]
        with col3:
            if st.button(
                PREP_ACTIVITIES[2],
                key="dp_split",
                type="primary" if selected == PREP_ACTIVITIES[2] else "secondary",
                use_container_width=True,
            ):
                st.session_state["data_prep_activity"] = PREP_ACTIVITIES[2]
        with col4:
            if st.button(
                PREP_ACTIVITIES[3],
                key="dp_preprocessing",
                type="primary" if selected == PREP_ACTIVITIES[3] else "secondary",
                use_container_width=True,
            ):
                st.session_state["data_prep_activity"] = PREP_ACTIVITIES[3]

        readiness = verify_xray_data_readiness()
        _render_xray_readiness_status(readiness)
        refresh_col, _ = st.columns([1, 4])
        with refresh_col:
            if st.button("Refresh verification", key="dp_refresh_verification"):
                clear_xray_data_readiness_cache()
                st.rerun()

        activity = st.session_state["data_prep_activity"]
        if activity == PREP_ACTIVITIES[0]:
            _prep_dataset_overview()
        elif activity == PREP_ACTIVITIES[1]:
            _prep_quality_checks()
        elif activity == PREP_ACTIVITIES[2]:
            _prep_safe_split()
        else:
            _prep_preprocessing()

        render_prep_governance_panel()

        st.markdown(
            """
            <div class="cxr-next-step">
              <p class="cxr-next-step__title">Ready for Model Training</p>
              <p class="cxr-next-step__text">
                The verified duplicate-free, patient-safe X-ray dataset is ready for baseline training and
                fine-tuning experiments.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Continue to Train & Validate",
            key="dp_continue_training",
            type="primary",
            use_container_width=False,
        ):
            st.session_state.active_stage = PIPELINE_STAGES[1][1]
            st.rerun()

    with tab_segmentation:
        _render_workstream_context("segmentation")
        _render_segmentation_data_prep()

    st.markdown("</div>", unsafe_allow_html=True)



@st.cache_data(show_spinner=False)
def _load_xray_model_comparison_artifact() -> dict[str, Any] | None:
    """Load the persisted validation-only model comparison record (read-only).

    Prefers ``model_comparison_with_vit_experimental.json``, the derived four-model
    record: it copies the three frozen entries verbatim from ``model_comparison.json``
    (including its original ``selected_model``) and adds the experimental ViT entry.
    Falls back to the frozen three-model file when the derived one is absent, so the
    page still renders the original comparison. Neither file is ever written here.
    """
    for path in (XRAY_MODEL_COMPARISON_WITH_VIT_JSON, XRAY_MODEL_COMPARISON_JSON):
        if not path.is_file():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Unable to load {path.name}: {exc}")
    return None


def _fmt_maybe_float(val: Any, *, digits: int = 4) -> str:
    """Format numbers for UI; missing/NaN values stay explicit."""
    if val is None:
        return "Unavailable"
    try:
        if isinstance(val, float) and (val != val):  # NaN
            return "Unavailable"
    except Exception:  # noqa: BLE001
        pass
    try:
        return f"{float(val):.{digits}f}"
    except Exception:  # noqa: BLE001
        return "Unavailable"


def _fmt_support(val: Any) -> str:
    """Format positive/negative support counts from saved metrics."""
    if val is None:
        return "Unavailable"
    try:
        if isinstance(val, float) and (val != val):
            return "Unavailable"
        return str(int(float(val)))
    except Exception:  # noqa: BLE001
        return "Unavailable"


def _fmt_threshold_pct(val: Any) -> str:
    """Format a saved probability threshold as a one-decimal percentage."""
    if val is None:
        return "Unavailable"
    try:
        if isinstance(val, float) and (val != val):
            return "Unavailable"
        return f"{float(val) * 100:.1f}%"
    except Exception:  # noqa: BLE001
        return "Unavailable"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` into an RGB triple."""
    raw = str(hex_color).lstrip("#")
    if len(raw) != 6:
        return (143, 163, 183)  # fallback: Baseline CNN slate
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def _threshold_heat_rgba(model_color: str, intensity: float) -> str:
    """Light→strong tint of a model hue; alpha capped so navy text stays readable."""
    r, g, b = _hex_to_rgb(model_color)
    t = max(0.0, min(1.0, float(intensity)))
    # Very light for the column minimum; stronger (but still soft) for the maximum.
    alpha = 0.07 + 0.32 * t
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def _threshold_column_intensity(value: float, lo: float, hi: float) -> float:
    """Map a threshold to [0, 1] within its model column (min→max)."""
    if hi <= lo:
        return 0.5
    return (float(value) - lo) / (hi - lo)


# Display order + shared model colours for every Train & Validate graph.
# These hex values are the single source of truth for bars, ROC/PR lines, CM accents,
# and heatmaps on this page (not the Inference Demo rank-colour table).
XRAY_EVAL_MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "display_name": "Baseline CNN",
        "comparison_key": "baseline_cnn",
        "artifact_dir": XRAY_BASELINE_ARTIFACT_DIR,
        "color": "#8FA3B7",  # slate blue
        "cm_accent": "baseline",
    },
    {
        "display_name": "Fine-tuned DenseNet",
        "comparison_key": "fine_tuned_densenet",
        "artifact_dir": XRAY_DENSENET_ARTIFACT_DIR,
        "color": "#62788D",  # deep slate
        "cm_accent": "densenet",
    },
    {
        "display_name": "Fine-tuned EfficientNet-B0",
        "comparison_key": "fine_tuned_efficientnet_b0",
        "artifact_dir": XRAY_EFFICIENTNET_ARTIFACT_DIR,
        "color": "#2E8585",  # teal — frozen three-model validation-selected
        "cm_accent": "selected",
    },
    {
        "display_name": XRAY_VIT_DISPLAY_NAME,
        "comparison_key": "fine_tuned_vit_b16",
        "artifact_dir": XRAY_VIT_ARTIFACT_DIR,
        # Copper — validation-leading experimental comparator (never abbreviated in legends).
        "color": "#AD5B2D",
        "cm_accent": "experimental",
    },
)


def _comparison_model_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Build Section 1 rows from the saved comparison record (no recomputation).

    Two distinct statuses, deliberately not merged:

    ``Validation-selected``
        The frozen three-model selection recorded in ``model_comparison.json``
        (EfficientNet-B0). It is historical and is not reassigned here.
    ``Validation-leading experimental comparator``
        ViT-B/16, which has the highest validation macro ROC-AUC of the four but is an
        experimental comparator that was never promoted to the frozen selection.
    """
    selected = str(comparison.get("selected_model") or "")
    rows: list[dict[str, Any]] = []
    for spec in XRAY_EVAL_MODEL_SPECS:
        display_name = str(spec["display_name"])
        macro = (comparison.get(spec["comparison_key"]) or {}).get("val_macro_metrics") or {}
        is_selected = display_name == selected
        is_experimental = display_name == XRAY_VIT_DISPLAY_NAME
        if is_experimental:
            status = XRAY_VIT_STATUS
        elif is_selected:
            status = "Validation-selected"
        else:
            status = "—"
        rows.append(
            {
                "model": display_name,
                "roc_auc": macro.get("roc_auc"),
                "pr_auc": macro.get("pr_auc"),
                "f1": macro.get("f1"),
                "sensitivity": macro.get("recall_sensitivity"),
                "specificity": macro.get("specificity"),
                "status": status,
                "is_winner": is_selected and not is_experimental,
                "is_experimental": is_experimental,
            }
        )
    return rows


def _four_model_macro_performance_figure(rows: list[dict[str, Any]]) -> go.Figure:
    """Grouped horizontal bar chart of saved macro ROC-AUC / PR-AUC / F1 only.

    Colours follow ``XRAY_EVAL_MODEL_SPECS``: slate / deep slate / teal / copper.
    Axis is fixed to [0, 1]; sensitivity/specificity are intentionally omitted.
    Legend is rendered outside the figure (HTML) so it can sit under the title.
    """
    metric_keys = (
        ("Macro ROC-AUC", "roc_auc"),
        ("Macro PR-AUC", "pr_auc"),
        ("Macro F1", "f1"),
    )
    metrics = [label for label, _ in metric_keys]
    color_by_model = {
        str(spec["display_name"]): str(spec["color"]) for spec in XRAY_EVAL_MODEL_SPECS
    }

    fig = go.Figure()
    for row in rows:
        model = str(row["model"])
        xs: list[float] = []
        texts: list[str] = []
        for _, key in metric_keys:
            raw = row.get(key)
            try:
                value = float(raw) if raw is not None else float("nan")
            except (TypeError, ValueError):
                value = float("nan")
            xs.append(value)
            texts.append(_fmt_maybe_float(raw))
        fig.add_trace(
            go.Bar(
                name=model,
                y=metrics,
                x=xs,
                orientation="h",
                marker_color=color_by_model.get(model, "#62788D"),
                text=texts,
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
                hovertemplate=(
                    f"<b>{html.escape(model)}</b><br>"
                    "%{y}: %{x:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="group",
        bargap=0.22,
        bargroupgap=0.06,
        height=520,
        margin=dict(l=132, r=78, t=20, b=88),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#16324a", size=12),
        showlegend=False,
        xaxis=dict(
            range=[0.0, 1.0],
            title=dict(
                text="Validation metric value",
                font=dict(size=12, color="#16324a"),
                standoff=18,
            ),
            tickformat=".1f",
            dtick=0.2,
            gridcolor="#e4ebf0",
            zeroline=False,
            fixedrange=True,
            automargin=True,
            tickfont=dict(size=11, color="#16324a"),
        ),
        yaxis=dict(
            title=None,
            categoryorder="array",
            categoryarray=list(reversed(metrics)),
            tickfont=dict(size=12, color="#16324a"),
            fixedrange=True,
            automargin=True,
        ),
    )
    return fig


def _model_legend_html() -> str:
    """Prominent horizontal model colour legend for the Train & Validate performance chart."""
    items: list[str] = []
    for spec in XRAY_EVAL_MODEL_SPECS:
        color = html.escape(str(spec["color"]))
        name = html.escape(str(spec["display_name"]))
        items.append(
            '<span class="cxr-model-legend__item">'
            f'<span class="cxr-model-legend__swatch" style="background:{color};"></span>'
            f'<span class="cxr-model-legend__label">{name}</span>'
            "</span>"
        )
    return f'<div class="cxr-model-legend" role="list">{"".join(items)}</div>'


@st.cache_data(show_spinner=False)
def _load_xray_metrics_validation_df(artifact_dir: Path) -> pd.DataFrame | None:
    """Load saved validation per-label metrics from an artifact root."""
    path = Path(artifact_dir) / "metrics" / "metrics_validation.csv"
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - surface read errors in UI
        st.warning(f"Unable to load validation CSV: {path.name}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def _load_xray_validation_predictions_df(artifact_dir: Path) -> pd.DataFrame | None:
    """Load saved validation predictions (labels + probabilities + hard preds)."""
    path = Path(artifact_dir) / "predictions" / "validation_predictions.csv"
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Unable to load validation predictions: {path.name}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def _load_xray_thresholds_from_validation_df(artifact_dir: Path) -> pd.DataFrame | None:
    """Load the recorded validation threshold-selection table for one run."""
    path = Path(artifact_dir) / "metrics" / "thresholds_from_validation.csv"
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Unable to load threshold table: {path.name}: {exc}")
        return None


def _metrics_by_label(df: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if df is None or "label" not in getattr(df, "columns", []):
        return {}
    return df.set_index("label").to_dict(orient="index")


def _selected_model_name(comparison: dict[str, Any] | None) -> str:
    if not comparison:
        return ""
    return str(comparison.get("selected_model") or "")


def _render_eval_pane_title(number: str, title: str) -> None:
    """Numbered pane title for Train & Validate Classification panels."""
    st.markdown(
        f'<div class="cxr-eval-pane__head">'
        f'<span class="cxr-eval-pane__num">{html.escape(number)}</span>'
        f'<p class="cxr-eval-pane__title">{html.escape(title)}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_four_model_performance_chart(comparison: dict[str, Any] | None) -> None:
    """Macro ROC-AUC / PR-AUC / F1 chart from the saved comparison artifact."""
    if not comparison:
        st.info("Validation comparison artifact not found (`model_comparison.json`).")
        return
    rows = _comparison_model_rows(comparison)
    st.markdown(_model_legend_html(), unsafe_allow_html=True)
    st.plotly_chart(
        _four_model_macro_performance_figure(rows),
        width="stretch",
        theme=None,
        config={"displayModeBar": False},
        key="cxr_four_model_macro_bars",
    )


def _per_finding_table_html(
    rows: list[dict[str, Any]],
    *,
    selected_model: str,
) -> str:
    """Long-format per-finding × model validation metrics table."""
    headers = [
        ("Finding", "cxr-text"),
        ("Model", "cxr-text"),
        ("ROC-AUC", "cxr-num"),
        ("PR-AUC", "cxr-num"),
        ("F1-score", "cxr-num"),
        ("Sensitivity", "cxr-num"),
        ("Specificity", "cxr-num"),
        ("Positive support", "cxr-num"),
        ("Negative support", "cxr-num"),
    ]
    head = "".join(f'<th class="{cls}">{label}</th>' for label, cls in headers)
    body: list[str] = []
    for row in rows:
        is_selected = bool(selected_model) and row.get("model") == selected_model
        if row.get("model") == XRAY_VIT_DISPLAY_NAME:
            row_class = ' class="cxr-table__experimental"'
        elif is_selected:
            row_class = ' class="cxr-table__winner"'
        else:
            row_class = ""
        cells = [
            ("cxr-text", str(row["finding"])),
            ("cxr-text", str(row["model"])),
            ("cxr-num", _fmt_maybe_float(row.get("roc_auc"))),
            ("cxr-num", _fmt_maybe_float(row.get("pr_auc"))),
            ("cxr-num", _fmt_maybe_float(row.get("f1"))),
            ("cxr-num", _fmt_maybe_float(row.get("sensitivity"))),
            ("cxr-num", _fmt_maybe_float(row.get("specificity"))),
            ("cxr-num", _fmt_support(row.get("n_positive"))),
            ("cxr-num", _fmt_support(row.get("n_negative"))),
        ]
        cell_html = "".join(
            f'<td class="{cls}">{html.escape(val)}</td>' for cls, val in cells
        )
        body.append(f"<tr{row_class}>{cell_html}</tr>")
    return (
        '<div class="cxr-table-wrap cxr-table-wrap--scroll">'
        '<table class="cxr-table cxr-table--comparison cxr-table--compact cxr-table--train">'
        f"<thead><tr>{head}</tr></thead>"
        f'<tbody>{"".join(body)}</tbody>'
        "</table></div>"
    )


def _render_per_finding_validation_metrics(comparison: dict[str, Any] | None) -> None:
    """Per-finding validation metrics from saved metrics_validation.csv (expander)."""
    selected = _selected_model_name(comparison)
    rows: list[dict[str, Any]] = []
    any_loaded = False

    for spec in XRAY_EVAL_MODEL_SPECS:
        metrics_df = _load_xray_metrics_validation_df(spec["artifact_dir"])
        by_label = _metrics_by_label(metrics_df)
        if by_label:
            any_loaded = True
        for finding in XRAY_NIH14_LABELS:
            m = by_label.get(finding)
            if not m:
                rows.append(
                    {
                        "finding": finding,
                        "model": spec["display_name"],
                        "roc_auc": None,
                        "pr_auc": None,
                        "f1": None,
                        "sensitivity": None,
                        "specificity": None,
                        "n_positive": None,
                        "n_negative": None,
                    }
                )
                continue
            rows.append(
                {
                    "finding": finding,
                    "model": spec["display_name"],
                    "roc_auc": m.get("roc_auc"),
                    "pr_auc": m.get("pr_auc"),
                    "f1": m.get("f1"),
                    "sensitivity": m.get("recall_sensitivity"),
                    "specificity": m.get("specificity"),
                    "n_positive": m.get("n_positive"),
                    "n_negative": m.get("n_negative"),
                }
            )

    if not any_loaded:
        st.info("Unavailable — saved per-finding validation metrics were not found.")
        return

    st.markdown('<div class="cxr-tv-expander-marker"></div>', unsafe_allow_html=True)
    with st.expander(
        f"Show per-finding validation metrics ({len(XRAY_NIH14_LABELS)} findings × "
        f"{len(XRAY_EVAL_MODEL_SPECS)} models)",
        expanded=False,
    ):
        st.markdown(
            _per_finding_table_html(rows, selected_model=selected),
            unsafe_allow_html=True,
        )


def _confusion_counts_from_predictions(
    pred_df: pd.DataFrame,
    labels: tuple[str, ...] = XRAY_NIH14_LABELS,
) -> dict[str, int] | None:
    """Aggregate multilabel one-vs-rest confusion counts from saved validation preds."""
    required = [f"true_{lab}" for lab in labels] + [f"pred_{lab}" for lab in labels]
    if any(col not in pred_df.columns for col in required):
        return None
    tp = fp = tn = fn = 0
    for lab in labels:
        true = pred_df[f"true_{lab}"].to_numpy(dtype=int)
        pred = pred_df[f"pred_{lab}"].to_numpy(dtype=int)
        tp += int(((true == 1) & (pred == 1)).sum())
        fp += int(((true == 0) & (pred == 1)).sum())
        tn += int(((true == 0) & (pred == 0)).sum())
        fn += int(((true == 1) & (pred == 0)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _confusion_heatmap_colorscale(model_color: str) -> list[list[float | str]]:
    """Soft→strong heatmap scale anchored on the shared Train & Validate model colour."""
    soft_mid = {
        "#8FA3B7": ("#f5f7f9", "#d5dde6"),
        "#62788D": ("#f3f5f7", "#c5d0db"),
        "#2E8585": ("#f2f8f8", "#b7d6d6"),
        "#AD5B2D": ("#fdf7f3", "#e8c9b5"),
    }
    soft, mid = soft_mid.get(model_color, ("#f8fafb", "#e2eaf0"))
    return [[0.0, soft], [0.45, mid], [1.0, model_color]]


def _confusion_matrix_figure(
    counts: dict[str, int],
    *,
    model_color: str,
    compact: bool = False,
) -> go.Figure:
    """Build a rendered 2x2 one-vs-rest aggregated confusion heatmap.

    Heatmap colours follow the shared Train & Validate model colour system.
    """
    tn = int(counts["tn"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    tp = int(counts["tp"])
    z = [[tn, fp], [fn, tp]]
    text_vals = [[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]]
    colorscale = _confusion_heatmap_colorscale(model_color)

    cell_font = 11 if compact else 15
    tick_font = 9 if compact else 11
    height = 210 if compact else 290
    # Extra left/top room so "Actual …" / "Predicted …" axis labels are not clipped
    # inside the compact 2 × 2 pane.
    margin = (
        dict(l=68, r=6, t=36, b=6) if compact else dict(l=8, r=8, t=8, b=8)
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted Negative", "Predicted Positive"],
            y=["Actual Negative", "Actual Positive"],
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": cell_font, "color": "#16324a"},
            colorscale=colorscale,
            showscale=False,
            hovertemplate="%{y} · %{x}<br>Count: %{text}<extra></extra>",
            xgap=2 if compact else 3,
            ygap=2 if compact else 3,
        )
    )
    fig.update_layout(
        margin=margin,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#16324a", size=10 if compact else 11),
        xaxis=dict(
            side="top",
            tickfont=dict(size=tick_font, color="#16324a"),
            automargin=True,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=tick_font, color="#16324a"),
            automargin=True,
        ),
    )
    return fig


def _render_validation_confusion_matrices(
    comparison: dict[str, Any] | None,
    *,
    compact: bool = False,
) -> None:
    """Aggregated multilabel confusion matrices from validation predictions.

    Compact 2 × 2 grid for the Train & Validate top pane; axis labels stay visible.
    """
    del comparison  # counts come from saved predictions; accent uses model identity
    specs = list(XRAY_EVAL_MODEL_SPECS)
    per_row = 2
    grid: list[Any] = []
    col_gap = "small" if compact else "medium"
    for index in range(len(specs)):
        if index % per_row == 0:
            grid.extend(st.columns(per_row, gap=col_gap))
    for col, spec in zip(grid, specs):
        display = str(spec["display_name"])
        model_color = str(spec["color"])
        accent = f" cxr-cm--{spec['cm_accent']}"
        pred_df = _load_xray_validation_predictions_df(spec["artifact_dir"])
        counts = (
            _confusion_counts_from_predictions(pred_df) if pred_df is not None else None
        )
        with col:
            compact_cls = " cxr-cm--compact" if compact else ""
            caption = (
                "14 findings · validation"
                if compact
                else "Aggregated across 14 findings · validation set"
            )
            st.markdown(
                f'<div class="cxr-cm{accent}{compact_cls}">'
                f'<p class="cxr-cm__title">{html.escape(display)}</p>'
                f'<p class="cxr-cm__caption">{caption}</p>'
                "</div>",
                unsafe_allow_html=True,
            )
            if counts is None:
                st.info("Unavailable — saved validation predictions not found.")
            else:
                st.plotly_chart(
                    _confusion_matrix_figure(
                        counts,
                        model_color=model_color,
                        compact=compact,
                    ),
                    width="stretch",
                    theme=None,
                    config={"displayModeBar": False, "staticPlot": True},
                    key=f"cxr_cm_{spec['comparison_key']}",
                )


def _prediction_arrays(
    pred_df: pd.DataFrame,
    labels: tuple[str, ...] = XRAY_NIH14_LABELS,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract (y_true, y_prob) matrices from a saved validation predictions table."""
    true_cols = [f"true_{lab}" for lab in labels]
    prob_cols = [f"prob_{lab}" for lab in labels]
    if any(c not in pred_df.columns for c in true_cols + prob_cols):
        return None
    y_true = pred_df[true_cols].to_numpy(dtype=float)
    y_prob = pred_df[prob_cols].to_numpy(dtype=float)
    return y_true, y_prob


def _micro_roc_points(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[pd.DataFrame, float] | None:
    """Micro-averaged ROC curve + its own AUC from flattened multilabel arrays.

    The AUC is computed from exactly the same flattened arrays that produce the
    plotted points, so the legend value is the area under the drawn curve. Macro
    metrics are NOT used here - they belong to the comparison table.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    yt = y_true.ravel()
    yp = y_prob.ravel()
    if len(np.unique(yt)) < 2:
        return None
    fpr, tpr, _ = roc_curve(yt, yp)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr}), float(roc_auc_score(yt, yp))


def _micro_pr_points(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[pd.DataFrame, float] | None:
    """Micro-averaged PR curve + its own average precision (micro AP).

    ``average_precision_score`` on the flattened arrays is the summary of exactly
    this curve. It is an average precision, not a macro PR-AUC, and is labelled
    as such in the UI.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    yt = y_true.ravel()
    yp = y_prob.ravel()
    if len(np.unique(yt)) < 2:
        return None
    precision, recall, _ = precision_recall_curve(yt, yp)
    return (
        pd.DataFrame({"recall": recall, "precision": precision}),
        float(average_precision_score(yt, yp)),
    )


def _collect_validation_roc_pr_frames() -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Build micro-averaged ROC/PR frames from saved validation predictions only."""
    roc_frames: list[pd.DataFrame] = []
    pr_frames: list[pd.DataFrame] = []

    for spec in XRAY_EVAL_MODEL_SPECS:
        display = str(spec["display_name"])
        pred_df = _load_xray_validation_predictions_df(spec["artifact_dir"])
        if pred_df is None:
            continue
        arrays = _prediction_arrays(pred_df)
        if arrays is None:
            continue
        y_true, y_prob = arrays

        roc_result = _micro_roc_points(y_true, y_prob)
        if roc_result is not None:
            roc_pts, roc_auc = roc_result
            frame = roc_pts.copy()
            frame["series"] = f"{display} (micro ROC-AUC = {roc_auc:.4f})"
            frame["model"] = display
            roc_frames.append(frame)

        pr_result = _micro_pr_points(y_true, y_prob)
        if pr_result is not None:
            pr_pts, micro_ap = pr_result
            frame = pr_pts.copy()
            frame["series"] = f"{display} (micro AP = {micro_ap:.4f})"
            frame["model"] = display
            pr_frames.append(frame)

    return roc_frames, pr_frames


def _roc_pr_series_color_scale(frame: pd.DataFrame) -> alt.Scale:
    color_by_model = {
        str(spec["display_name"]): str(spec["color"]) for spec in XRAY_EVAL_MODEL_SPECS
    }
    series_order = list(dict.fromkeys(frame["series"].tolist()))
    model_for_series = {
        row.series: row.model
        for row in frame[["series", "model"]].drop_duplicates().itertuples(index=False)
    }
    return alt.Scale(
        domain=series_order,
        range=[color_by_model.get(model_for_series[s], "#62788D") for s in series_order],
    )


def _white_curve_presentation(chart, *, height: int = 420):
    return (
        chart.properties(height=height)
        .configure(background="#ffffff")
        .configure_view(strokeWidth=0, fill="#ffffff")
        .configure_axis(
            labelFontSize=11,
            titleFontSize=12,
            labelColor="#0b1f36",
            titleColor="#0b1f36",
            gridColor="#e4ebf0",
            domainColor="#0b1f36",
            tickColor="#0b1f36",
        )
        .configure_legend(
            labelFontSize=11,
            titleFontSize=11,
            labelColor="#0b1f36",
            symbolStrokeWidth=2.4,
            orient="bottom",
            labelLimit=360,
        )
    )


def _render_validation_roc_curve(roc_frames: list[pd.DataFrame]) -> None:
    """Micro-averaged validation ROC curves (white presentation style)."""
    if not roc_frames:
        st.info("Unavailable — saved validation probabilities/labels were not found.")
        return
    roc_df = pd.concat(roc_frames, ignore_index=True)
    chance = pd.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})
    lines = (
        alt.Chart(roc_df)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X(
                "fpr:Q",
                title="1 − Specificity (false positive rate)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "tpr:Q",
                title="Sensitivity (true positive rate)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "series:N",
                title=None,
                scale=_roc_pr_series_color_scale(roc_df),
                legend=alt.Legend(labelLimit=360, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("series:N", title="Model"),
                alt.Tooltip("fpr:Q", format=".3f", title="FPR"),
                alt.Tooltip("tpr:Q", format=".3f", title="TPR"),
            ],
        )
    )
    diag = (
        alt.Chart(chance)
        .mark_line(strokeDash=[5, 4], color="#b8c5d1", strokeWidth=1.2)
        .encode(x="fpr:Q", y="tpr:Q")
    )
    st.altair_chart(
        _white_curve_presentation(diag + lines),
        width="stretch",
        theme=None,
    )


def _render_validation_pr_curve(pr_frames: list[pd.DataFrame]) -> None:
    """Micro-averaged validation precision–recall curves (white presentation style)."""
    if not pr_frames:
        st.info("Unavailable — saved validation probabilities/labels were not found.")
        return
    pr_df = pd.concat(pr_frames, ignore_index=True)
    lines = (
        alt.Chart(pr_df)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X(
                "recall:Q",
                title="Recall (sensitivity)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "precision:Q",
                title="Precision",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "series:N",
                title=None,
                scale=_roc_pr_series_color_scale(pr_df),
                legend=alt.Legend(labelLimit=360, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("series:N", title="Model"),
                alt.Tooltip("recall:Q", format=".3f", title="Recall"),
                alt.Tooltip("precision:Q", format=".3f", title="Precision"),
            ],
        )
    )
    st.altair_chart(
        _white_curve_presentation(lines),
        width="stretch",
        theme=None,
    )


def _load_model_threshold_map(artifact_dir: Path) -> dict[str, float] | None:
    """Load per-finding thresholds from validation CSV, falling back to thresholds.json."""
    thr_df = _load_xray_thresholds_from_validation_df(artifact_dir)
    if thr_df is not None and {"label", "threshold"}.issubset(thr_df.columns):
        out: dict[str, float] = {}
        for _, row in thr_df.iterrows():
            label = str(row["label"])
            try:
                out[label] = float(row["threshold"])
            except (TypeError, ValueError):
                continue
        if out:
            return out

    path = Path(artifact_dir) / "metrics" / "thresholds.json"
    if not path.is_file():
        return None
    try:
        raw = _read_json_file(path)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, dict):
        return None
    out = {}
    for label, value in raw.items():
        try:
            out[str(label)] = float(value)
        except (TypeError, ValueError):
            continue
    return out or None


def _threshold_table_html(rows: list[dict[str, Any]]) -> str:
    """14-finding threshold heatmap across every validation model.

    Column headers follow ``XRAY_EVAL_MODEL_SPECS``. Cell shade intensity is scaled
    within each model column from that column's min→max threshold only (not
    performance). Displayed text remains the exact saved threshold as one-decimal %.
    """
    head_cells = ['<th class="cxr-text">Finding</th>']
    col_ranges: dict[str, tuple[float, float]] = {}
    for spec in XRAY_EVAL_MODEL_SPECS:
        accent = html.escape(str(spec["cm_accent"]))
        name = html.escape(str(spec["display_name"]))
        display = str(spec["display_name"])
        head_cells.append(
            f'<th class="cxr-num cxr-thr-head cxr-thr-head--{accent}">{name}</th>'
        )
        vals = [
            float(row[display])
            for row in rows
            if row.get(display) is not None
            and not (
                isinstance(row[display], float) and row[display] != row[display]
            )
        ]
        if vals:
            col_ranges[display] = (min(vals), max(vals))
        else:
            col_ranges[display] = (0.0, 1.0)
    head = "".join(head_cells)

    body: list[str] = []
    for row in rows:
        finding = html.escape(str(row["finding"]))
        cells = [f'<td class="cxr-text">{finding}</td>']
        for spec in XRAY_EVAL_MODEL_SPECS:
            display = str(spec["display_name"])
            color = str(spec["color"])
            accent = html.escape(str(spec["cm_accent"]))
            raw = row.get(display)
            label = html.escape(_fmt_threshold_pct(raw))
            if raw is None or (
                isinstance(raw, float) and raw != raw
            ):
                cells.append(
                    f'<td class="cxr-num cxr-thr-heat cxr-thr-heat--{accent}">'
                    f"{label}</td>"
                )
                continue
            lo, hi = col_ranges[display]
            intensity = _threshold_column_intensity(float(raw), lo, hi)
            bg = _threshold_heat_rgba(color, intensity)
            cells.append(
                f'<td class="cxr-num cxr-thr-heat cxr-thr-heat--{accent}" '
                f'style="background:{bg};">{label}</td>'
            )
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="cxr-table-wrap">'
        '<table class="cxr-table cxr-table--comparison cxr-table--compact '
        'cxr-table--train cxr-table--thresholds cxr-table--threshold-heatmap">'
        f"<thead><tr>{head}</tr></thead>"
        f'<tbody>{"".join(body)}</tbody>'
        "</table></div>"
    )


def _render_decision_thresholds(comparison: dict[str, Any] | None) -> None:
    """Decision thresholds heatmap table from saved validation artifacts."""
    del comparison  # selection status is not used for the compact presentation table

    threshold_maps: dict[str, dict[str, float]] = {}
    missing_models: list[str] = []
    for spec in XRAY_EVAL_MODEL_SPECS:
        display = str(spec["display_name"])
        loaded = _load_model_threshold_map(spec["artifact_dir"])
        if loaded is None:
            missing_models.append(display)
            threshold_maps[display] = {}
        else:
            threshold_maps[display] = loaded

    if missing_models and len(missing_models) == len(XRAY_EVAL_MODEL_SPECS):
        st.info("Unavailable — saved validation thresholds were not found.")
        return

    rows: list[dict[str, Any]] = []
    missing_count = 0
    for finding in XRAY_NIH14_LABELS:
        row: dict[str, Any] = {"finding": finding}
        for spec in XRAY_EVAL_MODEL_SPECS:
            display = str(spec["display_name"])
            value = threshold_maps.get(display, {}).get(finding)
            if value is None:
                missing_count += 1
            row[display] = value
        rows.append(row)

    expected = len(XRAY_NIH14_LABELS) * len(XRAY_EVAL_MODEL_SPECS)
    present = expected - missing_count
    if present != expected:
        st.warning(
            f"Threshold coverage incomplete: found {present}/{expected} saved values "
            f"({len(XRAY_NIH14_LABELS)} findings × {len(XRAY_EVAL_MODEL_SPECS)} models)."
        )

    st.markdown(_threshold_table_html(rows), unsafe_allow_html=True)
    st.markdown(
        '<p class="cxr-microcopy cxr-microcopy--center cxr-threshold-footnote">'
        "Thresholds were selected on validation; threshold-based metrics are validation "
        "operating-point estimates."
        "</p>",
        unsafe_allow_html=True,
    )


def _render_classification_train_validate() -> None:
    """Six numbered Classification evaluation panes (validation artifacts only)."""
    st.markdown('<div class="cxr-tv-panes-marker"></div>', unsafe_allow_html=True)
    comparison = _load_xray_model_comparison_artifact()
    roc_frames, pr_frames = _collect_validation_roc_pr_frames()

    top_left, top_right = st.columns([1.05, 0.95], gap="medium")
    with top_left:
        with st.container(border=True, key="tv_cls_pane_01"):
            _render_eval_pane_title("01", "Four-model validation performance")
            _render_four_model_performance_chart(comparison)
    with top_right:
        with st.container(border=True, key="tv_cls_pane_02"):
            _render_eval_pane_title("02", "Confusion matrices")
            _render_validation_confusion_matrices(comparison, compact=True)

    mid_left, mid_right = st.columns(2, gap="medium")
    with mid_left:
        with st.container(border=True, key="tv_cls_pane_03"):
            _render_eval_pane_title("03", "Validation Micro-Averaged ROC Curves")
            _render_validation_roc_curve(roc_frames)
    with mid_right:
        with st.container(border=True, key="tv_cls_pane_04"):
            _render_eval_pane_title(
                "04", "Validation Micro-Averaged Precision–Recall Curves"
            )
            _render_validation_pr_curve(pr_frames)

    with st.container(border=True, key="tv_cls_pane_05"):
        _render_eval_pane_title("05", "Decision thresholds")
        _render_decision_thresholds(comparison)

    with st.container(border=True, key="tv_cls_pane_06"):
        _render_eval_pane_title("06", "Per-finding validation metrics")
        _render_per_finding_validation_metrics(comparison)


def _render_segmentation_train_validate() -> None:
    """MRI segmentation Train & Validate — six numbered panes from real artifacts."""
    _seg_ui_call(
        "render_segmentation_train_validate",
        fallback_stage="Validation training curves, Dice/IoU summaries, and qualitative examples",
    )


def render_train_validate() -> None:
    """Train & Validate — Classification (default) and Segmentation workstreams."""
    tab_classification, tab_segmentation = _render_workstream_tabs("tv_workstream_tabs")
    with tab_classification:
        _render_workstream_context("classification")
        _render_classification_train_validate()
    with tab_segmentation:
        _render_workstream_context("segmentation")
        _render_segmentation_train_validate()



def _format_file_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 ** 2):.2f} MB"


def _upload_file_signature(upload) -> str:
    """Stable signature for detecting a new upload without re-inferring on reruns."""
    import hashlib

    name = getattr(upload, "name", "uploaded_image")
    raw = upload.getvalue()
    digest = hashlib.md5(raw).hexdigest()
    return f"{name}:{len(raw)}:{digest}"


def _clear_xray_inference_state(*, bump_uploader: bool = True) -> None:
    """Clear prediction/preview state and optionally reset the uploader widget key."""
    st.session_state.xray_inference_results = None
    st.session_state.xray_processed_signature = None
    st.session_state.xray_preview = None
    st.session_state.xray_filename = None
    st.session_state.xray_upload_meta = None
    if bump_uploader:
        st.session_state.xray_uploader_nonce = int(
            st.session_state.get("xray_uploader_nonce", 0)
        ) + 1


def _open_uploaded_xray(upload) -> Image.Image:
    """Decode an uploaded file as grayscale then replicate to 3-channel RGB."""
    if Image is None:
        raise RuntimeError("Pillow is required to open uploaded images.")
    raw = upload.getvalue() if hasattr(upload, "getvalue") else upload
    with Image.open(io.BytesIO(raw)) as img:
        img.load()
        # Match training input convention: single-channel X-ray replicated to RGB.
        return img.convert("L").convert("RGB")


def _preprocess_xray_image(image: Image.Image, transform) -> Any:
    """Apply a model's exact saved validation transform (resize + normalization)."""
    tensor = transform(image)
    if hasattr(tensor, "unsqueeze"):
        return tensor.unsqueeze(0)
    raise TypeError("Preprocessing transform did not return a torch tensor.")


def _predict_xray_probabilities(bundle: dict[str, Any], batch_tensor: Any) -> list[float]:
    """Run sigmoid multilabel inference; returns probabilities in saved label order."""
    import torch

    model = bundle["model"]
    device = bundle["device"]
    model.eval()
    with torch.no_grad():
        logits = model(batch_tensor.to(device))
        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
    return [float(p) for p in probs]


def _build_prediction_rows(
    labels: list[str],
    probabilities: list[float],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, prob in zip(labels, probabilities):
        thr = float(thresholds.get(label, 0.5))
        predicted = prob >= thr
        rows.append(
            {
                "Finding": label,
                "Probability": prob,
                "Threshold": thr,
                "Predicted status": (
                    "Predicted finding" if predicted else "Below threshold"
                ),
                "predicted": predicted,
            }
        )
    return rows


def _run_single_model_inference(
    preview: Image.Image,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Run one frozen model on the uploaded image; never falls back to another checkpoint."""
    base = {
        "key": bundle.get("key"),
        "display_name": bundle.get("display_name"),
        "val_macro_roc_auc": bundle.get("val_macro_roc_auc"),
    }
    if not bundle.get("available"):
        return {
            **base,
            "available": False,
            "error": bundle.get("error")
            or "Checkpoint or matching preprocessing configuration unavailable.",
            "rows": None,
            "probabilities": None,
            "thresholds": dict(bundle.get("thresholds") or {}),
        }
    try:
        batch = _preprocess_xray_image(preview, bundle["transform"])
        probabilities = _predict_xray_probabilities(bundle, batch)
        labels = list(bundle["labels"])
        if len(probabilities) != len(labels):
            raise RuntimeError("Model output size does not match the saved label order.")
        if any(p < 0.0 or p > 1.0 for p in probabilities):
            raise RuntimeError("Model produced probabilities outside [0, 1].")
        rows = _build_prediction_rows(labels, probabilities, bundle.get("thresholds") or {})
        return {
            **base,
            "available": True,
            "error": None,
            "rows": rows,
            "probabilities": probabilities,
            "labels": labels,
            # Pass through frozen validation thresholds for Positive/Negative decisions only.
            "thresholds": dict(bundle.get("thresholds") or {}),
        }
    except Exception as exc:  # noqa: BLE001 - isolate per-model failures
        return {
            **base,
            "available": False,
            "error": str(exc),
            "rows": None,
            "probabilities": None,
            "thresholds": dict(bundle.get("thresholds") or {}),
        }


def _run_xray_inference_on_upload(
    upload,
    model_bundles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Run all four checkpoints once on the same uploaded image (no ensemble).

    Each model re-runs its own preprocessing from its own saved config, so the CNNs
    see 320 x 320 and ViT-B/16 sees 224 x 224 from the identical source image.
    """
    name = getattr(upload, "name", "uploaded_image")
    suffix = Path(name).suffix.lower()
    if suffix not in XRAY_INFERENCE_ACCEPTED_SUFFIXES:
        return {
            "results": [],
            "errors": [
                f"{name}: unsupported file type `{suffix or 'unknown'}`. "
                "Accepted: .png, .jpg, .jpeg."
            ],
            "preview": None,
            "filename": name,
        }

    try:
        raw = upload.getvalue() if hasattr(upload, "getvalue") else None
        file_size_bytes = len(raw) if isinstance(raw, (bytes, bytearray)) else None
        preview = _open_uploaded_xray(upload)
        image_width, image_height = preview.size
        upload_meta = {
            "filename": name,
            "file_size_bytes": file_size_bytes,
            "image_width": int(image_width),
            "image_height": int(image_height),
        }
        model_outputs: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        for spec in XRAY_COMPARISON_MODEL_SPECS:
            key = str(spec["key"])
            bundle = model_bundles.get(key) or _unavailable_xray_model_bundle(
                spec, "Model bundle was not loaded."
            )
            outcome = _run_single_model_inference(preview, bundle)
            model_outputs[key] = outcome
            # Surface only runtime failures; unavailable models show as Unavailable cells.
            if bundle.get("available") and not outcome.get("available"):
                errors.append(
                    f"{outcome.get('display_name') or key}: "
                    f"{outcome.get('error') or 'inference failed'}"
                )
        # Exact basename lookup in NIH test.csv only — never invent labels.
        ground_truth = _lookup_nih_test_ground_truth(name)
        return {
            "results": [
                {
                    "filename": name,
                    "preview": preview,
                    "model_outputs": model_outputs,
                    "ground_truth": ground_truth,
                    "upload_meta": upload_meta,
                    # Keep EfficientNet rows for any callers expecting single-model shape.
                    "rows": (model_outputs.get(XRAY_COMPARISON_DEFAULT_CHART_KEY) or {}).get(
                        "rows"
                    ),
                }
            ],
            "errors": errors,
            "preview": preview,
            "filename": name,
            "ground_truth": ground_truth,
            "upload_meta": upload_meta,
        }
    except Exception as exc:  # noqa: BLE001 - keep invalid uploads isolated
        return {
            "results": [],
            "errors": [f"{name}: {exc}"],
            "preview": None,
            "filename": name,
            "ground_truth": None,
            "upload_meta": None,
        }


@st.cache_data(show_spinner=False)
def _load_nih_test_ground_truth_index() -> dict[str, Any]:
    """Inspect NIH test.csv schema and index exact Image basenames → label vectors.

    Read-only: never modifies test.csv, never scores the full test set, never tunes
    thresholds. Label columns are discovered from the CSV header/values.
    """
    if not XRAY_TEST_CSV.is_file():
        raise FileNotFoundError(f"Missing NIH test CSV: {XRAY_TEST_CSV}")

    df = pd.read_csv(XRAY_TEST_CSV)
    columns = [str(c) for c in df.columns]
    if not columns:
        raise ValueError("NIH test.csv has no columns.")

    # Discover filename column from schema (prefer exact 'Image', else first string-like).
    image_col: str | None = None
    for candidate in columns:
        if candidate.strip().lower() == "image":
            image_col = candidate
            break
    if image_col is None:
        for candidate in columns:
            series = df[candidate].astype(str)
            if series.str.contains(r"\.(png|jpg|jpeg)$", case=False, regex=True, na=False).any():
                image_col = candidate
                break
    if image_col is None:
        raise ValueError("NIH test.csv schema has no recognizable image-filename column.")

    # Discover multilabel ground-truth columns: binary 0/1 fields that are not IDs.
    id_like = {image_col}
    for candidate in columns:
        lowered = candidate.strip().lower()
        if lowered in {"patientid", "patient_id", "patient id", "id"}:
            id_like.add(candidate)

    label_columns: list[str] = []
    for candidate in columns:
        if candidate in id_like:
            continue
        series = pd.to_numeric(df[candidate], errors="coerce")
        if series.isna().any():
            continue
        unique_vals = set(series.dropna().unique().tolist())
        if unique_vals and unique_vals.issubset({0, 1, 0.0, 1.0}):
            label_columns.append(candidate)

    if not label_columns:
        raise ValueError("NIH test.csv schema has no binary ground-truth label columns.")

    patient_col: str | None = None
    for candidate in columns:
        lowered = candidate.strip().lower()
        if lowered in {"patientid", "patient_id", "patient id"}:
            patient_col = candidate
            break

    by_basename: dict[str, dict[str, int]] = {}
    patient_id_by_basename: dict[str, str] = {}
    for _, row in df.iterrows():
        basename = Path(str(row[image_col])).name
        labels = {col: int(row[col]) for col in label_columns}
        # Exact basename keys only; do not alias or fuzzy-match uploads.
        by_basename[basename] = labels
        if patient_col is not None and pd.notna(row[patient_col]):
            patient_id_by_basename[basename] = str(row[patient_col]).strip()

    return {
        "image_column": image_col,
        "patient_column": patient_col,
        "label_columns": label_columns,
        "by_basename": by_basename,
        "patient_id_by_basename": patient_id_by_basename,
        "source_path": str(XRAY_TEST_CSV.as_posix()),
        "source_name": "NIH test.csv",
        "n_rows": int(len(df)),
    }


def _lookup_nih_test_ground_truth(uploaded_filename: str) -> dict[str, Any]:
    """Map an uploaded filename to NIH test.csv labels via exact basename only.

    Does not infer labels from pixels. Does not map arbitrary external images.
    """
    basename = Path(str(uploaded_filename or "")).name
    empty = {
        "found": False,
        "basename": basename,
        "labels": {},
        "positive_findings": [],
        "label_columns": [],
        "patient_id": None,
        "source_name": "NIH test.csv",
        "message": "Ground truth unavailable — external inference sample",
    }
    if not basename:
        return empty
    try:
        index = _load_nih_test_ground_truth_index()
    except Exception as exc:  # noqa: BLE001 - keep demo usable if CSV is missing
        return {
            **empty,
            "message": (
                "Ground truth unavailable — external inference sample "
                f"(test.csv lookup failed: {exc})"
            ),
        }

    labels = (index.get("by_basename") or {}).get(basename)
    if labels is None:
        return {
            **empty,
            "label_columns": list(index.get("label_columns") or []),
        }

    # Prefer NIH-14 display order for verified positives; fall back to CSV order.
    label_columns = list(index.get("label_columns") or [])
    csv_label_set = set(label_columns)
    ordered = [lab for lab in XRAY_NIH14_LABELS if lab in csv_label_set]
    for lab in label_columns:
        if lab not in ordered:
            ordered.append(lab)
    positive_findings = [col for col in ordered if int(labels.get(col, 0)) == 1]
    patient_id_raw = (index.get("patient_id_by_basename") or {}).get(basename)
    patient_id = str(patient_id_raw).strip() if patient_id_raw not in (None, "") else None
    return {
        "found": True,
        "basename": basename,
        "labels": dict(labels),
        "positive_findings": positive_findings,
        "label_columns": label_columns,
        "patient_id": patient_id,
        "source_name": str(index.get("source_name") or "NIH test.csv"),
        "message": "Ground truth source: NIH test.csv",
    }


def _prediction_outcome_label(*, ground_truth_positive: bool, predicted_positive: bool) -> str:
    """Classify one finding against official test.csv ground truth."""
    if ground_truth_positive and predicted_positive:
        return "Correct positive"
    if ground_truth_positive and not predicted_positive:
        return "Missed finding"
    if (not ground_truth_positive) and predicted_positive:
        return "False positive"
    return "Correct negative"


def _build_official_test_sample_evaluation(
    ground_truth: dict[str, Any],
    model_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Per-finding / per-model evaluation for one official NIH test.csv sample only.

    Uses each model's own saved validation threshold. Does not retune thresholds,
    retrain, or score the full protected test set.
    """
    if not ground_truth or not ground_truth.get("found"):
        return None

    labels_map: dict[str, int] = dict(ground_truth.get("labels") or {})
    # Display findings in the frozen NIH-14 order when present in this CSV row.
    csv_label_set = set(labels_map)
    findings = [lab for lab in XRAY_NIH14_LABELS if lab in csv_label_set]
    # Include any extra binary CSV labels not in the standard 14 (should be rare).
    for lab in ground_truth.get("label_columns") or []:
        if lab not in findings and lab in labels_map:
            findings.append(str(lab))

    rows: list[dict[str, Any]] = []
    per_model_summary: dict[str, dict[str, int]] = {}
    for spec in XRAY_COMPARISON_MODEL_SPECS:
        per_model_summary[str(spec["key"])] = {
            "correct_positives": 0,
            "missed_findings": 0,
            "false_positives": 0,
            "correct_negatives": 0,
            "unavailable": 0,
        }

    for finding in findings:
        gt_pos = int(labels_map.get(finding, 0)) == 1
        model_cells: dict[str, dict[str, Any]] = {}
        for spec in XRAY_COMPARISON_MODEL_SPECS:
            key = str(spec["key"])
            out = model_outputs.get(key) or {}
            prob = _model_probability_for_finding(out, finding)
            thr = _model_saved_threshold_for_finding(out, finding)
            if prob is None or thr is None or not out.get("available"):
                model_cells[key] = {
                    "available": False,
                    "predicted_positive": None,
                    "prediction_text": "Unavailable",
                    "outcome": "Unavailable",
                    "probability": prob,
                    "threshold": thr,
                }
                per_model_summary[key]["unavailable"] += 1
                continue
            predicted_positive = float(prob) >= float(thr)
            outcome = _prediction_outcome_label(
                ground_truth_positive=gt_pos,
                predicted_positive=predicted_positive,
            )
            model_cells[key] = {
                "available": True,
                "predicted_positive": predicted_positive,
                "prediction_text": (
                    "Predicted Positive" if predicted_positive else "Predicted Negative"
                ),
                "outcome": outcome,
                "probability": float(prob),
                "threshold": float(thr),
            }
            if outcome == "Correct positive":
                per_model_summary[key]["correct_positives"] += 1
            elif outcome == "Missed finding":
                per_model_summary[key]["missed_findings"] += 1
            elif outcome == "False positive":
                per_model_summary[key]["false_positives"] += 1
            else:
                per_model_summary[key]["correct_negatives"] += 1

        rows.append(
            {
                "finding": finding,
                "ground_truth_positive": gt_pos,
                "ground_truth_text": "Positive" if gt_pos else "Negative",
                "models": model_cells,
            }
        )

    gt_finding_count = sum(1 for f in findings if int(labels_map.get(f, 0)) == 1)
    total_label_decisions = len(findings)
    for key, summary in per_model_summary.items():
        correct_decisions = int(summary["correct_positives"]) + int(
            summary["correct_negatives"]
        )
        summary["correct_decisions"] = correct_decisions
        summary["total_label_decisions"] = total_label_decisions

    return {
        "findings": findings,
        "rows": rows,
        "ground_truth_finding_count": gt_finding_count,
        "positive_findings": list(ground_truth.get("positive_findings") or []),
        "per_model_summary": per_model_summary,
        "total_label_decisions": total_label_decisions,
    }


def _outcome_css_modifier(outcome: str) -> str:
    return {
        "Correct positive": "ok",
        "Correct negative": "ok",
        "Missed finding": "miss",
        "False positive": "fp",
    }.get(outcome, "na")


def _percentage_rank_indices(percentages: list[float]) -> list[int]:
    """Dense ranks for percentages: 0 = highest. Equal values share a rank."""
    unique_desc = sorted(set(percentages), reverse=True)
    rank_of = {value: index for index, value in enumerate(unique_desc)}
    return [rank_of[p] for p in percentages]


def _format_eval_cell_tooltip(
    *,
    ground_truth_text: str,
    prediction_text: str,
    outcome: str,
    probability: float | None,
    threshold: float | None,
) -> str:
    """Compact hover details for one official-evaluation model cell."""
    lines = [
        f"Ground truth: {ground_truth_text}",
        f"Prediction: {prediction_text}",
        f"Evaluation status: {outcome}",
    ]
    if probability is not None:
        lines.append(f"Probability: {float(probability):.4f}")
    if threshold is not None:
        lines.append(f"Saved threshold: {float(threshold):.4f}")
    return "\n".join(lines)


def _collect_model_prediction_lists(
    model_outputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Positive/negative finding lists per model using saved thresholds only."""
    collected: dict[str, dict[str, Any]] = {}
    for spec in XRAY_COMPARISON_MODEL_SPECS:
        key = str(spec["key"])
        out = model_outputs.get(key) or {}
        positives: list[str] = []
        negatives: list[str] = []
        available = bool(out.get("available"))
        for finding in XRAY_NIH14_LABELS:
            if not available:
                break
            prob = _model_probability_for_finding(out, finding)
            thr = _model_saved_threshold_for_finding(out, finding)
            if prob is None or thr is None:
                continue
            if float(prob) >= float(thr):
                positives.append(finding)
            else:
                negatives.append(finding)
        collected[key] = {
            "display_name": str(spec["display_name"]),
            "available": available,
            "positives": positives,
            "negatives": negatives,
            "error": out.get("error"),
        }
    return collected


def _render_external_inference_predictions(
    model_outputs: dict[str, dict[str, Any]],
) -> None:
    """External upload: predictions only — no GT, correctness, or accuracy totals."""
    st.markdown(
        '<p class="cxr-inference-gt cxr-inference-gt--external">'
        "Ground truth unavailable — external inference sample"
        "</p>",
        unsafe_allow_html=True,
    )

    header = "<th>Finding</th>" + "".join(
        f"<th>{html.escape(str(spec['display_name']))}</th>"
        for spec in XRAY_COMPARISON_MODEL_SPECS
    )
    body_rows: list[str] = []
    for finding in XRAY_NIH14_LABELS:
        cells = [f"<td>{html.escape(finding)}</td>"]
        for spec in XRAY_COMPARISON_MODEL_SPECS:
            out = model_outputs.get(str(spec["key"])) or {}
            prob = _model_probability_for_finding(out, finding)
            thr = _model_saved_threshold_for_finding(out, finding)
            if prob is None or thr is None or not out.get("available"):
                cells.append(
                    '<td class="cxr-eval-model cxr-eval-model--na">'
                    '<span class="cxr-eval-model__pred">Unavailable</span>'
                    "</td>"
                )
            elif float(prob) >= float(thr):
                cells.append(
                    '<td class="cxr-eval-model cxr-eval-model--pred-pos">'
                    '<span class="cxr-eval-model__pred">Predicted Positive</span>'
                    "</td>"
                )
            else:
                cells.append(
                    '<td class="cxr-eval-model cxr-eval-model--pred-neg">'
                    '<span class="cxr-eval-model__pred">Predicted Negative</span>'
                    "</td>"
                )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    st.markdown(
        '<div class="cxr-table-wrap cxr-table-wrap--eval">'
        '<table class="cxr-table cxr-table--eval">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>",
        unsafe_allow_html=True,
    )

    # Compact per-model positive lists (still no correctness claims).
    cards = _collect_model_prediction_lists(model_outputs)
    card_html: list[str] = []
    for spec in XRAY_COMPARISON_MODEL_SPECS:
        info = cards.get(str(spec["key"])) or {}
        if not info.get("available"):
            detail = html.escape(str(info.get("error") or "Model unavailable"))
        else:
            positives = info.get("positives") or []
            detail = (
                ", ".join(html.escape(p) for p in positives)
                if positives
                else "<em>None predicted positive</em>"
            )
        card_html.append(
            '<div class="cxr-pred-card">'
            f'<p class="cxr-pred-card__title">{html.escape(str(spec["display_name"]))}</p>'
            f'<p class="cxr-pred-card__label">Predicted Positive findings</p>'
            f'<p class="cxr-pred-card__value">{detail}</p>'
            "</div>"
        )
    st.markdown(
        f'<div class="cxr-pred-card-grid">{"".join(card_html)}</div>',
        unsafe_allow_html=True,
    )


def _render_official_image_metadata_card(
    ground_truth: dict[str, Any],
    upload_meta: dict[str, Any] | None,
) -> None:
    """Compact real metadata under the uploaded official test image."""
    if not ground_truth or not ground_truth.get("found"):
        return

    rows: list[tuple[str, str]] = [
        ("Official test filename", str(ground_truth.get("basename") or "")),
        ("Split", "Official test"),
        ("Ground-truth source", "NIH test.csv"),
    ]
    patient_id = ground_truth.get("patient_id")
    if patient_id not in (None, ""):
        rows.append(("Patient ID", str(patient_id)))

    meta = upload_meta or {}
    width = meta.get("image_width")
    height = meta.get("image_height")
    if width is not None and height is not None:
        rows.append(("Image dimensions", f"{int(width)} × {int(height)} px"))

    file_size_bytes = meta.get("file_size_bytes")
    if isinstance(file_size_bytes, int) and file_size_bytes >= 0:
        rows.append(("File size", _format_file_size(file_size_bytes)))

    positives = list(ground_truth.get("positive_findings") or [])
    rows.append(("Verified ground-truth findings", str(len(positives))))

    body = "".join(
        "<div class='cxr-photo-meta__row'>"
        f"<span class='cxr-photo-meta__k'>{html.escape(k)}</span>"
        f"<span class='cxr-photo-meta__v'>{html.escape(v)}</span>"
        "</div>"
        for k, v in rows
        if v != ""
    )
    st.markdown(
        f'<div class="cxr-photo-meta"><p class="cxr-photo-meta__title">Sample metadata</p>'
        f"{body}</div>",
        unsafe_allow_html=True,
    )


def _render_official_test_sample_evaluation(
    ground_truth: dict[str, Any] | None,
    model_outputs: dict[str, dict[str, Any]],
) -> None:
    """Official NIH test-sample matrix with GT emphasis and table summary rows."""
    if not ground_truth or not ground_truth.get("found"):
        _render_external_inference_predictions(model_outputs)
        return

    evaluation = _build_official_test_sample_evaluation(ground_truth, model_outputs)
    if evaluation is None:
        _render_external_inference_predictions(model_outputs)
        return

    positives = evaluation.get("positive_findings") or []
    if positives:
        badges = "".join(
            f'<span class="cxr-gt-badge">{html.escape(str(p))}</span>' for p in positives
        )
        badges_block = f'<div class="cxr-gt-badge-row">{badges}</div>'
    else:
        badges_block = (
            '<p class="cxr-gt-findings__empty">None (all findings labeled negative)</p>'
        )

    st.markdown(
        f"""
        <div class="cxr-gt-findings">
          <p class="cxr-gt-findings__heading">Verified ground-truth findings</p>
          {badges_block}
        </div>
        <h3 class="cxr-inference-results-title">Prediction Evaluation</h3>
        """,
        unsafe_allow_html=True,
    )

    header_cells = "".join(
        [
            "<th>Finding</th>",
            "<th>Ground Truth</th>",
        ]
        + [
            f"<th>{html.escape(str(spec['display_name']))}</th>"
            for spec in XRAY_COMPARISON_MODEL_SPECS
        ]
    )
    body_rows: list[str] = []
    for row in evaluation["rows"]:
        gt_pos = bool(row["ground_truth_positive"])
        row_class = "cxr-eval-row--gt-pos" if gt_pos else ""
        cells = [
            f"<td>{html.escape(str(row['finding']))}</td>",
            (
                f'<td class="cxr-eval-gt'
                f'{" cxr-eval-gt--pos" if gt_pos else ""}">'
                f"{html.escape(str(row['ground_truth_text']))}</td>"
            ),
        ]
        for spec in XRAY_COMPARISON_MODEL_SPECS:
            cell = (row.get("models") or {}).get(str(spec["key"])) or {}
            outcome = str(cell.get("outcome") or "Unavailable")
            pred_text = str(cell.get("prediction_text") or "Unavailable")
            mod = _outcome_css_modifier(outcome)
            tooltip = _format_eval_cell_tooltip(
                ground_truth_text=str(row["ground_truth_text"]),
                prediction_text=pred_text,
                outcome=outcome,
                probability=cell.get("probability"),
                threshold=cell.get("threshold"),
            )
            cells.append(
                f'<td class="cxr-eval-model cxr-eval-model--{mod}" '
                f'title="{html.escape(tooltip, quote=True)}">'
                f'<span class="cxr-eval-model__pred">{html.escape(pred_text)}</span>'
                f"</td>"
            )
        class_attr = f' class="{row_class}"' if row_class else ""
        body_rows.append(f"<tr{class_attr}>" + "".join(cells) + "</tr>")

    total = int(evaluation.get("total_label_decisions") or len(XRAY_NIH14_LABELS))
    summary_pcts: list[float] = []
    summary_correct: list[int] = []
    for spec in XRAY_COMPARISON_MODEL_SPECS:
        key = str(spec["key"])
        s = (evaluation.get("per_model_summary") or {}).get(key) or {}
        correct = int(s.get("correct_decisions", 0))
        pct = (100.0 * correct / total) if total else 0.0
        summary_correct.append(correct)
        summary_pcts.append(pct)

    ranks = _percentage_rank_indices(summary_pcts)
    correct_cells: list[str] = []
    pct_cells: list[str] = []
    # Dense ranks: 0 = highest = darkest green, then progressively lighter. Exact ties
    # share a rank and therefore the same shade. Clamped to the number of defined
    # shade steps so a fifth model would reuse the lightest rather than lose styling.
    max_rank = 3
    for correct, pct, rank in zip(summary_correct, summary_pcts, ranks):
        color_idx = min(int(rank), max_rank)
        correct_cells.append(
            f'<td class="cxr-eval-summary-val cxr-eval-summary-val--rank-{color_idx} '
            f'cxr-eval-summary-val--rank-light">{correct}/{total}</td>'
        )
        pct_cells.append(
            f'<td class="cxr-eval-summary-val cxr-eval-summary-val--rank-{color_idx}">'
            f"{pct:.1f}%</td>"
        )

    footer = (
        '<tr class="cxr-eval-summary-row cxr-eval-summary-row--correct">'
        '<td colspan="2" class="cxr-eval-summary-label">Correct label decisions</td>'
        + "".join(correct_cells)
        + "</tr>"
        '<tr class="cxr-eval-summary-row cxr-eval-summary-row--pct">'
        '<td colspan="2" class="cxr-eval-summary-label">'
        "Single-sample label-decision accuracy</td>"
        + "".join(pct_cells)
        + "</tr>"
    )

    st.markdown(
        '<div class="cxr-eval-legend" aria-label="Evaluation color legend">'
        '<span class="cxr-eval-legend__item cxr-eval-legend__item--gt">'
        "<i></i>Thick teal outline = verified ground-truth-positive finding</span>"
        '<span class="cxr-eval-legend__item cxr-eval-legend__item--ok">'
        "<i></i>Green = correct</span>"
        '<span class="cxr-eval-legend__item cxr-eval-legend__item--miss">'
        "<i></i>Amber = missed finding</span>"
        '<span class="cxr-eval-legend__item cxr-eval-legend__item--fp">'
        "<i></i>Red = false positive</span>"
        "</div>"
        '<div class="cxr-eval-legend cxr-eval-legend--accuracy" '
        'aria-label="Accuracy ranking legend">'
        '<span class="cxr-eval-legend__item cxr-eval-legend__item--rank-0">'
        "<i></i>Darkest green = highest</span>"
        '<span class="cxr-eval-legend__item cxr-eval-legend__item--rank-1">'
        "<i></i>then lighter by rank</span>"
        '<span class="cxr-eval-legend__item cxr-eval-legend__item--rank-2">'
        "<i></i><span aria-hidden=\"true\"></span></span>"
        '<span class="cxr-eval-legend__item cxr-eval-legend__item--rank-3">'
        "<i></i>lowest · ties share a shade</span>"
        "</div>"
        '<div class="cxr-table-wrap cxr-table-wrap--eval">'
        '<table class="cxr-table cxr-table--eval">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        f"<tfoot>{footer}</tfoot>"
        "</table></div>"
        '<p class="cxr-inference-gt-note cxr-sample-summary-disclaimer">'
        "This is a single official test-sample result. It is not the overall "
        "test-set accuracy or a clinical performance estimate."
        "</p>",
        unsafe_allow_html=True,
    )


def _model_probability_for_finding(
    model_output: dict[str, Any],
    finding: str,
) -> float | None:
    """Return one sigmoid probability for a finding, or None if unavailable."""
    if not model_output.get("available"):
        return None
    labels = list(model_output.get("labels") or XRAY_NIH14_LABELS)
    probs = model_output.get("probabilities")
    if probs is not None and len(probs) == len(labels) and finding in labels:
        return float(probs[labels.index(finding)])
    for row in model_output.get("rows") or []:
        if row.get("Finding") == finding:
            return float(row["Probability"])
    return None


def _model_saved_threshold_for_finding(
    model_output: dict[str, Any],
    finding: str,
) -> float | None:
    """Return that model's saved frozen validation threshold only (never invent 0.5)."""
    thresholds = model_output.get("thresholds") or {}
    if finding not in thresholds:
        return None
    return float(thresholds[finding])


def _render_upload_preview_card(preview: Image.Image, filename: str) -> bool:
    """Fixed-size image card with top-right remove control. Returns True if removed."""
    del filename  # preview only; do not show the uploaded filename under the image
    removed = False
    with st.container(key="xray_preview_card"):
        removed = st.button(
            "×",
            key="xray_remove_image",
            help="Remove image and clear results",
            type="secondary",
        )
        st.image(preview, width="stretch")
    return removed



def _resolve_inference_ground_truth(
    result_item: dict[str, Any],
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attach GT lookup for the current upload only (never scores the full test set)."""
    ground_truth = result_item.get("ground_truth")
    if ground_truth is None and payload is not None:
        ground_truth = payload.get("ground_truth")
    if ground_truth is None:
        ground_truth = _lookup_nih_test_ground_truth(
            str(
                result_item.get("filename")
                or st.session_state.get("xray_filename")
                or ""
            )
        )
    return ground_truth


def render_inference_demo() -> None:
    """Inference Demo — Classification (X-ray) and Segmentation (MRI) workstreams."""
    tab_classification, tab_segmentation = _render_workstream_tabs("inf_workstream_tabs")
    with tab_classification:
        _render_workstream_context("classification")
        _render_xray_inference_demo()
    with tab_segmentation:
        _render_workstream_context("segmentation")
        render_mri_inference_tab()


def _render_xray_inference_demo() -> None:
    """Upload/preview on the left; prediction evaluation beside the image."""
    model_error: str | None = None
    model_bundles: dict[str, dict[str, Any]] | None = None
    try:
        model_bundles = load_xray_comparison_models()
    except Exception as exc:  # noqa: BLE001 - surface load failures quietly in the UI
        model_error = str(exc)

    any_available = bool(
        model_bundles and any(b.get("available") for b in model_bundles.values())
    )
    if model_bundles is None:
        st.error(f"Unable to load classification models: {model_error}")
    elif not any_available:
        st.error(
            "No classification checkpoints are available. "
            "Download release `v1.0-presentation-demo` (or the shared asset bundle) and run "
            "`scripts/install_presentation_assets.ps1`. See docs/TEAM_SETUP.md."
        )
        for spec in XRAY_COMPARISON_MODEL_SPECS:
            bundle = (model_bundles or {}).get(str(spec["key"])) or {}
            if bundle.get("error"):
                st.caption(f"{spec['display_name']}: {bundle['error']}")

    left, right = st.columns([0.95, 1.35], gap="large")
    uploader_key = f"xray_inference_uploader_{st.session_state.xray_uploader_nonce}"

    with left:
        has_preview = st.session_state.xray_preview is not None

        if has_preview:
            removed = _render_upload_preview_card(
                st.session_state.xray_preview,
                str(st.session_state.xray_filename or "uploaded_image"),
            )
            if removed:
                _clear_xray_inference_state(bump_uploader=True)
                st.rerun()
            payload_left = st.session_state.get("xray_inference_results") or {}
            gt_left = None
            if payload_left.get("results"):
                gt_left = (payload_left["results"][0] or {}).get("ground_truth")
            if gt_left is None:
                gt_left = payload_left.get("ground_truth")
            if gt_left is None:
                gt_left = _lookup_nih_test_ground_truth(
                    str(st.session_state.xray_filename or "")
                )
            upload_meta = None
            if payload_left.get("results"):
                upload_meta = (payload_left["results"][0] or {}).get("upload_meta")
            if upload_meta is None:
                upload_meta = payload_left.get("upload_meta")
            if upload_meta is None:
                upload_meta = st.session_state.get("xray_upload_meta")
            _render_official_image_metadata_card(gt_left or {}, upload_meta)
        else:
            with st.container(key="xray_upload_card"):
                st.markdown(
                    """
                    <div class="cxr-upload-empty-copy">
                      <p class="cxr-upload-empty-copy__title">Click to upload chest X-ray</p>
                      <p class="cxr-upload-empty-copy__sub">PNG, JPG, or JPEG</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                uploaded = st.file_uploader(
                    "Click to upload chest X-ray",
                    type=["png", "jpg", "jpeg"],
                    accept_multiple_files=False,
                    key=uploader_key,
                    label_visibility="collapsed",
                    help="PNG, JPG, or JPEG",
                )

                if uploaded is not None and any_available and model_bundles is not None:
                    signature = _upload_file_signature(uploaded)
                    if signature != st.session_state.xray_processed_signature:
                        st.session_state.xray_inference_results = None
                        st.session_state.xray_preview = None
                        st.session_state.xray_filename = None
                        with st.spinner("Running four-model comparison…"):
                            outcome = _run_xray_inference_on_upload(uploaded, model_bundles)
                        st.session_state.xray_inference_results = {
                            "results": outcome["results"],
                            "errors": outcome["errors"],
                            "ground_truth": outcome.get("ground_truth"),
                            "upload_meta": outcome.get("upload_meta"),
                        }
                        st.session_state.xray_preview = outcome.get("preview")
                        st.session_state.xray_filename = outcome.get("filename")
                        st.session_state.xray_upload_meta = outcome.get("upload_meta")
                        st.session_state.xray_processed_signature = (
                            signature if outcome.get("preview") is not None else None
                        )
                        st.rerun()
                elif uploaded is not None and not any_available:
                    st.error("Classification models are not available.")

    payload = st.session_state.get("xray_inference_results")
    result_item = None
    if payload and payload.get("results"):
        result_item = payload["results"][0]

    with right:
        if payload and payload.get("errors"):
            for err in payload["errors"]:
                st.warning(err)

        if result_item is None:
            st.info("Upload an image to begin.")
        else:
            model_outputs = result_item.get("model_outputs") or {}
            ground_truth = _resolve_inference_ground_truth(result_item, payload)
            _render_official_test_sample_evaluation(ground_truth, model_outputs)

    if result_item is not None:
        st.markdown(
            f"""
            <div class="cxr-banner cxr-banner--disclaimer">
              <p class="cxr-banner__text">{XRAY_INFERENCE_DISCLAIMER}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )



def render_mri_inference_tab() -> None:
    """MRI segmentation inference — presentation samples and NIfTI upload."""
    _seg_ui_call(
        "render_segmentation_inference",
        fallback_stage="MRI slice selection, predicted tumour mask overlay, and demo controls",
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="cxr-footer">
          Medical Imaging AI Research Console · Academic prototype ·
          Not a medical device · Not for clinical use
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Governance & Limitations", expanded=False):
        st.markdown(
            """
            <div class="cxr-gov-item">
              <p class="cxr-gov-item__title">Intended use</p>
              <p class="cxr-gov-item__body">
                Academic AI engineering demonstration of a governed imaging workflow.
              </p>
            </div>
            <div class="cxr-gov-item">
              <p class="cxr-gov-item__title">Not intended for</p>
              <p class="cxr-gov-item__body">
                Diagnosis, triage, treatment, or autonomous clinical decisions.
              </p>
            </div>
            <div class="cxr-gov-item">
              <p class="cxr-gov-item__title">Dataset limitations</p>
              <p class="cxr-gov-item__body">
                Small course dataset, class/label imbalance, and limited generalizability.
              </p>
            </div>
            <div class="cxr-gov-item">
              <p class="cxr-gov-item__title">Human oversight</p>
              <p class="cxr-gov-item__body">
                Qualified clinicians retain responsibility for all medical decisions.
              </p>
            </div>
            <div class="cxr-gov-item">
              <p class="cxr-gov-item__title">Data privacy</p>
              <p class="cxr-gov-item__body">
                Do not upload real patient data without formal approval and governance.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Medical Imaging AI Research Console",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _init_state()
    load_styles()
    render_header()

    active_stage = render_stage_navigation()

    if active_stage == "Data Preparation":
        render_data_preparation()
    elif active_stage == "Train & Validate":
        render_train_validate()
    else:
        render_inference_demo()

    render_footer()


if __name__ == "__main__":
    main()
