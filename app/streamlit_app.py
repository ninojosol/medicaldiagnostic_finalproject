"""
Medical Imaging AI Research Console — Streamlit UI.

Academic decision-support prototype. Not for clinical diagnosis or treatment.
Shared AI-engineering lifecycle with Chest X-ray Classification and MRI Segmentation tracks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

try:
    from PIL import Image
except ImportError:  # pragma: no cover - reported at runtime if missing
    Image = None  # type: ignore[assignment, misc]


APP_DIR = Path(__file__).resolve().parent
STYLES_PATH = APP_DIR / "assets" / "styles.css"

# (stage number, display label used for state + content routing)
PIPELINE_STAGES = (
    ("01", "Data Preparation"),
    ("02", "Train & Validate"),
    ("03", "Inference Demo"),
)

PREP_ACTIVITIES = (
    "01. Task & Data Source",
    "02. Data Quality Audit",
    "03. Exploratory Review",
    "04. Cleaning & Validation",
    "05. Leakage-Safe Split",
    "06. Standardized Preprocessing",
)

TRAIN_ACTIVITIES = (
    "01. Training Objective",
    "02. Model & Input Design",
    "03. Loss & Optimization",
    "04. Validation Monitoring",
    "05. Performance Evaluation",
    "06. Model Selection Gate",
)


def load_styles() -> None:
    """Inject external CSS while keeping presentation out of Python logic."""
    if STYLES_PATH.is_file():
        css = STYLES_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def load_approved_model() -> Any:
    """Placeholder for connecting an explicitly approved final model.

    Only an explicitly approved final model may be connected here after
    validation review and a single evaluation on the untouched official test set.
    Do not wire the baseline checkpoint into this path.
    """
    raise NotImplementedError(
        "No approved final model is connected. Complete model selection on "
        "validation results and evaluate once on the untouched official test "
        "set before enabling research inference."
    )


def _init_state() -> None:
    if "active_stage" not in st.session_state:
        st.session_state.active_stage = PIPELINE_STAGES[0][1]
    if "prep_activity" not in st.session_state:
        st.session_state.prep_activity = PREP_ACTIVITIES[0]
    if "prep_activity_pills" not in st.session_state:
        st.session_state.prep_activity_pills = PREP_ACTIVITIES[0]
    if "train_activity" not in st.session_state:
        st.session_state.train_activity = TRAIN_ACTIVITIES[0]
    if "train_activity_pills" not in st.session_state:
        st.session_state.train_activity_pills = TRAIN_ACTIVITIES[0]


def render_header() -> None:
    st.markdown(
        """
        <div class="cxr-header">
          <p class="cxr-header__eyebrow">Medical Image Analysis · Final Project</p>
          <h1 class="cxr-header__title">Medical Imaging AI Research Console</h1>
          <p class="cxr-header__subtitle">
            Academic decision-support prototype · Not for clinical diagnosis or treatment
          </p>
        </div>
        """,
        unsafe_allow_html=True,
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


def _placeholder_panel(title: str, body: str) -> str:
    return f"""
    <div class="cxr-placeholder">
      <p class="cxr-placeholder__title">{title}</p>
      <p class="cxr-placeholder__body">{body}</p>
    </div>
    """


def _process_pipeline(steps: list[str]) -> str:
    parts: list[str] = []
    for index, step in enumerate(steps):
        if index:
            parts.append('<span class="cxr-pipeline__arrow">→</span>')
        parts.append(f'<span class="cxr-pipeline__step">{step}</span>')
    return f'<div class="cxr-pipeline">{"".join(parts)}</div>'


def _prep_activity_content(activity: str) -> tuple[str, str, str]:
    """Return (shared_principle, xray_card_html, mri_card_html)."""
    if activity == PREP_ACTIVITIES[0]:
        principle = (
            "Both tracks convert validated medical images into structured AI outputs. "
            "The learning target differs: disease probabilities for classification and "
            "spatial masks for segmentation."
        )
        xray = _track_card(
            "Track A",
            "Chest X-ray Classification",
            [
                ("Task", "Multi-label chest X-ray classification"),
                ("Input", "Chest X-ray image"),
                ("Output", "Probabilities across 14 disease findings"),
                ("Dataset scope", "1,422 X-ray images"),
                ("Training purpose", "Academic AI engineering research prototype"),
            ],
            accent="accent",
            badges=[("Classification", "ok")],
        )
        mri = _track_card(
            "Track B",
            "MRI Segmentation",
            [
                ("Task", "MRI tumor segmentation"),
                ("Input", "MRI volume"),
                ("Output", "Pixel/voxel-level predicted segmentation mask"),
                ("Dataset scope", "To be confirmed during pipeline implementation"),
                ("Training purpose", "Academic AI engineering research prototype"),
                (
                    "Status",
                    '<span class="cxr-badge cxr-badge--planned">Pipeline planned</span>',
                ),
            ],
            accent="muted",
            badges=[("Segmentation", "planned")],
        )
        return principle, xray, mri

    if activity == PREP_ACTIVITIES[1]:
        principle = (
            "Training data must be traceable, structurally valid, and correctly "
            "paired with its learning target."
        )
        xray = _track_card(
            "Track A",
            "Chest X-ray Classification",
            [],
            accent="accent",
            badges=[("Classification", "ok")],
            extra_html=_checklist(
                [
                    "Image–label matching",
                    "Image readability and supported-file validation",
                    "Patient identifier availability for safe splitting",
                    "Duplicate-record review",
                ]
            ),
        )
        mri = _track_card(
            "Track B",
            "MRI Segmentation",
            [
                (
                    "Status",
                    "Validation rules defined; execution pending",
                ),
            ],
            accent="muted",
            badges=[("Segmentation", "planned")],
            extra_html=_checklist(
                [
                    "MRI volume–mask pairing",
                    "Volume/mask dimensional consistency",
                    "Orientation consistency checks",
                    "Missing or invalid mask review",
                ]
            ),
        )
        return principle, xray, mri

    if activity == PREP_ACTIVITIES[2]:
        principle = (
            "Exploration identifies imbalance, representation issues, and whether "
            "the learning target is suitable before model development."
        )
        xray_extra = (
            _checklist(
                [
                    "Representative X-ray image review",
                    "Multi-label finding distribution",
                    "Class-imbalance assessment",
                ]
            )
            + '<p class="cxr-card__body" style="margin-top:0.85rem;">'
            "Disease findings occur at different frequencies. This imbalance informs "
            "the use of weighted binary cross-entropy and AUROC-based evaluation."
            "</p>"
            + _placeholder_panel(
                "Chart placeholder",
                "Label-distribution chart generated from verified notebook output — to be connected",
            )
        )
        mri_extra = (
            _checklist(
                [
                    "Representative MRI volume and mask-overlay review",
                    "Foreground/background coverage assessment",
                    "Tumor-size variation review",
                ]
            )
            + _placeholder_panel(
                "Example placeholder",
                "MRI image-and-mask examples will be connected after dataset preparation",
            )
        )
        xray = _track_card(
            "Track A",
            "Chest X-ray Classification",
            [],
            accent="accent",
            badges=[("Classification", "ok")],
            extra_html=xray_extra,
        )
        mri = _track_card(
            "Track B",
            "MRI Segmentation",
            [],
            accent="muted",
            badges=[("Segmentation", "planned")],
            extra_html=mri_extra,
        )
        return principle, xray, mri

    if activity == PREP_ACTIVITIES[3]:
        principle = (
            "Only records that meet defined validation rules proceed to model development."
        )
        xray = _track_card(
            "Track A",
            "Chest X-ray Classification",
            [],
            accent="accent",
            badges=[("Classification", "ok")],
            extra_html=_checklist(
                [
                    "Confirm supported image format and readable image data",
                    "Validate 14-label vector structure",
                    "Confirm PatientId availability",
                    "Preserve source-to-label traceability",
                ]
            )
            + '<p class="cxr-card__body" style="margin-top:0.75rem;">'
            "Resizing and DenseNet transforms belong to preprocessing, not cleaning."
            "</p>",
        )
        mri = _track_card(
            "Track B",
            "MRI Segmentation",
            [
                (
                    "Status",
                    '<span class="cxr-badge cxr-badge--planned">Validation workflow planned</span>',
                ),
            ],
            accent="muted",
            badges=[("Segmentation", "planned")],
            extra_html=_checklist(
                [
                    "Confirm readable MRI volume",
                    "Confirm corresponding mask exists",
                    "Validate image-mask dimensions",
                    "Preserve patient-level traceability",
                ]
            ),
        )
        return principle, xray, mri

    if activity == PREP_ACTIVITIES[4]:
        principle = (
            "Patient-level separation prevents the model from learning from the same "
            "patient in both training and validation data."
        )
        split_visual = """
        <div class="cxr-split">
          <div class="cxr-split__seg cxr-split__seg--train">
            <span class="cxr-split__label">Train</span>
            <span class="cxr-split__value">948 images · 744 patients</span>
          </div>
          <div class="cxr-split__seg cxr-split__seg--val">
            <span class="cxr-split__label">Validation</span>
            <span class="cxr-split__value">252 images · 186 patients</span>
          </div>
          <div class="cxr-split__seg cxr-split__seg--test">
            <span class="cxr-split__label">Official test</span>
            <span class="cxr-split__value">Untouched</span>
          </div>
        </div>
        <div class="cxr-note">
          The original course validation split contained patient overlap.
          A new GroupShuffleSplit using PatientId was created before model development.
        </div>
        """
        xray = (
            '<div class="cxr-emphasis">'
            + _track_card(
                "Track A · Leakage control",
                "Chest X-ray Classification",
                [
                    ("Method", "GroupShuffleSplit using PatientId"),
                    ("Training split", "948 images / 744 patients"),
                    ("Validation split", "252 images / 186 patients"),
                    ("Verification", "Zero patient overlap between training and validation"),
                    ("Official test set", "Untouched until final model selection"),
                ],
                accent="accent",
                badges=[("Classification", "ok")],
                extra_html=split_visual,
            )
            + "</div>"
        )
        mri = _track_card(
            "Track B",
            "MRI Segmentation",
            [
                ("Method", "Patient-level train/validation split"),
                ("Principle", "No patient appears in more than one development split"),
                ("Official test set", "Reserved for final evaluation"),
                (
                    "Status",
                    "Split will be created after MRI dataset preparation",
                ),
            ],
            accent="muted",
            badges=[("Segmentation", "planned")],
        )
        return principle, xray, mri

    # PREP_ACTIVITIES[5] — Standardized Preprocessing
    principle = (
        "The exact same approved preprocessing must be applied during training, "
        "validation, test evaluation, and future inference."
    )
    xray = _track_card(
        "Track A",
        "Chest X-ray Classification",
        [
            (
                "Consistency",
                "Training, validation, test, and inference use the same preprocessing specification",
            ),
        ],
        accent="accent",
        badges=[("Classification", "ok")],
        extra_html=_process_pipeline(
            [
                "Validate image",
                "Convert grayscale image to 3 channels",
                "Resize to 320 × 320",
                "Apply DenseNet preprocessing",
                "Model-ready tensor",
            ]
        ),
    )
    mri = _track_card(
        "Track B",
        "MRI Segmentation",
        [
            ("Alignment", "The mask remains spatially aligned with the MRI volume"),
            (
                "Status",
                "Preprocessing design planned; parameters will be finalized after dataset inspection.",
            ),
        ],
        accent="muted",
        badges=[("Segmentation", "planned")],
        extra_html=_process_pipeline(
            [
                "Validate MRI volume and mask",
                "Standardize orientation",
                "Prepare intensity values",
                "Resize or prepare patches/volume",
                "Model-ready image and mask tensors",
            ]
        ),
    )
    return principle, xray, mri


def render_prep_activity_stepper() -> str:
    """Compact six-activity preparation stepper (distinct from major pipeline tabs)."""
    st.markdown(
        '<p class="cxr-prep-label">Governed preparation workflow</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="cxr-prep-stepper-marker"></div>', unsafe_allow_html=True)

    selected = st.pills(
        "Preparation activity",
        options=list(PREP_ACTIVITIES),
        selection_mode="single",
        key="prep_activity_pills",
        label_visibility="collapsed",
    )
    if selected is None:
        selected = PREP_ACTIVITIES[0]
        st.session_state.prep_activity_pills = selected
        st.rerun()
    st.session_state.prep_activity = selected
    return selected


def render_readiness_gate() -> None:
    st.markdown(
        """
        <div class="cxr-gate">
          <p class="cxr-gate__eyebrow">Preparation gate</p>
          <h3 class="cxr-gate__title">Readiness Gate Before Model Development</h3>
          <ul class="cxr-gate__list">
            <li>Data and target pairing validated</li>
            <li>Data quality and imbalance reviewed</li>
            <li>Patient leakage prevented</li>
            <li>Preprocessing defined consistently</li>
            <li>Official test data protected</li>
          </ul>
          <p class="cxr-gate__closing">
            Only after these conditions are satisfied does the project proceed to
            training and validation.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_preparation() -> None:
    activity = render_prep_activity_stepper()
    principle, xray_html, mri_html = _prep_activity_content(activity)

    st.markdown(
        f'<p class="cxr-principle"><span class="cxr-principle__label">Shared principle</span>'
        f"{principle}</p>",
        unsafe_allow_html=True,
    )

    col_xray, col_mri = st.columns(2, gap="medium")
    with col_xray:
        st.markdown(xray_html, unsafe_allow_html=True)
    with col_mri:
        st.markdown(mri_html, unsafe_allow_html=True)

    render_readiness_gate()


def _metric_cards(metrics: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="cxr-metric-card"><p class="cxr-metric-card__value">{value}</p>'
        f'<p class="cxr-metric-card__label">{label}</p></div>'
        for label, value in metrics
    )
    return f'<div class="cxr-metric-grid">{cells}</div>'


def _train_activity_content(activity: str) -> tuple[str, str, str]:
    """Return (shared_principle, xray_card_html, mri_card_html)."""
    if activity == TRAIN_ACTIVITIES[0]:
        principle = (
            "Each track learns a different clinical-image target and must be evaluated "
            "using metrics appropriate to that target."
        )
        xray = _track_card(
            "Track A",
            "Chest X-ray Classification",
            [
                ("Task", "Multi-label classification"),
                ("Input", "Chest X-ray image"),
                ("Learning target", "Probabilities across 14 disease findings"),
                (
                    "Training objective",
                    "Learn independent disease-finding likelihoods from prepared X-ray images",
                ),
            ],
            accent="accent",
            badges=[("Classification", "ok")],
        )
        mri = _track_card(
            "Track B",
            "MRI Segmentation",
            [
                ("Task", "Tumor segmentation"),
                ("Input", "MRI volume"),
                ("Learning target", "Pixel/voxel-level segmentation mask"),
                (
                    "Training objective",
                    "Identify the spatial tumor region within an MRI volume",
                ),
                (
                    "Status",
                    '<span class="cxr-badge cxr-badge--planned">Pipeline planned</span>',
                ),
            ],
            accent="muted",
            badges=[("Segmentation", "planned")],
        )
        return principle, xray, mri

    if activity == TRAIN_ACTIVITIES[1]:
        principle = (
            "Architecture selection must match the structure of the input and the "
            "required output."
        )
        xray = _track_card(
            "Track A",
            "Chest X-ray Classification",
            [
                ("Architecture", "DenseNet121 transfer learning"),
                ("Input", "320 × 320, 3-channel converted X-ray image"),
                ("Output", "14 disease-finding probabilities"),
                (
                    "Rationale",
                    "Pretrained visual features support transfer learning on the course dataset",
                ),
                (
                    "Status",
                    '<span class="cxr-badge cxr-badge--ok">Baseline completed</span>',
                ),
                (
                    "Fine-tuning candidate",
                    '<span class="cxr-badge cxr-badge--pending">Under validation review</span>',
                ),
            ],
            accent="accent",
            badges=[("Classification", "ok")],
        )
        mri = _track_card(
            "Track B",
            "MRI Segmentation",
            [
                ("Candidate architecture", "3D U-Net"),
                ("Input", "MRI volume"),
                ("Output", "Predicted spatial segmentation mask"),
                (
                    "Rationale",
                    "The architecture preserves spatial information required for segmentation",
                ),
                (
                    "Status",
                    '<span class="cxr-badge cxr-badge--planned">Planned</span>',
                ),
            ],
            accent="muted",
            badges=[("Segmentation", "planned")],
        )
        return principle, xray, mri

    if activity == TRAIN_ACTIVITIES[2]:
        principle = (
            "The training objective must account for the task and the structure of "
            "the target labels."
        )
        xray = _track_card(
            "Track A",
            "Chest X-ray Classification",
            [
                ("Loss", "Weighted binary cross-entropy"),
                ("Reason", "Disease findings have unequal frequencies"),
                ("Output activation", "Independent probability per disease finding"),
            ],
            accent="accent",
            badges=[("Classification", "ok")],
            extra_html=(
                '<div class="cxr-note">'
                "Ordinary binary accuracy is not the primary evaluation metric for "
                "this multi-label task."
                "</div>"
            ),
        )
        mri = _track_card(
            "Track B",
            "MRI Segmentation",
            [
                ("Candidate training objective", "Dice-based segmentation loss"),
                (
                    "Reason",
                    "Segmentation evaluates overlap between the predicted and reference mask",
                ),
                (
                    "Primary consideration",
                    "Preserve spatial alignment between MRI volume and mask",
                ),
                (
                    "Status",
                    '<span class="cxr-badge cxr-badge--planned">Planned</span>',
                ),
            ],
            accent="muted",
            badges=[("Segmentation", "planned")],
        )
        return principle, xray, mri

    if activity == TRAIN_ACTIVITIES[3]:
        principle = (
            "Validation evidence monitors generalization during development without "
            "exposing the untouched official test set."
        )
        xray = _track_card(
            "Track A",
            "Chest X-ray Classification",
            [
                ("Validation design", "Patient-safe holdout"),
                ("Training split", "948 images / 744 patients"),
                ("Validation split", "252 images / 186 patients"),
                ("Patient overlap", "0"),
                (
                    "Monitoring focus",
                    "Training loss, validation loss, per-disease AUROC, macro AUROC",
                ),
            ],
            accent="accent",
            badges=[("Classification", "ok")],
            extra_html=_placeholder_panel(
                "Curve placeholder",
                "Verified training and validation curves will be connected from notebook output",
            ),
        )
        mri = _track_card(
            "Track B",
            "MRI Segmentation",
            [
                ("Validation design", "Patient-safe holdout"),
                ("Monitoring focus", "Training loss, validation loss, Dice score, IoU"),
                (
                    "Status",
                    '<span class="cxr-badge cxr-badge--planned">Planned</span>',
                ),
            ],
            accent="muted",
            badges=[("Segmentation", "planned")],
            extra_html=_placeholder_panel(
                "Curve placeholder",
                "Validation curves will be available after segmentation training",
            ),
        )
        return principle, xray, mri

    if activity == TRAIN_ACTIVITIES[4]:
        principle = (
            "Metrics must reflect the real task rather than relying on a misleading "
            "single accuracy value."
        )
        xray = _track_card(
            "Track A",
            "Chest X-ray Classification",
            [
                ("Primary metric", "Per-disease AUROC and macro AUROC"),
                ("Validation method", "Patient-safe holdout"),
            ],
            accent="accent",
            badges=[("Classification", "ok")],
            extra_html=(
                _metric_cards(
                    [
                        ("Baseline macro AUROC", "0.580"),
                        ("Best validation loss", "0.1707"),
                    ]
                )
                + '<div class="cxr-note">'
                "Rare validation classes may produce unstable AUROC estimates; "
                "binary accuracy is not the primary metric."
                "</div>"
                + _placeholder_panel(
                    "Metric placeholder",
                    "Per-disease AUROC table/chart from verified notebook output — to be connected",
                )
            ),
        )
        mri = _track_card(
            "Track B",
            "MRI Segmentation",
            [
                ("Primary metric", "Dice score"),
                ("Supporting metric", "Intersection over Union (IoU)"),
                (
                    "Evaluation focus",
                    "Overlap between predicted and reference masks",
                ),
                (
                    "Status",
                    '<span class="cxr-badge cxr-badge--planned">Planned</span>',
                ),
            ],
            accent="muted",
            badges=[("Segmentation", "planned")],
        )
        return principle, xray, mri

    # TRAIN_ACTIVITIES[5] — Model Selection Gate
    principle = (
        "Validation results select the final model. The official test set is used "
        "only once after selection."
    )
    xray = _track_card(
        "Track A",
        "Chest X-ray Classification",
        [
            (
                "Candidate 1",
                'DenseNet121 baseline — <span class="cxr-badge cxr-badge--ok">completed</span>',
            ),
            (
                "Candidate 2",
                'Fine-tuning model — <span class="cxr-badge cxr-badge--pending">under validation review</span>',
            ),
            (
                "Selection basis",
                "Validation loss, per-disease AUROC, macro AUROC, and stability across disease findings",
            ),
        ],
        accent="accent",
        badges=[("Classification", "ok")],
        extra_html=(
            '<div class="cxr-note">'
            "Final X-ray model selection is pending validation comparison."
            "</div>"
        ),
    )
    mri = _track_card(
        "Track B",
        "MRI Segmentation",
        [
            ("Candidate model", "3D U-Net"),
            (
                "Selection basis",
                "Validation Dice score, IoU, and qualitative mask review",
            ),
        ],
        accent="muted",
        badges=[("Segmentation", "planned")],
        extra_html=(
            '<div class="cxr-note">'
            "Segmentation training and validation are planned before model selection."
            "</div>"
        ),
    )
    return principle, xray, mri


def render_train_activity_stepper() -> str:
    """Compact six-activity train/validate stepper."""
    st.markdown(
        '<p class="cxr-prep-label">Shared ML development workflow</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="cxr-train-stepper-marker"></div>', unsafe_allow_html=True)

    selected = st.pills(
        "Training activity",
        options=list(TRAIN_ACTIVITIES),
        selection_mode="single",
        key="train_activity_pills",
        label_visibility="collapsed",
    )
    if selected is None:
        selected = TRAIN_ACTIVITIES[0]
        st.session_state.train_activity_pills = selected
        st.rerun()
    st.session_state.train_activity = selected
    return selected


def render_selection_gate() -> None:
    st.markdown(
        """
        <div class="cxr-gate">
          <p class="cxr-gate__eyebrow">Model selection gate</p>
          <h3 class="cxr-gate__title">Final Evaluation Protection Gate</h3>
          <ul class="cxr-gate__list">
            <li>Select one final model using validation evidence</li>
            <li>Freeze the selected model and preprocessing specification</li>
            <li>Evaluate the selected model once on the untouched official test set</li>
            <li>Do not tune or re-select models using official test results</li>
            <li>Only an explicitly approved final model may later connect to the research inference demo</li>
          </ul>
          <p class="cxr-gate__closing">
            Untouched test data provides the final unbiased estimate of model performance.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_train_validate() -> None:
    activity = render_train_activity_stepper()
    principle, xray_html, mri_html = _train_activity_content(activity)

    st.markdown(
        f'<p class="cxr-principle"><span class="cxr-principle__label">Shared principle</span>'
        f"{principle}</p>",
        unsafe_allow_html=True,
    )

    col_xray, col_mri = st.columns(2, gap="medium")
    with col_xray:
        st.markdown(xray_html, unsafe_allow_html=True)
    with col_mri:
        st.markdown(mri_html, unsafe_allow_html=True)

    if activity == TRAIN_ACTIVITIES[5]:
        render_selection_gate()


def _format_file_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 ** 2):.2f} MB"


def render_xray_inference_tab() -> None:
    st.markdown(
        """
        <div class="cxr-card cxr-card--accent">
          <p class="cxr-card__label">Research-use guardrail</p>
          <p class="cxr-card__title">Chest X-ray research inference</p>
          <ul class="cxr-card__list">
            <li>
              <span class="cxr-k">Intended use</span>
              <span class="cxr-v">Academic AI engineering demonstration</span>
            </li>
            <li>
              <span class="cxr-k">Input</span>
              <span class="cxr-v">Non-identifiable chest X-ray image</span>
            </li>
            <li>
              <span class="cxr-k">Future output</span>
              <span class="cxr-v">Probabilities across 14 disease findings</span>
            </li>
            <li>
              <span class="cxr-k">Clinical restriction</span>
              <span class="cxr-v">Not a diagnosis or clinical recommendation</span>
            </li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="cxr-prep-label">Upload workspace</p>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload a non-identifiable chest X-ray image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help="Use only approved academic sample data. Do not upload real patient data.",
        key="xray_inference_uploader",
    )

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        suffix = Path(uploaded.name).suffix.lower().lstrip(".") or "unknown"
        width = height = None
        preview_error = None

        if Image is None:
            preview_error = "Pillow is required to preview uploaded images."
        else:
            try:
                image = Image.open(uploaded)
                width, height = image.size
                st.image(
                    image,
                    caption="Uploaded image preview",
                    width="stretch",
                )
            except Exception as exc:  # noqa: BLE001 - surface user-facing preview failures
                preview_error = f"Unable to open image for preview: {exc}"

        meta_rows = [
            ("Filename", uploaded.name),
            ("Format", suffix.upper()),
            (
                "Dimensions",
                f"{width} × {height} px" if width and height else "Unavailable",
            ),
            ("File size", _format_file_size(len(file_bytes))),
        ]
        meta_html = "".join(
            f'<li><span class="cxr-k">{k}</span><span class="cxr-v">{v}</span></li>'
            for k, v in meta_rows
        )
        st.markdown(
            f"""
            <div class="cxr-card cxr-card--accent">
              <p class="cxr-card__label">File metadata</p>
              <p class="cxr-card__title">Upload summary</p>
              <ul class="cxr-card__list">{meta_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if preview_error:
            st.warning(preview_error)

    st.markdown(
        """
        <div class="cxr-card">
          <p class="cxr-card__label">Visible preprocessing flow</p>
          <p class="cxr-card__title">Approved DenseNet-compatible path</p>
          <div class="cxr-pipeline">
            <span class="cxr-pipeline__step">Validate image</span>
            <span class="cxr-pipeline__arrow">→</span>
            <span class="cxr-pipeline__step">Convert grayscale image to 3 channels</span>
            <span class="cxr-pipeline__arrow">→</span>
            <span class="cxr-pipeline__step">Resize to 320 × 320</span>
            <span class="cxr-pipeline__arrow">→</span>
            <span class="cxr-pipeline__step">Apply DenseNet preprocessing</span>
            <span class="cxr-pipeline__arrow">→</span>
            <span class="cxr-pipeline__step">Model output</span>
          </div>
          <p class="cxr-card__body">
            The approved preprocessing specification must match training, validation,
            official test evaluation, and future inference.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="cxr-status-board">
          <p class="cxr-status-board__label">Model connection status</p>
          <p class="cxr-status-board__title">Governance checkpoint</p>
          <div class="cxr-status-row">
            <span class="cxr-status-row__k">Final model selection</span>
            <span class="cxr-status-row__v">Pending validation comparison</span>
          </div>
          <div class="cxr-status-row">
            <span class="cxr-status-row__k">Official test evaluation</span>
            <span class="cxr-status-row__v">Required after model selection</span>
          </div>
          <div class="cxr-status-row">
            <span class="cxr-status-row__k">Research inference connection</span>
            <span class="cxr-status-row__v">Not yet approved</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    run_clicked = st.button(
        "Run Research Inference",
        type="primary",
        key="run_xray_research_inference",
    )

    if run_clicked:
        # Do not invoke load_approved_model() yet. Only an explicitly approved
        # final model may be connected here after final test evaluation.
        # Baseline weights must never be wired into this workspace.
        if uploaded is None:
            st.info("Upload an approved academic sample image before running research inference.")
        else:
            st.markdown(
                """
                <div class="cxr-info-state">
                  <p class="cxr-info-state__title">Inference connection pending</p>
                  <p class="cxr-info-state__text">
                    Inference connection pending: a final model must first be selected
                    using validation results and evaluated once on the untouched
                    official test set.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="cxr-future-output">
          <p class="cxr-future-output__label">Unavailable</p>
          <p class="cxr-future-output__title">Future Research Output</p>
          <ul class="cxr-future-output__list">
            <li>14 disease-finding probability outputs</li>
            <li>Clear probability labels for each finding</li>
            <li>Research-use interpretation notice</li>
            <li>No diagnosis or treatment recommendation</li>
          </ul>
          <p class="cxr-future-output__note">
            Output remains unavailable until model governance requirements are satisfied
            and an approved final model is connected through load_approved_model().
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mri_inference_tab() -> None:
    st.markdown(
        """
        <div class="cxr-card cxr-card--muted">
          <p class="cxr-card__label">Research-use guardrail</p>
          <p class="cxr-card__title">MRI segmentation research demo</p>
          <ul class="cxr-card__list">
            <li>
              <span class="cxr-k">Intended use</span>
              <span class="cxr-v">Academic AI engineering demonstration</span>
            </li>
            <li>
              <span class="cxr-k">Input</span>
              <span class="cxr-v">MRI volume</span>
            </li>
            <li>
              <span class="cxr-k">Future output</span>
              <span class="cxr-v">Predicted spatial segmentation mask</span>
            </li>
            <li>
              <span class="cxr-k">Clinical restriction</span>
              <span class="cxr-v">Not for diagnosis, treatment, triage, or autonomous care decisions</span>
            </li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="cxr-card">
          <p class="cxr-card__label">Intended workflow</p>
          <p class="cxr-card__title">Future segmentation path</p>
          <div class="cxr-pipeline">
            <span class="cxr-pipeline__step">MRI volume</span>
            <span class="cxr-pipeline__arrow">→</span>
            <span class="cxr-pipeline__step">Standardized preprocessing</span>
            <span class="cxr-pipeline__arrow">→</span>
            <span class="cxr-pipeline__step">Approved segmentation model</span>
            <span class="cxr-pipeline__arrow">→</span>
            <span class="cxr-pipeline__step">Predicted mask overlay</span>
          </div>
          <p class="cxr-card__body">
            Mask output will remain spatially aligned with the MRI volume and is
            intended only for academic model evaluation.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="cxr-empty-state">
          <p class="cxr-empty-state__title">MRI Segmentation Demo Not Yet Connected</p>
          <p class="cxr-empty-state__text">
            The segmentation pipeline is planned. This workspace will connect only
            after dataset preparation, patient-safe validation, model selection, and
            final evaluation on protected test data.
          </p>
          <span class="cxr-badge cxr-badge--planned">Pipeline planned</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button(
        "MRI Demo Not Yet Connected",
        disabled=True,
        use_container_width=False,
        key="mri_demo_disabled",
    )


def render_human_review_panel() -> None:
    st.markdown(
        """
        <div class="cxr-human-review">
          <p class="cxr-human-review__label">Responsible use</p>
          <h3 class="cxr-human-review__title">Human Review and Responsible Use</h3>
          <ul class="cxr-human-review__list">
            <li>Outputs are research artifacts, not clinical conclusions</li>
            <li>Qualified clinicians retain responsibility for patient-care decisions</li>
            <li>Model behavior must be interpreted with dataset limitations and uncertainty in mind</li>
            <li>Input data must be governed, approved, and non-identifiable</li>
            <li>Inference should record approved-model version and preprocessing version when connected in the future</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_inference_demo() -> None:
    st.markdown(
        """
        <div class="cxr-banner">
          <p class="cxr-banner__title">Research Prototype Only</p>
          <p class="cxr-banner__text">
            Results must not be used for diagnosis, treatment, triage, or patient-care
            decisions. Do not upload real patient data without formal approval and governance.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_xray, tab_mri = st.tabs(
        ["Chest X-ray Classification", "MRI Segmentation"]
    )
    with tab_xray:
        render_xray_inference_tab()
    with tab_mri:
        render_mri_inference_tab()

    render_human_review_panel()


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
