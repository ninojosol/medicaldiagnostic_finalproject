"""Headless render smoke test for all three stages of the Streamlit console.

Uses streamlit.testing.v1.AppTest so Classification rendering can be proven
unchanged before and after the segmentation work lands.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app" / "streamlit_app.py"

from streamlit.testing.v1 import AppTest  # noqa: E402

STAGES = ["Data Preparation", "Train & Validate", "Inference Demo"]


def run_stage(stage: str, timeout: float = 300.0) -> dict:
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.session_state["active_stage"] = stage
    at.run()
    return {
        "stage": stage,
        "exception": [f"{e.type}: {e.message}" for e in at.exception],
        "n_markdown": len(at.markdown),
        "n_tabs": len(at.tabs),
        "n_buttons": len(at.button),
        "n_dataframe": len(at.dataframe),
        "n_error": len(at.error),
        "errors": [e.value for e in at.error],
        "n_warning": len(at.warning),
        "n_info": len(at.info),
    }


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    results = []
    ok = True
    for stage in STAGES:
        try:
            r = run_stage(stage)
        except Exception as exc:  # noqa: BLE001
            r = {"stage": stage, "exception": [f"HARNESS: {type(exc).__name__}: {exc}"]}
        if r.get("exception"):
            ok = False
        results.append(r)
        print(f"[{stage}] exceptions={r.get('exception')} markdown={r.get('n_markdown')} "
              f"tabs={r.get('n_tabs')} buttons={r.get('n_buttons')} errors={r.get('errors')}")

    out_dir = REPO / "outputs" / "segmentation" / "_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"ui_smoke_{label}.json").write_text(
        json.dumps({"ok": ok, "results": results}, indent=2), encoding="utf-8"
    )
    print(f"[smoke] ok={ok} -> outputs/segmentation/_audit/ui_smoke_{label}.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
