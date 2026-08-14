"""Capture Streamlit UI screenshots for the Segmentation workstream.

Starts the app on a spare port, drives it with Playwright, clicks through the
three pipeline stages, switches to the Segmentation workstream tab on each, and
saves full-page PNGs. Also captures the Classification tabs so the report can
show they still render.

Output: outputs/segmentation/_audit/screenshots/

Usage::

    python scripts/mri_capture_screenshots.py [--port 8599] [--keep-open]
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHOTS = REPO / "outputs" / "segmentation" / "_audit" / "screenshots"

STAGES = ["Data Preparation", "Train & Validate", "Inference Demo"]


def free_port(preferred: int) -> int:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def wait_for(url: str, timeout: float = 180.0) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(1.5)
    return False


def _click_tab(page, needle: str) -> bool:
    """Click the first Streamlit tab whose label contains `needle`."""
    tabs = page.locator('[data-testid="stTab"]')
    for i in range(tabs.count()):
        try:
            if needle in tabs.nth(i).inner_text():
                tabs.nth(i).click()
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _settle(page) -> None:
    """Nudge the page so lazy charts/images finish painting before capture."""
    page.mouse.wheel(0, 900)
    page.wait_for_timeout(2000)
    page.mouse.wheel(0, -2000)
    page.wait_for_timeout(2000)


def _shoot(page, path: Path, base_height: int = 1150) -> None:
    """Screenshot the whole page.

    Streamlit scrolls inside its own container, so Playwright's `full_page` only
    yields one viewport. Grow the viewport to the app's scroll height first, then
    capture, then restore.
    """
    try:
        # section[data-testid="stMain"] is Streamlit's actual scroll container;
        # stAppViewContainer and <body> both report only the viewport height.
        height = page.evaluate(
            """() => {
                const main = document.querySelector('section[data-testid="stMain"]');
                let best = Math.max(
                    document.body.scrollHeight,
                    document.documentElement.scrollHeight
                );
                if (main) best = Math.max(best, main.scrollHeight);
                return best;
            }"""
        )
    except Exception:  # noqa: BLE001
        height = base_height
    target = int(min(max(height + 120, base_height), 14000))
    page.set_viewport_size({"width": 1680, "height": target})
    page.wait_for_timeout(2500)
    page.screenshot(path=str(path), full_page=True)
    page.set_viewport_size({"width": 1680, "height": base_height})
    page.wait_for_timeout(600)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8599)
    ap.add_argument("--keep-open", action="store_true")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}"

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(REPO / "app" / "streamlit_app.py"),
         "--server.port", str(port), "--server.headless", "true",
         "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        print(f"[app] starting on {url} ...")
        if not wait_for(f"{url}/_stcore/health"):
            print("ERROR: streamlit did not become healthy", file=sys.stderr)
            out = proc.stdout.read(4000) if proc.stdout else ""
            print(out, file=sys.stderr)
            return 2
        print("[app] healthy")

        captured: list[str] = []
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1680, "height": 1150},
                                    device_scale_factor=2)
            page.goto(url, wait_until="networkidle", timeout=120_000)
            page.wait_for_timeout(6000)

            for stage in STAGES:
                slug = stage.lower().replace(" & ", "_").replace(" ", "_")

                # click the stage button in the pipeline nav
                btn = page.get_by_role("button", name=stage).first
                btn.click()
                page.wait_for_timeout(4500)

                # --- Classification tab (default) ---
                page.wait_for_timeout(500)
                path = SHOTS / f"{slug}__classification.png"
                _shoot(page, path)
                captured.append(path.name)
                print(f"  captured {path.name}")

                # --- Segmentation tab ---
                # Streamlit 1.6x renders tabs as [data-testid="stTab"], not button[role=tab].
                if not _click_tab(page, "Segmentation"):
                    print(f"  WARNING: no Segmentation tab found on {stage}")
                    continue
                page.wait_for_timeout(7000)
                _settle(page)

                path = SHOTS / f"{slug}__segmentation.png"
                _shoot(page, path)
                captured.append(path.name)
                print(f"  captured {path.name}")

                # Inference demo has a nested upload tab worth capturing too.
                if stage == "Inference Demo" and _click_tab(page, "Upload NIfTI"):
                    page.wait_for_timeout(3500)
                    _settle(page)
                    path = SHOTS / f"{slug}__segmentation_upload.png"
                    _shoot(page, path)
                    captured.append(path.name)
                    print(f"  captured {path.name}")

            browser.close()

        print(f"\n[screenshots] {len(captured)} saved -> "
              f"{SHOTS.relative_to(REPO).as_posix()}/")
        for c in captured:
            size = (SHOTS / c).stat().st_size
            print(f"  {c:<48} {size / 1024:.0f} KB")
        return 0
    finally:
        if not args.keep_open:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
