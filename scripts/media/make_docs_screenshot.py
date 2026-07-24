"""Render `docs/media/docs-site.png`, the study-site screenshot shown in the README.

Serves the already-built `site/` on a loopback port, captures the landing page in a headless
Chromium at 2x device pixel ratio, then crops the empty right gutter and downscales to 1x. Needs
`uv run python -m mkdocs build` to have run first, and a local Chrome or Edge.

    uv run python scripts/media/make_docs_screenshot.py
"""

from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import socketserver
import subprocess
import tempfile
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_DIR = REPO_ROOT / "site"
DEFAULT_OUT = REPO_ROOT / "docs" / "media" / "docs-site.png"

VIEWPORT = (1440, 1100)
SCALE = 2
# Cropped in captured (2x) pixels: the theme's content column ends at 2599, so 2640 leaves a thin
# margin, and 2100 cuts just below the headline figure rather than mid-sentence.
CROP = (2640, 2100)

BROWSERS = (
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
)


def find_browser() -> str:
    for candidate in BROWSERS:
        if Path(candidate).exists():
            return candidate
    for name in ("google-chrome", "chromium", "chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("no Chrome or Edge found; install one or capture the screenshot by hand")


def serve(directory: Path) -> tuple[socketserver.TCPServer, int]:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    from PIL import Image

    if not (SITE_DIR / "index.html").exists():
        raise SystemExit(f"{SITE_DIR} is not built; run: uv run python -m mkdocs build")

    browser = find_browser()
    httpd, port = serve(SITE_DIR)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.png"
            subprocess.run(
                [
                    browser,
                    "--headless=new",
                    "--disable-gpu",
                    "--hide-scrollbars",
                    f"--force-device-scale-factor={SCALE}",
                    "--virtual-time-budget=6000",
                    f"--window-size={VIEWPORT[0]},{VIEWPORT[1]}",
                    f"--screenshot={raw}",
                    f"--user-data-dir={Path(tmp) / 'profile'}",
                    f"http://127.0.0.1:{port}/",
                ],
                check=True,
                capture_output=True,
            )
            shot = Image.open(raw).convert("RGB")
            shot = shot.crop((0, 0, *CROP)).resize(
                (CROP[0] // SCALE, CROP[1] // SCALE), Image.LANCZOS
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            shot.save(args.out, optimize=True)
    finally:
        httpd.shutdown()

    kib = args.out.stat().st_size / 1024
    print(f"wrote {args.out}")
    print(f"  {shot.width}x{shot.height}, {kib:.0f} KiB")


if __name__ == "__main__":
    main()
