"""Browser test of the static site (the GitHub Pages build), in headless Chromium.

    cd frontend
    $env:BASE_PATH = "/psagravity/"; $env:VITE_STATIC_DATA = "haifa"; npm run build; npx vite preview --port 4173
    real_fires\\.venv\\Scripts\\python israel_static\\tests\\test_site.py http://localhost:4173/psagravity/

(needs playwright: pip install -r real_fires/requirements-dev.txt, playwright install chromium-headless-shell)

Checks, with no server and no login: the map opens on the downloaded area, the wind loads, a polygon can be
drawn, Predict gives cells and arrival times, a cell can be clicked, the views switch, Extract Data shows the
inputs table, and a Start outside the downloaded period gives a clear message. Only requests to the site itself,
the map tiles, the fonts and the Pyodide CDN are allowed. Exit code 1 if a check fails.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ALLOWED_HOSTS = ("tile.openstreetmap.org", "fonts.googleapis.com", "fonts.gstatic.com", "cdn.jsdelivr.net")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--shots", default=None, help="folder for screenshots")
    ap.add_argument("--timeout", type=float, default=240.0, help="seconds to wait for Python + the prediction")
    ap.add_argument("--channel", default=None, help="use an installed browser instead: msedge or chrome")
    args = ap.parse_args(argv)
    shots = Path(args.shots) if args.shots else None
    fails: list[str] = []
    errors: list[str] = []
    foreign: set[str] = set()
    site_host = args.url.split("/")[2]

    def shot(pg, name):
        if shots:
            shots.mkdir(parents=True, exist_ok=True)
            pg.screenshot(path=str(shots / f"{name}.png"), full_page=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel)
        pg = browser.new_page(viewport={"width": 1400, "height": 1000})
        pg.on("pageerror", lambda e: errors.append(f"page error: {e}"))
        pg.on("console", lambda m: errors.append(f"console: {m.text}") if m.type == "error" else None)
        pg.on("request", lambda r: foreign.add(r.url.split("/")[2]) if r.url.startswith("http")
              and r.url.split("/")[2] != site_host and not r.url.split("/")[2].endswith(ALLOWED_HOSTS) else None)
        pg.goto(args.url)
        pg.wait_for_selector(".maplibregl-canvas")

        # the hint names the downloaded area once index.json is read
        pg.wait_for_function("document.querySelector('.hint')?.textContent.includes('downloaded for')", timeout=20_000)
        print("hint:", pg.inner_text(".hint"))
        print("start:", pg.input_value("label.start input >> nth=0"))

        # wind: loaded from the monthly file
        try:
            pg.wait_for_function("document.querySelector('.regions.wind')?.textContent.includes('MERRA-2, ')",
                                 timeout=20_000)
        except Exception:
            fails.append(f"wind did not load: {pg.inner_text('.regions.wind')!r}")
        print("wind:", pg.inner_text(".regions.wind"))

        # draw a polygon around the middle of the map (the middle of the downloaded area)
        box = pg.locator(".map").bounding_box()
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        pg.click("button:has-text('Draw')")
        for dx, dy in ((-25, -20), (25, -20), (25, 20), (-25, 20)):
            pg.mouse.click(cx + dx, cy + dy)
            pg.wait_for_timeout(150)
        pg.click("button:has-text('Finish')")
        shot(pg, "1_polygon")

        # Extract Data: the table of inputs under the map
        t0 = time.time()
        pg.click("button:has-text('Extract Data')")
        try:
            pg.wait_for_selector("pre.extracted", timeout=args.timeout * 1000)
            table = pg.inner_text("pre.extracted")
            print(f"extract ({time.time() - t0:.0f} s, includes loading Python):\n{table}")
            for word in ("elevation_m", "wind_speed_ms", "landcover"):
                if word not in table:
                    fails.append(f"Extract Data table has no {word}")
        except Exception:
            fails.append(f"Extract Data gave no table: {pg.inner_text('.panel')!r}")

        # Predict
        t0 = time.time()
        pg.click("button:has-text('Predict')")
        try:
            pg.wait_for_function("document.querySelector('.panel')?.textContent.includes('Most likely spread')"
                                 " || document.querySelector('.panel .error')", timeout=args.timeout * 1000)
        except Exception:
            fails.append("Predict gave no answer in time")
        panel = pg.inner_text(".panel")
        print(f"predict ({time.time() - t0:.0f} s):\n{panel}")
        if "Most likely spread" not in panel:
            fails.append("Predict failed")
        else:
            for word in ("Arrival time (survival model)", "NISAR:", "Weather:"):
                if word not in panel:
                    fails.append(f"prediction panel has no {word!r}")
            pg.wait_for_timeout(2500)          # the map zooms to the result
            shot(pg, "2_prediction")
            # every view of the cells
            for value in ("pe", "t50", "p6", "p12", "p24", "p48", "p"):
                pg.select_option(".view select", value)
                pg.wait_for_timeout(300)
            pg.select_option(".view select", "t50")
            pg.wait_for_timeout(800)
            shot(pg, "3_arrival")
            # click cells around the polygon until one answers (some cannot burn and are not drawn): its numbers
            box = pg.locator(".map").bounding_box()
            for dx, dy in ((150, 0), (-120, 30), (0, 130), (0, -130), (-150, -60)):
                pg.mouse.click(box["x"] + box["width"] / 2 + dx, box["y"] + box["height"] / 2 + dy)
                pg.wait_for_timeout(700)
                if pg.locator(".maplibregl-popup").count():
                    print("popup:", pg.inner_text(".maplibregl-popup >> nth=0").replace(chr(10), " | "))
                    break
            else:
                fails.append("clicking prediction cells opened no popup")

        # a start time with no downloaded data: a clear message, not a crash
        pg.fill("label.start input >> nth=0", "2020-01-01T12:00Z")
        pg.click("button:has-text('Predict')")
        try:
            pg.wait_for_selector(".panel .error", timeout=60_000)
            message = pg.inner_text(".panel .error")
            print("outside the period:", message)
            if "downloaded" not in message:
                fails.append(f"unclear message for a start outside the period: {message!r}")
        except Exception:
            fails.append("no error for a start outside the downloaded period")
        pg.wait_for_timeout(1500)
        print("wind outside the period:", pg.inner_text(".regions.wind"))
        shot(pg, "4_outside_period")
        browser.close()

    errors = [e for e in errors if "tile.openstreetmap.org" not in e]
    if errors:
        fails.append(f"{len(errors)} JS errors: " + " || ".join(errors[:5]))
    if foreign:
        fails.append(f"requests to other hosts (a server or an API?): {sorted(foreign)}")
    for f in fails:
        print("FAIL:", f)
    print("OK" if not fails else f"{len(fails)} checks failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
