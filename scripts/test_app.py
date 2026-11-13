"""Browser test of the fire site (app/index.html) or a single-fire page, in headless Chromium.

    pip install -r requirements-dev.txt && playwright install chromium-headless-shell
    python scripts/test_app.py app/index.html            # every fire in the selector
    python scripts/test_app.py data/fires/bug_fire/figures/bug_fire_spread.html

Checks, per fire: no JS errors, the map draws fire pixels, Play advances time, clicking the
timeline jumps, no horizontal overflow at desktop and phone width. Screenshots go to --shots.
Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HOT = """() => { const c = document.getElementById('map'), x = c.getContext('2d');
  const d = x.getImageData(0, 0, c.width, c.height).data; let n = 0;
  for (let i = 0; i < d.length; i += 4) if (d[i] > 200 && d[i+1] > 80 && d[i+2] < 120) n++; return n; }"""


def check_page(pg, label: str, shots: Path, fails: list[str]) -> None:
    hot = pg.evaluate(HOT)
    if hot < 50:
        fails.append(f"{label}: map shows no fire pixels ({hot})")
    c0 = pg.inner_text("#clock")
    pg.click("#play"); pg.wait_for_timeout(1200); c1 = pg.inner_text("#clock"); pg.click("#play")
    if c1 == c0:
        fails.append(f"{label}: Play did not advance time ({c0})")
    pg.locator("#tl").scroll_into_view_if_needed()
    box = pg.locator("#tl").bounding_box()
    pg.mouse.click(box["x"] + box["width"] * 0.15, box["y"] + box["height"] * 0.5); pg.wait_for_timeout(200)
    c2 = pg.inner_text("#clock")
    if c2 == c1:
        fails.append(f"{label}: timeline click did not change time ({c1})")
    if pg.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
        fails.append(f"{label}: horizontal overflow")
    print(f"  {label}: hot={hot} clock {c0!r} -> play {c1!r} -> click {c2!r}")
    pg.screenshot(path=str(shots / f"{label}.png"), full_page=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page", type=Path)
    ap.add_argument("--shots", type=Path, default=Path("/tmp/fire_app_shots"))
    a = ap.parse_args()
    a.shots.mkdir(parents=True, exist_ok=True)
    fails: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        for vp_name, vp in [("desktop", {"width": 1366, "height": 900}), ("phone", {"width": 390, "height": 844})]:
            pg = b.new_page(viewport=vp)
            errs: list[str] = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.on("console", lambda m: m.type == "error" and errs.append(m.text))
            pg.goto(a.page.resolve().as_uri()); pg.wait_for_timeout(1500)
            slugs = pg.evaluate("window.FIRE_INDEX ? window.FIRE_INDEX.map(f => f.slug) : []")
            if not slugs:  # single-fire page
                check_page(pg, f"{vp_name}", a.shots, fails)
            for slug in slugs:
                pg.select_option("#fire-pick", slug); pg.wait_for_timeout(1500)
                name = pg.inner_text("#f-name")
                check_page(pg, f"{vp_name}-{slug}", a.shots, fails)
                print(f"    title: {name}")
            fails += [f"{vp_name}: JS error: {e}" for e in errs]
            pg.close()
        b.close()
    print("FAIL:\n  " + "\n  ".join(fails) if fails else "PASS: all checks")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
