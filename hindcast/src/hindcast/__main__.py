"""Hindcast command line (run from the repo root or hindcast/).

  python -m hindcast windows bug_fire                       # offline: which windows / targets are scored
  python -m hindcast calibrate bug_fire                     # Earth Engine (once, then cached): fit N + threshold
  python -m hindcast evaluate bug_fire --skip-first         # test windows with the calibrated parameters
  python -m hindcast evaluate biscar
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import run
from .windows import EXCLUDED_FIRES, build_windows, describe

FEDS_NOTE = ("Scores are vs FEDS perimeters, which under-map the official WFIGS area "
             "(Bug FEDS = 37.8 % of official, Biscar 11.2 %).")
CALIBRATION = run.RESULTS_DIR / "calibration.json"


def _check(fire: str) -> None:
    if fire in EXCLUDED_FIRES:
        raise SystemExit(f"{fire} is left out of the scores: {EXCLUDED_FIRES[fire]}")


def cmd_windows(a) -> None:
    table = describe(build_windows(a.fire))
    print(table.to_string(index=False))
    if a.fire in EXCLUDED_FIRES:
        print(f"\nNOTE: {a.fire} is left out of the scores: {EXCLUDED_FIRES[a.fire]}")


def cmd_calibrate(a) -> None:
    _check(a.fire)
    window = build_windows(a.fire)[a.window]
    table = run.evaluate_window(window, run.spread_result(run.fetch_inputs(window), window))
    run.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(run.RESULTS_DIR / f"{a.fire}_calibration.csv", index=False)
    best = run.best_params(table)
    CALIBRATION.write_text(json.dumps({"fire": a.fire, "window": window.id, "hours_per_step": best.hours_per_step,
                                       "threshold": best.threshold, "mean_iou_new": round(best.iou, 4),
                                       "note": FEDS_NOTE}, indent=2))
    print(f"Calibrated on {window.id}: {best.hours_per_step} h/step, threshold {best.threshold}, "
          f"mean IoU (new area) {best.iou:.3f}")
    at = table[table.hours_per_step == best.hours_per_step]
    print("\nThreshold sweep at that h/step (mean over targets):")
    print(at.groupby("threshold")[["iou", "precision", "recall", "base_iou", "pred_new_km2"]].mean().round(3))
    print(f"\n{FEDS_NOTE}")


def cmd_evaluate(a) -> None:
    _check(a.fire)
    cal = json.loads(CALIBRATION.read_text())
    hps = a.hours_per_step or cal["hours_per_step"]
    thresholds = sorted({a.threshold or cal["threshold"], 0.3, 0.5, 0.7})   # report a few thresholds too
    windows = build_windows(a.fire)
    if a.skip_first:
        windows = windows[1:]
    if a.windows:
        windows = [w for w in windows if w.id in a.windows]
    tables = [run.evaluate_window(w, run.spread_result(run.fetch_inputs(w), w), hours_per_step=(hps,),
                                  thresholds=thresholds) for w in windows]
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(run.RESULTS_DIR / f"{a.fire}_scores.csv", index=False)
    main = table[table.threshold == (a.threshold or cal["threshold"])]
    cols = ["window", "horizon_h", "hours", "iou", "precision", "recall", "base_iou", "oracle_iou",
            "pred_new_km2", "obs_new_km2", "reached_grid_edge"]
    print(f"{a.fire}: {hps} h/step, threshold {a.threshold or cal['threshold']}")
    print(main[cols].round(3).to_string(index=False))
    beat = (main.iou > main.base_iou).mean()
    print(f"\nModel beats the same-area circle in {beat:.0%} of {len(main)} targets "
          f"(mean IoU {main.iou.mean():.3f} vs {main.base_iou.mean():.3f}).\n{FEDS_NOTE}")


def main() -> None:
    p = argparse.ArgumentParser(prog="hindcast")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("windows"); s.add_argument("fire"); s.set_defaults(func=cmd_windows)
    s = sub.add_parser("calibrate"); s.add_argument("fire"); s.add_argument("--window", type=int, default=0)
    s.set_defaults(func=cmd_calibrate)
    s = sub.add_parser("evaluate"); s.add_argument("fire")
    s.add_argument("--hours-per-step", type=float); s.add_argument("--threshold", type=float)
    s.add_argument("--skip-first", action="store_true", help="leave out the calibration window")
    s.add_argument("--windows", nargs="*", help="window ids, e.g. crosswhite_0720_0930")
    s.set_defaults(func=cmd_evaluate)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
