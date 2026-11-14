"""Build NISAR vegetation / fuel density layers for one or more fires.

For each fire and each pre-fire NISAR track, downloads the L2 GCOV granule,
computes the Radar Vegetation Index (RVI), and saves a GeoTIFF + heatmap.

Usage (run from real_fires/):
    python scripts/build_vegetation_layer.py bug_fire
    python scripts/build_vegetation_layer.py bug_fire --track A34
    python scripts/build_vegetation_layer.py bug_fire colwash biscar --phase pre
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running as a plain script from the real_fires/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firedata.config import FIRES
from firedata.vegetation import build_fuel_layer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    ap = argparse.ArgumentParser(
        description="Build NISAR vegetation / fuel density layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("fires", nargs="+", choices=list(FIRES),
                    help="One or more fire slugs")
    ap.add_argument("--track",
                    help="Process only this track ID (default: all tracks for the phase)")
    ap.add_argument("--phase", default="pre", choices=["pre", "during", "post"],
                    help="NISAR pass phase to process (default: pre)")
    args = ap.parse_args()

    errors = []

    for slug in args.fires:
        fire = FIRES[slug]
        passes_csv = fire.root / "processed" / "nisar_passes.csv"

        if not passes_csv.exists():
            print(
                f"  ✗  {slug}: nisar_passes.csv not found — "
                "run build_ground_truth.py first.",
                file=sys.stderr,
            )
            errors.append(slug)
            continue

        passes = pd.read_csv(passes_csv)
        pre = passes[passes["phase"] == args.phase]
        if args.track:
            pre = pre[pre["track_id"] == args.track]

        tracks = pre["track_id"].unique().tolist()
        if not tracks:
            print(
                f"  ✗  {slug}: no {args.phase!r} passes"
                + (f" for track {args.track!r}" if args.track else "") + ".",
                file=sys.stderr,
            )
            errors.append(slug)
            continue

        for track in tracks:
            try:
                tif = build_fuel_layer(fire, track, args.phase)
                print(f"  ✓  {slug}  {track}  →  {tif}")
            except Exception as exc:
                print(f"  ✗  {slug}  {track}  →  {exc}", file=sys.stderr)
                errors.append(f"{slug}/{track}")

    if errors:
        print(f"\n{len(errors)} error(s): {errors}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
