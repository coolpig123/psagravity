"""Build NISAR model inputs NetCDF (.nc) for the fire spread model.

Generates the exact variables requested by the spread model:
  - nisar_cover_pct     (%)     : vegetation cover -> density
  - nisar_soil_moisture (m3/m3) : soil moisture -> dryness
  - nisar_burned        (0/1)   : already burned -> can't burn again

Usage:
    python scripts/build_nisar_inputs.py bug_fire --track A34
    python scripts/build_nisar_inputs.py bug_fire --track A34 --out data/fires/bug_fire/processed/bug_fire_nisar.nc
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow importing firedata and geelayers
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "real_fires" / "src"))
sys.path.insert(0, str(repo_root / "gee_layers" / "src"))

from firedata.config import FIRES
from firedata.nisar_layers import extract_nisar_dataset


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build NISAR NetCDF inputs for fire spread model.")
    ap.add_argument("fire", choices=list(FIRES), help="Fire name, e.g. bug_fire")
    ap.add_argument("--track", default="A34", help="NISAR track ID (default: A34)")
    ap.add_argument("--phase", default="pre", choices=["pre", "post"], help="Pass phase (default: pre)")
    ap.add_argument("--res", type=float, default=100.0, help="Grid resolution in metres (default: 100m)")
    ap.add_argument("--radius-km", type=float, default=20.0, help="Grid radius in km (default: 20km)")
    ap.add_argument("--out", default=None, help="Output NetCDF file path (.nc)")
    args = ap.parse_args()

    F = FIRES[args.fire]
    out_nc = args.out or (F.root / "processed" / f"{F.slug}_nisar_{args.track}.nc")

    print(f"Building NISAR model inputs for {F.name} (track {args.track}, phase {args.phase})...")
    ds = extract_nisar_dataset(
        fire=F,
        track=args.track,
        phase=args.phase,
        res_m=args.res,
        radius_km=args.radius_km,
        out_nc=out_nc,
    )

    print("\n✓ Generated NISAR xarray Dataset:")
    print(ds)
    print(f"\n✓ Saved NetCDF file -> {out_nc}")
    print("Pass to model as:  predict(..., nisar=ds)  or  --nisar", out_nc)


if __name__ == "__main__":
    main()
