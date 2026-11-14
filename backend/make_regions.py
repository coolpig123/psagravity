"""Build frontend/public/regions.geojson: where get_fire_inputs has full data, and where not.

Available = cells covered by both US-only layers that get_fire_inputs needs:
gridMET (fuel dryness) and LANDFIRE EVC (fuel). The global layers (terrain, land cover,
Sentinel-2) and RTMA weather cover more than that, so these two set the limit.

Run once (needs the Earth Engine login, see frontend/README.md):
    backend\\.venv\\Scripts\\python backend\\make_regions.py
"""

import json
from pathlib import Path

import ee
from geelayers.auth import init

OUT = Path(__file__).resolve().parents[1] / "frontend" / "public" / "regions.geojson"
SCALE_M = 10_000      # vectorize on a 10 km grid: plenty for a world map
SIMPLIFY_M = 5_000    # then simplify the outline by up to 5 km
WORLD = [[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]]  # web-map world, counter-clockwise


def signed_area(ring: list) -> float:
    """> 0 for counter-clockwise rings (lon = x, lat = y)."""
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:])) / 2


def oriented(ring: list, ccw: bool) -> list:
    return ring if (signed_area(ring) > 0) == ccw else ring[::-1]


def polygons(geom: dict) -> list:
    return [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]


def split(available: dict) -> tuple[dict, dict]:
    """Available and unavailable (= world minus available) as MultiPolygons.

    Outer rings counter-clockwise and holes clockwise, as GeoJSON and MapLibre expect.
    Holes inside the available area (e.g. lakes) become unavailable islands.
    """
    avail, unavail_islands, cutouts = [], [], []
    for poly in polygons(available):
        outer, holes = poly[0], poly[1:]
        avail.append([oriented(outer, True)] + [oriented(h, False) for h in holes])
        cutouts.append(oriented(outer, False))
        unavail_islands += [[oriented(h, True)] for h in holes]
    unavailable = [[WORLD] + cutouts] + unavail_islands
    return ({"type": "MultiPolygon", "coordinates": avail},
            {"type": "MultiPolygon", "coordinates": unavailable})


def main() -> None:
    init()
    conus_box = ee.Geometry.Rectangle([-126, 24, -66, 50], None, False)
    gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").first().select("fm100").mask().gt(0)
    landfire = (ee.ImageCollection("LANDFIRE/Vegetation/EVC/v1_4_0")
                .filter(ee.Filter.eq("system:index", "CONUS")).first().mask().gt(0))
    covered = gridmet.And(landfire).selfMask().toInt()

    vectors = covered.reduceToVectors(geometry=conus_box, scale=SCALE_M, geometryType="polygon",
                                      eightConnected=True, maxPixels=1e9, bestEffort=True)
    available, unavailable = split(vectors.geometry(maxError=SIMPLIFY_M).simplify(SIMPLIFY_M).getInfo())

    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"available": True}, "geometry": available},
        {"type": "Feature", "properties": {"available": False}, "geometry": unavailable},
    ]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fc, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
