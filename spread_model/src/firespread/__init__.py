"""firespread: fire-spread probability (Alexandridis et al. 2008 cellular automaton).

    from firespread import predict, summary
    result = predict(polygon=drawn_geojson, start="2026-08-09T21:00Z")   # fetches inputs via geelayers
    print(summary(result))
    result["ignition_prob"]          # (y, x) chance each cell catches fire next step

See spread_model/README.md for the formulas.
"""
from .alexandridis import (DIRECTIONS, NISAR_VARIABLES, AlexandridisParams, p_den, p_dry, p_mask, p_slope,
                           p_veg, p_wind, slope_angle_deg, spread_probabilities, summary)
from .predict import MODEL_LAYERS, align_nisar, predict

__all__ = ["predict", "align_nisar", "spread_probabilities", "summary", "AlexandridisParams", "DIRECTIONS",
           "NISAR_VARIABLES", "MODEL_LAYERS", "p_veg", "p_den", "p_dry", "p_mask", "p_wind", "p_slope",
           "slope_angle_deg"]
