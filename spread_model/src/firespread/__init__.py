"""firespread: fire-spread probability (Alexandridis et al. 2008 cellular automaton + moisture).

    from firespread import predict, summary
    result = predict(polygon=drawn_geojson, start="2026-08-09T21:00Z")   # fetches inputs via geelayers
    result = predict(polygon=drawn_geojson, start=..., nisar=get_nisar)  # + NISAR (Dataset, path or provider)
    print(summary(result))
    result["ignition_prob"]          # (y, x) chance each cell catches fire next step

See spread_model/README.md for the formulas and nisar.py for the NISAR contract.
"""
from .alexandridis import (DIRECTIONS, AlexandridisParams, p_den, p_dry, p_emc, p_mask, p_moisture, p_slope,
                           p_veg, p_vwc, p_wind, simard_emc, slope_angle_deg, spread_probabilities, summary)
from .nisar import NISAR_VARIABLES, NisarRequest, align_nisar, attach_nisar, nisar_template
from .predict import MODEL_LAYERS, predict

__all__ = ["predict", "spread_probabilities", "summary", "AlexandridisParams", "DIRECTIONS", "MODEL_LAYERS",
           "NISAR_VARIABLES", "NisarRequest", "nisar_template", "align_nisar", "attach_nisar",
           "p_veg", "p_den", "p_dry", "p_vwc", "p_emc", "p_moisture", "p_mask", "p_wind", "p_slope",
           "simard_emc", "slope_angle_deg"]
