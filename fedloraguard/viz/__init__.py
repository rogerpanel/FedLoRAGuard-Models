"""Figure-regeneration utilities.

Each function in this package consumes the JSON / NumPy artifacts produced
by ``scripts/evaluate.py`` (or the LoRAchain-2026 builder) and emits a
matplotlib figure that reproduces one of the panels of the manuscript.

  * :func:`spectral_signature_plot`        -- Figure 3 (rank-1 trigger
    concentration in BA singular values).
  * :func:`privacy_utility_pareto_plot`    -- Figure 4 (privacy-utility
    Pareto frontier across the eps_r sweep).
  * :func:`achievement_radar_plot`         -- Figure 5 (multi-dimensional
    achievement profile).
  * :func:`reliability_diagram`            -- supplementary calibration
    diagram (15 equal-mass bins).

All figures are saved as both .pdf (publication) and .png (preview).
"""
from .figures import (
    spectral_signature_plot,
    privacy_utility_pareto_plot,
    achievement_radar_plot,
    reliability_diagram,
)

__all__ = [
    "spectral_signature_plot",
    "privacy_utility_pareto_plot",
    "achievement_radar_plot",
    "reliability_diagram",
]
