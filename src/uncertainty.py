"""
Volumetric Mining Auditor (VMA) - Uncertainty Propagation Module
Computes spatially correlated volumetric uncertainty and 95% confidence intervals.
"""

from __future__ import annotations
import math
from typing import Tuple


class VolumetricUncertaintyBounds(tuple):
    """
    Subclass of tuple returning (-1.96 * sigma_V, +1.96 * sigma_V).
    Allows clean 2-element tuple unpacking:
        lower, upper = compute_volumetric_uncertainty(...)
    while also exposing rich diagnostic attributes:
        res.lower_bound
        res.upper_bound
        res.sigma_v
        res.margin
        res.vif
    """
    def __new__(
        cls,
        lower_bound: float,
        upper_bound: float,
        sigma_v: float,
        margin: float,
        vif: float,
    ):
        instance = super().__new__(cls, (lower_bound, upper_bound))
        instance.lower_bound = lower_bound
        instance.upper_bound = upper_bound
        instance.sigma_v = sigma_v
        instance.margin = margin
        instance.vif = vif
        return instance

    def __repr__(self) -> str:
        return (
            f"VolumetricUncertaintyBounds(lower={self.lower_bound:.2f}, "
            f"upper={self.upper_bound:.2f}, sigma_V={self.sigma_v:.2f}, "
            f"margin=±{self.margin:.2f} m³)"
        )


def compute_volumetric_uncertainty(
    area: float,
    rmse_z: float = 2.0,
    alpha: float = 0.85,
    dx: float = 30.0,
    dy: float = 30.0,
) -> VolumetricUncertaintyBounds:
    """
    Compute volumetric error standard deviation (sigma_V) and 95% confidence bounds
    (+/- 1.96 * sigma_V) considering spatial autocorrelation across DEM grid cells.
    
    In traditional independent-pixel error models, errors cancel out via sqrt(N),
    severely underestimating real volumetric risk. Using the spatial autoregressive
    range model with autocorrelation parameter alpha:
        VIF = (1 + alpha) / (1 - alpha)
        N_eff = N / VIF
        sigma_V = rmse_z * sqrt(area * dx * dy * VIF)
        95% Confidence Bounds = (-1.96 * sigma_V, +1.96 * sigma_V)
        
    Args:
        area: Excavation planar surface area in square meters (m^2).
        rmse_z: Vertical root-mean-square error of elevation model (default 2.0 meters).
        alpha: Spatial autocorrelation coefficient [0.0, 1.0) (default 0.85).
        dx: Cell size in x-direction (default 30.0 meters, e.g. Copernicus 30m / SRTM).
        dy: Cell size in y-direction (default 30.0 meters).
        
    Returns:
        VolumetricUncertaintyBounds: 2-tuple (-1.96 * sigma_V, +1.96 * sigma_V)
        with properties .lower_bound, .upper_bound, .sigma_v, .margin.
    """
    if area <= 0.0 or dx <= 0.0 or dy <= 0.0 or rmse_z <= 0.0:
        return VolumetricUncertaintyBounds(0.0, 0.0, 0.0, 0.0, 1.0)

    # Bound alpha strictly to [0.0, 0.999]
    alpha = max(0.0, min(0.999, float(alpha)))

    # Spatial variance inflation factor (VIF)
    vif = (1.0 + alpha) / (1.0 - alpha)

    # Pixel footprint area
    pixel_area = dx * dy

    # sigma_V = rmse_z * sqrt(Area * pixel_area * VIF)
    sigma_v = rmse_z * math.sqrt(area * pixel_area * vif)

    # 95% expanded confidence margin (k = 1.96 for normal distribution)
    margin = 1.96 * sigma_v
    lower_bound = -margin
    upper_bound = margin

    return VolumetricUncertaintyBounds(lower_bound, upper_bound, sigma_v, margin, vif)
