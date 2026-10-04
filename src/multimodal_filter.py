"""
Volumetric Mining Auditor (VMA) - Multimodal Sentinel-1/Sentinel-2 Filter
Suppresses false alarms from seasonal agriculture and canopy clearing by fusing
optical spectral indices with C-band SAR backscatter shifts.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np


def apply_multimodal_filter(
    delta_z: np.ndarray,
    ndvi: np.ndarray,
    ndbi: np.ndarray,
    delta_sigma0: np.ndarray,
    threshold_z: float = 0.5,
) -> np.ndarray:
    """
    Applies multimodal optical & SAR fusion filter to isolate true ground excavation
    from vegetation canopy stripping or agricultural cycles:
    
        Valid Excavation Mask =
            (delta_z >= threshold_z) &
            (ndvi < 0.20) &
            (ndbi > 0.0) &
            (delta_sigma0 > 1.5)
            
    Physical rationale:
      - delta_z >= 0.5 m: Vertical elevation loss exceeding baseline DEM noise floor.
      - ndvi < 0.20: Bare rock / exposed mineral soil; rules out photosynthetic canopy.
      - ndbi > 0.0: Positive bare soil / mineral index (SWIR > NIR reflection).
      - delta_sigma0 > 1.5 dB: Significant surface roughness increase from blasted / broken
        rock textures in Sentinel-1 co-polarized (VV/VH) radar backscatter.
        
    Args:
        delta_z: 2D array of vertical elevation difference (pre_dem - post_dem in meters).
        ndvi: 2D array of Sentinel-2 Normalized Difference Vegetation Index [-1.0, 1.0].
        ndbi: 2D array of Sentinel-2 Normalized Difference Built-up/Bare Index [-1.0, 1.0].
        delta_sigma0: 2D array of Sentinel-1 radar backscatter shift in decibels (dB).
        threshold_z: Minimum vertical excavation threshold (default 0.5 m).
        
    Returns:
        np.ndarray: 2D boolean array of the same shape indicating confirmed mining excavation.
    """
    dz = np.asarray(delta_z, dtype=np.float64)
    vi = np.asarray(ndvi, dtype=np.float64)
    bi = np.asarray(ndbi, dtype=np.float64)
    s0 = np.asarray(delta_sigma0, dtype=np.float64)

    if not (dz.shape == vi.shape == bi.shape == s0.shape):
        raise ValueError(
            f"Array shapes must match: delta_z={dz.shape}, ndvi={vi.shape}, "
            f"ndbi={bi.shape}, delta_sigma0={s0.shape}"
        )

    # Core logical condition
    valid_mask = (
        (dz >= threshold_z)
        & (vi < 0.20)
        & (bi > 0.0)
        & (s0 > 1.5)
    )

    # Clean any NaN inputs
    nan_mask = np.isnan(dz) | np.isnan(vi) | np.isnan(bi) | np.isnan(s0)
    valid_mask = valid_mask & (~nan_mask)

    return valid_mask


def filter_elevation_diff(
    delta_z: np.ndarray,
    ndvi: np.ndarray,
    ndbi: np.ndarray,
    delta_sigma0: np.ndarray,
    threshold_z: float = 0.5,
) -> np.ndarray:
    """
    Apply multimodal filter and return masked elevation difference array,
    setting non-compliant / false-positive pixels to 0.0.
    """
    mask = apply_multimodal_filter(delta_z, ndvi, ndbi, delta_sigma0, threshold_z=threshold_z)
    return np.where(mask, delta_z, 0.0)


def multimodal_filter_diagnostics(
    delta_z: np.ndarray,
    ndvi: np.ndarray,
    ndbi: np.ndarray,
    delta_sigma0: np.ndarray,
    threshold_z: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute pixel breakdown statistics across each multimodal filtering criterion.
    """
    dz = np.asarray(delta_z, dtype=np.float64)
    vi = np.asarray(ndvi, dtype=np.float64)
    bi = np.asarray(ndbi, dtype=np.float64)
    s0 = np.asarray(delta_sigma0, dtype=np.float64)

    total_pixels = dz.size
    cond_dz = dz >= threshold_z
    cond_ndvi = vi < 0.20
    cond_ndbi = bi > 0.0
    cond_sar = s0 > 1.5

    confirmed = cond_dz & cond_ndvi & cond_ndbi & cond_sar
    canopy_false_positives = cond_dz & (~cond_ndvi)
    smooth_soil_false_positives = cond_dz & cond_ndvi & (~cond_sar)

    return {
        "total_pixels": int(total_pixels),
        "elevation_loss_pixels": int(np.sum(cond_dz)),
        "confirmed_mining_pixels": int(np.sum(confirmed)),
        "canopy_rejection_pixels": int(np.sum(canopy_false_positives)),
        "sar_rejection_pixels": int(np.sum(smooth_soil_false_positives)),
        "confirmation_rate_pct": float(np.sum(confirmed) / max(1, np.sum(cond_dz)) * 100.0),
    }
