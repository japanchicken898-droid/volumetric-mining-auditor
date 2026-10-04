"""
Volumetric Mining Auditor (VMA) - Numerical Integration Module
Implements Piecewise 2D Composite Simpson's 1/3 Double Integration with outer-product weights.
"""

from __future__ import annotations
import numpy as np


def simpson_weights_1d(n: int, d: float) -> np.ndarray:
    """
    Compute 1D composite Simpson's integration weights for n points spaced by d.
    
    For odd n >= 3 (even number of intervals), standard Simpson's 1/3 weights:
        w = (d / 3) * [1, 4, 2, 4, ..., 2, 4, 1]
    For even n >= 4 (odd number of intervals), uses composite Simpson's 1/3 for the
    first n-3 intervals combined with Simpson's 3/8 rule on the final 3 intervals,
    maintaining O(h^4) precision and exactness for polynomials up to degree 3.
    For n == 2, falls back to the trapezoidal rule.
    For n == 1, returns [d].
    
    Args:
        n: Number of sampling points along the axis.
        d: Grid step size along the axis (dx or dy).
        
    Returns:
        1D numpy array of integration weights with shape (n,).
    """
    if n <= 0:
        raise ValueError(f"Grid size n must be positive, got {n}")
    if d <= 0:
        raise ValueError(f"Step size d must be positive, got {d}")

    if n == 1:
        return np.array([d], dtype=np.float64)
    if n == 2:
        return np.array([d / 2.0, d / 2.0], dtype=np.float64)

    w = np.zeros(n, dtype=np.float64)

    if n % 2 == 1:
        # Standard Simpson's 1/3 rule (even number of subintervals: n - 1)
        w[0] = 1.0
        w[-1] = 1.0
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        return w * (d / 3.0)
    else:
        # Even number of points (odd number of subintervals: n - 1)
        # Use Simpson's 1/3 for first (n - 4) intervals and Simpson's 3/8 for last 3 intervals
        if n >= 4:
            # First (n - 3) points form (n - 4) intervals (which is an even number)
            num_simp_pts = n - 3
            w_simp = np.zeros(num_simp_pts, dtype=np.float64)
            w_simp[0] = 1.0
            w_simp[-1] = 1.0
            w_simp[1:-1:2] = 4.0
            w_simp[2:-1:2] = 2.0
            w[:num_simp_pts] += w_simp * (d / 3.0)

            # Simpson's 3/8 rule on the last 4 points (last 3 intervals): indices [n-4, n-3, n-2, n-1]
            w_38 = np.array([1.0, 3.0, 3.0, 1.0], dtype=np.float64) * (3.0 * d / 8.0)
            w[n - 4:] += w_38
            return w
        else:
            # Fallback for n == 3 handled above, n < 3 handled above
            return np.array([d / 2.0, d / 2.0], dtype=np.float64)


def compute_volume_simpson_2d(
    delta_z: np.ndarray,
    dx: float,
    dy: float,
    boundary_mask: np.ndarray | None = None,
) -> float:
    """
    Compute total excavated volume using Piecewise 2D Composite Simpson's 1/3
    Double Integration via tensor outer-product weights:
        W = w_y (x) w_x
        Volume = sum_{j, i} (W_{j, i} * delta_z_{j, i} * boundary_mask_{j, i})
        
    Args:
        delta_z: 2D array of elevation difference (pre_dem - post_dem in meters).
                 Shape is (Ny, Nx), where Ny corresponds to rows (y) and Nx to columns (x).
        dx: Spatial resolution / cell width along the x-axis (meters).
        dy: Spatial resolution / cell height along the y-axis (meters).
        boundary_mask: Optional 2D boolean or float mask of shape (Ny, Nx).
                       If boolean, 1 inside boundary, 0 outside.
                       If float, allows sub-pixel boundary weighting fractions [0.0, 1.0].
                       
    Returns:
        float: Computed volume in cubic meters (m^3).
    """
    delta_z_arr = np.asarray(delta_z, dtype=np.float64)
    if delta_z_arr.ndim != 2:
        raise ValueError(f"delta_z must be a 2D array, got shape {delta_z_arr.shape}")

    Ny, Nx = delta_z_arr.shape
    if Ny < 2 or Nx < 2:
        raise ValueError(f"Grid dimensions must be at least 2x2, got {Ny}x{Nx}")
    if dx <= 0 or dy <= 0:
        raise ValueError(f"Grid steps dx and dy must be positive, got dx={dx}, dy={dy}")

    # Generate 1D quadrature weight vectors
    wy = simpson_weights_1d(Ny, dy)
    wx = simpson_weights_1d(Nx, dx)

    # 2D tensor outer-product weight matrix
    W = np.outer(wy, wx)

    if boundary_mask is not None:
        mask_arr = np.asarray(boundary_mask, dtype=np.float64)
        if mask_arr.shape != delta_z_arr.shape:
            raise ValueError(
                f"boundary_mask shape {mask_arr.shape} does not match delta_z shape {delta_z_arr.shape}"
            )
        effective_z = delta_z_arr * mask_arr
    else:
        effective_z = delta_z_arr

    # Double integration via element-wise product and summation
    volume = float(np.sum(W * effective_z))
    return volume
