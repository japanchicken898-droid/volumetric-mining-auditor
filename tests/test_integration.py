"""
Unit tests for Volumetric Mining Auditor (VMA) numerical integration.
Asserts that compute_volume_simpson_2d achieves < 0.010% relative error
across analytical benchmark pit test cases E1, E2, E3, and E4.
"""

import pytest
import numpy as np

from src.integration import compute_volume_simpson_2d, simpson_weights_1d
from data.synthetic.generate_synthetic_pits import (
    generate_pit_e1,
    generate_pit_e2,
    generate_pit_e3,
    generate_pit_e4,
)

ERROR_TOLERANCE_PCT = 0.010  # 0.010%


def test_e1_circular_paraboloid():
    """
    Test Case E1: Circular Paraboloid
    Asserts relative error < 0.010% against exact analytical volume (1/2)*pi*R^2*H.
    """
    pit = generate_pit_e1()
    v_simpson = compute_volume_simpson_2d(pit["delta_z"], pit["dx"], pit["dy"])
    v_analytical = pit["analytical_volume"]

    rel_error_pct = abs(v_simpson - v_analytical) / v_analytical * 100.0
    assert rel_error_pct < ERROR_TOLERANCE_PCT, (
        f"E1 relative error {rel_error_pct:.6f}% exceeded {ERROR_TOLERANCE_PCT}%"
    )


def test_e2_elliptical_paraboloid():
    """
    Test Case E2: Elliptical Paraboloid
    Asserts relative error < 0.010% against exact analytical volume (1/2)*pi*a*b*H.
    """
    pit = generate_pit_e2()
    v_simpson = compute_volume_simpson_2d(pit["delta_z"], pit["dx"], pit["dy"])
    v_analytical = pit["analytical_volume"]

    rel_error_pct = abs(v_simpson - v_analytical) / v_analytical * 100.0
    assert rel_error_pct < ERROR_TOLERANCE_PCT, (
        f"E2 relative error {rel_error_pct:.6f}% exceeded {ERROR_TOLERANCE_PCT}%"
    )


def test_e3_deep_paraboloid():
    """
    Test Case E3: Deep Paraboloid (High Aspect Ratio H/R = 2.0)
    Asserts relative error < 0.010% against exact analytical volume (1/2)*pi*R^2*H.
    """
    pit = generate_pit_e3()
    v_simpson = compute_volume_simpson_2d(pit["delta_z"], pit["dx"], pit["dy"])
    v_analytical = pit["analytical_volume"]

    rel_error_pct = abs(v_simpson - v_analytical) / v_analytical * 100.0
    assert rel_error_pct < ERROR_TOLERANCE_PCT, (
        f"E3 relative error {rel_error_pct:.6f}% exceeded {ERROR_TOLERANCE_PCT}%"
    )


def test_e4_asymmetric_polynomial():
    """
    Test Case E4: Asymmetric Polynomial Pit
    Asserts relative error < 0.010% against exact analytical integration.
    """
    pit = generate_pit_e4()
    v_simpson = compute_volume_simpson_2d(pit["delta_z"], pit["dx"], pit["dy"])
    v_analytical = pit["analytical_volume"]

    rel_error_pct = abs(v_simpson - v_analytical) / v_analytical * 100.0
    assert rel_error_pct < ERROR_TOLERANCE_PCT, (
        f"E4 relative error {rel_error_pct:.8f}% exceeded {ERROR_TOLERANCE_PCT}%"
    )


def test_boundary_mask():
    """
    Test that boundary_mask properly limits integration volume.
    """
    pit = generate_pit_e1(N=51)
    dz = pit["delta_z"]
    dx, dy = pit["dx"], pit["dy"]

    # Full volume
    v_full = compute_volume_simpson_2d(dz, dx, dy)

    # Half mask (x >= 0, with 0.5 weight along the splitting boundary x == 0)
    X, _ = np.meshgrid(pit["x"], pit["y"])
    mask_half = np.where(X > 0, 1.0, np.where(X == 0, 0.5, 0.0))
    v_half = compute_volume_simpson_2d(dz, dx, dy, boundary_mask=mask_half)

    # For a circular paraboloid centered at origin, right half should be exactly 50%
    ratio = v_half / v_full
    assert 0.499 <= ratio <= 0.501


def test_even_grid_dimensions():
    """
    Test that even number of points (odd number of intervals) integrates polynomials correctly.
    """
    # 2D plane: z(x, y) = 10.0 on [0, 4] x [0, 6] -> Volume = 240.0
    Ny, Nx = 50, 40  # both even
    x = np.linspace(0, 4, Nx)
    y = np.linspace(0, 6, Ny)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    Z = np.full((Ny, Nx), 10.0)

    v = compute_volume_simpson_2d(Z, dx, dy)
    assert abs(v - 240.0) < 1e-9


def test_invalid_inputs():
    """
    Test that invalid inputs to compute_volume_simpson_2d raise appropriate errors.
    """
    with pytest.raises(ValueError):
        compute_volume_simpson_2d(np.ones(10), 1.0, 1.0)  # 1D array

    with pytest.raises(ValueError):
        compute_volume_simpson_2d(np.ones((5, 5)), -1.0, 1.0)  # negative dx

    with pytest.raises(ValueError):
        compute_volume_simpson_2d(np.ones((5, 5)), 1.0, 1.0, boundary_mask=np.ones((6, 6)))
