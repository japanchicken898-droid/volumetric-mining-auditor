"""
Unit tests for Marching Squares, Uncertainty Propagation, Multimodal Filter,
SQLite Ledger, and DXF Export.
"""

import os
import tempfile
import numpy as np
import pytest
from shapely.geometry import Polygon

from src.marching_squares import extract_boundary_marching_squares, shoelace_area
from src.uncertainty import compute_volumetric_uncertainty
from src.multimodal_filter import apply_multimodal_filter, filter_elevation_diff
from src.ledger import AuditLedger, generate_verification_qr, compute_sha256
from src.dxf_export import export_contours_to_dxf, dxf_to_bytes


def test_shoelace_area_known_rectangle():
    # 10m x 20m rectangle -> Area = 200 m^2
    coords = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 20.0],
        [0.0, 20.0],
        [0.0, 0.0],
    ])
    area = shoelace_area(coords)
    assert abs(area - 200.0) < 1e-6


def test_marching_squares_circular_contour():
    # Circular depression: R = 20m, depth = 10m
    N = 81
    x = np.linspace(-30, 30, N)
    y = np.linspace(-30, 30, N)
    X, Y = np.meshgrid(x, y)
    Z = np.maximum(0.0, 10.0 * (1.0 - (X**2 + Y**2) / (20.0**2)))

    # Threshold at 5.0m -> Radius at cutoff = 20 * sqrt(1 - 5/10) = 20 * sqrt(0.5) = 14.1421 m
    # Exact circle area = pi * (14.1421)^2 = pi * 200 = 628.3185 m^2
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    res = extract_boundary_marching_squares(
        Z, threshold=5.0, dx=dx, dy=dy, x_origin=float(x[0]), y_origin=float(y[0])
    )
    assert len(res.contours) >= 1
    expected_area = np.pi * 200.0
    rel_err = abs(res.area - expected_area) / expected_area
    assert rel_err < 0.02  # within 2% on an 81x81 discrete grid


def test_volumetric_uncertainty():
    area = 10000.0  # 10,000 m^2
    bounds = compute_volumetric_uncertainty(area, rmse_z=2.0, alpha=0.85, dx=30.0, dy=30.0)
    lower, upper = bounds
    assert lower < 0.0
    assert upper > 0.0
    assert abs(lower + upper) < 1e-9  # symmetric +/- 1.96 * sigma_V
    assert bounds.margin == upper
    assert bounds.sigma_v > 0.0


def test_multimodal_filter_conditions():
    # Create 4 pixels testing each individual threshold
    # Condition: (delta_z >= 0.5) & (ndvi < 0.20) & (ndbi > 0.0) & (delta_sigma0 > 1.5)
    dz = np.array([[1.0, 0.2], [2.0, 3.0]])
    ndvi = np.array([[0.10, 0.10], [0.35, 0.05]])
    ndbi = np.array([[0.20, 0.20], [0.20, -0.10]])
    sar = np.array([[2.0, 2.0], [2.0, 0.5]])

    mask = apply_multimodal_filter(dz, ndvi, ndbi, sar, threshold_z=0.5)
    # Pixel [0, 0]: dz=1.0 (>=0.5), ndvi=0.1 (<0.2), ndbi=0.2 (>0), sar=2.0 (>1.5) -> True
    # Pixel [0, 1]: dz=0.2 (<0.5) -> False
    # Pixel [1, 0]: ndvi=0.35 (>=0.2) -> False
    # Pixel [1, 1]: ndbi=-0.1 (<=0) or sar=0.5 (<=1.5) -> False
    assert mask[0, 0] == True
    assert mask[0, 1] == False
    assert mask[1, 0] == False
    assert mask[1, 1] == False

    filtered_dz = filter_elevation_diff(dz, ndvi, ndbi, sar, threshold_z=0.5)
    assert filtered_dz[0, 0] == 1.0
    assert filtered_dz[0, 1] == 0.0


def test_audit_ledger_and_qr():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_ledger.db")
        ledger = AuditLedger(db_path=db_path)

        # Record 1
        rec1 = ledger.append_record(
            pre_dem_hash="a" * 64,
            post_dem_hash="b" * 64,
            lease_geojson={"type": "Polygon", "coordinates": []},
            metrics={"volume_m3": 125000.0, "status": "COMPLIANT"},
        )
        assert rec1["record_id"] == 1
        assert rec1["previous_hash"] == "0" * 64

        # Record 2 (chained)
        rec2 = ledger.append_record(
            pre_dem_hash="c" * 64,
            post_dem_hash="d" * 64,
            lease_geojson={"type": "Polygon", "coordinates": []},
            metrics={"volume_m3": 145000.0, "status": "BREACH"},
        )
        assert rec2["record_id"] == 2
        assert rec2["previous_hash"] == rec1["current_hash"]

        # Chain verification
        is_valid, errors = ledger.verify_chain()
        assert is_valid
        assert len(errors) == 0

        # QR code generation
        qr_path = os.path.join(tmpdir, "certificate_qr.png")
        img = generate_verification_qr(rec2["current_hash"], rec2["metrics"], output_path=qr_path)
        assert os.path.isfile(qr_path)
        assert img.size[0] > 100 and img.size[1] > 100


def test_dxf_export():
    contours = [np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])]
    lease = np.array([[-5, -5], [15, -5], [15, 15], [-5, 15], [-5, -5]])

    doc = export_contours_to_dxf(contours, lease_polygon_coords=lease)
    raw_bytes = dxf_to_bytes(doc)
    assert len(raw_bytes) > 0
    assert b"VMA_PIT_BOUNDARY" in raw_bytes
    assert b"VMA_LEASE_BOUNDARY" in raw_bytes
