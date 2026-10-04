"""
Volumetric Mining Auditor (VMA) - Marching Squares Module
Extracts sub-pixel contours for excavation boundary demarcation and calculates
planar area using the Shoelace formula.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Any
import numpy as np
from shapely.geometry import Polygon, MultiPolygon


def shoelace_area(coords: np.ndarray) -> float:
    """
    Compute planar area of a closed 2D polygon using the Shoelace formula
    (Gauss's area formula).
    
    Args:
        coords: (N, 2) array of (x, y) vertex coordinates.
        
    Returns:
        float: Non-negative planar area.
    """
    coords_arr = np.asarray(coords, dtype=np.float64)
    if len(coords_arr) < 3:
        return 0.0
    x = coords_arr[:, 0]
    y = coords_arr[:, 1]
    # Sum over cross products
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return float(area)


class MarchingSquaresResult(tuple):
    """
    Container for marching squares boundary extraction results.
    Inherits from tuple for backward-compatible unpacking:
        contours, area = extract_boundary_marching_squares(...)
    Also exposes attributes:
        result.contours
        result.area
        result.polygon
    """
    def __new__(
        cls,
        contours: List[np.ndarray],
        area: float,
        polygon: Optional[Any] = None,
    ):
        instance = super().__new__(cls, (contours, area))
        instance.contours = contours
        instance.area = area
        instance.polygon = polygon
        return instance

    def __getitem__(self, item):
        if isinstance(item, str):
            if item == "contours":
                return self.contours
            elif item == "area":
                return self.area
            elif item == "polygon":
                return self.polygon
            raise KeyError(f"Invalid key '{item}'")
        return super().__getitem__(item)


def extract_boundary_marching_squares(
    elevation_diff: np.ndarray,
    threshold: float = 0.5,
    dx: float = 1.0,
    dy: float = 1.0,
    x_origin: float = 0.0,
    y_origin: float = 0.0,
    connect_tol: float = 1e-4,
) -> MarchingSquaresResult:
    """
    Extract boundary contours from 2D elevation difference matrix using the
    Marching Squares algorithm with sub-pixel linear interpolation on cell edges,
    and compute the enclosed planar area using the Shoelace formula.
    
    Args:
        elevation_diff: 2D array of elevation change (pre_dem - post_dem) with shape (Ny, Nx).
        threshold: Height difference cutoff to identify excavation boundaries (default 0.5 m).
        dx: Cell resolution in x-direction (meters).
        dy: Cell resolution in y-direction (meters).
        x_origin: Minimum x-coordinate corresponding to column index 0 (meters).
        y_origin: Minimum y-coordinate corresponding to row index 0 (meters).
        connect_tol: Metric distance tolerance for chaining contour line segments.
        
    Returns:
        MarchingSquaresResult containing:
            - contours: List of (K, 2) numpy arrays representing closed perimeter rings (x, y).
            - area: Total planar area enclosed by the boundaries in square meters.
            - polygon: Shapely Polygon or MultiPolygon representation.
    """
    diff = np.asarray(elevation_diff, dtype=np.float64)
    if diff.ndim != 2:
        raise ValueError(f"elevation_diff must be a 2D array, got shape {diff.shape}")

    Ny, Nx = diff.shape
    if Ny < 2 or Nx < 2:
        return MarchingSquaresResult([], 0.0, None)

    # 1D coordinate vectors
    x_coords = x_origin + np.arange(Nx, dtype=np.float64) * dx
    y_coords = y_origin + np.arange(Ny, dtype=np.float64) * dy

    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    eps = 1e-12

    def interp(
        p_a: Tuple[float, float],
        p_b: Tuple[float, float],
        v_a: float,
        v_b: float,
    ) -> Tuple[float, float]:
        span = v_b - v_a
        if abs(span) < eps:
            t = 0.5
        else:
            t = (threshold - v_a) / span
            t = max(0.0, min(1.0, float(t)))
        return (p_a[0] + t * (p_b[0] - p_a[0]), p_a[1] + t * (p_b[1] - p_a[1]))

    # Iterate through all grid cells
    for j in range(Ny - 1):
        for i in range(Nx - 1):
            v0 = diff[j, i]          # top-left
            v1 = diff[j, i + 1]      # top-right
            v2 = diff[j + 1, i + 1]  # bottom-right
            v3 = diff[j + 1, i]      # bottom-left

            b0 = 1 if v0 >= threshold else 0
            b1 = 1 if v1 >= threshold else 0
            b2 = 1 if v2 >= threshold else 0
            b3 = 1 if v3 >= threshold else 0

            case_idx = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)
            if case_idx == 0 or case_idx == 15:
                continue

            p0 = (x_coords[i], y_coords[j])
            p1 = (x_coords[i + 1], y_coords[j])
            p2 = (x_coords[i + 1], y_coords[j + 1])
            p3 = (x_coords[i], y_coords[j + 1])

            # Interpolate edge intersection points
            e0 = interp(p0, p1, v0, v1)  # Top edge
            e1 = interp(p1, p2, v1, v2)  # Right edge
            e2 = interp(p3, p2, v3, v2)  # Bottom edge
            e3 = interp(p0, p3, v0, v3)  # Left edge

            # Look-up table for contour segments
            if case_idx == 1:
                segments.append((e3, e0))
            elif case_idx == 2:
                segments.append((e0, e1))
            elif case_idx == 3:
                segments.append((e3, e1))
            elif case_idx == 4:
                segments.append((e1, e2))
            elif case_idx == 5:
                v_avg = (v0 + v1 + v2 + v3) / 4.0
                if v_avg >= threshold:
                    segments.append((e3, e2))
                    segments.append((e1, e0))
                else:
                    segments.append((e3, e0))
                    segments.append((e1, e2))
            elif case_idx == 6:
                segments.append((e0, e2))
            elif case_idx == 7:
                segments.append((e3, e2))
            elif case_idx == 8:
                segments.append((e2, e3))
            elif case_idx == 9:
                segments.append((e2, e0))
            elif case_idx == 10:
                v_avg = (v0 + v1 + v2 + v3) / 4.0
                if v_avg >= threshold:
                    segments.append((e0, e1))
                    segments.append((e2, e3))
                else:
                    segments.append((e0, e3))
                    segments.append((e2, e1))
            elif case_idx == 11:
                segments.append((e2, e1))
            elif case_idx == 12:
                segments.append((e1, e3))
            elif case_idx == 13:
                segments.append((e1, e0))
            elif case_idx == 14:
                segments.append((e0, e3))

    if not segments:
        return MarchingSquaresResult([], 0.0, None)

    # Chain segments into continuous polygonal rings
    chains: List[List[Tuple[float, float]]] = []
    pool = list(segments)
    tol = max(connect_tol, min(dx, dy) * 0.1)

    while pool:
        first_seg = pool.pop(0)
        chain = [first_seg[0], first_seg[1]]
        extended = True

        while extended:
            extended = False
            head = chain[0]
            tail = chain[-1]

            # Try extending tail
            for idx, seg in enumerate(pool):
                d0 = np.hypot(seg[0][0] - tail[0], seg[0][1] - tail[1])
                d1 = np.hypot(seg[1][0] - tail[0], seg[1][1] - tail[1])
                if d0 < tol:
                    chain.append(seg[1])
                    pool.pop(idx)
                    extended = True
                    break
                elif d1 < tol:
                    chain.append(seg[0])
                    pool.pop(idx)
                    extended = True
                    break

            # Try extending head
            if not extended:
                for idx, seg in enumerate(pool):
                    d0 = np.hypot(seg[1][0] - head[0], seg[1][1] - head[1])
                    d1 = np.hypot(seg[0][0] - head[0], seg[0][1] - head[1])
                    if d0 < tol:
                        chain.insert(0, seg[0])
                        pool.pop(idx)
                        extended = True
                        break
                    elif d1 < tol:
                        chain.insert(0, seg[1])
                        pool.pop(idx)
                        extended = True
                        break

        # Ensure ring is closed if endpoints are close
        if len(chain) >= 3:
            if np.hypot(chain[0][0] - chain[-1][0], chain[0][1] - chain[-1][1]) < 2.0 * tol:
                chain[-1] = chain[0]
            chains.append(chain)

    # Convert chains to numpy arrays and compute area via Shoelace formula
    contour_arrays: List[np.ndarray] = []
    total_area = 0.0
    polygons = []

    for ch in chains:
        arr = np.array(ch, dtype=np.float64)
        if len(arr) < 3:
            continue
        contour_arrays.append(arr)
        poly_area = shoelace_area(arr)
        total_area += poly_area

        try:
            poly_geom = Polygon(arr)
            if poly_geom.is_valid and not poly_geom.is_empty:
                polygons.append(poly_geom)
        except Exception:
            pass

    shapely_geom = None
    if polygons:
        if len(polygons) == 1:
            shapely_geom = polygons[0]
        else:
            shapely_geom = MultiPolygon(polygons)

    return MarchingSquaresResult(contour_arrays, total_area, shapely_geom)
