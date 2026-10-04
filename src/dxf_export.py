"""
Volumetric Mining Auditor (VMA) - CAD & GIS Interoperability Module
Exports 3D excavation contours and compliance boundary polylines to AutoCAD DXF files.
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import ezdxf


def export_contours_to_dxf(
    pit_contours: List[np.ndarray],
    lease_polygon_coords: Optional[np.ndarray] = None,
    encroachment_contours: Optional[List[np.ndarray]] = None,
    output_path: Optional[str] = None,
    elevation: float = 0.0,
) -> ezdxf.document.Drawing:
    """
    Export excavation boundary contours and cadastral lease lines to an AutoCAD DXF drawing.
    
    Layers:
      - VMA_PIT_BOUNDARY (Cyan): Extracted pit contours.
      - VMA_LEASE_BOUNDARY (Green): Cadastral lease concession boundary.
      - VMA_ENCROACHMENT (Red): Unauthorized excavation breach outside lease.
    
    Args:
        pit_contours: List of (N, 2) or (N, 3) arrays of pit boundary coordinates.
        lease_polygon_coords: Optional (M, 2) array of legal lease vertices.
        encroachment_contours: Optional list of (K, 2) arrays for encroachment zones.
        output_path: Optional path to save .dxf file.
        elevation: Default Z-elevation to apply if 2D coordinates are given.
        
    Returns:
        ezdxf.document.Drawing: Generated DXF drawing document.
    """
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Setup layers with standard engineering colors
    # AutoCAD Color Index (ACI): 4 = Cyan, 3 = Green, 1 = Red
    doc.layers.add("VMA_PIT_BOUNDARY", color=4)
    doc.layers.add("VMA_LEASE_BOUNDARY", color=3)
    doc.layers.add("VMA_ENCROACHMENT", color=1)

    # Helper to convert to 3D tuples
    def to_3d_points(coords: np.ndarray) -> List[Tuple[float, float, float]]:
        pts = []
        for row in coords:
            x, y = float(row[0]), float(row[1])
            z = float(row[2]) if len(row) > 2 else elevation
            pts.append((x, y, z))
        return pts

    # Add pit contours
    for contour in pit_contours:
        if len(contour) >= 2:
            pts = to_3d_points(contour)
            msp.add_polyline3d(pts, dxfattribs={"layer": "VMA_PIT_BOUNDARY"})

    # Add lease boundary
    if lease_polygon_coords is not None and len(lease_polygon_coords) >= 2:
        pts = to_3d_points(lease_polygon_coords)
        if pts[0] != pts[-1]:
            pts.append(pts[0])  # Close ring
        msp.add_polyline3d(pts, dxfattribs={"layer": "VMA_LEASE_BOUNDARY"})

    # Add encroachment contours
    if encroachment_contours:
        for contour in encroachment_contours:
            if len(contour) >= 2:
                pts = to_3d_points(contour)
                msp.add_polyline3d(pts, dxfattribs={"layer": "VMA_ENCROACHMENT"})

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        doc.saveas(output_path)

    return doc


def dxf_to_bytes(doc: ezdxf.document.Drawing) -> bytes:
    """Serialize DXF document to in-memory bytes for direct web download."""
    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode("utf-8")
