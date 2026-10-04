"""
Volumetric Mining Auditor (VMA) Core Package
"""

from src.integration import compute_volume_simpson_2d
from src.marching_squares import extract_boundary_marching_squares, shoelace_area
from src.uncertainty import compute_volumetric_uncertainty
from src.multimodal_filter import apply_multimodal_filter
from src.ledger import AuditLedger, generate_verification_qr

__all__ = [
    "compute_volume_simpson_2d",
    "extract_boundary_marching_squares",
    "shoelace_area",
    "compute_volumetric_uncertainty",
    "apply_multimodal_filter",
    "AuditLedger",
    "generate_verification_qr",
]
