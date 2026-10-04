import os
import math
import numpy as np
import geopandas as gpd
from mock_gen import create_mock_data
from mining_engine import process_mining_data, generate_pdf_report

def run_tests():
    print("========================================")
    print("Starting Mining Geospatial Core Verification")
    print("========================================")
    
    # Paths
    dem_pre = "dem_pre.tif"
    dem_post = "dem_post.tif"
    boundary = "lease_boundary.geojson"
    report_pdf = "test_audit_report.pdf"
    
    # Remove files if they exist
    for f in [dem_pre, dem_post, boundary, report_pdf]:
        if os.path.exists(f):
            os.remove(f)
            
    # Test 1: Mock Data Generation
    print("\n[Test 1] Generating mock dataset...")
    create_mock_data(dem_pre, dem_post, boundary)
    
    assert os.path.exists(dem_pre), "Pre-mining DEM TIF not generated"
    assert os.path.exists(dem_post), "Post-mining DEM TIF not generated"
    assert os.path.exists(boundary), "Boundary GeoJSON not generated"
    print("✅ Mock data generation successful.")
    
    # Test 2: Core Processing Engine
    print("\n[Test 2] Running mining core processing...")
    metrics, geoms = process_mining_data(dem_pre, dem_post, boundary)
    
    # Print metrics
    print("Calculated Metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")
        
    # Validation checks
    # 1. Pit area
    # Since we filter for dz > 2.0, the outer boundary of the pit is detected where
    # 15 * (1 - (d/25)^2) > 2.0 => d < 25 * sqrt(13/15) = 23.2737 meters.
    # Analytical detected area is pi * (23.2737)^2 = 1701.69 m².
    calculated_area = metrics['total_pit_area_m2']
    expected_area = 1701.0
    print(f" - Analytical Detected Pit Area: ~{expected_area} m²")
    print(f" - Calculated Pit Area: {calculated_area:.2f} m²")
    assert abs(calculated_area - expected_area) < 5.0, f"Pit area is far from expected: {calculated_area}"
    
    # 2. Volume
    # Analytical volume integrated where dz > 2.0 is:
    # Int_{0}^{23.2737} 2 * pi * r * 15 * (1 - r^2 / 625) dr = 30 * pi * [r^2 / 2 - r^4 / 2500]_{0}^{23.2737}
    # = 30 * pi * [270.84 - 117.36] = 30 * pi * 153.48 = 14464.7 m³.
    analytical_volume = 14464.7
    calculated_volume = metrics['total_volume_m3']
    print(f" - Analytical Volume (>2m depth): {analytical_volume:.2f} m³")
    print(f" - Calculated Volume (Simpson's 1/3): {calculated_volume:.2f} m³")
    
    percent_diff = abs(calculated_volume - analytical_volume) / analytical_volume * 100
    print(f" - Percentage Difference: {percent_diff:.4f}%")
    assert percent_diff < 0.5, f"Volume calculations differ from analytical volume by {percent_diff:.2f}%"
    
    # 3. Partitioning
    # Authorized + Illegal area = Total pit area
    area_sum = metrics['authorized_area_m2'] + metrics['illegal_area_m2']
    print(f" - Area Partitioning Sum: {area_sum:.2f} m² vs Total: {metrics['total_pit_area_m2']:.2f} m²")
    assert abs(area_sum - metrics['total_pit_area_m2']) < 1e-5, "Area partition does not equal total area"
    
    # Authorized + Illegal volume = Total volume
    volume_sum = metrics['authorized_volume_m3'] + metrics['illegal_volume_m3']
    print(f" - Volume Partitioning Sum: {volume_sum:.2f} m³ vs Total: {metrics['total_volume_m3']:.2f} m³")
    assert abs(volume_sum - metrics['total_volume_m3']) < 1e-5, "Volume partition does not equal total volume"
    
    # 4. Encroachment flag
    assert metrics['illegal_area_m2'] > 0, "Illegal area should be greater than 0 for this mock test setup"
    assert metrics['encroachment_ratio'] > 0, "Encroachment ratio should be greater than 0"
    
    print("✅ Core processing tests passed successfully.")
    
    # Test 3: PDF Exporter
    print("\n[Test 3] Compiling ReportLab PDF audit report...")
    generate_pdf_report(metrics, geoms, dem_pre, dem_post, boundary, report_pdf)
    assert os.path.exists(report_pdf), "PDF Report not generated"
    assert os.path.getsize(report_pdf) > 1000, "PDF Report is empty or too small"
    print("✅ PDF Report generation successful.")
    
    # Clean up test files
    for f in [dem_pre, dem_post, boundary, report_pdf]:
         if os.path.exists(f):
             os.remove(f)
             
    print("\n========================================")
    print("All tests passed successfully!")
    print("========================================")

if __name__ == "__main__":
    run_tests()
