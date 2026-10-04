import numpy as np
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import Polygon

def create_mock_data(
    dem_pre_path="dem_pre.tif",
    dem_post_path="dem_post.tif",
    boundary_path="lease_boundary.geojson"
):
    """
    Generates synthetic dataset for testing the mining volume core.
    
    1. dem_pre.tif: A 100x100 flat elevation grid at 100m.
    2. dem_post.tif: A 100x100 grid showing a 15m deep excavation pit in the center.
    3. lease_boundary.geojson: A 4-point square boundary slicing through the pit.
    """
    width = 100
    height = 100
    
    # Coordinate system: UTM Zone 32N (EPSG:32632), coordinates in meters
    west = 500000.0
    south = 4999900.0
    east = 500100.0
    north = 5000000.0
    
    # Transform grid mapping columns/rows to UTM coordinates (1 pixel = 1 meter)
    transform = from_bounds(west, south, east, north, width, height)
    
    # Generate Pre-mining DEM: Flat plane at 100.0 meters elevation
    dem_pre = np.full((height, width), 100.0, dtype=np.float32)
    
    # Generate Post-mining DEM: Smooth parabolic excavation pit of depth 15m at center
    dem_post = dem_pre.copy()
    center_r, center_c = 50, 50
    radius = 25.0
    max_depth = 15.0
    
    # Generate grid coordinates
    r_coords, c_coords = np.ogrid[:height, :width]
    distances = np.sqrt((r_coords - center_r) ** 2 + (c_coords - center_c) ** 2)
    
    # Inside the radius, depth is parabolic: depth = max_depth * (1 - (d/radius)^2)
    pit_mask = distances < radius
    pit_depth = max_depth * (1.0 - (distances / radius) ** 2)
    
    dem_post[pit_mask] -= pit_depth[pit_mask]
    
    # Add a bit of small random noise to elevations (e.g. standard deviation 0.05m) to simulate real DEMs
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, size=(height, width)).astype(np.float32)
    dem_pre += noise
    dem_post += noise
    
    # Save DEM Pre
    meta = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': -9999.0,
        'width': width,
        'height': height,
        'count': 1,
        'crs': 'EPSG:32632',
        'transform': transform
    }
    
    with rasterio.open(dem_pre_path, 'w', **meta) as dst:
        dst.write(dem_pre, 1)
        
    # Save DEM Post
    with rasterio.open(dem_post_path, 'w', **meta) as dst:
        dst.write(dem_post, 1)
        
    # Create lease boundary (4-point polygon)
    # The pit center is at X = 500050, Y = 4999950.
    # The pit extends from X = 500025 to 500075 and Y = 4999925 to 4999975.
    # Let's slice the pit: Lease boundary covers X: [500010, 500055], Y: [4999920, 4999980].
    # This leaves the eastern part of the pit (X > 500055) outside the lease boundary (illegal encroachment).
    poly = Polygon([
        (500010.0, 4999920.0),
        (500055.0, 4999920.0),
        (500055.0, 4999980.0),
        (500010.0, 4999980.0),
        (500010.0, 4999920.0) # Close polygon
    ])
    
    gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:32632", geometry=[poly])
    gdf.to_file(boundary_path, driver="GeoJSON")
    
    print(f"Mock data successfully written:")
    print(f" - {dem_pre_path}")
    print(f" - {dem_post_path}")
    print(f" - {boundary_path}")

if __name__ == "__main__":
    create_mock_data()
