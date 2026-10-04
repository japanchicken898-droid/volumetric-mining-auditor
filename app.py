import streamlit as st
import os
import tempfile
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Import core functionalities
from mock_gen import create_mock_data
from mining_engine import process_mining_data, generate_pdf_report

# Page Config
st.set_page_config(
    page_title="Volumetric Mining Auditor",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
    .main {
        background-color: #F8FAFC;
    }
    .reportview-container {
        font-family: "Outfit", "Inter", sans-serif;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1E293B;
    }
    .stAlert {
        border-radius: 8px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 15px;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 8px;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .metric-label {
        color: #64748B;
        font-weight: 500;
    }
    .metric-val {
        font-weight: 600;
        color: #0F172A;
    }
    .metric-val-danger {
        font-weight: 600;
        color: #DC2626;
    }
    .metric-val-success {
        font-weight: 600;
        color: #16A34A;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⛏️ Volumetric Mining Auditor & Compliance Engine")
st.write("Perform geospatial audits on quarries and mines using high-resolution pre-mining and post-mining Digital Elevation Models (DEMs).")

# Sidebar settings
st.sidebar.header("Control Panel")

# Mock data generation
st.sidebar.subheader("Testing & Demo Data")
if st.sidebar.button("⚙️ Generate Mock Datasets", help="Create synthetic pre-mining and post-mining DEMs and lease boundary files for testing."):
    with st.spinner("Generating mock rasters..."):
        create_mock_data("dem_pre.tif", "dem_post.tif", "lease_boundary.geojson")
        st.sidebar.success("Successfully generated demo files in workspace:")
        st.sidebar.info("- `dem_pre.tif` (100x100 grid)\n- `dem_post.tif` (15m excavation)\n- `lease_boundary.geojson` (4-point polygon)")
        st.session_state['dem_pre_path'] = "dem_pre.tif"
        st.session_state['dem_post_path'] = "dem_post.tif"
        st.session_state['boundary_path'] = "lease_boundary.geojson"

st.sidebar.subheader("Inputs Upload")

# Callback to clear results when inputs change
def clear_results():
    if 'results' in st.session_state:
        del st.session_state['results']

# File upload or mock paths with clear callbacks
dem_pre_file = st.sidebar.file_uploader("Upload Pre-Mining DEM (.tif)", type=["tif", "tiff"], on_change=clear_results)
dem_post_file = st.sidebar.file_uploader("Upload Post-Mining DEM (.tif)", type=["tif", "tiff"], on_change=clear_results)
boundary_files = st.sidebar.file_uploader(
    "Upload Lease Boundary (.geojson, .shp, .shx, .dbf, .prj, .kml)", 
    type=["geojson", "shp", "shx", "dbf", "prj", "kml"], 
    accept_multiple_files=True,
    on_change=clear_results
)

# Define default paths or uploaded files
dem_pre_path = None
dem_post_path = None
boundary_path = None

# If user uploaded files, write them to temp files
temp_dir = tempfile.mkdtemp()

if dem_pre_file:
    dem_pre_path = os.path.join(temp_dir, "uploaded_pre.tif")
    with open(dem_pre_path, "wb") as f:
        f.write(dem_pre_file.getbuffer())
elif 'dem_pre_path' in st.session_state:
    dem_pre_path = st.session_state['dem_pre_path']

if dem_post_file:
    dem_post_path = os.path.join(temp_dir, "uploaded_post.tif")
    with open(dem_post_path, "wb") as f:
        f.write(dem_post_file.getbuffer())
elif 'dem_post_path' in st.session_state:
    dem_post_path = st.session_state['dem_post_path']

if boundary_files:
    # Save all uploaded shapefile/geojson components to the temp folder
    for f in boundary_files:
        filepath = os.path.join(temp_dir, f.name)
        with open(filepath, "wb") as out_f:
            out_f.write(f.getbuffer())
        if f.name.lower().endswith(('.shp', '.geojson', '.kml')):
            boundary_path = filepath
elif 'boundary_path' in st.session_state:
    boundary_path = st.session_state['boundary_path']

# Trigger analysis button
run_analysis = st.sidebar.button("🚀 Run Auditing Engine", use_container_width=True)

if dem_pre_path and dem_post_path and boundary_path:
    # Run analysis automatically if requested or mock loaded
    if run_analysis or 'results' in st.session_state or ('dem_pre_path' in st.session_state and run_analysis):
        with st.spinner("Processing elevation grids and performing spatial checks..."):
            try:
                metrics, geoms = process_mining_data(dem_pre_path, dem_post_path, boundary_path)
                st.session_state['results'] = (metrics, geoms)
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.stop()
                
        metrics, geoms = st.session_state['results']
        
        # Encroachment check
        encroached = metrics['illegal_area_m2'] > 1.0
        
        # Alert Box
        if encroached:
            st.error(f"⚠️ **COMPLIANCE ALERT: INCROACHMENT DETECTED!** \nExcavation pit extends outside the lease boundary by **{metrics['illegal_area_m2']:.2f} m²** ({metrics['encroachment_ratio']*100:.1f}% of pit area). \nTotal illegal excavation volume is **{metrics['illegal_volume_m3']:.2f} m³**.")
        else:
            st.success("✅ **COMPLIANCE CHECK: PASSED** \nExcavation activity is fully contained within the official lease boundary. No encroachment detected.")
            
        # Metrics Display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="card">
                    <div class="card-header">📐 Excavation Area Metrics</div>
                    <div class="metric-row"><span class="metric-label">Total Pit Area:</span><span class="metric-val">{metrics['total_pit_area_m2']:.1f} m²</span></div>
                    <div class="metric-row"><span class="metric-label">Authorized Area:</span><span class="metric-val-success">{metrics['authorized_area_m2']:.1f} m²</span></div>
                    <div class="metric-row"><span class="metric-label">Encroached Area:</span><span class="metric-val-danger" style="color: {'#DC2626' if encroached else '#0F172A'}">{metrics['illegal_area_m2']:.1f} m²</span></div>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div class="card">
                    <div class="card-header">📊 Excavation Volume Metrics</div>
                    <div class="metric-row"><span class="metric-label">Total Excavated Vol:</span><span class="metric-val">{metrics['total_volume_m3']:.1f} m³</span></div>
                    <div class="metric-row"><span class="metric-label">Authorized Vol:</span><span class="metric-val-success">{metrics['authorized_volume_m3']:.1f} m³</span></div>
                    <div class="metric-row"><span class="metric-label">Encroached Vol:</span><span class="metric-val-danger" style="color: {'#DC2626' if encroached else '#0F172A'}">{metrics['illegal_volume_m3']:.1f} m³</span></div>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
                <div class="card">
                    <div class="card-header">📏 Pit Depth Metrics</div>
                    <div class="metric-row"><span class="metric-label">Maximum Depth:</span><span class="metric-val">{metrics['max_depth_m']:.2f} m</span></div>
                    <div class="metric-row"><span class="metric-label">Average Depth:</span><span class="metric-val">{metrics['avg_depth_m']:.2f} m</span></div>
                    <div class="metric-row"><span class="metric-label">Encroachment Ratio:</span><span class="metric-val-danger" style="color: {'#DC2626' if encroached else '#0F172A'}">{metrics['encroachment_ratio']*100:.1f} %</span></div>
                </div>
            """, unsafe_allow_html=True)
            
        # Visualizations Tabs
        tab1, tab2 = st.tabs(["🗺️ Compliance Mapping", "🧊 3D Terrain Analysis"])
        
        with tab1:
            st.subheader("Spatial Verification Map")
            st.write("Lease Boundary (dashed black), Authorized excavation (green), and Illegal encroachment (red).")
            
            # Matplotlib Plot
            with rasterio.open(dem_pre_path) as src_pre:
                dem_pre = src_pre.read(1)
                
            with rasterio.open(dem_post_path) as src_post:
                dem_post = src_post.read(1)
                
            dz = dem_pre - dem_post
            if src_pre.nodata is not None:
                dz[(dem_pre == src_pre.nodata) | (dem_post == src_post.nodata)] = 0.0
                
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(dz, cmap='YlOrRd')
            fig.colorbar(im, ax=ax, label="Depth Difference (m)")
            
            lease_shape = shape(geoms['lease_geojson']) if geoms['lease_geojson'] else None
            auth_shape = shape(geoms['authorized_geojson']) if geoms['authorized_geojson'] else None
            illegal_shape = shape(geoms['illegal_geojson']) if geoms['illegal_geojson'] else None
            
            if lease_shape:
                # Need to plot coordinates in pixel index space or UTM space?
                # Because the image is plotted in pixel coordinates, we should re-project shapes or plot shapes in UTM!
                # Ah! It is much cleaner to plot in UTM space by using rasterio's extent for the image!
                # Rasterio bounds:
                extent = [src_pre.bounds.left, src_pre.bounds.right, src_pre.bounds.bottom, src_pre.bounds.top]
                
                # Replot with extent!
                ax.clear()
                im = ax.imshow(dz, cmap='YlOrRd', extent=extent)
                
                # Plot lease boundary outline
                x, y = lease_shape.exterior.xy
                ax.plot(x, y, color='black', linestyle='--', linewidth=2.0, label='Lease Boundary')
                
                # Plot authorized pit
                if auth_shape and not auth_shape.is_empty:
                    if isinstance(auth_shape, Polygon):
                        x, y = auth_shape.exterior.xy
                        ax.fill(x, y, color='green', alpha=0.5, label='Authorized Pit')
                    elif isinstance(auth_shape, MultiPolygon):
                        for poly in auth_shape.geoms:
                            x, y = poly.exterior.xy
                            ax.fill(x, y, color='green', alpha=0.5)
                            
                # Plot illegal pit
                if illegal_shape and not illegal_shape.is_empty:
                    if isinstance(illegal_shape, Polygon):
                        x, y = illegal_shape.exterior.xy
                        ax.fill(x, y, color='red', alpha=0.5, label='Illegal Encroachment')
                    elif isinstance(illegal_shape, MultiPolygon):
                        for poly in illegal_shape.geoms:
                            x, y = poly.exterior.xy
                            ax.fill(x, y, color='red', alpha=0.5)
                
                # Setup labels and legends
                ax.set_title("Volumetric Difference Map overlaid with Lease Boundary", fontsize=12, fontweight='bold')
                ax.set_xlabel("UTM Easting (m)")
                ax.set_ylabel("UTM Northing (m)")
                
                legend_elements = [
                    Patch(facecolor='none', edgecolor='black', linestyle='--', label='Lease Boundary'),
                    Patch(facecolor='green', alpha=0.5, label='Authorized Area'),
                ]
                if encroached:
                    legend_elements.append(Patch(facecolor='red', alpha=0.5, label='Illegal Encroachment'))
                    
                ax.legend(handles=legend_elements, loc='upper right')
                
            st.pyplot(fig)
            
        with tab2:
            st.subheader("3D Interactive Surface Plot")
            st.write("Rotate and zoom to audit the excavation geometry.")
            
            # Select DEM to display
            dem_choice = st.radio("Choose 3D Surface Model:", ["Pre-Mining Terrain", "Post-Mining Terrain", "Excavation Depth Difference (Δz)"], horizontal=True)
            
            with rasterio.open(dem_pre_path) as src_pre:
                d_pre = src_pre.read(1)
            with rasterio.open(dem_post_path) as src_post:
                d_post = src_post.read(1)
                
            # Keep NaNs as np.nan so Plotly renders clean elevation surfaces without fake vertical walls
            d_pre[d_pre == -9999.0] = np.nan
            d_post[d_post == -9999.0] = np.nan
            
            d_diff = d_pre - d_post
            
            if dem_choice == "Pre-Mining Terrain":
                z_data = d_pre
                title = "3D Pre-Mining Elevation"
                colorscale = "earth"
            elif dem_choice == "Post-Mining Terrain":
                z_data = d_post
                title = "3D Post-Mining Elevation"
                colorscale = "earth"
            else:
                z_data = d_diff
                title = "3D Excavation Depth (Δz)"
                colorscale = "Viridis"
                
            # Create Plotly Surface
            fig_3d = go.Figure(data=[go.Surface(z=z_data, colorscale=colorscale)])
            fig_3d.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='Easting Grid',
                    yaxis_title='Northing Grid',
                    zaxis_title='Elevation / Depth (m)'
                ),
                width=900,
                height=700,
                margin=dict(l=0, r=0, b=0, t=50)
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
        # PDF Generation
        st.subheader("⚖️ Export Compliance Report")
        st.write("Generate a printable, audit-grade PDF document containing legal flags, data tables, and embedded maps.")
        
        pdf_filename = "Mining_Audit_Report.pdf"
        
        # We can write to a temporary file, then read and expose to Streamlit download button
        temp_pdf = os.path.join(temp_dir, pdf_filename)
        
        if st.button("📄 Compile PDF Audit Report"):
            with st.spinner("Generating high-resolution report using ReportLab..."):
                try:
                    generate_pdf_report(metrics, geoms, dem_pre_path, dem_post_path, boundary_path, temp_pdf)
                    
                    with open(temp_pdf, "rb") as pdf_file:
                        pdf_data = pdf_file.read()
                        
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_data,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("Report compiled successfully! Click the button above to download.")
                except Exception as e:
                    st.error(f"Failed to compile PDF: {str(e)}")
                    
else:
    # Display message if files are missing
    st.info("👋 Welcome! Please upload Pre-Mining DEM, Post-Mining DEM, and Lease Boundary files in the Sidebar. \nAlternatively, click **'Generate Mock Datasets'** in the sidebar to load synthetic test data instantly.")
