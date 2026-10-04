import os
from datetime import datetime
import numpy as np
import rasterio
import rasterio.crs
from rasterio.features import shapes, geometry_mask
import geopandas as gpd
import shapely
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union

# PDF and Plotting imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import io

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def get_simpson_weights(n):
    """
    Generates 1D weights for Simpson's 1/3 integration rule of length n.
    Handles both odd and even lengths gracefully:
    - For odd n: Standard Simpson's 1/3 rule weights (1, 4, 2, 4, ..., 1) with multiplier h/3.
    - For even n: Simpson's 1/3 rule on the first n-1 points, and Trapezoidal rule on the last interval.
    - For n < 3: Trapezoidal rule fallback.
    
    Weights are returned in scaled form such that:
    Integral = (h/3) * sum(weights * values)
    """
    if n <= 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    if n == 2:
        # Trapezoidal rule: Integral = h/2 * (f0 + f1).
        # Scaled to h/3 multiplier: h/3 * (1.5 * f0 + 1.5 * f1).
        return np.array([1.5, 1.5])
    
    if n % 2 == 1:
        # Odd number of points (even number of intervals)
        w = np.ones(n)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        return w
    else:
        # Even number of points (odd number of intervals)
        # Apply Simpson's 1/3 rule to the first n-1 points (indices 0 to n-2)
        w = np.zeros(n)
        w_simps = np.ones(n - 1)
        w_simps[1:-1:2] = 4.0
        w_simps[2:-1:2] = 2.0
        w[:-1] += w_simps
        
        # Apply Trapezoidal rule to the last interval (indices n-2 to n-1)
        # The trapezoidal rule has weight 0.5 * h, which matches 1.5 * (h/3).
        # We add 1.5 to the weights of the last two points (index n-2 and n-1).
        w[-2:] += np.array([1.5, 1.5])
        return w

def process_mining_data(dem_pre_path, dem_post_path, boundary_file_path):
    """
    Core engine to process mining elevations and lease boundaries.
    
    Parameters:
      dem_pre_path (str): Path to pre-mining DEM TIFF.
      dem_post_path (str): Path to post-mining DEM TIFF.
      boundary_file_path (str): Path to lease boundary GeoJSON/Shapefile.
      
    Returns:
      metrics (dict): Comprehensive excavation metrics.
      geometries (dict): GeoJSON dicts of Authorized and Illegal zones.
    """
    # 1. Load Pre and Post DEMs using Rasterio
    with rasterio.open(dem_pre_path) as src_pre:
        dem_pre = src_pre.read(1)
        pre_meta = src_pre.meta
        pre_transform = src_pre.transform
        pre_crs = src_pre.crs
        pre_res = src_pre.res  # Resolution (dx, dy)
        pre_nodata = src_pre.nodata
        
    with rasterio.open(dem_post_path) as src_post:
        dem_post = src_post.read(1)
        post_nodata = src_post.nodata
        post_crs = src_post.crs
        post_transform = src_post.transform
        
    # Handle None CRS gracefully
    if pre_crs is None:
        pre_crs = rasterio.crs.CRS.from_epsg(4326)
    if post_crs is None:
        post_crs = rasterio.crs.CRS.from_epsg(4326)
        
    # Assert transforms and CRS match
    if pre_transform != post_transform or pre_crs != post_crs:
        raise ValueError("Raster transforms or CRS do not match!")
        
    # Check shape alignment
    if dem_pre.shape != dem_post.shape:
        raise ValueError(
            f"Pre-mining DEM shape {dem_pre.shape} does not match Post-mining DEM shape {dem_post.shape}."
        )
        
    # 2. Calculate elevation difference grid: Δz = DEM_pre - DEM_post
    # Handle NaN and nodata values gracefully
    invalid_mask = np.isnan(dem_pre) | np.isnan(dem_post)
    if pre_nodata is not None:
        invalid_mask |= (dem_pre == pre_nodata)
    if post_nodata is not None:
        invalid_mask |= (dem_post == post_nodata)
        
    dz = dem_pre - dem_post
    dz[invalid_mask] = 0.0  # Reset invalid differences to 0
    
    # 3. Load Lease Boundary and reproject to match DEM CRS
    if isinstance(boundary_file_path, list):
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()
        main_path = None
        for f in boundary_file_path:
            if hasattr(f, 'name'):
                name = f.name
                content = f.getvalue() if hasattr(f, 'getvalue') else f.read()
                filepath = os.path.join(temp_dir, name)
                with open(filepath, 'wb') as temp_f:
                    temp_f.write(content)
            elif isinstance(f, str):
                name = os.path.basename(f)
                filepath = os.path.join(temp_dir, name)
                shutil.copy(f, filepath)
            else:
                continue
            
            if name.lower().endswith(('.shp', '.geojson', '.kml')):
                main_path = filepath
                
        if main_path is None:
            raise ValueError("No valid shapefile (.shp), GeoJSON, or KML file found in the uploaded boundary files.")
        boundary_gdf = gpd.read_file(main_path)
    else:
        boundary_gdf = gpd.read_file(boundary_file_path)
    
    if boundary_gdf.crs != pre_crs:
        boundary_gdf = boundary_gdf.to_crs(pre_crs)
    
    # Dissolve to a single lease geometry (handling multiple features)
    lease_geom = boundary_gdf.geometry.unary_union
    if not lease_geom.is_valid:
        lease_geom = lease_geom.buffer(0.0)
        
    # 4. Detect pit area where Δz > 2.0 meters
    pit_mask = (dz > 2.0) & ~invalid_mask
    
    # Vectorize the pit mask into polygons
    if np.any(pit_mask):
        # shapes() returns (geometry, value) tuples for connected components
        shape_gen = shapes(pit_mask.astype(np.uint8), mask=pit_mask, transform=pre_transform)
        pit_polygons = [shape(g) for g, v in shape_gen if v == 1]
    else:
        pit_polygons = []
        
    if len(pit_polygons) > 0:
        detected_pit_geom = unary_union(pit_polygons)
        if not detected_pit_geom.is_valid:
            detected_pit_geom = detected_pit_geom.buffer(0.0)
    else:
        detected_pit_geom = Polygon()
        
    # 5. Perform spatial operations with Shapely
    if not detected_pit_geom.is_empty:
        authorized_geom = detected_pit_geom.intersection(lease_geom)
        illegal_geom = detected_pit_geom.difference(lease_geom)
    else:
        authorized_geom = Polygon()
        illegal_geom = Polygon()
        
    # Clean up results
    if not authorized_geom.is_valid:
        authorized_geom = authorized_geom.buffer(0.0)
    if not illegal_geom.is_valid:
        illegal_geom = illegal_geom.buffer(0.0)
        
    # 6. Calculate Area Metrics (m²)
    total_pit_area = detected_pit_geom.area
    authorized_area = authorized_geom.area
    illegal_area = illegal_geom.area
    encroachment_ratio = illegal_area / total_pit_area if total_pit_area > 0 else 0.0
    
    # 7. Calculate Depth Metrics within the detected pit polygon
    if not detected_pit_geom.is_empty:
        # Use rasterio geometry_mask to mask pixels outside the detected pit geometry
        pit_pixel_mask = geometry_mask([detected_pit_geom], out_shape=dz.shape, transform=pre_transform, invert=True)
        pit_dz = dz[pit_pixel_mask & ~invalid_mask]
        
        if len(pit_dz) > 0:
            max_depth = float(np.max(pit_dz))
            avg_depth = float(np.mean(pit_dz))
        else:
            max_depth = 0.0
            avg_depth = 0.0
    else:
        max_depth = 0.0
        avg_depth = 0.0
        
    # 8. Calculate total excavated volume (m³) using Simpson's 2D double integration
    # Formula: V = (dx * dy / 9) * sum(weights * Δz)
    height, width = dz.shape
    dx, dy = pre_res[0], pre_res[1]
    
    # Generate 1D weights
    w_r = get_simpson_weights(height)  # Row weights (Y axis)
    w_c = get_simpson_weights(width)   # Col weights (X axis)
    
    # 2D weights matrix
    weights_2d = np.outer(w_r, w_c)
    
    # Total Volume (integrate clean positive changes)
    dz_excavation = np.where(pit_mask, dz, 0.0)
    total_volume = (dx * dy / 9.0) * np.sum(weights_2d * dz_excavation)
    
    # Authorized Volume
    if not authorized_geom.is_empty:
        auth_pixel_mask = geometry_mask([authorized_geom], out_shape=dz.shape, transform=pre_transform, invert=True)
        dz_auth = np.where(auth_pixel_mask & pit_mask, dz, 0.0)
        authorized_volume = (dx * dy / 9.0) * np.sum(weights_2d * dz_auth)
    else:
        authorized_volume = 0.0
        
    # Illegal/Encroached Volume
    if not illegal_geom.is_empty:
        illegal_pixel_mask = geometry_mask([illegal_geom], out_shape=dz.shape, transform=pre_transform, invert=True)
        dz_illegal = np.where(illegal_pixel_mask & pit_mask, dz, 0.0)
        illegal_volume = (dx * dy / 9.0) * np.sum(weights_2d * dz_illegal)
    else:
        illegal_volume = 0.0
        
    # Output metrics dictionary
    metrics = {
        'total_pit_area_m2': float(total_pit_area),
        'authorized_area_m2': float(authorized_area),
        'illegal_area_m2': float(illegal_area),
        'encroachment_ratio': float(encroachment_ratio),
        'max_depth_m': float(max_depth),
        'avg_depth_m': float(avg_depth),
        'total_volume_m3': float(total_volume),
        'authorized_volume_m3': float(authorized_volume),
        'illegal_volume_m3': float(illegal_volume),
    }
    
    # Convert Shapely geometries to GeoJSON features
    geoms = {
        'authorized_geojson': mapping(authorized_geom) if not authorized_geom.is_empty else None,
        'illegal_geojson': mapping(illegal_geom) if not illegal_geom.is_empty else None,
        'lease_geojson': mapping(lease_geom) if not lease_geom.is_empty else None
    }
    
    return metrics, geoms

def generate_pdf_report(metrics, geoms, dem_pre_path, dem_post_path, boundary_path, output_pdf_path):
    """
    Generates a professional, print-ready PDF compliance report using ReportLab.
    Includes metadata, metrics comparison tables, compliance warnings, and embedded figures.
    """
    # 1. Generate standard Matplotlib visualization for the report
    with rasterio.open(dem_pre_path) as src_pre:
        dem_pre = src_pre.read(1)
        transform = src_pre.transform
        nodata = src_pre.nodata
        crs_str = src_pre.crs.to_string() if src_pre.crs else "Unknown CRS"
        
    with rasterio.open(dem_post_path) as src_post:
        dem_post = src_post.read(1)
        
    dz = dem_pre - dem_post
    if nodata is not None:
        dz[(dem_pre == nodata) | (dem_post == nodata)] = 0.0
        
    fig, axs = plt.subplots(2, 2, figsize=(10, 8.5))
    
    # Pre-mining DEM
    im0 = axs[0, 0].imshow(dem_pre, cmap='terrain')
    axs[0, 0].set_title("Pre-Mining Terrain (DEM)", fontsize=10, fontweight='bold')
    axs[0, 0].axis('off')
    fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04).set_label("Elevation (m)", fontsize=8)
    
    # Post-mining DEM
    im1 = axs[0, 1].imshow(dem_post, cmap='terrain')
    axs[0, 1].set_title("Post-Mining Terrain (DEM)", fontsize=10, fontweight='bold')
    axs[0, 1].axis('off')
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04).set_label("Elevation (m)", fontsize=8)
    
    # Difference Map
    im2 = axs[1, 0].imshow(dz, cmap='YlOrRd')
    axs[1, 0].set_title("Excavation Depth (Δz)", fontsize=10, fontweight='bold')
    axs[1, 0].axis('off')
    fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04).set_label("Depth (m)", fontsize=8)
    
    # Spatial Compliance Overlay
    axs[1, 1].set_title("Compliance & Encroachment Map", fontsize=10, fontweight='bold')
    axs[1, 1].axis('off')
    
    # Recreate geometries to plot using matplotlib
    lease_shape = shape(geoms['lease_geojson']) if geoms['lease_geojson'] else None
    auth_shape = shape(geoms['authorized_geojson']) if geoms['authorized_geojson'] else None
    illegal_shape = shape(geoms['illegal_geojson']) if geoms['illegal_geojson'] else None
    
    # Plot lease boundary outline
    if lease_shape:
        x, y = lease_shape.exterior.xy
        axs[1, 1].plot(x, y, color='black', linestyle='--', linewidth=2.0, label='Lease Boundary')
        
    # Plot authorized pit
    if auth_shape:
        if isinstance(auth_shape, Polygon):
            x, y = auth_shape.exterior.xy
            axs[1, 1].fill(x, y, color='green', alpha=0.6, label='Authorized Pit')
        elif isinstance(auth_shape, MultiPolygon):
            for poly in auth_shape.geoms:
                x, y = poly.exterior.xy
                axs[1, 1].fill(x, y, color='green', alpha=0.6)
                
    # Plot illegal pit
    if illegal_shape and not illegal_shape.is_empty:
        if isinstance(illegal_shape, Polygon):
            x, y = illegal_shape.exterior.xy
            axs[1, 1].fill(x, y, color='red', alpha=0.6, label='Illegal Encroachment')
        elif isinstance(illegal_shape, MultiPolygon):
            for poly in illegal_shape.geoms:
                x, y = poly.exterior.xy
                axs[1, 1].fill(x, y, color='red', alpha=0.6)
                
    legend_patches = [
        Patch(facecolor='none', edgecolor='black', linestyle='--', label='Lease Boundary'),
        Patch(facecolor='green', alpha=0.6, label='Authorized Area'),
    ]
    if illegal_shape and not illegal_shape.is_empty:
        legend_patches.append(Patch(facecolor='red', alpha=0.6, label='Illegal Encroachment'))
        
    axs[1, 1].legend(handles=legend_patches, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    
    # Save image to buffer
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=300)
    img_buf.seek(0)
    plt.close()
    
    # 2. Build ReportLab PDF
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=0.5*inch,
        rightMargin=0.5*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#1A365D")  # Slate Blue
    c_secondary = colors.HexColor("#4A5568") # Charcoal
    c_danger = colors.HexColor("#C53030")    # Crimson Red
    c_success = colors.HexColor("#2F855A")   # Forest Green
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=c_primary,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=c_secondary,
        spaceAfter=10
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748")
    )
    
    warning_style = ParagraphStyle(
        'WarningBox',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=c_danger
    )
    
    success_style = ParagraphStyle(
        'SuccessBox',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=c_success
    )
    
    story = []
    
    # Title
    story.append(Paragraph("MINING EXCAVATION AUDIT REPORT", title_style))
    story.append(Paragraph("Standard Compliance & Encroachment Assessment", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Extract name for display
    if isinstance(boundary_path, list):
        main_name = "Lease Boundary"
        for f in boundary_path:
            name = f.name if hasattr(f, 'name') else str(f)
            if name.lower().endswith(('.shp', '.geojson', '.kml')):
                main_name = os.path.basename(name)
                break
    else:
        main_name = os.path.basename(boundary_path)

    # Metadata Table
    meta_data = [
        [Paragraph("<b>Pre-Mining DEM:</b>", body_style), Paragraph(os.path.basename(dem_pre_path), body_style),
         Paragraph("<b>Date of Analysis:</b>", body_style), Paragraph(datetime.now().strftime("%B %d, %Y - %H:%M UTC"), body_style)],
        [Paragraph("<b>Post-Mining DEM:</b>", body_style), Paragraph(os.path.basename(dem_post_path), body_style),
         Paragraph("<b>Coordinate System:</b>", body_style), Paragraph(crs_str, body_style)],
        [Paragraph("<b>Lease Boundary:</b>", body_style), Paragraph(main_name, body_style),
         Paragraph("<b>Elevation Cutoff:</b>", body_style), Paragraph("> 2.0 meters", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[1.5*inch, 2.25*inch, 1.5*inch, 2.25*inch])
    meta_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LINEBELOW', (0,-1), (-1,-1), 1, c_secondary),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))
    
    # Compliance Alert Banner
    encroached = metrics['illegal_area_m2'] > 1.0
    if encroached:
        alert_text = f"<b>WARNING: ILLEGAL ENCROACHMENT DETECTED</b><br/>The excavation pit extends outside the official lease boundary by {metrics['illegal_area_m2']:.2f} m² ({metrics['encroachment_ratio']*100:.1f}% of total excavation area). A total of {metrics['illegal_volume_m3']:.2f} m³ has been illegally excavated."
        alert_style = warning_style
        bg_color = colors.HexColor("#FFF5F5")
        border_color = c_danger
    else:
        alert_text = "<b>COMPLIANCE STATUS: PASSED</b><br/>All detected mining excavation is completely contained within the official lease boundary limits. No encroachment detected."
        alert_style = success_style
        bg_color = colors.HexColor("#F0FFF4")
        border_color = c_success
        
    alert_table = Table([[Paragraph(alert_text, alert_style)]], colWidths=[7.5*inch])
    alert_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_color),
        ('BOX', (0,0), (-1,-1), 1.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(alert_table)
    story.append(Spacer(1, 15))
    
    # Metrics Table
    story.append(Paragraph("Excavation Metrics Summary", heading_style))
    
    headers = [
        Paragraph("<b>Metric Description</b>", body_style),
        Paragraph("<b>Total Pit</b>", body_style),
        Paragraph("<b>Authorized Zone</b>", body_style),
        Paragraph("<b>Encroached Zone</b>", body_style)
    ]
    
    metrics_rows = [
        headers,
        [Paragraph("Excavation Area (m²)", body_style),
         f"{metrics['total_pit_area_m2']:.2f}",
         f"{metrics['authorized_area_m2']:.2f}",
         f"{metrics['illegal_area_m2']:.2f}"],
        [Paragraph("Excavated Volume (m³)", body_style),
         f"{metrics['total_volume_m3']:.2f}",
         f"{metrics['authorized_volume_m3']:.2f}",
         f"{metrics['illegal_volume_m3']:.2f}"],
        [Paragraph("Maximum Depth (m)", body_style),
         f"{metrics['max_depth_m']:.2f}",
         f"-",
         f"-"],
        [Paragraph("Average Depth (m)", body_style),
         f"{metrics['avg_depth_m']:.2f}",
         f"-",
         f"-"]
    ]
    
    metrics_table = Table(metrics_rows, colWidths=[3.0*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EDF2F7")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (0,0), (0,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 15))
    
    # Visualizations Section
    story.append(Paragraph("Spatial Visualizations", heading_style))
    
    # Embed the saved matplotlib figure
    story.append(Image(img_buf, width=7.2*inch, height=6.1*inch))
    
    # Build Document with Digital Signature Checksum Footer
    import hashlib
    def get_file_hash(filepath):
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    pre_hash = get_file_hash(dem_pre_path)

    def draw_page_decorations(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor("#718096"))
        
        # Divider line
        canvas.setStrokeColor(colors.HexColor("#CBD5E0"))
        canvas.setLineWidth(0.5)
        canvas.line(36, 45, 612 - 36, 45)
        
        # Digital Signature / Dataset Hash
        sig_text = f"Digital Signature / Dataset Hash: {pre_hash}"
        canvas.drawString(36, 30, sig_text)
        
        # Page Number
        page_num = canvas.getPageNumber()
        canvas.drawRightString(612 - 36, 30, f"Page {page_num}")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page_decorations, onLaterPages=draw_page_decorations)
    print(f"Report PDF successfully written to {output_pdf_path}")
