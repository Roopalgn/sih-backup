"""
map_widget.py
-------------
Folium map component for displaying regional forecast confidence over India.
Design: Professional government meteorological forecasting system (light palette,
Esri satellite basemap with administrative boundaries and clear floating legend).
"""

from __future__ import annotations
import numpy as np

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

CONFIDENCE_COLORMAP = [
    (0.0,   "#E5484D"),  # Low (≤ 40%) - Red
    (40.0,  "#E5484D"),
    (40.1,  "#F59E0B"),  # Moderate (40 - 80%) - Amber / Yellow
    (80.0,  "#F59E0B"),
    (80.1,  "#16A36A"),  # High (≥ 80%) - Green
    (100.0, "#16A36A"),
]

BUST_COLORMAP = [
    (0.00, "#16A36A"),  # Low bust risk - Green
    (0.25, "#3B82C4"),  # Moderate blue
    (0.50, "#F59E0B"),  # Moderate risk - Amber
    (0.75, "#E5484D"),  # High bust risk - Red
    (1.00, "#991B1B"),  # Severe bust risk
]

SUBDIVISIONS_BBOX = [
    {"name": "Odisha", "bounds": [[17.0, 82.0], [22.0, 88.0]], "center": [19.5, 85.0]},
    {"name": "Gangetic West Bengal", "bounds": [[21.0, 85.0], [25.0, 90.0]], "center": [23.0, 87.5]},
    {"name": "Konkan & Goa", "bounds": [[14.0, 72.0], [20.0, 76.0]], "center": [17.0, 74.0]},
    {"name": "Northwest India", "bounds": [[26.0, 68.0], [32.0, 78.0]], "center": [29.0, 73.0]},
]


def _interpolate_color(value: float, colormap: list) -> str:
    """Linearly interpolate or step a hex color from a colormap list."""
    vmin, vmax = colormap[0][0], colormap[-1][0]
    v = max(vmin, min(vmax, value))
    for i in range(len(colormap) - 1):
        v0, c0 = colormap[i]
        v1, c1 = colormap[i + 1]
        if v0 <= v <= v1:
            if v0 == v1:
                return c1
            t = (v - v0) / (v1 - v0 + 1e-10)
            r0, g0, b0 = int(c0[1:3], 16), int(c0[3:5], 16), int(c0[5:7], 16)
            r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
            r = int(r0 + t * (r1 - r0))
            g = int(g0 + t * (g1 - g0))
            b = int(b0 + t * (b1 - b0))
            return f"#{r:02x}{g:02x}{b:02x}"
    return colormap[-1][1]


def create_confidence_map(
    lats: list,
    lons: list,
    confidence_map: list,
    bust_prob_map: list,
    error_map: list = None,
    high_bust_regions: list = None,
    title: str = "Regional Forecast Confidence",
    mode: str = "confidence",
    subsample: int = 2,
):
    """
    Create the operational satellite-based Folium map matching the reference dashboard.
    """
    if not FOLIUM_AVAILABLE:
        return None

    lats_arr = np.array(lats)
    lons_arr = np.array(lons)
    conf_arr = np.array(confidence_map)
    bust_arr = np.array(bust_prob_map)
    err_arr  = np.array(error_map) if error_map is not None else None

    # Center on India
    m = folium.Map(
        location=[22.8, 80.5],
        zoom_start=5,
        tiles=None,
        control_scale=True,
    )

    # 1. Base Layer: Esri World Imagery (Satellite)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="&copy; Esri &bull; Earthstar Geographics",
        name="Satellite",
        control=False,
    ).add_to(m)

    # 2. Administrative Boundaries & Geographical Labels Overlay
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap &copy; CARTO",
        name="Boundaries & Labels",
        overlay=True,
        control=False,
    ).add_to(m)

    H, W = len(lats_arr), len(lons_arr)
    step = max(1, subsample)
    cell_deg = 0.25 * step

    # Grid Rectangles with Forecast Confidence / Bust overlay
    for i in range(0, H, step):
        for j in range(0, W, step):
            lat = float(lats_arr[i])
            lon = float(lons_arr[j])
            conf_val = float(conf_arr[i, j]) if conf_arr.shape == (H, W) else 50.0
            bust_val = float(bust_arr[i, j]) if bust_arr.shape == (H, W) else 0.5
            err_val  = float(err_arr[i, j]) if (err_arr is not None and err_arr.shape == (H, W)) else 0.0

            if mode == "confidence":
                if conf_val >= 80.0:
                    color = "#16A36A"  # High
                    fill_op = 0.72
                elif conf_val >= 40.0:
                    color = "#F59E0B"  # Moderate
                    fill_op = 0.68
                else:
                    color = "#E5484D"  # Low
                    fill_op = 0.78
            else:
                color = _interpolate_color(bust_val, BUST_COLORMAP)
                fill_op = 0.70

            tip_html = f"""
            <div style="font-family:Inter,sans-serif;font-size:11px;color:#102A43;background:#FFFFFF;padding:6px 8px;border-radius:4px;box-shadow:0 2px 6px rgba(16,42,67,0.15);border:1px solid #D9E3EA;">
                <div style="font-weight:700;color:#123B6D;margin-bottom:2px;">COORDINATE: {lat:.2f}&deg;N, {lon:.2f}&deg;E</div>
                <div>Confidence: <strong style="color:{'#16A36A' if conf_val>=80 else ('#F59E0B' if conf_val>=40 else '#E5484D')};">{conf_val:.1f}%</strong></div>
                <div>Bust Probability: <strong>{bust_val:.1%}</strong></div>
                <div>Predicted Error: <strong>{err_val:.1f} mm/day</strong></div>
            </div>
            """

            folium.Rectangle(
                bounds=[[lat - cell_deg/2, lon - cell_deg/2], [lat + cell_deg/2, lon + cell_deg/2]],
                color=None,
                fill=True,
                fill_color=color,
                fill_opacity=fill_op,
                tooltip=folium.Tooltip(tip_html, sticky=True),
            ).add_to(m)

    # Subdivision Boundary Lines
    for sub in SUBDIVISIONS_BBOX:
        folium.Rectangle(
            bounds=sub["bounds"],
            color="#FFFFFF",
            weight=1.5,
            dash_array="3, 3",
            fill=False,
            tooltip=f"{sub['name']} (IMD Subdivision)",
        ).add_to(m)

    # Floating Legend on Bottom-Left (matching reference image)
    legend_html = """
    <div style="position:fixed;bottom:24px;left:20px;z-index:1000;
                background:rgba(16,42,67,0.85);backdrop-filter:blur(4px);
                padding:10px 14px;border-radius:6px;border:1px solid rgba(255,255,255,0.2);
                font-family:'Inter',sans-serif;color:#FFFFFF;box-shadow:0 4px 12px rgba(0,0,0,0.3);min-width:140px;">
        <div style="font-size:11px;font-weight:700;color:#FFFFFF;margin-bottom:6px;letter-spacing:0.3px;">
            Forecast Confidence
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin:3px 0;font-size:11px;">
            <span style="background:#16A36A;width:12px;height:12px;display:inline-block;border-radius:2px;"></span>
            <span>High (&ge; 80%)</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin:3px 0;font-size:11px;">
            <span style="background:#F59E0B;width:12px;height:12px;display:inline-block;border-radius:2px;"></span>
            <span>Moderate (40 &ndash; 80%)</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin:3px 0;font-size:11px;">
            <span style="background:#E5484D;width:12px;height:12px;display:inline-block;border-radius:2px;"></span>
            <span>Low (&le; 40%)</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m
