"""
map_widget.py
-------------
Reusable Folium map component for displaying forecast confidence
and bust probability maps over India.
"""

from __future__ import annotations
import numpy as np

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

CONFIDENCE_COLORMAP = [
    (0,   "#d73027"),
    (25,  "#f46d43"),
    (50,  "#fdae61"),
    (70,  "#a6d96a"),
    (85,  "#1a9850"),
    (100, "#006837"),
]

BUST_COLORMAP = [
    (0.0, "#ffffb2"),
    (0.2, "#fed976"),
    (0.4, "#fd8d3c"),
    (0.6, "#f03b20"),
    (0.8, "#bd0026"),
    (1.0, "#7a0023"),
]


def _interpolate_color(value: float, colormap: list) -> str:
    """Linearly interpolate a hex color from a colormap list."""
    vmin, vmax = colormap[0][0], colormap[-1][0]
    v = max(vmin, min(vmax, value))
    for i in range(len(colormap) - 1):
        v0, c0 = colormap[i]
        v1, c1 = colormap[i + 1]
        if v0 <= v <= v1:
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
    high_bust_regions: list = None,
    title: str = "Forecast Confidence Map",
    mode: str = "confidence",
    subsample: int = 4,
):
    """
    Create an interactive Folium map with confidence/bust probability overlay.

    Args:
        lats: Latitude values
        lons: Longitude values
        confidence_map: [H, W] confidence values 0–100
        bust_prob_map: [H, W] bust probability 0–1
        high_bust_regions: List of region dicts with lat/lon centers
        title: Map title
        mode: 'confidence' or 'bust'
        subsample: Draw every Nth grid point (lower = more detail, slower)

    Returns:
        Folium Map object (or None if folium not installed)
    """
    if not FOLIUM_AVAILABLE:
        return None

    lats_arr = np.array(lats)
    lons_arr = np.array(lons)
    conf_arr = np.array(confidence_map)
    bust_arr = np.array(bust_prob_map)

    m = folium.Map(
        location=[22.0, 82.0],
        zoom_start=5,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Title overlay
    title_html = f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                background:rgba(255,255,255,0.92);padding:8px 18px;
                border-radius:8px;border:2px solid #333;z-index:1000;
                font-family:Arial;font-size:14px;font-weight:bold;">
        {title}
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    H, W = len(lats_arr), len(lons_arr)

    # Grid rectangles (subsampled for performance)
    for i in range(0, H, subsample):
        for j in range(0, W, subsample):
            lat = float(lats_arr[i])
            lon = float(lons_arr[j])
            conf = float(conf_arr[i, j]) if conf_arr.shape == (H, W) else 50.0
            bust = float(bust_arr[i, j]) if bust_arr.shape == (H, W) else 0.5

            if mode == "confidence":
                color = _interpolate_color(conf, CONFIDENCE_COLORMAP)
                tip = f"Lat:{lat:.2f} Lon:{lon:.2f} | Conf:{conf:.1f}% | Bust:{bust:.2f}"
            else:
                color = _interpolate_color(bust, BUST_COLORMAP)
                tip = f"Lat:{lat:.2f} Lon:{lon:.2f} | Bust:{bust:.2f} | Conf:{conf:.1f}%"

            cell = subsample * 0.25
            folium.Rectangle(
                bounds=[[lat - cell/2, lon - cell/2], [lat + cell/2, lon + cell/2]],
                color=None,
                fill=True,
                fill_color=color,
                fill_opacity=0.65,
                tooltip=tip,
            ).add_to(m)

    # High-bust region markers
    if high_bust_regions:
        for r in high_bust_regions:
            prob = r.get("mean_bust_prob", 0)
            name = r.get("name", "")
            lat_c = r.get("lat_center", 20.0)
            lon_c = r.get("lon_center", 80.0)
            area_frac = r.get("area_fraction", 0)
            if prob < 0.4:
                continue
            popup_html = f"""<div style="font-family:Arial;min-width:180px;">
                <b style="color:#d62728;">⚠ High Bust Risk</b><br>
                <b>Region:</b> {name}<br>
                <b>Bust Probability:</b> {prob:.1%}<br>
                <b>Area Affected:</b> {area_frac:.1%}
            </div>"""
            folium.Marker(
                location=[lat_c, lon_c],
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"⚠ {name}: {prob:.1%} bust prob",
                icon=folium.Icon(
                    color="red" if prob > 0.7 else "orange",
                    icon="exclamation-sign",
                    prefix="glyphicon",
                ),
            ).add_to(m)

    # Legend
    if mode == "confidence":
        vals = [0, 25, 50, 75, 100]
        lbls = ["0% (Bust)", "25%", "50%", "75%", "100% (High conf)"]
        colors = [_interpolate_color(v, CONFIDENCE_COLORMAP) for v in vals]
        leg_title = "Forecast Confidence"
    else:
        vals = [0, 0.25, 0.5, 0.75, 1.0]
        lbls = ["0 (No risk)", "0.25", "0.5", "0.75", "1.0 (Certain bust)"]
        colors = [_interpolate_color(v, BUST_COLORMAP) for v in vals]
        leg_title = "Bust Probability"

    items = "".join(
        f'<div><span style="background:{c};width:20px;height:12px;display:inline-block;'
        f'margin-right:5px;border-radius:2px;"></span>{l}</div>'
        for c, l in zip(colors, lbls)
    )
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:15px;z-index:1000;
                background:rgba(255,255,255,0.95);padding:10px 14px;
                border-radius:8px;border:1px solid #999;font-family:Arial;font-size:12px;">
        <b>{leg_title}</b><br>{items}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m
