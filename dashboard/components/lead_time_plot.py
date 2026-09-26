"""
lead_time_plot.py
-----------------
Plotly charts: forecast confidence and bust probability vs. lead day (Day 1–10)
for multiple Indian regions.
"""

from __future__ import annotations
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

REGION_COLORS = {
    "Bay of Bengal":  "#1f77b4",
    "Central India":  "#2ca02c",
    "NW India":       "#d62728",
    "NE India":       "#9467bd",
    "Peninsular India": "#8c564b",
    "Konkan & Goa":   "#e377c2",
    "Odisha":         "#7f7f7f",
    "Vidarbha":       "#bcbd22",
    "All India":      "#17becf",
}


def create_lead_time_confidence_plot(
    lead_days: list,
    confidence_by_region: dict,
    event_name: str = "",
):
    """
    Multi-region confidence vs. lead time line chart.

    Args:
        lead_days: [1, 2, ..., 10]
        confidence_by_region: {region_name: [conf_day1, ..., conf_day10]}
        event_name: Optional title suffix

    Returns:
        Plotly Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly not installed")

    fig = go.Figure()

    for region, conf_vals in confidence_by_region.items():
        color = REGION_COLORS.get(region, "#333")
        fig.add_trace(go.Scatter(
            x=lead_days, y=conf_vals,
            mode="lines+markers", name=region,
            line=dict(color=color, width=2.5),
            marker=dict(size=7),
            hovertemplate=f"<b>{region}</b><br>Day %{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    # Uncertainty band for All India
    if "All India" in confidence_by_region:
        ai = confidence_by_region["All India"]
        upper = [min(c + 12, 100) for c in ai]
        lower = [max(c - 12, 0) for c in ai]
        fig.add_trace(go.Scatter(
            x=lead_days + lead_days[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(23,190,207,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", showlegend=False,
        ))

    # Low-confidence zone shading
    fig.add_hrect(
        y0=0, y1=40,
        fillcolor="rgba(214,39,40,0.07)", layer="below", line_width=0,
        annotation_text="Low confidence zone",
        annotation_position="left",
        annotation_font=dict(size=10, color="#d62728"),
    )

    title = "Forecast Confidence by Lead Day"
    if event_name:
        title += f" — {event_name}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(
            title="Lead Day",
            tickmode="array", tickvals=lead_days,
            ticktext=[f"Day {d}" for d in lead_days],
            gridcolor="#eee",
        ),
        yaxis=dict(title="Confidence (%)", range=[0, 105], gridcolor="#eee"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        plot_bgcolor="white", paper_bgcolor="white",
        height=380, margin=dict(l=50, r=20, t=50, b=80),
        hovermode="x unified",
    )
    return fig


def create_bust_probability_trend_plot(
    lead_days: list,
    bust_prob_by_region: dict,
    event_name: str = "",
):
    """
    Multi-region bust probability vs. lead day line chart.

    Args:
        lead_days: [1, 2, ..., 10]
        bust_prob_by_region: {region_name: [bust_prob_day1, ..., bust_prob_day10]}
        event_name: Optional title suffix

    Returns:
        Plotly Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly not installed")

    fig = go.Figure()
    for region, probs in bust_prob_by_region.items():
        color = REGION_COLORS.get(region, "#333")
        fig.add_trace(go.Scatter(
            x=lead_days, y=[p * 100 for p in probs],
            mode="lines+markers", name=region,
            line=dict(color=color, width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate=f"<b>{region}</b><br>Day %{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    # 50% danger threshold
    fig.add_hline(
        y=50, line_dash="dash", line_color="red", line_width=1.5,
        annotation_text="Bust threshold (50%)",
        annotation_position="right",
        annotation_font=dict(color="red"),
    )

    title = "Bust Probability by Lead Day"
    if event_name:
        title += f" — {event_name}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(
            title="Lead Day",
            tickmode="array", tickvals=lead_days,
            ticktext=[f"Day {d}" for d in lead_days],
            gridcolor="#eee",
        ),
        yaxis=dict(title="Bust Probability (%)", range=[0, 105], gridcolor="#eee"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        plot_bgcolor="white", paper_bgcolor="white",
        height=340, margin=dict(l=50, r=20, t=50, b=80),
    )
    return fig


def generate_simulated_lead_time_data(
    event_type: str = "cyclone",
    base_bust: float = 0.6,
):
    """
    Generate realistic simulated lead-time data for a given event type.
    Used as fallback when no precomputed data is available.

    Returns:
        lead_days, confidence_by_region, bust_prob_by_region
    """
    lead_days = list(range(1, 11))

    decay_profiles = {
        "cyclone":             {"Bay of Bengal": 0.85, "Central India": 0.92, "All India": 0.93},
        "monsoon_depression":  {"Bay of Bengal": 0.90, "Central India": 0.88, "All India": 0.92},
        "heat_wave":           {"NW India": 0.86,      "Vidarbha": 0.91,      "All India": 0.94},
        "western_disturbance": {"NW India": 0.88,      "NE India": 0.96,      "All India": 0.95},
        "active_break":        {"Central India": 0.87, "NE India": 0.89,      "All India": 0.91},
    }
    profile = decay_profiles.get(event_type, {"All India": 0.93})

    conf_by_region, bust_by_region = {}, {}
    for region, decay in profile.items():
        start_conf = 88 - (1 - decay) * 50
        conf_by_region[region] = [max(5, start_conf * (decay ** (d - 1))) for d in lead_days]
        start_bust = base_bust * 0.3
        bust_by_region[region] = [min(0.95, start_bust * ((1.0 / decay) ** (d - 1))) for d in lead_days]

    return lead_days, conf_by_region, bust_by_region
