"""
explain_panel.py
----------------
Streamlit component for the synoptic explainability panel.

Displays:
  1. Bar chart of top meteorological drivers (Integrated Gradients attribution)
  2. Textual meteorological interpretation per event type
  3. Event type badge
  4. Operational guidance for forecasters
"""

from __future__ import annotations
from typing import Optional

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


EVENT_TYPE_CONFIG = {
    "cyclone": {
        "emoji": "🌀",
        "label": "Tropical Cyclone",
        "color": "#d62728",
        "bg": "#fde0e0",
        "description": (
            "Rapid intensification (RI) and track uncertainty are primary bust drivers. "
            "NWP models fail due to inadequate inner-core resolution, insufficient air-sea "
            "coupling, and misrepresentation of Ocean Heat Content feedback."
        ),
        "operational_note": (
            "Issue probabilistic intensity forecasts for D3–D5. Monitor ensemble spread "
            "for RI signals. Add 15–20% uncertainty envelope to wind radii forecasts."
        ),
    },
    "monsoon_depression": {
        "emoji": "🌧️",
        "label": "Monsoon Depression",
        "color": "#1f77b4",
        "bg": "#deeaf5",
        "description": (
            "Track misplacement of 50–100 km shifts heavy rainfall entirely across river basins. "
            "SW quadrant precipitation (80% of total) is extremely sensitive to vortex position. "
            "Convective parameterisation errors dominate at D3+."
        ),
        "operational_note": (
            "Monitor track uncertainty carefully — 100 km error can shift flood risk from "
            "Mahanadi to Krishna basin. Increase precipitation uncertainty by 30% for D4+."
        ),
    },
    "heat_wave": {
        "emoji": "☀️",
        "label": "Heat Wave",
        "color": "#e6550d",
        "bg": "#fde8d5",
        "description": (
            "Land–atmosphere coupling deficiency causes systematic Tmax underestimation (2–4°C) "
            "in NWP models. Soil moisture feedback and dry PBL mixing errors accumulate at D4+."
        ),
        "operational_note": (
            "Apply +2 to +4°C empirical bias correction to D4+ Tmax forecasts over NW India "
            "during April–June. Issue Heat Wave Watch when corrected Tmax > 44°C for D5."
        ),
    },
    "western_disturbance": {
        "emoji": "❌",
        "label": "Western Disturbance",
        "color": "#756bb1",
        "bg": "#e9e6f5",
        "description": (
            "Himalayan orographic interaction and phase-locking with tropical easterlies create "
            "complex precipitation patterns. Sparse high-altitude observations cause initial "
            "condition errors that amplify through D3."
        ),
        "operational_note": (
            "Increase precipitation uncertainty by 40% over J&K and HP for D3+ WD forecasts. "
            "Use orographic enhancement factors for windward slopes."
        ),
    },
    "active_break": {
        "emoji": "⇄",
        "label": "Active/Break Transition",
        "color": "#2ca02c",
        "bg": "#dff5e1",
        "description": (
            "Intraseasonal oscillation (BSISO/MJO) transitions poorly captured — break onset/withdrawal "
            "errors of 2–3 days cause widespread regional rainfall busts. "
            "Monsoon trough position is the key predictor."
        ),
        "operational_note": (
            "Issue probabilistic timing forecasts for break onset/withdrawal. "
            "Avoid deterministic day-specific rainfall predictions beyond D5 during transitions."
        ),
    },
}


def render_event_badge(event_type: Optional[str]) -> None:
    """Render a coloured event-type badge."""
    if not STREAMLIT_AVAILABLE:
        return
    if event_type and event_type in EVENT_TYPE_CONFIG:
        cfg = EVENT_TYPE_CONFIG[event_type]
        st.markdown(
            f'<div style="display:inline-block;padding:6px 14px;border-radius:20px;'
            f'background:{cfg["bg"]};border:2px solid {cfg["color"]};'
            f'font-family:Arial;font-size:14px;font-weight:bold;color:{cfg["color"]};'
            f'margin-bottom:10px;">{cfg["emoji"]} {cfg["label"]}</div>',
            unsafe_allow_html=True,
        )


def render_attribution_chart(top_drivers: list) -> None:
    """Horizontal bar chart of Integrated Gradients attribution scores."""
    if not STREAMLIT_AVAILABLE or not PLOTLY_AVAILABLE or not top_drivers:
        return

    names  = [d.get("channel_name", "?") for d in top_drivers]
    scores = [d.get("attribution_pct", 0) for d in top_drivers]
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"][: len(top_drivers)]

    fig = go.Figure(go.Bar(
        x=scores[::-1], y=names[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{s:.1f}%" for s in scores[::-1]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Attribution: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Integrated Gradients — Meteorological Driver Attribution", font=dict(size=13)),
        xaxis=dict(title="Attribution (%)", range=[0, max(scores) * 1.35 + 5], gridcolor="#eee"),
        yaxis=dict(tickfont=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        height=240, margin=dict(l=10, r=70, t=45, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_driver_descriptions(top_drivers: list, event_type: Optional[str] = None) -> None:
    """Render textual descriptions and operational guidance."""
    if not STREAMLIT_AVAILABLE:
        return

    rank_emojis = ["🥇", "🥈", "🥉"]
    st.markdown("##### Key Contributing Factors")
    for d in top_drivers:
        rank = d.get("rank", 1)
        name = d.get("channel_name", "Unknown")
        desc = d.get("description", "")
        pct  = d.get("attribution_pct", 0)
        emoji = rank_emojis[rank - 1] if rank <= 3 else f"{rank}."
        st.markdown(f"**{emoji} {name}** ({pct:.1f}%)  \n🔹 _{desc}_")

    if event_type and event_type in EVENT_TYPE_CONFIG:
        cfg = EVENT_TYPE_CONFIG[event_type]
        st.info(f"**📋 Operational Note:** {cfg['operational_note']}")
        st.caption(f"📝 {cfg['description']}")


def render_explainability_panel(
    top_drivers: list,
    event_type: Optional[str] = None,
    mean_bust_prob: float = 0.5,
    mean_confidence: float = 50.0,
    high_bust_regions: list = None,
) -> None:
    """
    Full explainability panel: badge + alert + attribution chart + driver text + region summary.
    """
    if not STREAMLIT_AVAILABLE:
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        render_event_badge(event_type)
    with col2:
        if mean_bust_prob > 0.6:
            st.error(f"🚨 High bust risk — {mean_bust_prob:.0%} probability")
        elif mean_bust_prob > 0.4:
            st.warning(f"⚠️ Moderate bust risk — {mean_bust_prob:.0%} probability")
        else:
            st.success(f"✅ Low bust risk — {mean_bust_prob:.0%} probability")

    if top_drivers:
        render_attribution_chart(top_drivers)
        render_driver_descriptions(top_drivers, event_type)

    if high_bust_regions:
        st.markdown("##### 📍 High-Risk Regions")
        for region in sorted(high_bust_regions, key=lambda r: r.get("mean_bust_prob", 0), reverse=True):
            prob  = region.get("mean_bust_prob", 0)
            name  = region.get("name", "?")
            frac  = region.get("area_fraction", 0)
            bar   = "█" * int(prob * 10) + "░" * (10 - int(prob * 10))
            st.markdown(f"`{bar}` **{name}** — {prob:.0%} bust probability, {frac:.0%} area affected")
