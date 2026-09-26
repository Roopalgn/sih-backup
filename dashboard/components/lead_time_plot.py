"""
lead_time_plot.py
-----------------
Forecast Reliability Lead-Time charts matching the reference dashboard.
Grouped bar chart comparing Forecast Confidence and Bust Probability
strictly across operational horizons: Day 3, Day 5, Day 7, Day 10.
"""

from __future__ import annotations
import plotly.graph_objects as go

VALID_LEAD_DAYS = [3, 5, 7, 10]


def create_lead_time_reliability_barchart(
    confidence_vals: list[float],
    bust_vals: list[float],
    lead_days: list[int] = None,
):
    """
    Grouped bar chart matching the reference dashboard's 'FORECAST RELIABILITY (LEAD TIME)' card.
    """
    leads = [f"Day {d}" for d in (lead_days or VALID_LEAD_DAYS)]

    # Format percentages
    conf_pct = [c if c <= 100.0 else 100.0 for c in confidence_vals]
    bust_pct = [b * 100.0 if b <= 1.0 else b for b in bust_vals]

    fig = go.Figure()

    # Confidence Bars (Blue)
    fig.add_trace(go.Bar(
        name="Confidence",
        x=leads,
        y=conf_pct,
        marker=dict(
            color="#1769AA",
            line=dict(color="#123B6D", width=0.5),
        ),
        text=[f"{v:.1f}%" for v in conf_pct],
        textposition="outside",
        textfont=dict(size=10, color="#102A43", family="Inter, sans-serif"),
        hovertemplate="<b>Confidence</b><br>%{x}: <b>%{y:.1f}%</b><extra></extra>",
    ))

    # Bust Probability Bars (Coral / Red)
    fig.add_trace(go.Bar(
        name="Bust Probability",
        x=leads,
        y=bust_pct,
        marker=dict(
            color="#E5484D",
            line=dict(color="#B91C1C", width=0.5),
        ),
        text=[f"{v:.1f}%" for v in bust_pct],
        textposition="outside",
        textfont=dict(size=10, color="#102A43", family="Inter, sans-serif"),
        hovertemplate="<b>Bust Probability</b><br>%{x}: <b>%{y:.1f}%</b><extra></extra>",
    ))

    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=220,
        margin=dict(l=25, r=10, t=10, b=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=10, color="#526777", family="Inter, sans-serif"),
        ),
        xaxis=dict(
            tickfont=dict(size=11, color="#526777", family="Inter, sans-serif"),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=[0, 115],
            tickmode="array",
            tickvals=[0, 50, 100],
            ticktext=["0%", "50%", "100%"],
            tickfont=dict(size=10, color="#718596", family="Inter, sans-serif"),
            gridcolor="#EAF1F6",
            zeroline=False,
        ),
    )
    return fig
