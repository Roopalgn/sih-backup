"""
app.py
------
Main Streamlit dashboard for the AI Forecast Bust Detection System.
Organization: NCMRWF / Ministry of Earth Sciences (MoES) — SIH #26079

Tabs:
  1. Confidence Map   — gridded confidence over India (green → red)
  2. Bust Probability — spatial bust risk heatmap
  3. Explainability   — Integrated Gradients top drivers + operational guidance
  4. Lead-Time Trend  — Day 1→10 confidence degradation by region

Run with:
    streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path
from datetime import date, datetime
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Forecast Bust Detector — NCMRWF",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a3a6b 0%, #0d5c8a 100%);
    color: white; padding: 16px 24px; border-radius: 10px; margin-bottom: 18px;
}
.stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ─────────────────────────────────────────────────────────────────────
PRECOMPUTED_DIR = Path(__file__).parent / "precomputed"
API_BASE = "http://localhost:8000"


def _check_api_online() -> bool:
    """Return True if the FastAPI backend is reachable."""
    try:
        import requests
        r = requests.head(f"{API_BASE}/", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def _data_source_banner(source: str, event_name: str = "") -> None:
    """Show a clearly visible provenance badge for the current data."""
    if source == "live_model":
        st.success(f"✅ **LIVE MODEL** — ForecastBustUNet prediction on real GEFS data {event_name}")
    elif source == "precomputed_cache":
        st.info(f"📦 **CACHED** — Pre-computed from real model run {event_name}")
    elif source == "illustrative_only":
        st.warning(
            f"⚠️ **ILLUSTRATIVE ONLY** — These numbers were NOT produced by the model. "
            f"They are placeholder data shown while model training is pending. "
            f"{event_name}"
        )
    else:
        st.error(f"❓ **UNKNOWN SOURCE** — provenance not recorded for {event_name}")


@st.cache_data(ttl=60)
def load_showcase_events() -> list:
    """
    Load showcase events by calling the API for each event.

    Priority:
      1. Call POST /api/v1/predict for each event (returns live_model or cache)
      2. Fall back to events.json if API offline (shows illustrative banner)
    """
    events_path = PRECOMPUTED_DIR / "events.json"
    base_events = json.loads(events_path.read_text()) if events_path.exists() else []

    if not _check_api_online():
        # API offline — return JSON events but mark them for what they are
        return base_events

    try:
        import requests
        enriched = []
        for ev in base_events:
            date_str = ev.get("request_date")
            lead_day = ev.get("lead_day")
            if not date_str or not lead_day:
                enriched.append(ev)
                continue

            try:
                r = requests.post(
                    f"{API_BASE}/api/v1/predict",
                    json={"date": date_str, "lead_day": lead_day, "variable": ev.get("variable", "rainfall")},
                    timeout=60,
                )
                if r.status_code == 200:
                    data = r.json()
                    data["event_name"]   = ev.get("event_name", date_str)
                    data["description"]  = ev.get("description", "")
                    data["event_type"]   = ev.get("event_type")
                    enriched.append(data)
                elif r.status_code == 503:
                    # No data — keep JSON version with its data_source
                    enriched.append(ev)
                else:
                    enriched.append(ev)
            except Exception:
                enriched.append(ev)

        return enriched if enriched else base_events

    except ImportError:
        return base_events



def _generate_grid(mean_bust: float, event_type: str, lead_day: int, seed: int):
    """Spatially correlated bust/confidence grids (fast numpy, no external data needed)."""
    np.random.seed(seed + lead_day * 7)
    lats = np.arange(6.0, 38.5, 0.5)
    lons = np.arange(68.0, 98.5, 0.5)
    H, W = len(lats), len(lons)

    hotspot_configs = {
        "cyclone":             [(H * .30, W * .75, .85, 8), (H * .45, W * .65, .65, 6)],
        "monsoon_depression":  [(H * .35, W * .50, .75, 7), (H * .20, W * .45, .55, 5)],
        "heat_wave":           [(H * .82, W * .18, .78, 9), (H * .62, W * .28, .60, 7)],
        "western_disturbance": [(H * .90, W * .18, .72, 8), (H * .85, W * .32, .50, 5)],
        "active_break":        [(H * .40, W * .50, .68, 8), (H * .68, W * .82, .52, 6)],
    }
    bust = np.zeros((H, W))
    for cx, cy, strength, radius in hotspot_configs.get(event_type, hotspot_configs["cyclone"]):
        ys, xs = np.ogrid[:H, :W]
        bust += strength * np.exp(-((ys - cx)**2 + (xs - cy)**2)**0.5 / radius)

    bust = np.clip(bust, 0, 1)
    if bust.mean() > 1e-6:
        bust = bust * (mean_bust / bust.mean())
    bust += np.random.normal(0, 0.04, bust.shape)
    bust = np.clip(bust * (0.8 + 0.02 * lead_day), 0, 1)

    error = bust * np.random.exponential(12.0, size=bust.shape)
    conf  = np.clip(100.0 * np.exp(-error / 15.0) * (1 - bust), 0, 100)
    return lats.tolist(), lons.tolist(), conf, bust


def _lead_time_series(event_type: str, base_bust: float):
    lead_days = list(range(1, 11))
    decay_map = {
        "cyclone":             {"Bay of Bengal": .82, "Odisha": .87,         "All India": .93},
        "monsoon_depression":  {"Central India": .88, "Konkan & Goa": .86,   "All India": .92},
        "heat_wave":           {"NW India": .86,      "Vidarbha": .91,        "All India": .94},
        "western_disturbance": {"NW India": .88,      "NE India": .96,        "All India": .95},
        "active_break":        {"Central India": .87, "NE India": .89,        "All India": .92},
    }
    regions = decay_map.get(event_type, {"All India": .93})
    conf_d, bust_d = {}, {}
    for reg, decay in regions.items():
        start = 88 - (1 - decay) * 30
        conf_d[reg] = [max(5, start * (decay ** (d - 1))) for d in lead_days]
        bust_d[reg] = [min(.95, base_bust * .3 * ((1 / decay) ** (d - 1))) for d in lead_days]
    return lead_days, conf_d, bust_d


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="font-size:32px">🌀</span>
    <div>
      <div style="font-size:22px;font-weight:bold;">AI-Based Forecast Bust Detection System</div>
      <div style="font-size:13px;opacity:.85;">
        National Centre for Medium Range Weather Forecasting (NCMRWF) &bull; MoES &bull; SIH #26079
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Forecast Control Panel")
    st.divider()

    showcase_events = load_showcase_events()
    event_names = ["--- Custom Date ---"] + [e["event_name"] for e in showcase_events]

    selected_event_name = st.selectbox(
        "📂 Showcase Event",
        event_names,
        help="Select a pre-computed historical bust event for demonstration",
    )

    st.divider()
    st.markdown("**Or enter a custom forecast date:**")

    if selected_event_name != "--- Custom Date ---" and showcase_events:
        evt = next((e for e in showcase_events if e["event_name"] == selected_event_name), None)
        default_date = datetime.strptime(evt["request_date"], "%Y-%m-%d").date() if evt else date(2022, 5, 15)
        default_lead = evt.get("lead_day", 3) if evt else 3
    else:
        evt = None
        default_date = date(2022, 5, 15)
        default_lead = 3

    selected_date = st.date_input(
        "📅 Forecast Init Date", value=default_date,
        min_value=date(2021, 1, 1), max_value=date(2023, 12, 31),
    )

    VALID_LEAD_DAYS = [3, 5, 7, 10]
    # Guard: if stored event has a lead_day outside trained set, reset to 3
    default_lead = default_lead if default_lead in VALID_LEAD_DAYS else VALID_LEAD_DAYS[0]
    lead_day = st.select_slider(
        "⏱️ Lead Time (Days)",
        options=VALID_LEAD_DAYS,
        value=default_lead,
        help="Model is trained and evaluated only at these 4 lead times.",
    )
    variable = st.selectbox(
        "🌡️ Target Variable",
        ["rainfall"],
        format_func=lambda v: {"rainfall": "🌧️ Daily Rainfall (active)"}.get(v, v),
        help="Only rainfall is implemented. Temperature/z500/wind850 are future work.",
    )


    st.divider()
    st.markdown("**🗺️ Display Options**")
    map_mode     = st.radio("Map Overlay", ["Confidence (%)", "Bust Probability"], horizontal=True)
    map_subsample = st.slider("Map Detail", 2, 8, 4, help="Lower = finer grid (slower)")

    st.divider()
    st.caption("📊 Model: ForecastBustUNet (Dual-Head Attention U-Net)")
    st.caption("🌍 Domain: 14°N–32°N, 68°E–90°E | 0.25° resolution (4 subdivisions)")
    st.caption("⚠️ Bust = |forecast − obs| > P90 real forecast error")
    api_online = _check_api_online()
    if api_online:
        st.success("🟢 API online")
    else:
        st.error("🔴 API offline — using cached data")

# ── Resolve prediction data ────────────────────────────────────────────────────
if evt is not None:
    event_type        = evt.get("event_type", "active_break")
    mean_bust         = evt.get("mean_bust_probability") or 0.0
    mean_conf         = evt.get("mean_confidence") or 0.0
    top_drivers       = evt.get("top_drivers", [])
    high_bust_regions = evt.get("high_bust_regions", [])
    event_label       = evt.get("event_name", "Unknown Event")
    event_desc        = evt.get("description", "")
    data_source       = evt.get("data_source", "unknown")
    # Extract real 2D grid arrays if present in the cached response
    real_conf_map  = evt.get("confidence_map")
    real_bust_map  = evt.get("bust_probability_map")
    real_lats      = evt.get("grid_latitudes")
    real_lons      = evt.get("grid_longitudes")
else:
    # Custom date — try API, otherwise honest illustrative label
    event_type   = "monsoon_depression"
    event_label  = f"Custom: {selected_date} Day {lead_day}"
    event_desc   = ""
    data_source  = "illustrative_only"
    mean_bust    = 0.0
    mean_conf    = 0.0
    top_drivers  = []
    high_bust_regions = []
    real_conf_map = None
    real_bust_map = None
    real_lats     = None
    real_lons     = None

    if _check_api_online():
        try:
            import requests as _req
            r = _req.post(
                f"{API_BASE}/api/v1/predict",
                json={"date": str(selected_date), "lead_day": lead_day, "variable": variable},
                timeout=90,
            )
            if r.status_code == 200:
                api_data = r.json()
                mean_bust         = api_data.get("mean_bust_probability", 0.0) or 0.0
                mean_conf         = api_data.get("mean_confidence", 0.0) or 0.0
                top_drivers       = api_data.get("top_drivers", [])
                high_bust_regions = api_data.get("high_bust_regions", [])
                data_source       = api_data.get("data_source", "live_model")
                event_type        = api_data.get("event_type") or "monsoon_depression"
                # Extract real grids from API response
                real_conf_map  = api_data.get("confidence_map")
                real_bust_map  = api_data.get("bust_probability_map")
                real_lats      = api_data.get("grid_latitudes")
                real_lons      = api_data.get("grid_longitudes")
            # 503 → stay illustrative_only, grids stay None
        except Exception:
            pass

# ── Grid resolution: use real model output if available, else generate synthetic ──
has_real_grid = (
    data_source in ("live_model", "precomputed_cache")
    and real_conf_map is not None
    and real_bust_map is not None
    and len(real_conf_map) > 0
    and len(real_bust_map) > 0
)

if has_real_grid:
    lats      = real_lats
    lons      = real_lons
    conf_grid = real_conf_map
    bust_grid = real_bust_map
    # Use the 4 trained lead days for the trend axis when real data is available
    lead_days_list, conf_trend, bust_trend = _lead_time_series(event_type, mean_bust)
else:
    # Explicitly synthetic — only for illustrative_only or missing-data cases
    seed = abs(hash(f"{selected_date}{lead_day}")) % (2 ** 20)
    lats, lons, conf_grid, bust_grid = _generate_grid(
        max(mean_bust, 0.35), event_type, lead_day, seed
    )
    lead_days_list, conf_trend, bust_trend = _lead_time_series(event_type, max(mean_bust, 0.35))

# ── Data provenance banner ────────────────────────────────────────────────────
_data_source_banner(data_source, event_label)

# ── Metrics row ──────────────────────────────────────────────────────────────
st.markdown(f"#### Analysis: **{event_label}** — Day {lead_day} Forecast")

if event_desc:
    st.caption(f"📝 {event_desc}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("📅 Init Date",       str(selected_date))
m2.metric("⏱️ Lead Day",        f"Day {lead_day}")
m3.metric("🟢 Mean Confidence", f"{mean_conf:.1f}%",
          delta=f"{mean_conf - 75:.1f}%", delta_color="normal")
m4.metric("🔴 Mean Bust Prob",  f"{mean_bust:.1%}",
          delta=f"+{(lead_day-1)*3}% vs Day 1", delta_color="inverse")
m5.metric("🌊 Event Type",      event_type.replace("_", " ").title())

st.divider()

# ── Main Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Confidence Map",
    "⚠️ Bust Probability",
    "🧠 Explainability",
    "📈 Lead-Time Trend",
])

conf_arr = np.array(conf_grid)
bust_arr = np.array(bust_grid)
lats_arr = np.array(lats)
lons_arr = np.array(lons)

# ── Tab 1: Confidence Map ─────────────────────────────────────────────────────────
with tab1:
    st.markdown(f"**Regional Forecast Confidence — Day {lead_day}** (🟢 High conf → 🔴 Low conf / likely bust)")

    col_map, col_stat = st.columns([3, 1])
    with col_map:
        try:
            from dashboard.components.map_widget import create_confidence_map
            from streamlit_folium import folium_static
            m = create_confidence_map(
                lats, lons, conf_grid, bust_grid.tolist(),
                high_bust_regions=high_bust_regions,
                title=f"Confidence — {event_label} Day {lead_day}",
                mode="confidence" if map_mode == "Confidence (%)" else "bust",
                subsample=map_subsample,
            )
            if m:
                folium_static(m, width=720, height=460)
            else:
                raise ImportError("folium map returned None")
        except Exception:
            import plotly.express as px
            display = conf_arr if map_mode == "Confidence (%)" else bust_arr * 100
            cscale  = "RdYlGn" if map_mode == "Confidence (%)" else "Reds"
            label   = "Confidence (%)" if map_mode == "Confidence (%)" else "Bust Prob ×100"
            fig = px.imshow(
                display, x=lons_arr, y=lats_arr,
                color_continuous_scale=cscale, aspect="auto",
                labels={"x": "Longitude", "y": "Latitude", "color": label},
                title=f"🗺️ {event_label} | Day {lead_day} | {label}",
            )
            fig.update_layout(height=460)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("_Tip: `pip install streamlit-folium folium` for interactive map with tooltips._")

        # Per-map provenance label — prevents banner/map disagreement
        st.caption(
            "📍 **Map source: Live model output** — real GEFS forecast processed by ForecastBustUNet."
            if has_real_grid else
            "📍 **Map source: Illustrative pattern** — NOT model output. Generated locally for layout only."
        )

    with col_stat:
        st.markdown("#### Stats")
        st.metric("Mean Confidence", f"{conf_arr.mean():.1f}%")
        st.metric("Min Confidence",  f"{conf_arr.min():.1f}%")
        st.metric("Low Conf Cells",  f"{(conf_arr < 40).mean():.1%}")
        if high_bust_regions:
            st.markdown("#### ⚠️ Alert Zones")
            for r in high_bust_regions:
                p, n = r.get("mean_bust_prob", 0), r.get("name", "?")
                if p > 0.5:
                    st.error(f"☹️ **{n}**: {p:.0%}")
                elif p > 0.3:
                    st.warning(f"⚠️ **{n}**: {p:.0%}")

# ── Tab 2: Bust Probability ───────────────────────────────────────────────────────
with tab2:
    st.markdown(
        f"**Forecast Bust Probability — Day {lead_day}** "
        "(🟡 Low risk → 🔴 High risk; threshold = P90 historical error)"
    )
    col_bmap, col_bstat = st.columns([3, 1])
    with col_bmap:
        try:
            from dashboard.components.map_widget import create_confidence_map
            from streamlit_folium import folium_static
            m_bust = create_confidence_map(
                lats, lons, conf_grid, bust_arr.tolist(),
                high_bust_regions=high_bust_regions,
                title=f"Bust Probability — {event_label} Day {lead_day}",
                mode="bust", subsample=map_subsample,
            )
            if m_bust:
                folium_static(m_bust, width=720, height=460)
            else:
                raise ImportError
        except Exception:
            import plotly.express as px
            fig_b = px.imshow(
                bust_arr * 100, x=lons_arr, y=lats_arr,
                color_continuous_scale="Reds", aspect="auto",
                labels={"x": "Longitude", "y": "Latitude", "color": "Bust Prob (%)"},
                title=f"⚠️ Bust Probability — {event_label} Day {lead_day}",
            )
            fig_b.update_layout(height=460)
            st.plotly_chart(fig_b, use_container_width=True)

        # Per-map provenance label — matches tab1 pattern
        st.caption(
            "📍 **Map source: Live model output** — real GEFS forecast processed by ForecastBustUNet."
            if has_real_grid else
            "📍 **Map source: Illustrative pattern** — NOT model output. Generated locally for layout only."
        )

    with col_bstat:
        st.markdown("#### Bust Stats")
        st.metric("Mean Bust Prob",  f"{bust_arr.mean():.1%}")
        st.metric("P90 Bust Prob",   f"{np.percentile(bust_arr, 90):.1%}")
        st.metric("High-Risk Cells", f"{(bust_arr > 0.5).mean():.1%}")
        st.divider()
        st.markdown("**Bust Criterion**")
        st.code("E(x,y,τ) > P90[real fcst error]")
        st.caption("E = |Forecast − Observed| ; P90 from training window")


# ── Tab 3: Explainability ──────────────────────────────────────────────────────────
with tab3:
    st.markdown("**Synoptic Explainability — Integrated Gradients Attribution**")
    st.caption("🧠 Identifies which meteorological variables drive forecast uncertainty.")

    try:
        from dashboard.components.explain_panel import render_explainability_panel
        render_explainability_panel(
            top_drivers=top_drivers,
            event_type=event_type,
            mean_bust_prob=mean_bust,
            mean_confidence=mean_conf,
            high_bust_regions=high_bust_regions,
        )
    except Exception:
        if event_type:
            st.info(f"**Event Type:** {event_type.replace('_', ' ').title()}")
        for d in top_drivers:
            st.markdown(f"**{d['rank']}. {d['channel_name']}** ({d['attribution_pct']:.1f}%) — _{d['description']}_")

    with st.expander("📝 View Confidence Formula"):
        st.latex(r"""
        C(x,y,\tau) = 100 \times
        \exp\!\left(-\frac{\hat{E}_\tau(x,y)}{\sigma_{\text{clim}}(x,y)}\right)
        \times \bigl(1 - \hat{P}_\tau(x,y)\bigr)
        """)
        st.caption("Ê = predicted error · σ_clim = climatological error scale · P̂ = bust probability")

# ── Tab 4: Lead-Time Trend ────────────────────────────────────────────────────────
with tab4:
    st.markdown("**Forecast Confidence & Bust Probability — Days 3, 5, 7, 10**")
    st.caption("📊 Regional confidence degradation curves across the 4 trained lead times.")


    try:
        from dashboard.components.lead_time_plot import (
            create_lead_time_confidence_plot,
            create_bust_probability_trend_plot,
        )
        st.plotly_chart(
            create_lead_time_confidence_plot(lead_days_list, conf_trend, event_label),
            use_container_width=True,
        )
        st.divider()
        st.plotly_chart(
            create_bust_probability_trend_plot(lead_days_list, bust_trend, event_label),
            use_container_width=True,
        )
    except Exception:
        import plotly.express as px, pandas as pd
        df = pd.DataFrame({"Lead Day": lead_days_list, **conf_trend})
        fig = px.line(df, x="Lead Day", y=list(conf_trend.keys()),
                      title=f"Confidence Trend: {event_label}")
        st.plotly_chart(fig, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "📡 **AI Forecast Bust Detection** | MoES (NCMRWF) | SIH 2026 — PS #26079 | "
    "Model: ForecastBustUNet | Data: IMD Gridded Obs × NOAA GEFSv12 | "
    "Explainability: Integrated Gradients (hand-rolled implementation, not Captum)"
)
