# AI-Based Forecast Bust Detection
### SIH #26079 | Ministry of Earth Sciences (MoES) / NCMRWF

> **Problem:** Medium-range NWP forecasts sometimes show catastrophic errors during rapidly evolving weather events (monsoon depressions, cyclones, heat waves, western disturbances). These "forecast busts" can critically impact operational decision-making.

> **Solution:** An AI/ML system that identifies *where* and *how far in advance* a forecast is likely to fail — producing spatial confidence maps, bust probability estimates, and synoptic explainability via Integrated Gradients.

---

## 📌 Scope & Status

> This is a working prototype. All numbers and maps it produces are either (a) real model output or (b) explicitly labelled `illustrative_only`. No silent substitution of fabricated data occurs.

| Dimension | Value |
|---|---|
| **Domain** | 14–32°N, 68–90°E — 4 IMD subdivisions: Odisha, Gangetic WB, Konkan & Goa, NW India |
| **Resolution** | 0.25° × 0.25° (native GEFSv12 grid) |
| **Forecast source** | NOAA GEFSv12 — control member (c00), available from **2020-09-01 onward** |
| **Ground truth** | IMD gridded daily rainfall at 0.25° |
| **Training window** | Jun–Sep 2021 |
| **Validation window** | Aug 2022 |
| **Test window** | Jun–Jul + Sep 2022 |
| **Lead times** | **Day 3, 5, 7, 10 only** — not Day 1–10 |
| **Variable** | Rainfall only (temperature pipeline not yet implemented) |
| **Bust definition** | `|GEFS forecast − IMD obs| > P90` where P90 is computed from real forecast errors over the training window |
| **Explainability** | Hand-rolled Integrated Gradients (manual Riemann sum, **not** using Captum library) |

### What's real
- ✅ P90 threshold from actual `|forecast − observation|` error distribution
- ✅ Dual-Head Attention U-Net trained on real paired forecast/obs data
- ✅ Integrated Gradients driver attribution computed live per prediction
- ✅ FastAPI and Streamlit dashboard wired together; `data_source` field in every response

### What's illustrative
- 🟡 Showcase events before model training completes — labelled `data_source: illustrative_only`
- 🟡 Dashboard maps for `illustrative_only` events — generated locally, not model output (captioned)

### Not yet implemented
- ❌ Temperature bust detection
- ❌ Ensemble spread (control member only)
- ❌ Per-cell P90 (currently pooled across domain)

---

## 🏆 Expected Outcomes (Problem Statement #26079)

| Output | Implementation |
|---|---|
| **Forecast confidence map** | `C(x,y,τ) = 100 × exp(−Ê/σ) × (1 − P̂_bust)` per grid cell, Day 3/5/7/10 |
| **Bust probability** | ForecastBustUNet classification head, Focal Loss, threshold = P90 of real forecast error |
| **Error-prone area detection** | Spatial bust labels: `E(x,y,τ) > P90[real forecast error, training window only]` |
| **Explainable output** | Integrated Gradients (hand-rolled) — top 3 meteorological drivers per prediction |
| **Prototype dashboard / API** | Streamlit (4-tab dashboard) + FastAPI REST API with Swagger UI |

---

## 🌟 Architecture

```
IMD Gridded Obs (ground truth)  +  NOAA GEFSv12 Forecasts (input)
         ↓  xarray / imdlib / s3fs (byte-range reads)
Feature Engineering
  • Error E(x,y,τ) = |Forecast_τ − Obs|
  • Bust label = 1 where E > P90(real forecast error, training window)
  • 6 channels: [forecast_precip, lat_norm, lon_norm, lead_norm, sin_doy, cos_doy]
         ↓
Dual-Head Attention U-Net  (PyTorch, in_channels=6)
  ├─ Head A: Error Regression  → Huber loss       → predicted |error| map
  └─ Head B: Bust Classification → Focal Loss (γ=2) → bust probability [0,1]
         ↓
Integrated Gradients (hand-rolled, n_steps=25) → Top 3 drivers per prediction
         ↓
  FastAPI REST API  ←→  Streamlit Dashboard (4 tabs)
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install streamlit plotly fastapi uvicorn pydantic pyyaml requests folium streamlit-folium
```

### 2. Download Data (validate with 5-date stub first)
```bash
# Validate pipeline with 5 real dates before full-season download
python data/download_gefs_stub.py

# Full monsoon season download (Jun–Sep 2021 + Jun–Sep 2022)
python data/download_gefs.py --config config/settings.yaml

# IMD observations
python data/download_imd.py --variables rain
```

> **Note:** NOAA GEFS 0.25° data (GEFSv12) is only available from **2020-09-01** onward.
> The downloader uses AWS S3 with GRIB2 index byte-range reads — no full-globe downloads.
> NOMADS (`nomads.ncep.noaa.gov`) only serves recent real-time data, not 2021 archives.

### 3. Build ML Dataset
```bash
# Full pipeline (real P90 from real forecast error)
python data/preprocess.py --config config/settings.yaml

# Quick 5-date stub validation
python data/preprocess.py --stub
```

### 4. Train Model
```bash
# Full training (~10 epochs, ~15-30 min on CPU)
python model/train.py --config config/settings.yaml
```

### 5. Start API + Dashboard
```bash
# Start API
uvicorn api.main:app --port 8000 &

# Generate real showcase events from trained model
python dashboard/generate_showcase_events.py

# Start dashboard
streamlit run dashboard/app.py --server.port 8501
# Opens at: http://localhost:8501
# Swagger UI: http://localhost:8000/docs
```

### 6. Run Tests
```bash
# Unit tests
python -m pytest tests/test_pipeline.py tests/test_model.py -v

# Integration tests (require preprocessed data + trained checkpoint)
python -m pytest tests/test_integration_end_to_end.py -v
```

---

## 📊 Data Sources (all free / publicly accessible)

| Dataset | Source | Variables | Access |
|---|---|---|---|
| IMD Gridded Rainfall (0.25°) | IMD Pune via `imdlib` | Daily rainfall (mm/day) | `pip install imdlib` |
| NOAA GEFSv12 (0.25°) | AWS S3 `noaa-gefs-pds` (2020-09-01+) | Precip (APCP), T2m | Free, anonymous S3 |

---

## 🧠 Model Details

**ForecastBustUNet** — Dual-Head Attention U-Net

| Component | Detail |
|---|---|
| Input channels | 6: [forecast_precip, lat_norm, lon_norm, lead_norm, sin_doy, cos_doy] |
| Encoder | 4-level Conv (32→64→128→256), BatchNorm + ReLU, MaxPool2d |
| Attention | Oktay et al. 2018 attention gates on all skip connections |
| Decoder | ConvTranspose2d + skip-connected DoubleConv |
| Head A | Conv → ReLU → Conv → **Softplus** → error magnitude ≥ 0 |
| Head B | Conv → ReLU → Conv → **Sigmoid** → bust probability ∈ [0, 1] |
| Loss | `α × Huber(Head A) + (1−α) × FocalLoss(Head B, γ=2)` |
| Explainability | Hand-rolled Integrated Gradients (Riemann sum, n_steps=25 — **not** using Captum) |

**Bust Label Definition:**
$$\text{Bust}(x,y,\tau) = \mathbb{1}\bigl(|F_\tau(x,y) - O(x,y)| > \mathcal{P}_{90}(\tau)\bigr)$$

where $\mathcal{P}_{90}(\tau)$ is the 90th percentile of real `|forecast − observation|` errors pooled across the training domain for lead day $\tau$. Computed from training window only (no data leakage).

**Confidence Indicator:**
$$C(x,y,\tau) = 100 \times \exp\!\left(-\frac{\hat{E}(x,y)}{\sigma_{\text{clim}}}\right) \times (1 - \hat{P}_{\text{bust}}(x,y))$$

---

## 🌐 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/api/v1/info` | GET | Model scope, bust definition, lead times, explainability method |
| `/api/v1/predict` | POST | Bust probability + confidence maps + IG driver attribution |
| `/api/v1/historical` | GET | Reference catalog of curated historical events (metadata only, not model output) |

> **Note on `/api/v1/historical`:** This endpoint returns factual event metadata (names, dates, severity) from a curated reference catalog. It is **not** model-generated output. Use `POST /api/v1/predict` for live model predictions.

**POST /api/v1/predict** example (in-training-window date):
```json
{
  "date": "2021-08-10",
  "lead_day": 5,
  "variable": "rainfall",
  "use_cache": true
}
```

Every response includes `"data_source"`:
- `"live_model"` — ForecastBustUNet ran on real GEFS data
- `"precomputed_cache"` — loaded from pre-saved JSON
- `"illustrative_only"` — placeholder, NOT model output (visibly labelled in UI)

---

## 🌊 Showcase Demo Events

Current contents of [`dashboard/precomputed/events.json`](file:///c:/Users/roopa/OneDrive/Desktop/hackathon/sih-backup/dashboard/precomputed/events.json):

| Event | Date | Lead | data_source | Note |
|---|---|---|---|---|
| Monsoon Onset 2021 | 2021-06-15 | Day 5 | `illustrative_only` | Replace by running `generate_showcase_events.py` after training |
| Peak Monsoon 2021 | 2021-08-10 | Day 5 | `illustrative_only` | Replace by running `generate_showcase_events.py` after training |
| Monsoon Break 2021 | 2021-09-15 | Day 7 | `illustrative_only` | Replace by running `generate_showcase_events.py` after training |
| Monsoon Onset 2022 | 2022-06-25 | Day 3 | `illustrative_only` | Replace by running `generate_showcase_events.py` after training |
| Cyclone Amphan OOD | 2020-05-16 | Day 4 | `illustrative_only` | Pre-GEFSv12 era — permanently illustrative |

> Amphan (May 2020) predates the GEFSv12 cutoff (2020-09-01) and cannot be run through the real pipeline. It is permanently labelled `illustrative_only`. The other 4 events will flip to `live_model` once training completes and `generate_showcase_events.py` is run.

---

## 📁 Project Structure

```
sih-backup/
├── config/
│   └── settings.yaml              ← All configuration (domain=14-32N/68-90E, lead=[3,5,7,10])
├── data/
│   ├── config_loader.py           ← YAML config + GEFSv12 date guard + channel count validation
│   ├── download_imd.py            ← IMD rain via imdlib
│   ├── download_gefs.py           ← GEFSv12 from AWS S3 (byte-range reads, not full globe)
│   ├── download_gefs_stub.py      ← 5-date stub downloader for pipeline validation
│   ├── download_era5.py           ← ERA5 via cdsapi (optional, not used in core pipeline)
│   ├── preprocess.py              ← Real P90 from forecast error, 6-channel features, Zarr output
│   └── event_catalog.py           ← 9 curated Indian weather events (reference metadata)
├── model/
│   ├── architecture.py            ← ForecastBustUNet (Dual-Head Attention U-Net, in_channels=6)
│   ├── loss.py                    ← MultiTaskBustLoss (Huber + Focal γ=2)
│   ├── dataset.py                 ← PyTorch BustDataset (auto-detects per-split Zarr files)
│   ├── train.py                   ← Training loop (AMP, early stopping, AUC checkpoint)
│   ├── evaluate.py                ← ROC-AUC, CSI, FAR by lead day and subdivision
│   └── explain.py                 ← Integrated Gradients (hand-rolled, not Captum)
├── api/
│   ├── main.py                    ← FastAPI app (lifespan model load, CORS, /api/v1/info)
│   ├── schemas.py                 ← Pydantic schemas with data_source field
│   └── routes/
│       ├── predict.py             ← POST /api/v1/predict (no synthetic fallback; 503 if unavailable)
│       └── history.py             ← GET /api/v1/historical (reference_catalog, not model output)
├── dashboard/
│   ├── app.py                     ← Streamlit dashboard (real grid from API; illustrative pattern only if no data)
│   ├── generate_showcase_events.py ← Calls live API to replace illustrative events with real model output
│   ├── components/
│   │   ├── map_widget.py          ← Folium confidence/bust map
│   │   ├── lead_time_plot.py      ← Plotly Day 3/5/7/10 trend charts
│   │   └── explain_panel.py       ← Attribution bar chart + operational guidance
│   └── precomputed/
│       └── events.json            ← Showcase events (illustrative_only until generate_showcase_events.py run)
├── tests/
│   ├── test_pipeline.py           ← Data pipeline unit tests
│   ├── test_model.py              ← Model / loss / confidence unit tests
│   └── test_integration_end_to_end.py ← Full pipeline integration test (requires data + checkpoint)
├── environment.yml                ← Conda environment (reference)
└── README.md

```

---

## ✅ Honesty Framework

Every API response and dashboard panel carries a `data_source` field:

| Value | Meaning | UI treatment |
|---|---|---|
| `live_model` | ForecastBustUNet ran on real GEFS data | ✅ green banner + "Live model output" map caption |
| `precomputed_cache` | Loaded from pre-saved real-model JSON | 📦 blue banner + "Live model output" map caption |
| `illustrative_only` | Placeholder, NOT model output | ⚠️ orange banner + "Illustrative pattern" map caption |
| `reference_catalog` | `/api/v1/historical` metadata only | Note in response body |

---

*Built for Smart India Hackathon 2026 | Problem Statement #26079*
*Organization: Ministry of Earth Sciences (MoES) — NCMRWF*
