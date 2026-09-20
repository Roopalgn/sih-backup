# AI-Based Forecast Bust Detection
### SIH #26079 | Ministry of Earth Sciences (MoES) / NCMRWF

> **Problem:** Medium-range NWP forecasts sometimes show catastrophic errors during rapidly evolving weather events (monsoon depressions, cyclones, heat waves, western disturbances). These "forecast busts" can critically impact operational decision-making.

> **Solution:** An AI/ML system that identifies *where* and *how far in advance* a forecast is likely to fail — producing spatial confidence maps, bust probability estimates, and synoptic explainability via Integrated Gradients.

---

## 🏆 Expected Outcomes (Problem Statement #26079)

| Output | Implementation |
|---|---|
| **Forecast confidence map** | `C(x,y,τ) = 100 × exp(−Ê/σ) × (1 − P̂_bust)` per grid cell, Day 1–10 |
| **Bust probability** | ForecastBustUNet classification head, Focal Loss, threshold = P90 |
| **Error-prone area detection** | Spatial bust labels: `E(x,y,τ) > P90[historical error]` |
| **Explainable output** | Integrated Gradients (Captum) — top meteorological drivers per forecast |
| **Prototype dashboard / API** | Streamlit (4-tab dashboard) + FastAPI REST API with Swagger UI |

---

## 🌟 Architecture

```
IMD Gridded Obs (ground truth)  +  NOAA GEFS Forecasts (input)
         ↓  xarray / imdlib / s3fs
Feature Engineering
  • Error E(x,y,τ) = |Forecast_τ − Obs|
  • Bust label = 1 where E > P90(historical error per cell per lead day)
  • Coord channels (lat, lon), cyclical DOY encoding, lead-time normalisation
         ↓
Dual-Head Attention U-Net  (PyTorch)
  ├─ Head A: Error Regression  → Huber loss       → predicted |error| map
  └─ Head B: Bust Classification → Focal Loss (γ=2) → bust probability [0,1]
         ↓
Integrated Gradients (Captum) → Top meteorological drivers per prediction
         ↓
  FastAPI REST API  ←→  Streamlit Dashboard (4 tabs)
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
conda env create -f environment.yml
conda activate forecast-bust
```

### 2. Download Data
```bash
# IMD gridded observations (0.25° daily rainfall, requires no credentials)
python data/download_imd.py --variables rain tmax tmin

# NOAA GEFS forecast archives from AWS S3 (free, no credentials)
python data/download_gefs.py --method auto --start 2018-01-01 --end 2023-12-31

# ERA5 reanalysis — register free at https://cds.climate.copernicus.eu/
# then save ~/.cdsapirc with your UID:key
python data/download_era5.py --type both
```

### 3. Build ML Dataset
```bash
python data/preprocess.py --variables rain
python data/event_catalog.py     # exports dashboard/precomputed/event_catalog.json
```

### 4. Train Model
```bash
# Full training (GPU recommended)
python model/train.py --config config/settings.yaml

# 5-epoch quick test (CPU, 50 samples)
python model/train.py --fast
```

### 5. Evaluate
```bash
python model/evaluate.py --split test --checkpoint checkpoints/best_model.pt
# Outputs: results/metrics_by_lead_test.csv, results/metrics_by_subdivision_test.csv
```

### 6. Run Dashboard (works offline with precomputed cache)
```bash
streamlit run dashboard/app.py
# Opens at: http://localhost:8501
```

### 7. Run API
```bash
uvicorn api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### 8. Run Tests
```bash
python -m pytest tests/ -v
```

---

## 📊 Data Sources (all free / publicly accessible)

| Dataset | Source | Variables | Access |
|---|---|---|---|
| IMD Gridded Rainfall (0.25°) | IMD Pune via `imdlib` | Daily rainfall (mm/day) | `pip install imdlib` |
| IMD Gridded Temp (0.5°) | IMD Pune via `imdlib` | Daily Tmax, Tmin (°C) | `pip install imdlib` |
| NOAA GEFS (0.25°) | AWS S3 `noaa-gefs-pds` | Precip, T2m, Z500, U/V850 | Free, anonymous |
| ERA5 Reanalysis | Copernicus CDS | Z500, T850, CAPE, MSLP | Free registration |

---

## 🧠 Model Details

**ForecastBustUNet** — Dual-Head Attention U-Net

| Component | Detail |
|---|---|
| Encoder | 4-level Conv (64→128→256→512), BatchNorm + ReLU, MaxPool2d |
| Attention | Oktay et al. 2018 attention gates on all skip connections |
| Bottleneck | DoubleConv(1024) with Dropout |
| Decoder | ConvTranspose2d + skip-connected DoubleConv |
| Head A | Conv → ReLU → Conv → **Softplus** → error magnitude ≥ 0 |
| Head B | Conv → ReLU → Conv → **Sigmoid** → bust probability ∈ [0, 1] |
| Loss | `α × Huber(Head A) + (1−α) × FocalLoss(Head B, γ=2)` |
| Explainability | Integrated Gradients (`captum.attr.IntegratedGradients`) |

**Bust Label Definition:**
$$\text{Bust}(x,y,\tau) = \mathbb{1}\bigl(|F_\tau(x,y) - O(x,y)| > \mathcal{P}_{90}(x,y,\tau)\bigr)$$

**Confidence Indicator:**
$$C(x,y,\tau) = 100 \times \exp\!\left(-\frac{\hat{E}(x,y)}{\sigma_{\text{clim}}}\right) \times (1 - \hat{P}_{\text{bust}}(x,y))$$

---

## 🌐 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/api/v1/info` | GET | Model and dataset info |
| `/api/v1/predict` | POST | Bust probability + confidence maps |
| `/api/v1/historical` | GET | Historical bust event catalog |

**POST /api/v1/predict** example:
```json
{
  "date": "2020-05-16",
  "lead_day": 4,
  "variable": "rainfall",
  "use_cache": true
}
```

---

## 🌊 Showcase Demo Events

| Event | Date | Type | Key Finding |
|---|---|---|---|
| **Cyclone Amphan** | May 2020 | Cyclone | D3–D5 bust over Bay of Bengal / Odisha |
| **Kerala Floods** | Aug 2018 | Monsoon Depression | D5–D7 severe underestimation over Western Ghats |
| **NI Heat Wave** | May 2022 | Heat Wave | D4–D7 Tmax underestimation of 2–4°C over NW India |
| **Cyclone Fani** | May 2019 | Cyclone | RI event caused D3 bust over Bay of Bengal |
| **Delhi WD Snowfall** | Jan 2023 | Western Disturbance | Himalayan orographic bust J&K / HP |

---

## 📁 Project Structure

```
sih-backup/
├── config/
│   └── settings.yaml              ← All configuration (domain, thresholds, model)
├── data/
│   ├── config_loader.py           ← YAML config utility
│   ├── download_imd.py            ← IMD rain/temp via imdlib
│   ├── download_gefs.py           ← GEFS from AWS S3 + Open-Meteo fallback
│   ├── download_era5.py           ← ERA5 via cdsapi
│   ├── preprocess.py              ← Feature engineering, P90 bust labels, Zarr output
│   └── event_catalog.py           ← 9 curated Indian weather events
├── model/
│   ├── architecture.py            ← ForecastBustUNet (Dual-Head Attention U-Net)
│   ├── loss.py                    ← MultiTaskBustLoss (Huber + Focal)
│   ├── dataset.py                 ← PyTorch BustDataset (Zarr)
│   ├── train.py                   ← Training loop (AMP, early stopping, AUC checkpoint)
│   ├── evaluate.py                ← ROC-AUC, CSI, FAR by lead day and subdivision
│   └── explain.py                 ← Integrated Gradients (Captum)
├── api/
│   ├── main.py                    ← FastAPI app (lifespan model load, CORS)
│   ├── schemas.py                 ← Pydantic request/response schemas
│   └── routes/
│       ├── predict.py             ← POST /api/v1/predict
│       └── history.py             ← GET /api/v1/historical
├── dashboard/
│   ├── app.py                     ← Streamlit dashboard (4 tabs, offline-ready)
│   ├── components/
│   │   ├── map_widget.py          ← Folium confidence/bust map
│   │   ├── lead_time_plot.py      ← Plotly Day 1→10 trend charts
│   │   └── explain_panel.py       ← Attribution bar chart + operational guidance
│   └── precomputed/
│       └── events.json            ← 5 showcase event predictions (demo cache)
├── tests/
│   ├── test_pipeline.py           ← Data pipeline unit tests
│   └── test_model.py              ← Model / loss / confidence unit tests
├── environment.yml                ← Conda environment
└── README.md
```

---

*Built for Smart India Hackathon 2024 | Problem Statement #26079*
*Organization: Ministry of Earth Sciences (MoES) — NCMRWF*
