"""
main.py
-------
FastAPI application for the Forecast Bust Detection system.

Endpoints:
  GET  /             - Health check
  GET  /api/v1/info  - Model and dataset info
  POST /api/v1/predict - Get bust probability and confidence maps
  GET  /api/v1/historical - Get historical bust events catalog

Runs with: uvicorn api.main:app --reload --port 8000
"""

import json
import os
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.predict import router as predict_router, load_model_at_startup
from api.routes.history import router as history_router


# ── App Lifespan (model loading) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model on startup, release on shutdown."""
    print("[API] Starting up — loading model...")
    load_model_at_startup()
    print("[API] Model ready.")
    yield
    print("[API] Shutting down.")


# ── FastAPI App ───────────────────────────────────────────
app = FastAPI(
    title="AI Forecast Bust Detection API",
    description=(
        "REST API for NCMRWF AI-Based Forecast Bust Detection System (SIH #26079). "
        "Provides gridded forecast confidence maps and bust probability estimates for "
        "medium-range weather forecasts over India (Day 1–Day 10)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit dashboard on same machine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(predict_router, prefix="/api/v1", tags=["Prediction"])
app.include_router(history_router, prefix="/api/v1", tags=["Historical"])


@app.get("/", tags=["Health"])
async def root():
    """Health check."""
    return {
        "status": "ok",
        "service": "AI Forecast Bust Detection API",
        "version": "1.0.0",
        "organization": "NCMRWF / MoES",
        "problem_statement": "SIH #26079",
    }


@app.get("/api/v1/info", tags=["Info"])
async def info():
    """Return model and dataset information — all strings reflect actual implementation."""
    model_loaded = False
    try:
        from api.routes.predict import _model
        model_loaded = _model is not None
    except Exception:
        pass

    return {
        "model": "ForecastBustUNet (Dual-Head Attention U-Net, 6→32→64→128→256 channels)",
        "model_loaded": model_loaded,
        "checkpoint": str(Path("checkpoints/best_model.pt").resolve()),
        "inputs": [
            "NOAA GEFSv12 control-member rainfall forecast (0.25°, APCP)",
            "Coordinate fields: latitude (norm), longitude (norm)",
            "Temporal: lead-day (norm), sin(DOY), cos(DOY)",
        ],
        "n_channels": 6,
        "outputs": [
            "Bust Probability [0, 1] per grid cell (Sigmoid head)",
            "Forecast Error Map (Softplus head, mm/day)",
            "Confidence = 100 × exp(−error/σ) × (1 − bust_prob) per cell",
        ],
        "domain": "14–32°N, 68–90°E (union bounding box: Odisha, Gangetic WB, Konkan & Goa, NW India)",
        "resolution": "0.25° × 0.25°",
        "lead_times": "Days 3, 5, 7, 10 (4 lead times)",
        "data_window": (
            "Training: Jun–Sep 2021 | Validation: Aug 2022 | Test: Jun–Jul + Sep 2022"
        ),
        "gefs_era": "GEFSv12 (available from 2020-09-01 onward)",
        "bust_definition": (
            "Bust = 1 where |GEFS forecast − IMD obs| > P90. "
            "P90 is the 90th percentile of real |forecast − observation| errors "
            "over the training window, pooled across the domain (not per-cell). "
            "Per-cell P90 requires a larger training window and is future work."
        ),
        "explainability": (
            "Integrated Gradients (hand-rolled, Captum-compatible implementation in model/explain.py). "
            "NOT using Captum library — manual Riemann approximation with n_steps=25. "
            "Top 3 meteorological drivers per prediction computed live."
        ),
        "data_source_values": {
            "live_model": "ForecastBustUNet ran on real preprocessed forecast data",
            "precomputed_cache": "Loaded from pre-saved JSON (may be live_model or illustrative)",
            "illustrative_only": "Hand-authored numbers, NOT model output — flagged in UI",
        },
        "status": (
            "Prototype — trained on 2 monsoon seasons (~120 dates). "
            "Temperature variable not implemented. Ensemble spread not implemented. "
            "See README Scope & Status section."
        ),
    }

