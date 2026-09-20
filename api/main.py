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
    """Return model and dataset information."""
    return {
        "model": "ForecastBustUNet (Dual-Head Attention U-Net)",
        "inputs": ["GFS/GEFS Forecast Fields", "IMD Observations (ground truth for training)"],
        "outputs": [
            "Bust Probability Map [0-1] per grid cell",
            "Forecast Error Map (predicted RMSE)",
            "Confidence Indicator [0-100%] per grid cell",
        ],
        "domain": "Indian Subcontinent (6°N-38°N, 68°E-98°E) at 0.25° resolution",
        "lead_times": "Day 1 to Day 10",
        "bust_definition": "P90 of climatological gridded forecast error per cell and lead day",
        "explainability": "Integrated Gradients (Captum) — top meteorological drivers per prediction",
    }
