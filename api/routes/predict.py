"""
predict.py
----------
POST /api/v1/predict route for forecast bust prediction.

Fallback chain (no synthetic data):
  1. Precomputed JSON cache  → data_source: "precomputed_cache" | "illustrative_only"
  2. Live model inference    → data_source: "live_model"
  3. HTTP 503 error          → fails loudly, never fakes data

The data_source field in every response tells the client exactly where
the numbers came from. Illustrative data is flagged as such.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas import PredictRequest, PredictResponse, MeteoDriver, BustRegion

router = APIRouter()

# Module-level model + config cache
_model  = None
_device = None
_cfg    = None

PRECOMPUTED_DIR = Path("dashboard/precomputed")
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")


def load_model_at_startup() -> None:
    """Load trained model into memory. Called once at app startup."""
    global _model, _device, _cfg
    try:
        import torch
        sys.path.insert(0, ".")
        from model.architecture import ForecastBustUNet
        from data.config_loader import load_config

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _cfg    = load_config("config/settings.yaml")

        if CHECKPOINT_PATH.exists():
            ckpt   = torch.load(CHECKPOINT_PATH, map_location=_device)
            _model = ForecastBustUNet(
                in_channels=_cfg["model"]["in_channels"],
                encoder_channels=tuple(_cfg["model"]["encoder_channels"]),
            ).to(_device)
            _model.load_state_dict(ckpt["model_state_dict"])
            _model.eval()
            print(f"[API] Model loaded from {CHECKPOINT_PATH} on {_device}")
        else:
            print(
                f"[API] No checkpoint at {CHECKPOINT_PATH}. "
                f"Live inference unavailable. Run: python model/train.py"
            )

    except Exception as e:
        print(f"[API] Model load failed: {e}. Live inference will return 503.")
        _model = None


def _build_feature_tensor(date_str: str, lead_day: int):
    """
    Load preprocessed Zarr dataset and find the sample matching (date, lead_day).

    Returns:
        (feature_tensor, lats, lons) where feature_tensor is [1, C, H, W] torch.Tensor
        or None if the sample is not found.
    """
    try:
        import torch
        import xarray as xr
        import pandas as pd

        processed_dir = Path(_cfg["paths"]["processed_data"])

        # Try test zarr first, then val, then train
        for split in ("test", "val", "train"):
            zarr_path = processed_dir / f"dataset_{split}.zarr"
            if not zarr_path.exists():
                zarr_path = processed_dir / "dataset.zarr"
            if not zarr_path.exists():
                continue

            ds = xr.open_zarr(str(zarr_path))
            if "init_date" not in ds or "lead_day" not in ds:
                ds.close()
                continue

            dates    = pd.DatetimeIndex(ds["init_date"].values)
            leads    = ds["lead_day"].values
            target_d = pd.Timestamp(date_str).normalize()

            # Find exact match
            mask = (dates.normalize() == target_d) & (leads == lead_day)
            idxs = np.where(mask)[0]

            if len(idxs) == 0:
                ds.close()
                continue

            idx   = idxs[0]
            feats = ds["features"].values[idx]       # [C, H, W]
            lats  = ds.latitude.values.tolist()
            lons  = ds.longitude.values.tolist()
            ds.close()

            tensor = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0)  # [1, C, H, W]
            print(f"[API] Feature tensor found in {split} split, sample {idx}")
            return tensor, lats, lons

        return None, None, None

    except Exception as e:
        print(f"[API] Feature load failed: {e}")
        return None, None, None


def _detect_bust_regions(bust_np: np.ndarray, lats: list, lons: list) -> list:
    """
    Find spatially contiguous high-bust regions using connected components.
    Returns list of BustRegion objects.
    """
    try:
        from scipy.ndimage import label
        labeled, n_feat = label(bust_np > 0.5)
    except ImportError:
        return []

    regions = []
    for rid in range(1, min(n_feat + 1, 5)):
        mask = labeled == rid
        if mask.sum() < 4:
            continue
        lat_arr = np.array(lats)
        lon_arr = np.array(lons)
        lat_c   = float((lat_arr[mask.any(axis=1)].min() + lat_arr[mask.any(axis=1)].max()) / 2)
        lon_c   = float((lon_arr[mask.any(axis=0)].min() + lon_arr[mask.any(axis=0)].max()) / 2)
        regions.append(BustRegion(
            name=f"High-risk region {rid}",
            lat_center=lat_c,
            lon_center=lon_c,
            mean_bust_prob=round(float(bust_np[mask].mean()), 3),
            area_fraction=round(float(mask.mean()), 3),
        ))
    return sorted(regions, key=lambda r: r.mean_bust_prob, reverse=True)


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Generate forecast bust probability and confidence maps.

    Fallback chain:
      1. Precomputed JSON cache (honors data_source from cached JSON)
      2. Live model inference (if checkpoint loaded + preprocessed data available)
      3. HTTP 503 — fails loudly, never returns fabricated data

    The `data_source` field in the response tells you exactly where the numbers came from.
    """
    date_str = request.date
    lead_day = request.lead_day

    # ── 1. Precomputed cache ───────────────────────────────────────────────────
    cache_file = PRECOMPUTED_DIR / f"prediction_{date_str}_day{lead_day:02d}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        # Preserve data_source from JSON exactly — do NOT overwrite it
        print(f"[API] Cache hit: {cache_file} (data_source={data.get('data_source', 'unknown')})")
        return PredictResponse(**data)

    # ── 2. Live model inference ────────────────────────────────────────────────
    if _model is not None:
        try:
            import torch
            from model.architecture import compute_confidence_map
            from model.explain import (
                compute_integrated_gradients,
                summarise_attributions,
                CHANNEL_NAMES,
            )

            features, lats, lons = _build_feature_tensor(date_str, lead_day)
            if features is None:
                raise RuntimeError(
                    f"No preprocessed data for date={date_str}, lead_day={lead_day}. "
                    f"Run: python data/preprocess.py to build the dataset."
                )

            features = features.to(_device)

            with torch.no_grad():
                error_map, bust_prob = _model(features)
                conf_map = compute_confidence_map(error_map, bust_prob)

            # Integrated Gradients — real attribution, not hardcoded
            attributions = compute_integrated_gradients(
                _model, features, target="bust", n_steps=25, device=_device
            )
            drivers = summarise_attributions(
                attributions, channel_names=CHANNEL_NAMES, top_n=3
            )

            bust_np = bust_prob[0, 0].cpu().numpy()
            err_np  = error_map[0, 0].cpu().numpy()
            conf_np = conf_map[0, 0].cpu().numpy()

            return PredictResponse(
                request_date=date_str,
                lead_day=lead_day,
                variable=request.variable,
                grid_latitudes=lats,
                grid_longitudes=lons,
                bust_probability_map=bust_np.tolist(),
                error_magnitude_map=err_np.tolist(),
                confidence_map=conf_np.tolist(),
                mean_bust_probability=round(float(bust_np.mean()), 3),
                mean_confidence=round(float(conf_np.mean()), 1),
                high_bust_regions=_detect_bust_regions(bust_np, lats, lons),
                top_drivers=[
                    MeteoDriver(
                        rank=d["rank"],
                        channel_name=d["channel_name"],
                        attribution_pct=d["attribution_pct"],
                        description=d["description"],
                    )
                    for d in drivers
                ],
                event_type=None,
                data_source="live_model",
            )

        except Exception as e:
            print(f"[API] Live inference failed: {e}")
            # Fall through to 503

    # ── 3. Fail loudly — never fake data ──────────────────────────────────────
    reason = "model not loaded" if _model is None else "feature data unavailable or inference error"
    raise HTTPException(
        status_code=503,
        detail=(
            f"No prediction available for date={date_str}, lead_day={lead_day}. "
            f"Reason: {reason}. "
            f"This API does not return synthetic or fabricated data. "
            f"To fix: (1) run python data/preprocess.py, "
            f"(2) run python model/train.py, "
            f"(3) restart the API."
        ),
    )
