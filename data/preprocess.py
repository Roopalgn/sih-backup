"""
preprocess.py
-------------
Builds the ML-ready training dataset from raw IMD and GEFS data.

Pipeline:
  1. Load raw IMD observations (ground truth) → xarray
  2. Load raw GEFS forecasts for each lead day → xarray
  3. Regrid GEFS to match IMD 0.25° grid (if needed)
  4. Compute gridded forecast error: E(x,y,τ) = |F_τ(x,y) - O(x,y)|
  5. Compute climatological 90th percentile error thresholds per cell per lead day
  6. Generate binary bust labels: bust = 1 where E > P90(x,y,τ)
  7. Assemble multi-channel input tensor with static fields
  8. Save as Zarr dataset to data/processed/dataset.zarr

Output Zarr variables:
  features[time, lead_day, channel, lat, lon]  - ML inputs
  error_map[time, lead_day, lat, lon]           - regression target
  bust_map[time, lead_day, lat, lon]            - classification target
  p90_threshold[lead_day, lat, lon]             - percentile threshold
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from config_loader import load_config


# ─────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────
def regrid_to_target(source: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    """
    Bilinearly regrid source to match target grid using xarray interpolation.
    (Use xesmf for production; this is the lightweight fallback.)
    """
    return source.interp(
        latitude=target.latitude,
        longitude=target.longitude,
        method="linear",
    )


def add_coord_channels(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate normalised lat/lon arrays matching the dataset grid.
    Returns:
        lat_norm: [H, W] in range [-1, 1]
        lon_norm: [H, W] in range [-1, 1]
    """
    lats = ds.latitude.values
    lons = ds.longitude.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lat_norm = (lat_grid - lat_grid.mean()) / (lat_grid.max() - lat_grid.min() + 1e-8)
    lon_norm = (lon_grid - lon_grid.mean()) / (lon_grid.max() - lon_grid.min() + 1e-8)
    return lat_norm.astype(np.float32), lon_norm.astype(np.float32)


def day_of_year_encoding(doy: int) -> tuple[float, float]:
    """Cyclical sine/cosine encoding for day of year."""
    angle = 2 * np.pi * doy / 365.0
    return float(np.sin(angle)), float(np.cos(angle))


def compute_bust_labels(
    error: xr.DataArray,
    p90_threshold: xr.DataArray,
) -> xr.DataArray:
    """
    Binary bust label: 1 where absolute error exceeds P90 threshold.

    Args:
        error: Gridded absolute forecast error [time, lat, lon]
        p90_threshold: Climatological 90th percentile [lat, lon]

    Returns:
        bust: Binary DataArray [time, lat, lon]
    """
    bust = (error > p90_threshold).astype(np.float32)
    bust.attrs["description"] = "1 = forecast bust (error > P90 climatological threshold)"
    return bust


# ─────────────────────────────────────────────────────
# Main Preprocessing
# ─────────────────────────────────────────────────────
def run_preprocessing(cfg: dict, variables: list[str] = None) -> None:
    """
    Full preprocessing pipeline.

    Args:
        cfg: Configuration dict from settings.yaml
        variables: List of variables to process (default: all)
    """
    if variables is None:
        variables = ["rain"]  # Start with rainfall as primary target

    domain = cfg["domain"]
    processed_dir = Path(cfg["paths"]["processed_data"])
    raw_dir = Path(cfg["paths"]["raw_data"])
    lead_days = cfg["data"]["lead_days"]
    bust_percentile = cfg["bust"]["percentile_threshold"]

    processed_dir.mkdir(parents=True, exist_ok=True)

    print("[Preprocess] Loading IMD rainfall observations...")
    # ── Load observations ──────────────────────────────────
    obs_files = sorted((raw_dir / "imd_rain").glob("*.nc"))
    if not obs_files:
        raise FileNotFoundError(f"No IMD rain NetCDF files in {raw_dir / 'imd_rain'}")

    obs_ds = xr.open_mfdataset(
        obs_files,
        combine="by_coords",
        engine="netcdf4",
    )

    # Standardise coordinate names
    for old, new in [("lat", "latitude"), ("lon", "longitude")]:
        if old in obs_ds.coords:
            obs_ds = obs_ds.rename({old: new})

    rain_obs = obs_ds["rain"] if "rain" in obs_ds else list(obs_ds.data_vars)[0]
    print(f"[Preprocess] Observations: {rain_obs.dims} | shape={rain_obs.shape}")

    # ── Load GEFS forecasts ────────────────────────────────
    print("[Preprocess] Loading GEFS forecasts...")
    gefs_dir = raw_dir / "gefs"
    gefs_files = sorted(gefs_dir.glob("*.nc"))
    if not gefs_files:
        print("[Preprocess] WARNING: No GEFS files found. Generating synthetic placeholder for structure testing.")
        # Create synthetic GEFS data matching obs grid for testing
        _create_synthetic_gefs(obs_ds, gefs_dir, lead_days)
        gefs_files = sorted(gefs_dir.glob("*.nc"))

    # ── Compute P90 threshold (training years only) ────────
    print(f"[Preprocess] Computing P{bust_percentile} error thresholds (training data)...")
    train_years = cfg["data"]["train_years"]
    train_obs = rain_obs.sel(time=rain_obs.time.dt.year.isin(train_years))

    # For each lead day, compute climatological percentile error
    # Here we use observed daily variability as a proxy for climatological spread
    p90_maps = {}
    for lead_day in lead_days:
        # Seasonal P90 of daily rain anomalies (proxy for error percentile)
        clim_mean = train_obs.groupby("time.dayofyear").mean()
        anomalies = train_obs.groupby("time.dayofyear") - clim_mean
        abs_anomalies = np.abs(anomalies)
        # Scale by lead day (errors grow with lead time)
        lead_scaling = 1.0 + 0.1 * (lead_day - 1)
        p90_map = abs_anomalies.quantile(bust_percentile / 100.0, dim="time") * lead_scaling
        p90_maps[lead_day] = p90_map.values.astype(np.float32)

    # ── Assemble ML dataset ────────────────────────────────
    print("[Preprocess] Assembling ML features and labels...")
    lat_norm, lon_norm = add_coord_channels(obs_ds)

    all_dates = pd.DatetimeIndex(
        [
            pd.Timestamp(f"{y}-{m:02d}-{d:02d}")
            for y in range(min(train_years), max(cfg["data"]["test_years"]) + 1)
            for m in range(1, 13)
            for d in range(1, 32)
            if _valid_date(y, m, d)
        ]
    )

    feature_list = []
    error_list = []
    bust_list = []
    date_list = []
    lead_day_list = []

    lats = obs_ds.latitude.values
    lons = obs_ds.longitude.values
    H, W = len(lats), len(lons)

    for date in tqdm(all_dates, desc="Processing dates"):
        try:
            obs_day = rain_obs.sel(time=date, method="nearest").values.astype(np.float32)
        except KeyError:
            continue

        doy = date.dayofyear
        sin_doy, cos_doy = day_of_year_encoding(doy)

        for lead_day in lead_days:
            # Try to load GEFS forecast for this date and lead
            gefs_file = gefs_dir / f"gefs_{date.strftime('%Y%m%d')}_f{lead_day*24:03d}.nc"
            if not gefs_file.exists():
                # Try synthetic or skip
                synth_file = gefs_dir / f"synth_{date.strftime('%Y%m%d')}.nc"
                if not synth_file.exists():
                    continue
                gefs_file = synth_file

            try:
                gefs_ds = xr.open_dataset(gefs_file)
                # Find precipitation variable
                precip_var = _find_var(gefs_ds, ["tp", "precip", "apcp", "precipitation_sum", "APCP_surface"])
                if precip_var is None:
                    gefs_ds.close()
                    continue

                forecast_field = gefs_ds[precip_var]
                # Regrid to obs grid if needed
                if "latitude" not in forecast_field.coords or len(forecast_field.latitude) != H:
                    forecast_field = regrid_to_target(
                        forecast_field.to_dataset(),
                        obs_ds,
                    )[precip_var]

                fcast = forecast_field.values.astype(np.float32)
                if fcast.ndim > 2:
                    fcast = fcast[0]  # Take first time slice if stacked
                fcast = fcast[:H, :W]  # Ensure matching shape

                # Compute absolute error
                error = np.abs(fcast - obs_day)

                # Generate bust label
                p90 = p90_maps[lead_day]
                bust = (error > p90).astype(np.float32)

                # Build feature channels:
                # [0] Forecast precipitation
                # [1] Lat (normalised)
                # [2] Lon (normalised)
                # [3] Lead day (normalised 0-1)
                # [4] sin(day_of_year)
                # [5] cos(day_of_year)
                lead_norm = (lead_day - 1) / 9.0  # Normalise to [0, 1]
                sin_ch = np.full((H, W), sin_doy, dtype=np.float32)
                cos_ch = np.full((H, W), cos_doy, dtype=np.float32)
                lead_ch = np.full((H, W), lead_norm, dtype=np.float32)

                feature = np.stack(
                    [fcast, lat_norm, lon_norm, lead_ch, sin_ch, cos_ch],
                    axis=0,
                )  # [6, H, W]

                feature_list.append(feature)
                error_list.append(error[np.newaxis])  # [1, H, W]
                bust_list.append(bust[np.newaxis])  # [1, H, W]
                date_list.append(date)
                lead_day_list.append(lead_day)

                gefs_ds.close()

            except Exception as e:
                print(f"[Preprocess] Skip {date.date()} lead={lead_day}: {e}")
                continue

    if not feature_list:
        print("[Preprocess] WARNING: No samples assembled. Check data paths.")
        return

    print(f"[Preprocess] Assembled {len(feature_list)} samples | shape={feature_list[0].shape}")

    # ── Save as Zarr ───────────────────────────────────────
    features_arr = np.stack(feature_list, axis=0)  # [N, C, H, W]
    errors_arr = np.stack(error_list, axis=0)  # [N, 1, H, W]
    busts_arr = np.stack(bust_list, axis=0)  # [N, 1, H, W]

    out_ds = xr.Dataset(
        {
            "features": (["sample", "channel", "latitude", "longitude"], features_arr),
            "error_map": (["sample", "channel", "latitude", "longitude"], errors_arr),
            "bust_map": (["sample", "channel", "latitude", "longitude"], busts_arr),
        },
        coords={
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "description": "SIH #26079 Forecast Bust Detection ML Dataset",
            "n_samples": len(feature_list),
            "channels": "[forecast_precip, lat_norm, lon_norm, lead_norm, sin_doy, cos_doy]",
            "bust_threshold": f"P{bust_percentile} of climatological error per cell per lead day",
        },
    )

    # Add date and lead_day as coordinate arrays
    out_ds["init_date"] = xr.DataArray(pd.DatetimeIndex(date_list), dims=["sample"])
    out_ds["lead_day"] = xr.DataArray(lead_day_list, dims=["sample"])

    zarr_path = processed_dir / "dataset.zarr"
    out_ds.to_zarr(zarr_path, mode="w")
    print(f"[Preprocess] Saved ML dataset: {zarr_path}")
    print(f"[Preprocess] Bust rate: {busts_arr.mean():.3f} ({busts_arr.mean()*100:.1f}%)")

    # ── Save P90 threshold maps ────────────────────────────
    p90_arr = np.stack([p90_maps[d] for d in lead_days], axis=0)  # [L, H, W]
    p90_ds = xr.Dataset(
        {"p90_threshold": (["lead_day", "latitude", "longitude"], p90_arr)},
        coords={"lead_day": lead_days, "latitude": lats, "longitude": lons},
    )
    p90_path = processed_dir / "p90_thresholds.nc"
    p90_ds.to_netcdf(p90_path)
    print(f"[Preprocess] Saved P90 thresholds: {p90_path}")


def _find_var(ds: xr.Dataset, candidates: list[str]) -> Optional[str]:
    """Find the first matching variable name in a dataset."""
    for c in candidates:
        if c in ds.data_vars:
            return c
    return None


def _valid_date(y: int, m: int, d: int) -> bool:
    """Check if date is valid."""
    import calendar
    return d <= calendar.monthrange(y, m)[1]


def _create_synthetic_gefs(obs_ds: xr.Dataset, output_dir: Path, lead_days: list) -> None:
    """
    Create synthetic GEFS-like NetCDF files for pipeline testing.
    Generates Gaussian noise + seasonal cycle on the obs grid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lats = obs_ds.latitude.values
    lons = obs_ds.longitude.values
    H, W = len(lats), len(lons)

    print("[Preprocess] Generating synthetic GEFS data for pipeline testing...")
    dates = pd.date_range("2022-06-01", "2022-08-31", freq="D")
    for date in dates:
        ds_out = xr.Dataset(
            {"tp": (["latitude", "longitude"], np.random.exponential(5.0, size=(H, W)).astype(np.float32))},
            coords={"latitude": lats, "longitude": lons},
            attrs={"note": "Synthetic data for testing"},
        )
        synth_path = output_dir / f"synth_{date.strftime('%Y%m%d')}.nc"
        ds_out.to_netcdf(synth_path)


def main():
    parser = argparse.ArgumentParser(description="Preprocess NWP + IMD data for ML")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--variables", nargs="+", default=["rain"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_preprocessing(cfg, args.variables)


if __name__ == "__main__":
    main()
