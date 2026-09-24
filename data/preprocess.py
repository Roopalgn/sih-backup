"""
preprocess.py
-------------
Builds the ML-ready training dataset from real IMD observations and GEFSv12 forecasts.

Pipeline:
  1. Load IMD observations (ground truth) → xarray
  2. Load GEFS forecasts for each lead day → xarray
  3. Compute gridded FORECAST ERROR: E(x,y,τ) = |F_τ(x,y) − O(x,y)|
  4. Compute P90 forecast error threshold (training data ONLY)
  5. Generate binary bust labels: bust(x,y,τ) = 1 where E > P90(τ)
  6. Assemble 6-channel input tensor
  7. Save as Zarr dataset to data/processed/dataset.zarr

Bust Definition (CORRECT):
  P90(τ) = 90th percentile of |forecast − observation| error over training window,
            computed from real paired forecast/obs data.
            If fewer than min_samples_per_cell samples exist for a lead day,
            P90 is pooled across all subdivisions (not per-cell).
            This is stated explicitly in /api/v1/info and README.

Feature channels (6 total, matching in_channels: 6 in config):
  [0] forecast_precip  — GEFS rainfall forecast (mm/day)
  [1] lat_norm         — Latitude normalised to [-1, 1]
  [2] lon_norm         — Longitude normalised to [-1, 1]
  [3] lead_norm        — Lead day normalised to [0, 1] (Day 3→0.22, 5→0.44, 7→0.67, 10→1.0)
  [4] sin_doy          — sin(2π × day_of_year / 365)
  [5] cos_doy          — cos(2π × day_of_year / 365)

Usage:
  python data/preprocess.py                   # Full pipeline
  python data/preprocess.py --stub            # Use only stub dates (5 dates, fast)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, ".")
from data.config_loader import load_config


# ── Utility ──────────────────────────────────────────────────────────────────────

def regrid_to_target(source: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    """Bilinear regrid source to match target lat/lon grid (lightweight fallback)."""
    return source.interp(
        latitude=target.latitude,
        longitude=target.longitude,
        method="linear",
    )


def add_coord_channels(ds: xr.Dataset):
    """Return normalised lat_norm, lon_norm arrays [H, W] in [-1, 1]."""
    lats = ds.latitude.values
    lons = ds.longitude.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lat_norm = (lat_grid - lat_grid.mean()) / (lat_grid.max() - lat_grid.min() + 1e-8)
    lon_norm = (lon_grid - lon_grid.mean()) / (lon_grid.max() - lon_grid.min() + 1e-8)
    return lat_norm.astype(np.float32), lon_norm.astype(np.float32)


def day_of_year_encoding(doy: int):
    """Cyclical sin/cos encoding of day-of-year."""
    angle = 2 * np.pi * doy / 365.0
    return float(np.sin(angle)), float(np.cos(angle))


def compute_bust_labels(error: xr.DataArray, p90_threshold: xr.DataArray) -> xr.DataArray:
    """
    Binary bust label: 1 where absolute forecast error exceeds P90 threshold.

    Args:
        error: Gridded absolute forecast error [time, lat, lon]
        p90_threshold: 90th percentile of historical forecast error [lat, lon]

    Returns:
        bust: Binary DataArray [time, lat, lon], dtype float32
    """
    bust = (error > p90_threshold).astype(np.float32)
    bust.attrs["description"] = "1 = forecast bust (|error| > P90 of historical forecast error)"
    return bust


def _find_var(ds: xr.Dataset, candidates: list) -> Optional[str]:
    """Return the first matching variable name found in the dataset."""
    for c in candidates:
        if c in ds.data_vars:
            return c
    return None


def _valid_date(y: int, m: int, d: int) -> bool:
    import calendar
    return d <= calendar.monthrange(y, m)[1]


# ── P90 computation from REAL forecast error ─────────────────────────────────────

def compute_p90_from_forecast_error(
    rain_obs: xr.DataArray,
    gefs_dir: Path,
    lead_days: list,
    train_years: list,
    train_months: list,
    bust_percentile: int,
    min_samples: int,
    H: int,
    W: int,
) -> dict:
    """
    Compute P90 threshold from REAL |forecast − observation| error distribution.

    Only uses training-window data to avoid data leakage.

    Strategy:
      - Collect all |forecast − obs| error values for each lead day over the
        training window.
      - If total samples ≥ min_samples: compute global P90 across all cells
        (pooled — sufficient for a prototype).
      - If fewer samples: use a conservative fallback (12 mm, documented).

    Returns:
        p90_maps: {lead_day: np.ndarray [H, W]}
    """
    p90_maps = {}
    fallback_used = []

    # Build training dates
    train_dates = []
    for y in train_years:
        for m in train_months:
            import calendar
            _, n_days = calendar.monthrange(y, m)
            for d in range(1, n_days + 1):
                train_dates.append(pd.Timestamp(f"{y}-{m:02d}-{d:02d}"))

    for lead_day in lead_days:
        fhr = lead_day * 24
        errors_flat = []

        for date in train_dates:
            # Load obs for this date
            try:
                obs_day = rain_obs.sel(time=date, method="nearest").values.astype(np.float32)
            except Exception:
                continue

            # Load matching GEFS forecast
            gefs_file = gefs_dir / f"gefs_{date.strftime('%Y%m%d')}_f{fhr:03d}.nc"
            if not gefs_file.exists():
                continue

            try:
                gefs_ds = xr.open_dataset(gefs_file)
                precip_var = _find_var(
                    gefs_ds, ["tp", "precip", "apcp", "APCP_surface",
                              "precipitation_sum", "tprate"]
                )
                if precip_var is None:
                    gefs_ds.close()
                    continue

                fcast = gefs_ds[precip_var]

                # Regrid if spatial dims differ
                if (len(fcast.latitude) != H) or (len(fcast.longitude) != W):
                    obs_template = rain_obs.isel(time=0).to_dataset(name="rain")
                    fcast = regrid_to_target(fcast.to_dataset(name=precip_var), obs_template)[precip_var]

                fcast_vals = fcast.values.astype(np.float32)
                if fcast_vals.ndim > 2:
                    fcast_vals = fcast_vals[0]

                # ── Units conversion (mirrors sample assembly) ───────────────
                # GEFSv12 `tp` is in METRES; IMD rain is in mm.
                # If max value < 2.0, assume metres and convert to mm.
                if float(np.nanmax(fcast_vals)) < 2.0 and precip_var == "tp":
                    fcast_vals = fcast_vals * 1000.0  # m → mm

                # Compute absolute error and accumulate
                # Use nanpercentile-safe values: mask NaN/inf and physical outliers
                obs_clip  = obs_day[:H, :W].copy()
                fcast_clip = fcast_vals[:H, :W].copy()
                error_vals = np.abs(fcast_clip - obs_clip)
                # Keep only finite, non-negative, physically plausible values
                valid_mask = (
                    np.isfinite(error_vals)
                    & np.isfinite(obs_clip)
                    & (obs_clip >= 0)
                    & (error_vals < 500)   # sanity cap: 500 mm/day max error
                )
                errors_flat.extend(error_vals[valid_mask].flatten().tolist())
                gefs_ds.close()

            except Exception as e:
                continue


        # Compute P90 from collected real errors
        if len(errors_flat) >= min_samples:
            p90_val = float(np.percentile(errors_flat, bust_percentile))
            print(
                f"  Lead Day {lead_day}: P{bust_percentile} = {p90_val:.3f} mm "
                f"(n={len(errors_flat)} real error samples) [pooled across domain]"
            )
        else:
            # Fallback — documented, not hidden
            p90_val = 12.0
            fallback_used.append(lead_day)
            print(
                f"  Lead Day {lead_day}: WARNING — only {len(errors_flat)} samples "
                f"(< {min_samples} minimum). Using fallback P90 = {p90_val} mm. "
                f"Download more forecast data for this lead day."
            )

        # Assign same P90 to all cells (pooled, not per-cell — documented)
        p90_maps[lead_day] = np.full((H, W), p90_val, dtype=np.float32)

    if fallback_used:
        print(
            f"\n  [P90 NOTICE] Fallback used for lead days {fallback_used}. "
            f"These days had insufficient paired forecast/obs data. "
            f"Run full-season download (python data/download_gefs.py) to fix.\n"
        )

    return p90_maps


# ── Feature assembly ──────────────────────────────────────────────────────────────

def assemble_dataset(
    rain_obs: xr.DataArray,
    obs_ds: xr.Dataset,
    gefs_dir: Path,
    p90_maps: dict,
    lead_days: list,
    date_list_all: list,
    lat_norm: np.ndarray,
    lon_norm: np.ndarray,
    H: int,
    W: int,
) -> tuple:
    """
    Build feature tensors, error maps, and bust labels for all dates and lead days.

    Returns:
        features_arr  [N, 6, H, W]
        errors_arr    [N, 1, H, W]
        busts_arr     [N, 1, H, W]
        init_dates    [N]
        lead_days_out [N]
    """
    feature_list, error_list, bust_list = [], [], []
    date_out, lead_out = [], []

    for date in tqdm(date_list_all, desc="Assembling samples"):
        try:
            obs_day = rain_obs.sel(time=date, method="nearest").values.astype(np.float32)
        except Exception:
            continue

        doy = date.dayofyear
        sin_doy, cos_doy = day_of_year_encoding(doy)

        for lead_day in lead_days:
            fhr = lead_day * 24
            gefs_file = gefs_dir / f"gefs_{date.strftime('%Y%m%d')}_f{fhr:03d}.nc"
            if not gefs_file.exists():
                continue

            try:
                gefs_ds = xr.open_dataset(gefs_file)
                precip_var = _find_var(
                    gefs_ds, ["tp", "precip", "apcp", "APCP_surface",
                              "precipitation_sum", "tprate"]
                )
                if precip_var is None:
                    gefs_ds.close()
                    continue

                fcast = gefs_ds[precip_var]
                if (len(fcast.latitude) != H) or (len(fcast.longitude) != W):
                    obs_template = rain_obs.isel(time=0).to_dataset(name="rain")
                    fcast = regrid_to_target(fcast.to_dataset(name=precip_var), obs_template)[precip_var]

                fcast_vals = fcast.values.astype(np.float32)
                if fcast_vals.ndim > 2:
                    fcast_vals = fcast_vals[0]
                fcast_vals = fcast_vals[:H, :W]

                # ── Units conversion ────────────────────────────────────────
                # GEFSv12 `tp` is total accumulated precipitation in METRES.
                # IMD rain is in mm/day. Convert GEFS → mm before computing error.
                # Heuristic: if max value < 2.0 assume metres, multiply by 1000.
                if float(np.nanmax(fcast_vals)) < 2.0 and precip_var == "tp":
                    fcast_vals = fcast_vals * 1000.0   # m → mm


                # ── NaN/missing-value mask ──────────────────────────────────
                # IMD uses NaN or negative values for missing grid cells (sea).
                # Replace missing obs with 0 so errors are finite and bust=0 there.
                obs_day_clean = obs_day[:H, :W].copy()
                missing = ~np.isfinite(obs_day_clean) | (obs_day_clean < 0)
                obs_day_clean[missing] = 0.0
                fcast_vals[missing] = 0.0   # treat missing-obs cells as no-error

                error = np.abs(fcast_vals - obs_day_clean)
                p90   = p90_maps[lead_day]
                bust  = (error > p90).astype(np.float32)
                bust[missing] = 0.0          # no bust label for missing-obs cells

                # lead_day normalised: [3→0.2, 5→0.4, 7→0.6, 10→0.9] relative to Day 10
                lead_norm = (lead_day - 1) / 9.0
                lead_ch   = np.full((H, W), lead_norm, dtype=np.float32)
                sin_ch    = np.full((H, W), sin_doy,  dtype=np.float32)
                cos_ch    = np.full((H, W), cos_doy,  dtype=np.float32)

                feature = np.stack(
                    [fcast_vals, lat_norm, lon_norm, lead_ch, sin_ch, cos_ch],
                    axis=0,
                )  # [6, H, W]

                feature_list.append(feature)
                error_list.append(error[np.newaxis])      # [1, H, W]
                bust_list.append(bust[np.newaxis])         # [1, H, W]

                date_out.append(date)
                lead_out.append(lead_day)
                gefs_ds.close()

            except Exception as e:
                print(f"  Skip {date.date()} lead={lead_day}: {e}")
                continue

    if not feature_list:
        return None, None, None, None, None

    return (
        np.stack(feature_list, axis=0),   # [N, 6, H, W]
        np.stack(error_list,   axis=0),   # [N, 1, H, W]
        np.stack(bust_list,    axis=0),   # [N, 1, H, W]
        pd.DatetimeIndex(date_out),
        lead_out,
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────────

def run_preprocessing(cfg: dict, stub_mode: bool = False) -> None:
    """
    Full preprocessing pipeline.

    Args:
        cfg: Configuration from settings.yaml
        stub_mode: If True, only processes the 5 stub dates (fast validation)
    """
    # Only rainfall is implemented — temperature removed until that pipeline exists
    domain        = cfg["domain"]
    paths         = cfg["paths"]
    lead_days     = cfg["data"]["lead_days"]
    bust_pct      = cfg["bust"]["percentile_threshold"]
    min_samples   = cfg["bust"].get("min_samples_per_cell", 20)
    train_years   = cfg["data"]["train_years"]
    train_months  = cfg["data"].get("train_months", [6, 7, 8, 9])
    val_years     = cfg["data"]["val_years"]
    val_months    = cfg["data"].get("val_months", [8])
    test_years    = cfg["data"]["test_years"]
    test_months   = cfg["data"].get("test_months", [6, 7, 9])

    processed_dir = Path(paths["processed_data"])
    raw_dir       = Path(paths["raw_data"])
    gefs_dir      = raw_dir / "gefs"

    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Load IMD observations ──────────────────────────────────────────────────
    print("[Preprocess] Loading IMD rainfall observations...")
    # download_imd.py saves converted NetCDF as:
    #   data/processed/imd_rain_YYYY.nc  (from convert_to_netcdf)
    # Raw binary files go to data/raw/imd_rain/ (imdlib format, not NetCDF)
    # Search both locations so the pipeline works regardless of layout.
    obs_files = (
        sorted((raw_dir / "imd_rain").glob("*.nc"))           # legacy path
        or sorted(processed_dir.glob("imd_rain_*.nc"))        # download_imd.py output
    )
    if not obs_files:
        raise FileNotFoundError(
            f"No IMD rain NetCDF files found in:\n"
            f"  {raw_dir / 'imd_rain'}/*.nc\n"
            f"  {processed_dir}/imd_rain_*.nc\n"
            f"Run: python data/download_imd.py --variables rain"
        )
    print(f"[Preprocess] Found {len(obs_files)} IMD obs files: {[f.name for f in obs_files]}")

    obs_ds = xr.open_mfdataset(obs_files, combine="by_coords", engine="netcdf4")
    for old, new in [("lat", "latitude"), ("lon", "longitude")]:
        if old in obs_ds.coords:
            obs_ds = obs_ds.rename({old: new})

    # Clip observations to config domain
    obs_ds = obs_ds.sel(
        latitude=slice(domain["lat_min"] - 0.5, domain["lat_max"] + 0.5),
        longitude=slice(domain["lon_min"] - 0.5, domain["lon_max"] + 0.5),
    )

    rain_obs = obs_ds["rain"] if "rain" in obs_ds else list(obs_ds.data_vars)[0]
    lats = obs_ds.latitude.values
    lons = obs_ds.longitude.values
    H, W = len(lats), len(lons)
    lat_norm, lon_norm = add_coord_channels(obs_ds)
    print(f"[Preprocess] Observations: shape=[{H}, {W}] | "
          f"lat=[{lats[0]:.2f}, {lats[-1]:.2f}] lon=[{lons[0]:.2f}, {lons[-1]:.2f}]")


    # ── Build date lists ──────────────────────────────────────────────────────
    if stub_mode:
        from data.download_gefs_stub import STUB_DATES
        import calendar
        train_dates = [pd.Timestamp(d) for d in STUB_DATES if d.year in train_years and d.month in train_months]
        val_dates   = [pd.Timestamp(d) for d in STUB_DATES if d.year in val_years   and d.month in val_months]
        test_dates  = [pd.Timestamp(d) for d in STUB_DATES if d.year in test_years  and d.month in test_months]
        print(f"[Preprocess] STUB MODE: {len(train_dates)} train, {len(val_dates)} val, {len(test_dates)} test dates")
    else:
        import calendar

        def _month_dates(years, months):
            dates = []
            for y in years:
                for m in months:
                    _, n_days = calendar.monthrange(y, m)
                    for d in range(1, n_days + 1):
                        dates.append(pd.Timestamp(f"{y}-{m:02d}-{d:02d}"))
            return dates

        train_dates = _month_dates(train_years,  train_months)
        val_dates   = _month_dates(val_years,    val_months)
        test_dates  = _month_dates(test_years,   test_months)
        print(f"[Preprocess] {len(train_dates)} train | {len(val_dates)} val | {len(test_dates)} test dates")

    # ── Compute P90 from REAL forecast error (training window only) ───────────
    print(f"\n[Preprocess] Computing P{bust_pct} thresholds from real forecast error...")
    print(f"  Using training window: years={train_years}, months={train_months}")
    print(f"  Data leakage note: val/test dates are excluded from P90 computation.\n")

    p90_maps = compute_p90_from_forecast_error(
        rain_obs=rain_obs,
        gefs_dir=gefs_dir,
        lead_days=lead_days,
        train_years=train_years,
        train_months=train_months,
        bust_percentile=bust_pct,
        min_samples=min_samples,
        H=H, W=W,
    )

    # ── Assemble features for each split ──────────────────────────────────────
    for split_name, split_dates in [
        ("train", train_dates),
        ("val",   val_dates),
        ("test",  test_dates),
    ]:
        print(f"\n[Preprocess] Assembling {split_name} split ({len(split_dates)} candidate dates)...")

        features_arr, errors_arr, busts_arr, init_dates, lead_days_out = assemble_dataset(
            rain_obs=rain_obs,
            obs_ds=obs_ds,
            gefs_dir=gefs_dir,
            p90_maps=p90_maps,
            lead_days=lead_days,
            date_list_all=split_dates,
            lat_norm=lat_norm,
            lon_norm=lon_norm,
            H=H, W=W,
        )

        if features_arr is None:
            print(f"[Preprocess] WARNING: {split_name} split has no valid samples. Skipping.")
            continue

        N = features_arr.shape[0]
        bust_rate = float(busts_arr.mean())
        print(f"[Preprocess] {split_name}: {N} samples | bust rate = {bust_rate:.1%}")

        if bust_rate < 0.03:
            print(f"  WARNING: Very low bust rate ({bust_rate:.1%}). "
                  f"P90 threshold may be too high — check that forecast data loaded correctly.")
        elif bust_rate > 0.30:
            print(f"  WARNING: Unusually high bust rate ({bust_rate:.1%}). "
                  f"Check that forecast and obs data are correctly aligned in time/space.")

        # Save as Zarr
        zarr_path = processed_dir / f"dataset_{split_name}.zarr"
        # features: [N, 6, H, W] — 6 channels
        # error_map / bust_map: [N, 1, H, W] → squeeze to [N, H, W]
        # Using distinct dim sizes avoids xarray "conflicting sizes for dimension 'channel'" error
        out_ds = xr.Dataset(
            {
                "features":  (["sample", "channel", "latitude", "longitude"],
                               features_arr.astype(np.float32)),
                "error_map": (["sample", "latitude", "longitude"],
                               errors_arr[:, 0].astype(np.float32)),
                "bust_map":  (["sample", "latitude", "longitude"],
                               busts_arr[:, 0].astype(np.float32)),
            },
            coords={"latitude": lats, "longitude": lons},
            attrs={
                "description": f"SIH #26079 Forecast Bust Detection — {split_name} split",
                "n_samples":   N,
                "bust_rate":   bust_rate,
                "channels":    str(["forecast_precip", "lat_norm", "lon_norm",
                                    "lead_norm", "sin_doy", "cos_doy"]),
                "bust_definition": (
                    f"P{bust_pct} of |forecast - observation| error over training window. "
                    f"Pooled across domain (not per-cell). "
                    f"See /api/v1/info for status of each lead day."
                ),
                "p90_source": "real_forecast_error",
            },
        )
        out_ds["init_date"] = xr.DataArray(init_dates, dims=["sample"])
        out_ds["lead_day"]  = xr.DataArray(lead_days_out, dims=["sample"])
        out_ds.to_zarr(zarr_path, mode="w")
        print(f"[Preprocess] Saved: {zarr_path}")


    # ── Save P90 threshold maps ────────────────────────────────────────────────
    p90_arr = np.stack([p90_maps[d] for d in lead_days], axis=0)  # [L, H, W]
    p90_ds = xr.Dataset(
        {"p90_threshold": (["lead_day", "latitude", "longitude"], p90_arr)},
        coords={"lead_day": lead_days, "latitude": lats, "longitude": lons},
        attrs={
            "description": (
                f"P{bust_pct} of real |forecast - observation| error. "
                f"Pooled across domain. Computed from training window only."
            ),
            "p90_source": "real_forecast_error",
        },
    )
    p90_path = processed_dir / "p90_thresholds.nc"
    p90_ds.to_netcdf(p90_path)
    print(f"\n[Preprocess] Saved P90 thresholds: {p90_path}")
    print("[Preprocess] Complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess GEFSv12 + IMD data for Forecast Bust Detection ML"
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument(
        "--stub", action="store_true",
        help="Use only the 5 stub dates (fast pipeline validation)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_preprocessing(cfg, stub_mode=args.stub)


if __name__ == "__main__":
    main()
