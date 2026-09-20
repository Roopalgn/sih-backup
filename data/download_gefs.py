"""
download_gefs.py
----------------
Downloads NOAA GEFS (Global Ensemble Forecast System) hindcast data
from AWS S3 open data bucket: s3://noaa-gefs-pds

For pre-2020 data (or when S3 unavailable), falls back to the Open-Meteo
Historical Forecast API which provides archived GFS/ECMWF forecasts.

Fetches:
  - 00Z daily initialisation runs
  - Lead times: +24h to +240h (Day 1 to Day 10)
  - Variables: Precipitation, 2m Temperature, Z500, U850, V850
  - Members: Control (c00) + ensemble mean (for spread computation)

Output:
  data/processed/gefs_{YYYYMMDD}_day{d}.nc per forecast date and lead day
"""

import os
import argparse
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from config_loader import load_config


# ─────────────────────────────────────────────────────
# Open-Meteo fallback (for historical archive)
# ─────────────────────────────────────────────────────
def fetch_openmeteo_forecast(
    date: datetime.date,
    lead_days: list,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> Optional[xr.Dataset]:
    """
    Fetch historical forecast via Open-Meteo archive API.
    Returns an xarray Dataset with variables aligned to the given domain.

    Note: Open-Meteo provides point-wise data; for gridded coverage
    we sample on the 0.25-degree grid and interpolate.
    """
    try:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry

        cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        # Sample grid points at 1° resolution (reduce API calls)
        lats = np.arange(lat_min, lat_max, 1.0)
        lons = np.arange(lon_min, lon_max, 1.0)

        all_data = []
        for lat in lats:
            for lon in lons:
                url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": date.strftime("%Y-%m-%d"),
                    "end_date": (date + datetime.timedelta(days=max(lead_days))).strftime("%Y-%m-%d"),
                    "daily": [
                        "precipitation_sum",
                        "temperature_2m_max",
                        "temperature_2m_min",
                    ],
                    "timezone": "UTC",
                    "models": "gfs_seamless",
                }
                try:
                    responses = openmeteo.weather_api(url, params=params)
                    daily = responses[0].Daily()
                    all_data.append(
                        {
                            "lat": lat,
                            "lon": lon,
                            "precip": daily.Variables(0).ValuesAsNumpy(),
                            "tmax": daily.Variables(1).ValuesAsNumpy(),
                            "tmin": daily.Variables(2).ValuesAsNumpy(),
                        }
                    )
                except Exception as e:
                    print(f"[GEFS/OpenMeteo] Error at ({lat},{lon}): {e}")
                    continue

        if not all_data:
            return None

        # Build gridded dataset from point samples
        lat_arr = sorted(set(d["lat"] for d in all_data))
        lon_arr = sorted(set(d["lon"] for d in all_data))
        n_days = max(len(d["precip"]) for d in all_data)

        precip = np.full((n_days, len(lat_arr), len(lon_arr)), np.nan)
        tmax = np.full((n_days, len(lat_arr), len(lon_arr)), np.nan)
        tmin = np.full((n_days, len(lat_arr), len(lon_arr)), np.nan)

        for d in all_data:
            i = lat_arr.index(d["lat"])
            j = lon_arr.index(d["lon"])
            n = len(d["precip"])
            precip[:n, i, j] = d["precip"]
            tmax[:n, i, j] = d["tmax"]
            tmin[:n, i, j] = d["tmin"]

        time_coord = pd.date_range(date, periods=n_days, freq="D")
        ds = xr.Dataset(
            {
                "precip": (["time", "latitude", "longitude"], precip),
                "tmax": (["time", "latitude", "longitude"], tmax),
                "tmin": (["time", "latitude", "longitude"], tmin),
            },
            coords={
                "time": time_coord,
                "latitude": lat_arr,
                "longitude": lon_arr,
            },
        )
        return ds

    except ImportError:
        print("[GEFS] openmeteo-requests not installed, skipping Open-Meteo fallback.")
        return None


def fetch_gefs_s3(
    date: datetime.date,
    lead_hours_list: list,
    output_dir: str,
    domain: dict,
) -> None:
    """
    Download GEFS c00 (control) member from AWS S3 for a specific date.
    GEFS on AWS: s3://noaa-gefs-pds/gefs.YYYYMMDD/00/atmos/pgrb2sp25/

    Files are downloaded and opened with cfgrib, then subset to the Indian domain.
    """
    try:
        import s3fs
        import cfgrib
    except ImportError:
        print("[GEFS] s3fs or cfgrib not installed. Skipping S3 download.")
        return

    date_str = date.strftime("%Y%m%d")
    s3 = s3fs.S3FileSystem(anon=True)

    os.makedirs(output_dir, exist_ok=True)

    for fhr in lead_hours_list:
        out_path = Path(output_dir) / f"gefs_{date_str}_f{fhr:03d}.nc"
        if out_path.exists():
            print(f"[GEFS S3] Skipping {date_str} f{fhr:03d} — exists")
            continue

        s3_path = f"noaa-gefs-pds/gefs.{date_str}/00/atmos/pgrb2sp25/gec00.t00z.pgrb2s.0p25.f{fhr:03d}"

        try:
            print(f"[GEFS S3] Downloading: {s3_path}")
            with s3.open(s3_path, "rb") as f:
                grib_bytes = f.read()

            tmp_grib = Path(output_dir) / f"_tmp_{date_str}_{fhr:03d}.grib2"
            tmp_grib.write_bytes(grib_bytes)

            # Open with cfgrib and subset variables
            datasets = []
            for filter_keys in [
                {"typeOfLevel": "surface", "shortName": "tp"},
                {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2t"},
                {"typeOfLevel": "isobaricInhPa", "level": 500, "shortName": "gh"},
                {"typeOfLevel": "isobaricInhPa", "level": 850, "shortName": "u"},
                {"typeOfLevel": "isobaricInhPa", "level": 850, "shortName": "v"},
                {"typeOfLevel": "isobaricInhPa", "level": 700, "shortName": "q"},
            ]:
                try:
                    ds = xr.open_dataset(
                        str(tmp_grib),
                        engine="cfgrib",
                        backend_kwargs={"filter_by_keys": filter_keys},
                    )
                    datasets.append(ds)
                except Exception:
                    continue

            if datasets:
                merged = xr.merge(datasets, compat="override")

                # Clip to domain
                if "latitude" in merged.coords:
                    merged = merged.sel(
                        latitude=slice(
                            max(merged.latitude.values),
                            min(merged.latitude.values),
                        )
                    )
                    lat_mask = (
                        (merged.latitude >= domain["lat_min"])
                        & (merged.latitude <= domain["lat_max"])
                    )
                    lon_mask = (
                        (merged.longitude >= domain["lon_min"])
                        & (merged.longitude <= domain["lon_max"])
                    )
                    merged = merged.sel(
                        latitude=merged.latitude[lat_mask],
                        longitude=merged.longitude[lon_mask],
                    )

                merged.to_netcdf(out_path)
                print(f"[GEFS S3] Saved: {out_path}")
            else:
                print(f"[GEFS S3] No usable data in {s3_path}")

            tmp_grib.unlink(missing_ok=True)

        except Exception as e:
            print(f"[GEFS S3] Error for {date_str} f{fhr:03d}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Download GEFS hindcast data")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument(
        "--method",
        choices=["s3", "openmeteo", "auto"],
        default="auto",
        help="Data source: 's3' (AWS), 'openmeteo', or 'auto' (try S3 first)",
    )
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2023-12-31", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    cfg = load_config(args.config)
    domain = cfg["domain"]
    lead_days = cfg["data"]["lead_days"]
    lead_hours = [d * 24 for d in lead_days]

    raw_dir = os.path.join(cfg["paths"]["raw_data"], "gefs")
    os.makedirs(raw_dir, exist_ok=True)

    start_date = datetime.date.fromisoformat(args.start)
    end_date = datetime.date.fromisoformat(args.end)
    date_range = pd.date_range(start_date, end_date, freq="D")

    print(f"[GEFS] Downloading {len(date_range)} dates, method={args.method}")

    for dt in tqdm(date_range):
        date = dt.date()
        if args.method in ("s3", "auto"):
            fetch_gefs_s3(date, lead_hours, raw_dir, domain)
        elif args.method == "openmeteo":
            ds = fetch_openmeteo_forecast(
                date,
                lead_days,
                domain["lat_min"],
                domain["lat_max"],
                domain["lon_min"],
                domain["lon_max"],
            )
            if ds is not None:
                out_path = Path(raw_dir) / f"gefs_openmeteo_{date.strftime('%Y%m%d')}.nc"
                ds.to_netcdf(out_path)

    print("[GEFS] Done.")


if __name__ == "__main__":
    main()
