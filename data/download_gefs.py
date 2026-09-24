"""
download_gefs.py
----------------
Downloads NOAA GEFSv12 0.25° forecast data from AWS S3 open data bucket.

Data availability: GEFSv12 (2020-09-01 onward) at 0.25° resolution.
S3 bucket: s3://noaa-gefs-pds  (free, anonymous access)

Spatial subsetting strategy:
  1. Fetch the GRIB2 index file (.idx) for each forecast — this is a small text file.
  2. Parse the index to find byte offsets of only the required variables (APCP, TMP).
  3. Issue an HTTP Range request to download only those bytes from the full GRIB2.
  This avoids pulling ~100MB global files — each India-only variable slice is ~1–3MB.

Output: data/raw/gefs/gefs_{YYYYMMDD}_f{HHH}.nc  (one NetCDF per date × lead hour)

NOTE: NOMADS filter URLs (nomads.ncep.noaa.gov/cgi-bin/filter_gefs_0p25.pl) serve
real-time data only (last ~10 days). For archived 2021–2022 data, use this S3 path.
"""

import os
import io
import re
import datetime
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm

from config_loader import load_config

S3_BASE = "https://noaa-gefs-pds.s3.amazonaws.com"


# ── GRIB2 index-based byte-range download ─────────────────────────────────────

def _fetch_idx(date: datetime.date, fhr: int) -> Optional[str]:
    """
    Download the GRIB2 index file (.idx) for a GEFSv12 control member file.
    Index files are small (~20 KB) text files listing byte offsets per message.
    """
    date_str = date.strftime("%Y%m%d")
    path = (
        f"gefs.{date_str}/00/atmos/pgrb2sp25/"
        f"gec00.t00z.pgrb2s.0p25.f{fhr:03d}.idx"
    )
    url = f"{S3_BASE}/{path}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"[GEFS] Index fetch failed for {date_str} f{fhr:03d}: {e}")
        return None


def _parse_idx_for_vars(idx_text: str, wanted: list) -> list:
    """
    Parse GRIB2 index text to extract byte-range entries for wanted variables.

    Index format (one entry per line):
        {msg_num}:{byte_offset}:d={YYYYMMDD}{HH}:{var_name}:{level}:{description}:

    Args:
        idx_text: Raw .idx file contents
        wanted: List of variable name strings to match (e.g. ['APCP', 'TMP'])

    Returns:
        List of (byte_start, byte_end, var_name) tuples.
        byte_end is start of next message (or None = end of file).
    """
    lines = [l.strip() for l in idx_text.strip().splitlines() if l.strip()]
    entries = []
    for line in lines:
        parts = line.split(":")
        if len(parts) < 4:
            continue
        try:
            byte_offset = int(parts[1])
            var_name = parts[3]
            entries.append((byte_offset, var_name, line))
        except (ValueError, IndexError):
            continue

    # Find entries for wanted variables
    results = []
    for i, (offset, var, line) in enumerate(entries):
        for w in wanted:
            if w in var:
                next_offset = entries[i + 1][0] if i + 1 < len(entries) else None
                results.append((offset, next_offset, var))
                break

    return results


def _download_grib_bytes(
    date: datetime.date, fhr: int, byte_start: int, byte_end: Optional[int]
) -> Optional[bytes]:
    """Download a specific byte range from a GEFSv12 GRIB2 file on S3."""
    date_str = date.strftime("%Y%m%d")
    path = (
        f"gefs.{date_str}/00/atmos/pgrb2sp25/"
        f"gec00.t00z.pgrb2s.0p25.f{fhr:03d}"
    )
    url = f"{S3_BASE}/{path}"
    headers = {
        "Range": f"bytes={byte_start}-{byte_end - 1}" if byte_end else f"bytes={byte_start}-"
    }
    try:
        r = requests.get(url, headers=headers, timeout=60)
        if r.status_code in (200, 206):
            return r.content
        return None
    except Exception as e:
        print(f"[GEFS] Byte-range fetch failed ({byte_start}-{byte_end}): {e}")
        return None


def _grib_bytes_to_xarray(grib_bytes: bytes) -> Optional[xr.Dataset]:
    """Decode raw GRIB2 bytes into an xarray Dataset using cfgrib."""
    try:
        import cfgrib
        with io.BytesIO(grib_bytes) as buf:
            ds = cfgrib.open_dataset(buf, engine="cfgrib")
        return ds
    except Exception:
        # cfgrib may not handle all in-memory cases; write to temp file
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tf:
                tf.write(grib_bytes)
                tmp_path = tf.name
            # IMPORTANT: load all data into memory before deleting the temp file.
            # xarray opens lazily by default — deleting first causes FileNotFoundError.
            ds = xr.open_dataset(tmp_path, engine="cfgrib")
            ds.load()       # force eager load while file still exists
            os.unlink(tmp_path)
            return ds
        except Exception as e2:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            print(f"[GEFS] cfgrib decode failed: {e2}")
            return None



def fetch_one_day(
    date: datetime.date,
    lead_hours: list,
    output_dir: Path,
    domain: dict,
    wanted_vars: list = None,
) -> int:
    """
    Download and save GEFS data for one initialisation date.

    Uses byte-range reads from S3 index files. Only downloads variables in
    wanted_vars, cropped to the domain bounding box.

    Args:
        date: Forecast init date
        lead_hours: List of forecast hours to fetch (e.g. [72, 120, 168, 240])
        output_dir: Directory to save output NetCDF files
        domain: dict with lat_min, lat_max, lon_min, lon_max
        wanted_vars: GRIB2 variable names to fetch (default: APCP + TMP_2m)

    Returns:
        Number of successfully downloaded files.
    """
    if wanted_vars is None:
        wanted_vars = ["APCP", "TMP"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = date.strftime("%Y%m%d")
    n_saved = 0

    for fhr in lead_hours:
        out_nc = output_dir / f"gefs_{date_str}_f{fhr:03d}.nc"
        if out_nc.exists():
            n_saved += 1
            continue

        # 1. Fetch index file
        idx_text = _fetch_idx(date, fhr)
        if idx_text is None:
            print(f"[GEFS] No index for {date_str} f{fhr:03d} — skipping")
            continue

        # 2. Parse byte offsets for wanted variables
        byte_ranges = _parse_idx_for_vars(idx_text, wanted_vars)
        if not byte_ranges:
            print(f"[GEFS] Vars {wanted_vars} not found in index for {date_str} f{fhr:03d}")
            continue

        # 3. Download each variable, decode, collect
        datasets = []
        for byte_start, byte_end, var_name in byte_ranges:
            raw_bytes = _download_grib_bytes(date, fhr, byte_start, byte_end)
            if raw_bytes is None:
                continue
            ds = _grib_bytes_to_xarray(raw_bytes)
            if ds is not None:
                datasets.append((var_name, ds))

        if not datasets:
            print(f"[GEFS] No data decoded for {date_str} f{fhr:03d}")
            continue

        # 4. Combine variables, clip to domain, save as NetCDF
        try:
            merged = {}
            lat_min, lat_max = domain["lat_min"] - 1, domain["lat_max"] + 1
            lon_min, lon_max = domain["lon_min"] - 1, domain["lon_max"] + 1

            for var_name, ds in datasets:
                # Find spatial coordinates
                lat_coord = next(
                    (c for c in ds.coords if c in ("latitude", "lat")), None
                )
                lon_coord = next(
                    (c for c in ds.coords if c in ("longitude", "lon")), None
                )
                if lat_coord is None or lon_coord is None:
                    continue

                # Clip to India domain.
                # GEFSv12 GRIB2 stores latitudes DESCENDING (90 → -90).
                # xarray slice() requires ascending order, so we sort first.
                lat_vals = ds[lat_coord].values
                if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
                    # Descending — sort so slice works correctly
                    ds = ds.sortby(lat_coord)

                ds_clipped = ds.sel(
                    {lat_coord: slice(lat_min, lat_max),
                     lon_coord: slice(lon_min, lon_max)}
                )
                # Rename coords to standard names
                if lat_coord != "latitude":
                    ds_clipped = ds_clipped.rename({lat_coord: "latitude"})
                if lon_coord != "longitude":
                    ds_clipped = ds_clipped.rename({lon_coord: "longitude"})

                for v in ds_clipped.data_vars:
                    merged[v] = ds_clipped[v]


            if not merged:
                continue

            out_ds = xr.Dataset(merged)
            out_ds.attrs.update({
                "source": "NOAA GEFSv12",
                "init_date": date_str,
                "forecast_hour": fhr,
                "domain": f"{lat_min}–{lat_max}°N, {lon_min}–{lon_max}°E",
            })
            out_ds.to_netcdf(out_nc)
            print(f"[GEFS] Saved {out_nc.name} ({out_ds.dims})")
            n_saved += 1

        except Exception as e:
            print(f"[GEFS] Error saving {date_str} f{fhr:03d}: {e}")

    return n_saved


# ── Open-Meteo fallback (kept for completeness — use for pre-2020 dates only) ──
# Not used by the main pipeline; available if you need a quick sanity check on
# forecast data structure without S3 access.
def fetch_openmeteo_point(
    lat: float, lon: float, date: datetime.date
) -> Optional[dict]:
    """
    Fetch historical forecast for a single point via Open-Meteo API.
    Returns dict with precipitation and temperature for that date.
    NOTE: Point-wise only — not suitable for gridded feature computation.
    """
    try:
        import requests
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": date.isoformat(),
            "end_date": date.isoformat(),
            "daily": ["precipitation_sum", "temperature_2m_max"],
            "timezone": "UTC",
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ── Main entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download GEFSv12 forecast data for India via S3 byte-range reads"
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--start", help="Override start date YYYY-MM-DD")
    parser.add_argument("--end",   help="Override end date YYYY-MM-DD")
    args = parser.parse_args()

    cfg = load_config(args.config)
    domain = cfg["domain"]
    lead_days = cfg["data"]["lead_days"]
    lead_hours = [d * 24 for d in lead_days]

    gefs_min = datetime.date.fromisoformat(cfg["data"].get("gefs_min_date", "2020-09-01"))

    # Build date range from config
    if args.start:
        start = datetime.date.fromisoformat(args.start)
    else:
        train_years  = cfg["data"]["train_years"]
        train_months = cfg["data"].get("train_months", [6, 7, 8, 9])
        start = datetime.date(min(train_years), min(train_months), 1)

    if args.end:
        end = datetime.date.fromisoformat(args.end)
    else:
        test_years   = cfg["data"]["test_years"]
        test_months  = cfg["data"].get("test_months", [6, 7, 9])
        import calendar
        last_month = max(test_months)
        last_year  = max(test_years)
        end = datetime.date(last_year, last_month, calendar.monthrange(last_year, last_month)[1])

    # GEFSv12 date guard
    if start < gefs_min:
        raise ValueError(
            f"Start date {start} is before GEFSv12 cutoff {gefs_min}. "
            f"GEFSv12 0.25° data only available from {gefs_min}. "
            f"NOMADS serves only real-time data — for archived hindcasts use this S3 path."
        )

    print(f"[GEFS] Downloading {start} → {end} | lead hours: {lead_hours}")
    print(f"[GEFS] Domain: {domain['lat_min']}–{domain['lat_max']}°N, "
          f"{domain['lon_min']}–{domain['lon_max']}°E")

    output_dir = Path(cfg["paths"]["raw_data"]) / "gefs"
    dates = pd.date_range(start, end, freq="D")
    total_saved = 0

    for dt in tqdm(dates, desc="GEFS days"):
        n = fetch_one_day(dt.date(), lead_hours, output_dir, domain)
        total_saved += n

    print(f"[GEFS] Complete. {total_saved} files saved to {output_dir}")


if __name__ == "__main__":
    main()
