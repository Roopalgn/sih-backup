"""
download_era5.py
----------------
Downloads ERA5 reanalysis upper-air and surface data from the
Copernicus Climate Data Store (CDS) via cdsapi.

Requirements:
  1. Register at https://cds.climate.copernicus.eu/
  2. Save API key to ~/.cdsapirc:
       url: https://cds.climate.copernicus.eu/api/v2
       key: <UID>:<API-KEY>

Downloads monthly GRIB files, converts to NetCDF, clips to Indian domain.

Variables:
  Pressure levels: z (geopotential), t (temperature), u, v, q
  Single levels: msl, 2m_temperature, total_precipitation, cape, tcwv

Output:
  data/raw/era5/era5_pressure_{YYYY}_{MM}.nc
  data/raw/era5/era5_surface_{YYYY}_{MM}.nc
"""

import os
import argparse
from pathlib import Path

import cdsapi
import xarray as xr
from tqdm import tqdm

from config_loader import load_config


PRESSURE_LEVELS = ["500", "700", "850", "200", "1000"]

PRESSURE_VARIABLES = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]

SURFACE_VARIABLES = [
    "mean_sea_level_pressure",
    "2m_temperature",
    "total_precipitation",
    "convective_available_potential_energy",
    "total_column_water_vapour",
]


def download_era5_pressure(client: cdsapi.Client, year: int, month: int, domain: dict, output_dir: str) -> Path:
    """Download ERA5 pressure-level data for a given month."""
    out_path = Path(output_dir) / f"era5_pressure_{year}_{month:02d}.nc"
    if out_path.exists():
        print(f"[ERA5] Skipping pressure {year}-{month:02d} — exists")
        return out_path

    print(f"[ERA5] Requesting pressure levels {year}-{month:02d}")
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": PRESSURE_VARIABLES,
            "pressure_level": PRESSURE_LEVELS,
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
            "area": [
                domain["lat_max"],
                domain["lon_min"],
                domain["lat_min"],
                domain["lon_max"],
            ],
        },
        str(out_path),
    )
    print(f"[ERA5] Saved: {out_path}")
    return out_path


def download_era5_surface(client: cdsapi.Client, year: int, month: int, domain: dict, output_dir: str) -> Path:
    """Download ERA5 single-level (surface) data for a given month."""
    out_path = Path(output_dir) / f"era5_surface_{year}_{month:02d}.nc"
    if out_path.exists():
        print(f"[ERA5] Skipping surface {year}-{month:02d} — exists")
        return out_path

    print(f"[ERA5] Requesting surface {year}-{month:02d}")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": SURFACE_VARIABLES,
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
            "area": [
                domain["lat_max"],
                domain["lon_min"],
                domain["lat_min"],
                domain["lon_max"],
            ],
        },
        str(out_path),
    )
    print(f"[ERA5] Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 reanalysis data")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument(
        "--type",
        choices=["pressure", "surface", "both"],
        default="both",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    domain = cfg["domain"]
    output_dir = os.path.join(cfg["paths"]["raw_data"], "era5")
    os.makedirs(output_dir, exist_ok=True)

    client = cdsapi.Client()

    year_month_pairs = [
        (y, m)
        for y in range(args.start_year, args.end_year + 1)
        for m in range(1, 13)
    ]

    for year, month in tqdm(year_month_pairs, desc="ERA5 months"):
        if args.type in ("pressure", "both"):
            download_era5_pressure(client, year, month, domain, output_dir)
        if args.type in ("surface", "both"):
            download_era5_surface(client, year, month, domain, output_dir)

    print("[ERA5] All downloads complete.")


if __name__ == "__main__":
    main()
