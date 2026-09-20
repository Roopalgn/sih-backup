"""
download_imd.py
---------------
Downloads IMD (India Meteorological Department) gridded observational data:
  - Daily rainfall at 0.25° x 0.25° (Pai et al. 2014)
  - Daily max/min temperature at 0.5° x 0.5° (Srivastava 2009)

Uses: imdlib (https://github.com/iamsaswata/imdlib)

Output:
  data/raw/imd_rain_{year}.nc    - Daily rainfall (mm/day)
  data/raw/imd_tmax_{year}.nc    - Daily max temperature (°C)
  data/raw/imd_tmin_{year}.nc    - Daily min temperature (°C)
"""

import os
import argparse
from pathlib import Path

import xarray as xr
import imdlib as imd

from config_loader import load_config


def download_imd_data(variable: str, start_year: int, end_year: int, data_dir: str) -> None:
    """
    Download IMD gridded data using imdlib.

    Args:
        variable: One of 'rain', 'tmax', 'tmin'
        start_year: First year to download
        end_year: Last year to download (inclusive)
        data_dir: Directory to save raw binary files
    """
    os.makedirs(data_dir, exist_ok=True)
    print(f"[IMD] Downloading {variable} | {start_year}–{end_year} → {data_dir}")

    imd.get_data(
        var_type=variable,
        start_yr=start_year,
        end_yr=end_year,
        fn_format="yearwise",
        file_dir=data_dir,
    )
    print(f"[IMD] Download complete for {variable}")


def convert_to_netcdf(
    variable: str,
    start_year: int,
    end_year: int,
    data_dir: str,
    output_dir: str,
    domain: dict,
) -> None:
    """
    Convert imdlib binary to xarray Dataset, clip to domain, save as NetCDF.

    Args:
        variable: 'rain', 'tmax', or 'tmin'
        start_year: First year
        end_year: Last year (inclusive)
        data_dir: Directory containing raw imdlib binary files
        output_dir: Directory to write NetCDF output
        domain: dict with lat_min, lat_max, lon_min, lon_max
    """
    os.makedirs(output_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        out_path = Path(output_dir) / f"imd_{variable}_{year}.nc"
        if out_path.exists():
            print(f"[IMD] Skipping {year} — already exists: {out_path}")
            continue

        print(f"[IMD] Converting {variable} {year} → {out_path}")
        data = imd.open_data(variable, year, year, "yearwise", data_dir)
        ds = data.get_xarray()

        # Rename coordinates to standard names if needed
        rename_map = {}
        if "lat" in ds.coords:
            rename_map["lat"] = "latitude"
        if "lon" in ds.coords:
            rename_map["lon"] = "longitude"
        if rename_map:
            ds = ds.rename(rename_map)

        # Clip to domain
        ds = ds.sel(
            latitude=slice(domain["lat_min"], domain["lat_max"]),
            longitude=slice(domain["lon_min"], domain["lon_max"]),
        )

        # Add metadata
        var_long_names = {
            "rain": "IMD Daily Rainfall",
            "tmax": "IMD Daily Maximum Temperature",
            "tmin": "IMD Daily Minimum Temperature",
        }
        var_units = {
            "rain": "mm day-1",
            "tmax": "degree_Celsius",
            "tmin": "degree_Celsius",
        }
        ds.attrs.update(
            {
                "source": "India Meteorological Department (IMD)",
                "long_name": var_long_names.get(variable, variable),
                "units": var_units.get(variable, "unknown"),
                "reference": "Pai et al. 2014 (rainfall); Srivastava et al. 2009 (temperature)",
                "domain": f"{domain['lat_min']}N-{domain['lat_max']}N, {domain['lon_min']}E-{domain['lon_max']}E",
            }
        )

        ds.to_netcdf(out_path)
        print(f"[IMD] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Download IMD gridded data")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file path")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["rain", "tmax", "tmin"],
        choices=["rain", "tmax", "tmin"],
        help="IMD variables to download",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    domain = cfg["domain"]
    start_year = cfg["data"]["imd_start"]
    end_year = cfg["data"]["imd_end"]

    raw_dir = cfg["paths"]["raw_data"]
    processed_dir = cfg["paths"]["processed_data"]

    for variable in args.variables:
        var_raw_dir = os.path.join(raw_dir, f"imd_{variable}")
        download_imd_data(variable, start_year, end_year, var_raw_dir)
        convert_to_netcdf(variable, start_year, end_year, var_raw_dir, processed_dir, domain)

    print("[IMD] All downloads and conversions complete.")


if __name__ == "__main__":
    main()
