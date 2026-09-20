"""
test_pipeline.py
----------------
Unit tests for the data pipeline — config, bust labels, coordinates, event catalog.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, ".")


def test_config_loader():
    """Config YAML loads and contains required keys."""
    from data.config_loader import load_config
    cfg = load_config("config/settings.yaml")
    assert "domain" in cfg
    assert cfg["domain"]["lat_min"] == 6.0
    assert cfg["domain"]["lon_min"] == 68.0
    assert "bust" in cfg
    assert cfg["bust"]["percentile_threshold"] == 90


def test_bust_label_definition():
    """
    P90 bust labels should flag ~10 % of cells.
    Acceptable range: 5–20 % (small-sample variance expected).
    """
    import xarray as xr
    from data.preprocess import compute_bust_labels

    np.random.seed(42)
    H, W = 20, 20
    lats = np.linspace(10, 30, H)
    lons = np.linspace(70, 90, W)

    errors = np.random.exponential(5.0, size=(30, H, W))
    threshold = np.percentile(errors, 90, axis=0)

    error_da = xr.DataArray(
        errors, dims=["time", "latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
    )
    thresh_da = xr.DataArray(
        threshold, dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
    )

    bust = compute_bust_labels(error_da, thresh_da)
    bust_rate = float(bust.mean())
    assert 0.05 < bust_rate < 0.20, f"Bust rate {bust_rate:.3f} out of expected [0.05, 0.20]"


def test_coord_channels():
    """Normalised lat/lon channels must be in [-1, 1] and have the right shape."""
    import xarray as xr
    from data.preprocess import add_coord_channels

    lats = np.linspace(6, 38, 10)
    lons = np.linspace(68, 98, 12)
    ds = xr.Dataset(coords={"latitude": lats, "longitude": lons})
    lat_norm, lon_norm = add_coord_channels(ds)

    assert lat_norm.shape == (10, 12)
    assert lon_norm.shape == (10, 12)
    assert lat_norm.min() >= -1 and lat_norm.max() <= 1
    assert lon_norm.min() >= -1 and lon_norm.max() <= 1


def test_day_of_year_encoding():
    """Cyclical DOY encoding must stay in [-1, 1] and be approximately circular."""
    from data.preprocess import day_of_year_encoding

    for doy in [1, 90, 180, 270, 365]:
        s, c = day_of_year_encoding(doy)
        assert -1 <= s <= 1
        assert -1 <= c <= 1

    # Day 1 ≈ Day 366 (circular)
    s1, c1 = day_of_year_encoding(1)
    s366, c366 = day_of_year_encoding(366)
    assert abs(s1 - s366) < 0.1


def test_event_catalog_integrity():
    """Catalog has expected events and all required fields."""
    from data.event_catalog import EVENT_CATALOG, get_events_by_type, get_showcase_events

    assert len(EVENT_CATALOG) >= 5

    for e in EVENT_CATALOG:
        assert e.event_type in {
            "cyclone", "monsoon_depression", "heat_wave",
            "western_disturbance", "active_break",
        }
        assert 1 <= e.severity <= 5
        assert e.start_date <= e.end_date

    cyclones = get_events_by_type("cyclone")
    assert len(cyclones) >= 2

    showcase = get_showcase_events()
    assert len(showcase) == 5


def test_config_bust_threshold_range():
    """Bust percentile threshold must be between 50 and 99."""
    from data.config_loader import load_config
    cfg = load_config("config/settings.yaml")
    p = cfg["bust"]["percentile_threshold"]
    assert 50 <= p <= 99, f"Bust threshold {p} outside valid range [50, 99]"
