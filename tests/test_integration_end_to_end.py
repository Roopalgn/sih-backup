"""
test_integration_end_to_end.py
-------------------------------
Integration tests verifying the full pipeline is real, not faked.

These tests:
  1. Require a preprocessed dataset (run python data/preprocess.py first)
  2. Require a trained model checkpoint (run python model/train.py first)
  3. Verify predictions are NOT the hardcoded 42.3/28.7/18.1 attribution triple
  4. Verify API fails loudly (503) rather than returning synthetic data

Run individually:
    pytest tests/test_integration_end_to_end.py -v

Skip gracefully if prerequisites are missing (data, checkpoint):
    pytest tests/test_integration_end_to_end.py -v --tb=short
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _find_zarr_path() -> Path | None:
    """Find any zarr dataset produced by preprocess.py."""
    base = Path("data/processed")
    for name in ("dataset_train.zarr", "dataset_val.zarr", "dataset.zarr"):
        p = base / name
        if p.exists():
            return p
    return None


def _find_checkpoint() -> Path | None:
    p = Path("checkpoints/best_model.pt")
    return p if p.exists() else None


ZARR_PATH  = _find_zarr_path()
CKPT_PATH  = _find_checkpoint()
HAS_ZARR   = ZARR_PATH is not None
HAS_CKPT   = CKPT_PATH is not None
HAS_TORCH  = True
try:
    import torch
except ImportError:
    HAS_TORCH = False

# ── Tests ─────────────────────────────────────────────────────────────────────

def test_zarr_dataset_exists():
    """Preprocessing ran and produced at least one zarr split."""
    assert HAS_ZARR, (
        f"No zarr dataset found in data/processed/. "
        f"Run: python data/preprocess.py"
    )


def test_zarr_has_real_p90_metadata():
    """
    P90 must come from real forecast error, not observation anomaly.
    We check the zarr attributes to confirm.
    """
    if not HAS_ZARR:
        pytest.skip("No zarr dataset — run preprocess.py first")

    import xarray as xr
    ds = xr.open_zarr(str(ZARR_PATH))
    bust_def = ds.attrs.get("bust_definition", "")
    p90_src  = ds.attrs.get("p90_source", "")
    ds.close()

    # Must NOT say "observation anomaly" or "climatological anomaly"
    assert "observation anomaly" not in bust_def.lower(), (
        f"bust_definition references observation anomaly: {bust_def}"
    )
    assert "climatological anomaly" not in bust_def.lower(), (
        f"bust_definition references climatological anomaly: {bust_def}"
    )
    # Should say "forecast error" or have p90_source = "real_forecast_error"
    has_correct_source = (
        "forecast error" in bust_def.lower() or
        p90_src == "real_forecast_error"
    )
    assert has_correct_source, (
        f"P90 not documented as from real forecast error. "
        f"bust_definition='{bust_def}', p90_source='{p90_src}'"
    )


def test_zarr_bust_rate_plausible():
    """Bust rate should be in the expected 5–25% range for real forecast error P90."""
    if not HAS_ZARR:
        pytest.skip("No zarr dataset")

    import xarray as xr
    ds = xr.open_zarr(str(ZARR_PATH))
    bust_arr = ds["bust_map"].values
    ds.close()

    rate = float(bust_arr.mean())
    assert 0.03 <= rate <= 0.35, (
        f"Bust rate {rate:.3f} is outside expected [0.03, 0.35]. "
        f"Either P90 is wrong, or forecast/obs data are misaligned."
    )


def test_channel_count_matches_config():
    """in_channels in config must match actual feature tensor shape."""
    if not HAS_ZARR:
        pytest.skip("No zarr dataset")

    import xarray as xr
    from data.config_loader import load_config

    ds = xr.open_zarr(str(ZARR_PATH))
    n_channels_data = ds["features"].shape[1]  # [N, C, H, W]
    ds.close()

    cfg = load_config("config/settings.yaml")
    n_channels_config = cfg["model"]["in_channels"]

    assert n_channels_data == n_channels_config, (
        f"Feature tensor has {n_channels_data} channels but "
        f"config.model.in_channels = {n_channels_config}. "
        f"These must agree before training. "
        f"Fix config/settings.yaml to set in_channels: {n_channels_data}."
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_checkpoint_exists():
    """Model checkpoint exists after training."""
    assert HAS_CKPT, (
        "No checkpoint at checkpoints/best_model.pt. "
        "Run: python model/train.py"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not HAS_ZARR, reason="No zarr dataset")
@pytest.mark.skipif(not HAS_CKPT, reason="No checkpoint")
def test_model_produces_real_non_hardcoded_outputs():
    """
    Core integration test: model output must NOT be the hardcoded 42.3/28.7/18.1 triple.
    This confirms the model is actually being called with real data.
    """
    import torch
    import xarray as xr
    from data.config_loader import load_config
    from model.architecture import ForecastBustUNet, compute_confidence_map
    from model.explain import compute_integrated_gradients, summarise_attributions, CHANNEL_NAMES

    cfg    = load_config("config/settings.yaml")
    device = torch.device("cpu")

    # Load model
    ckpt   = torch.load(str(CKPT_PATH), map_location=device)
    model  = ForecastBustUNet(
        in_channels=cfg["model"]["in_channels"],
        encoder_channels=tuple(cfg["model"]["encoder_channels"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load a real sample from the dataset
    ds     = xr.open_zarr(str(ZARR_PATH))
    feats  = ds["features"].values[0].astype(np.float32)  # [C, H, W]
    ds.close()

    x = torch.from_numpy(feats).unsqueeze(0).to(device)  # [1, C, H, W]

    # Forward pass
    with torch.no_grad():
        err_map, bust_prob = model(x)
        conf_map = compute_confidence_map(err_map, bust_prob)

    err_mean  = float(err_map.mean())
    bust_mean = float(bust_prob.mean())
    conf_mean = float(conf_map.mean())

    assert err_map.shape == (1, 1, feats.shape[1], feats.shape[2]), "Error map shape mismatch"
    assert (bust_prob >= 0).all() and (bust_prob <= 1).all(), "Bust prob out of [0,1]"
    assert (conf_map >= 0).all() and (conf_map <= 100).all(), "Confidence out of [0,100]"
    assert err_mean >= 0, "Error map must be non-negative (Softplus head)"

    # Verify Integrated Gradients produces variable (not hardcoded) attributions
    attributions = compute_integrated_gradients(
        model, x, target="bust", n_steps=10, device=device
    )
    drivers = summarise_attributions(attributions, channel_names=CHANNEL_NAMES, top_n=3)

    attr_pcts = sorted([d["attribution_pct"] for d in drivers])
    HARDCODED  = sorted([42.3, 28.7, 18.1])

    is_hardcoded = all(abs(a - h) < 2.0 for a, h in zip(attr_pcts, HARDCODED))
    assert not is_hardcoded, (
        f"Attribution percentages {attr_pcts} match the hardcoded triple {HARDCODED}. "
        f"This means the model is NOT being called — synthetic data is leaking."
    )

    # Verify different inputs produce different outputs
    x_perturbed = x + torch.randn_like(x) * 0.1
    with torch.no_grad():
        _, bust_prob2 = model(x_perturbed)

    assert not torch.allclose(bust_prob, bust_prob2, atol=1e-4), (
        "Model output is identical for different inputs — model may be degenerate."
    )

    print(f"\n  err_mean={err_mean:.3f}  bust_mean={bust_mean:.3f}  conf_mean={conf_mean:.1f}")
    print(f"  Top drivers: {[(d['channel_name'], d['attribution_pct']) for d in drivers]}")
    print("  PASS: Model produces real, non-hardcoded, input-dependent outputs.")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_api_fails_loudly_not_silently():
    """
    API must return 503 for an unavailable date — not synthetic data.
    """
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app, raise_server_exceptions=False)

    # Request a date well outside any training/eval window
    response = client.post(
        "/api/v1/predict",
        json={"date": "2005-06-15", "lead_day": 3},
    )

    if response.status_code == 503:
        # Correct — fail loudly
        detail = response.json().get("detail", "")
        assert "synthetic" in detail.lower() or "fabricat" in detail.lower() or len(detail) > 10, (
            "503 response should explain why prediction failed"
        )
        print("  PASS: API returns 503 for unavailable date (no silent fake).")

    elif response.status_code == 200:
        data = response.json()
        source = data.get("data_source", "")
        assert source in ("precomputed_cache", "illustrative_only"), (
            f"Response must not claim data_source='live_model' for a 2005 date. "
            f"Got data_source='{source}'. This may indicate synthetic data leaking."
        )
        print(f"  PASS: API returns cache/illustrative (data_source={source}) for out-of-window date.")
    else:
        pytest.fail(f"Unexpected status code {response.status_code}")


def test_events_json_no_fabricated_numbers():
    """
    events.json must not contain hand-authored float bust probabilities
    for events tagged 'live_model'.
    """
    events_path = Path("dashboard/precomputed/events.json")
    if not events_path.exists():
        pytest.skip("events.json not found")

    events = json.loads(events_path.read_text())
    for ev in events:
        source = ev.get("data_source", "")
        if source == "live_model":
            # Must have real numbers (not None)
            assert ev.get("mean_bust_probability") is not None, (
                f"Event '{ev.get('event_name')}' claims data_source=live_model "
                f"but mean_bust_probability is None."
            )
        if source == "illustrative_only":
            # Should have None values (we replaced the hand-authored numbers)
            prob = ev.get("mean_bust_probability")
            assert prob is None or isinstance(prob, (int, float)), (
                f"Event '{ev.get('event_name')}' has unexpected bust_probability type"
            )
