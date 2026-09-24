"""
explain.py
----------
Explainability module for ForecastBustUNet using Integrated Gradients (hand-rolled).

NOTE: This is a manual Riemann-sum implementation of Integrated Gradients.
It is NOT using the Captum library. The implementation is Captum-compatible
in terms of the IG formula but has no dependency on captum.


Outputs:
  - Per-channel attribution maps [C, H, W] showing which input variable
    contributed most to the bust probability prediction
  - Top-N meteorological driver summary in JSON format
  - Attribution heatmap PNG overlaid on a India map

Meteorological interpretation:
  Channel 0: forecast_precip -> Direct precipitation forecast influence
  Channel 1: lat_norm         -> Latitudinal position sensitivity
  Channel 2: lon_norm         -> Longitudinal position sensitivity
  Channel 3: lead_norm        -> Lead time influence
  Channel 4: sin_doy          -> Seasonal component (summer)
  Channel 5: cos_doy          -> Seasonal component (winter)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from model.architecture import ForecastBustUNet

CHANNEL_NAMES = [
    "Forecast Precipitation",
    "Latitude (normalised)",
    "Longitude (normalised)",
    "Lead Time",
    "Seasonal (sin DOY)",
    "Seasonal (cos DOY)",
]

CHANNEL_DESCRIPTIONS = [
    "Direct model precipitation forecast value",
    "Geographic latitude influence",
    "Geographic longitude influence",
    "Forecast lead time (longer = less reliable)",
    "Monsoon/summer seasonal component",
    "Winter/post-monsoon seasonal component",
]


def compute_integrated_gradients(
    model: ForecastBustUNet,
    input_tensor: torch.Tensor,
    target: str = "bust",
    n_steps: int = 50,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Compute Integrated Gradients for a single input sample.

    IG formula:
        Attr_i(x) = (x_i - x'_i) * integral_0^1 dF/dx_i(x' + alpha*(x - x')) d_alpha

    Uses the zero tensor as baseline (x' = 0).

    Args:
        model: Trained ForecastBustUNet
        input_tensor: Single sample [1, C, H, W]
        target: 'bust' for classification head, 'error' for regression head
        n_steps: Number of Riemann approximation steps
        device: Torch device

    Returns:
        attributions: [C, H, W] attribution map
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    x = input_tensor.to(device).requires_grad_(False)
    baseline = torch.zeros_like(x)

    # Interpolate between baseline and input
    alphas = torch.linspace(0, 1, n_steps, device=device)
    integrated_grads = torch.zeros_like(x[0])  # [C, H, W]

    for alpha in alphas:
        x_interp = baseline + alpha * (x - baseline)
        x_interp.requires_grad_(True)

        model.zero_grad()
        error_map, bust_prob = model(x_interp)

        # Select target output
        if target == "bust":
            output = bust_prob.mean()  # Scalar
        else:
            output = error_map.mean()

        output.backward()

        if x_interp.grad is not None:
            integrated_grads += x_interp.grad[0].detach()  # [C, H, W]

    # Scale by input difference
    attributions = (x[0] - baseline[0]).detach().cpu().numpy() * (
        integrated_grads.cpu().numpy() / n_steps
    )
    return attributions  # [C, H, W]


def summarise_attributions(
    attributions: np.ndarray,
    channel_names: list[str] = None,
    top_n: int = 3,
) -> list[dict]:
    """
    Summarise spatial attributions into per-channel importance scores.

    Args:
        attributions: [C, H, W] attribution array
        channel_names: Human-readable names for each channel
        top_n: Number of top drivers to return

    Returns:
        List of dicts: {channel, name, attribution_score, description}
    """
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(attributions.shape[0])]

    # Use mean absolute attribution per channel as importance score
    importance = np.abs(attributions).mean(axis=(1, 2))  # [C]
    total = importance.sum() + 1e-10
    normalized = importance / total

    top_indices = np.argsort(importance)[::-1][:top_n]

    drivers = []
    for rank, idx in enumerate(top_indices):
        ch_name = channel_names[idx] if idx < len(channel_names) else f"Channel {idx}"
        ch_desc = CHANNEL_DESCRIPTIONS[idx] if idx < len(CHANNEL_DESCRIPTIONS) else ""
        drivers.append({
            "rank": rank + 1,
            "channel_idx": int(idx),
            "channel_name": ch_name,
            "attribution_score": float(round(importance[idx], 4)),
            "attribution_pct": float(round(normalized[idx] * 100, 1)),
            "description": ch_desc,
        })
    return drivers


def explain_sample(
    model: ForecastBustUNet,
    input_tensor: torch.Tensor,
    output_dir: str,
    sample_id: str = "sample",
    target: str = "bust",
    n_steps: int = 50,
    channel_names: list[str] = None,
    save_png: bool = True,
) -> dict:
    """
    Full explainability pipeline for a single sample:
      1. Compute Integrated Gradients
      2. Summarise top meteorological drivers
      3. Save attribution maps as PNG and JSON

    Args:
        model: Trained ForecastBustUNet
        input_tensor: Input sample [1, C, H, W]
        output_dir: Directory to save outputs
        sample_id: Identifier for filenames
        target: 'bust' or 'error'
        n_steps: IG integration steps
        channel_names: Channel display names
        save_png: Whether to save PNG heatmap

    Returns:
        result: Dict with attributions and top drivers
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if channel_names is None:
        n_ch = input_tensor.shape[1]
        channel_names = CHANNEL_NAMES[:n_ch] + [f"Channel {i}" for i in range(len(CHANNEL_NAMES), n_ch)]

    print(f"[Explain] Computing Integrated Gradients ({n_steps} steps) for {sample_id}...")
    attributions = compute_integrated_gradients(model, input_tensor, target=target, n_steps=n_steps)

    drivers = summarise_attributions(attributions, channel_names=channel_names)

    result = {
        "sample_id": sample_id,
        "target": target,
        "top_drivers": drivers,
        "attribution_shape": list(attributions.shape),
    }

    # Save JSON
    json_path = Path(output_dir) / f"{sample_id}_attribution.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Explain] Attribution JSON: {json_path}")

    # Save PNG heatmap
    if save_png:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            # Channel importance bar chart
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Bar chart of top drivers
            names = [d["channel_name"] for d in drivers]
            scores = [d["attribution_pct"] for d in drivers]
            colors = ["#d62728", "#ff7f0e", "#1f77b4"][:len(drivers)]
            axes[0].barh(names[::-1], scores[::-1], color=colors[::-1])
            axes[0].set_xlabel("Attribution (%)")
            axes[0].set_title(f"Top Meteorological Drivers ({target})", fontsize=12)
            axes[0].axvline(x=0, color="k", linewidth=0.5)

            # Spatial attribution heatmap (sum of absolute attributions across channels)
            spatial_attr = np.abs(attributions).sum(axis=0)  # [H, W]
            im = axes[1].imshow(
                spatial_attr, cmap="Reds", aspect="auto", origin="lower"
            )
            plt.colorbar(im, ax=axes[1], label="Attribution magnitude")
            axes[1].set_title("Spatial Attribution Heatmap", fontsize=12)
            axes[1].set_xlabel("Longitude index")
            axes[1].set_ylabel("Latitude index")

            plt.suptitle(f"Explainability: {sample_id} ({target})", fontsize=14)
            plt.tight_layout()

            png_path = Path(output_dir) / f"{sample_id}_attribution.png"
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[Explain] Attribution PNG: {png_path}")
            result["png_path"] = str(png_path)
        except Exception as e:
            print(f"[Explain] PNG generation failed: {e}")

    return result


if __name__ == "__main__":
    # Smoke test with random data
    model = ForecastBustUNet(in_channels=6, encoder_channels=(32, 64, 128, 256))
    model.eval()
    dummy_input = torch.randn(1, 6, 32, 32)

    result = explain_sample(
        model,
        dummy_input,
        output_dir="results/explain_test",
        sample_id="test_sample",
        n_steps=10,
    )
    print("Top drivers:")
    for d in result["top_drivers"]:
        print(f"  {d['rank']}. {d['channel_name']}: {d['attribution_pct']:.1f}%")
    print("Explainability test PASSED")
