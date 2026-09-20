"""
evaluate.py
-----------
Comprehensive model evaluation for ForecastBustUNet.

Outputs:
  - ROC-AUC, F1, CSI, FAR per lead day
  - Regression RMSE, MAE per IMD meteorological subdivision
  - Calibration reliability diagram data
  - Summary CSV saved to results/evaluation_{split}.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from tqdm import tqdm

from model.architecture import ForecastBustUNet
from model.dataset import BustDataset

try:
    import sys; sys.path.insert(0, ".")
    from data.config_loader import load_config
except ImportError:
    import yaml
    def load_config(p):
        with open(p) as f: return yaml.safe_load(f)


# Approximate bounding boxes for IMD meteorological subdivisions over India
SUBDIVISIONS = {
    "Gangetic WB": (21, 24, 85, 89),
    "Odisha": (18, 22, 82, 87),
    "Vidarbha": (19, 22, 77, 82),
    "Konkan & Goa": (14, 20, 72, 77),
    "NW India": (25, 35, 68, 78),
    "NE India": (22, 30, 88, 98),
    "Peninsular": (8, 18, 75, 82),
    "Bay of Bengal": (6, 20, 82, 98),
}


def extract_region(arr: np.ndarray, lats: np.ndarray, lons: np.ndarray, bbox: tuple) -> np.ndarray:
    """Extract spatial subset within bounding box (lat_min, lat_max, lon_min, lon_max)."""
    lat_min, lat_max, lon_min, lon_max = bbox
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    return arr[..., lat_mask, :][..., lon_mask]


@torch.no_grad()
def run_evaluation(
    model: ForecastBustUNet,
    loader: DataLoader,
    device: torch.device,
    lats: np.ndarray,
    lons: np.ndarray,
):
    """Run full evaluation pass, collecting predictions and targets."""
    model.eval()
    all_probs = []
    all_busts = []
    all_pred_errors = []
    all_true_errors = []
    all_lead_days = []

    for batch in tqdm(loader, desc="Evaluating"):
        features = batch["features"].to(device)
        pred_error, pred_prob = model(features)

        all_probs.append(pred_prob.cpu().numpy())
        all_busts.append(batch["bust_map"].numpy())
        all_pred_errors.append(pred_error.cpu().numpy())
        all_true_errors.append(batch["error_map"].numpy())
        if "lead_day" in batch:
            all_lead_days.extend(batch["lead_day"].tolist())

    probs = np.concatenate(all_probs, axis=0)        # [N, 1, H, W]
    busts = np.concatenate(all_busts, axis=0)        # [N, 1, H, W]
    pred_errors = np.concatenate(all_pred_errors, axis=0)
    true_errors = np.concatenate(all_true_errors, axis=0)
    lead_days = np.array(all_lead_days) if all_lead_days else None

    return probs, busts, pred_errors, true_errors, lead_days


def lead_day_metrics(probs, busts, pred_errors, true_errors, lead_days) -> pd.DataFrame:
    """Compute metrics stratified by lead day."""
    rows = []
    for lead in sorted(set(lead_days.tolist())):
        mask = lead_days == lead
        p = probs[mask].ravel()
        b = busts[mask].ravel().astype(int)
        pe = pred_errors[mask].ravel()
        te = true_errors[mask].ravel()

        auc = roc_auc_score(b, p) if b.sum() > 0 else float("nan")
        ap = average_precision_score(b, p) if b.sum() > 0 else float("nan")
        pred_bin = (p >= 0.5).astype(int)
        f1 = f1_score(b, pred_bin, zero_division=0)
        H = int(((pred_bin == 1) & (b == 1)).sum())
        M = int(((pred_bin == 0) & (b == 1)).sum())
        FA = int(((pred_bin == 1) & (b == 0)).sum())
        csi = H / (H + M + FA + 1e-8)
        far = FA / (H + FA + 1e-8)
        rmse = float(np.sqrt(((pe - te) ** 2).mean()))
        mae = float(np.abs(pe - te).mean())

        rows.append({
            "lead_day": lead,
            "auc": round(auc, 4),
            "ap": round(ap, 4),
            "f1": round(f1, 4),
            "csi": round(csi, 4),
            "far": round(far, 4),
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "bust_rate": round(b.mean(), 4),
            "n_samples": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def subdivision_metrics(pred_errors, true_errors, lats, lons) -> pd.DataFrame:
    """Compute RMSE/MAE per IMD meteorological subdivision."""
    rows = []
    for name, bbox in SUBDIVISIONS.items():
        pe_reg = extract_region(pred_errors[:, 0], lats, lons, bbox)
        te_reg = extract_region(true_errors[:, 0], lats, lons, bbox)
        if pe_reg.size == 0:
            continue
        rmse = float(np.sqrt(((pe_reg - te_reg) ** 2).mean()))
        mae = float(np.abs(pe_reg - te_reg).mean())
        rows.append({"subdivision": name, "rmse": round(rmse, 3), "mae": round(mae, 3)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ForecastBustUNet")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = str(Path(cfg["paths"]["processed_data"]) / "dataset.zarr")
    dataset = BustDataset(zarr_path, split=args.split)
    loader = DataLoader(dataset, batch_size=cfg["model"]["batch_size"], shuffle=False)

    # Load model
    model = ForecastBustUNet(
        in_channels=dataset.n_channels,
        encoder_channels=tuple(cfg["model"]["encoder_channels"]),
    ).to(device)

    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Evaluate] Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"[Evaluate] WARNING: Checkpoint {args.checkpoint} not found. Using random weights.")

    # Load grid coordinates from dataset
    import xarray as xr
    ds = xr.open_zarr(zarr_path)
    lats = ds.latitude.values
    lons = ds.longitude.values

    probs, busts, pred_errors, true_errors, lead_days = run_evaluation(
        model, loader, device, lats, lons
    )

    # ── Lead-day metrics ──────────────────────────────────
    if lead_days is not None and len(set(lead_days.tolist())) > 1:
        lead_df = lead_day_metrics(probs, busts, pred_errors, true_errors, lead_days)
        lead_path = results_dir / f"metrics_by_lead_{args.split}.csv"
        lead_df.to_csv(lead_path, index=False)
        print(f"\n[Evaluate] Lead-day metrics:\n{lead_df.to_string(index=False)}")
        print(f"Saved: {lead_path}")

    # ── Subdivision metrics ──────────────────────────────
    sub_df = subdivision_metrics(pred_errors, true_errors, lats, lons)
    sub_path = results_dir / f"metrics_by_subdivision_{args.split}.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"\n[Evaluate] Subdivision metrics:\n{sub_df.to_string(index=False)}")
    print(f"Saved: {sub_path}")

    # ── Overall metrics ───────────────────────────────────
    p_flat = probs.ravel()
    b_flat = busts.ravel().astype(int)
    auc = roc_auc_score(b_flat, p_flat) if b_flat.sum() > 0 else float("nan")
    rmse = float(np.sqrt(((pred_errors - true_errors) ** 2).mean()))
    print(f"\n[Evaluate] Overall | AUC={auc:.4f} | RMSE={rmse:.3f}")


if __name__ == "__main__":
    main()
