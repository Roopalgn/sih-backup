"""
train.py
--------
Training loop for ForecastBustUNet.

Features:
  - Multi-task loss (Huber regression + Focal classification)
  - AdamW optimizer with cosine annealing LR schedule
  - Automatic Mixed Precision (AMP) for GPU efficiency
  - Gradient clipping
  - Best checkpoint saving by validation AUC
  - Per-epoch metrics: AUC, F1, CSI, FAR, regression RMSE
  - Optional Weights & Biases logging

Usage:
  python model/train.py --config config/settings.yaml
  python model/train.py --config config/settings.yaml --fast  # Quick 5-epoch test
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from tqdm import tqdm

from model.architecture import ForecastBustUNet
from model.loss import MultiTaskBustLoss
from model.dataset import BustDataset

try:
    import sys
    sys.path.insert(0, ".")
    from data.config_loader import load_config
except ImportError:
    import yaml
    def load_config(p):
        with open(p) as f:
            return yaml.safe_load(f)


def compute_metrics(
    pred_prob: np.ndarray,
    true_bust: np.ndarray,
    pred_error: np.ndarray,
    true_error: np.ndarray,
) -> dict[str, float]:
    """
    Compute evaluation metrics for bust detection and error regression.

    Returns dict with: auc, ap, f1, csi, far, rmse, mae
    """
    # Flatten spatial dims
    prob_flat = pred_prob.ravel()
    bust_flat = true_bust.ravel().astype(int)
    err_pred_flat = pred_error.ravel()
    err_true_flat = true_error.ravel()

    metrics = {}

    # — Classification metrics —
    if bust_flat.sum() > 0 and bust_flat.sum() < len(bust_flat):
        metrics["auc"] = roc_auc_score(bust_flat, prob_flat)
        metrics["ap"] = average_precision_score(bust_flat, prob_flat)
    else:
        metrics["auc"] = 0.5
        metrics["ap"] = bust_flat.mean()

    # F1 at 0.5 threshold
    pred_binary = (prob_flat >= 0.5).astype(int)
    metrics["f1"] = f1_score(bust_flat, pred_binary, zero_division=0)

    # Critical Success Index (CSI) and False Alarm Ratio (FAR)
    H = int(((pred_binary == 1) & (bust_flat == 1)).sum())   # Hits
    M = int(((pred_binary == 0) & (bust_flat == 1)).sum())   # Misses
    FA = int(((pred_binary == 1) & (bust_flat == 0)).sum())  # False Alarms
    metrics["csi"] = H / (H + M + FA + 1e-8)
    metrics["far"] = FA / (H + FA + 1e-8)

    # — Regression metrics —
    metrics["rmse"] = float(np.sqrt(((err_pred_flat - err_true_flat) ** 2).mean()))
    metrics["mae"] = float(np.abs(err_pred_flat - err_true_flat).mean())

    return metrics


def train_epoch(
    model: ForecastBustUNet,
    loader: DataLoader,
    criterion: MultiTaskBustLoss,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    reg_loss_sum = 0.0
    cls_loss_sum = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        features = batch["features"].to(device)
        error_map = batch["error_map"].to(device)
        bust_map = batch["bust_map"].to(device)

        optimizer.zero_grad()

        with autocast(enabled=(device.type == "cuda")):
            pred_error, pred_prob = model(features)
            loss, loss_dict = criterion(pred_error, error_map, pred_prob, bust_map)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_dict["total"]
        reg_loss_sum += loss_dict["regression"]
        cls_loss_sum += loss_dict["classification"]
        n_batches += 1

    return {
        "train/loss": total_loss / n_batches,
        "train/reg_loss": reg_loss_sum / n_batches,
        "train/cls_loss": cls_loss_sum / n_batches,
    }


@torch.no_grad()
def validate(
    model: ForecastBustUNet,
    loader: DataLoader,
    criterion: MultiTaskBustLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_probs = []
    all_busts = []
    all_pred_errors = []
    all_true_errors = []
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        features = batch["features"].to(device)
        error_map = batch["error_map"].to(device)
        bust_map = batch["bust_map"].to(device)

        pred_error, pred_prob = model(features)
        _, loss_dict = criterion(pred_error, error_map, pred_prob, bust_map)
        total_loss += loss_dict["total"]
        n_batches += 1

        all_probs.append(pred_prob.cpu().numpy())
        all_busts.append(bust_map.cpu().numpy())
        all_pred_errors.append(pred_error.cpu().numpy())
        all_true_errors.append(error_map.cpu().numpy())

    probs = np.concatenate(all_probs)
    busts = np.concatenate(all_busts)
    pred_errors = np.concatenate(all_pred_errors)
    true_errors = np.concatenate(all_true_errors)

    metrics = compute_metrics(probs, busts, pred_errors, true_errors)
    metrics["val/loss"] = total_loss / n_batches
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ForecastBustUNet")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--fast", action="store_true", help="Quick test run (5 epochs, 50 samples)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    paths_cfg = cfg["paths"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # Checkpoints dir
    ckpt_dir = Path(paths_cfg["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = str(Path(paths_cfg["processed_data"]) / "dataset.zarr")
    max_samples = 50 if args.fast else None
    n_epochs = 5 if args.fast else model_cfg["num_epochs"]

    # ── Datasets ────────────────────────────────────────────
    train_ds = BustDataset(zarr_path, split="train", max_samples=max_samples)
    val_ds = BustDataset(zarr_path, split="val", max_samples=max_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=model_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────
    model = ForecastBustUNet(
        in_channels=train_ds.n_channels,
        encoder_channels=tuple(model_cfg["encoder_channels"]),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model params: {total_params:,}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Train] Resumed from {args.resume}")

    # ── Loss, Optimizer, Scheduler ───────────────────────────
    criterion = MultiTaskBustLoss(
        alpha=model_cfg["loss_alpha"],
        focal_gamma=model_cfg["focal_gamma"],
    )
    optimizer = AdamW(
        model.parameters(),
        lr=model_cfg["learning_rate"],
        weight_decay=model_cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Training Loop ────────────────────────────────────
    best_auc = 0.0
    patience_counter = 0
    history = []

    print(f"[Train] Starting training for {n_epochs} epochs")
    print(f"[Train] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    for epoch in range(1, n_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            grad_clip=model_cfg.get("grad_clip", 1.0)
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Log
        lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "lr": lr, **train_metrics, **val_metrics}
        history.append(row)

        print(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"Loss: {train_metrics['train/loss']:.4f} | "
            f"Val Loss: {val_metrics['val/loss']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"CSI: {val_metrics['csi']:.4f} | "
            f"RMSE: {val_metrics['rmse']:.2f}"
        )

        # Save best checkpoint
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            patience_counter = 0
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_auc,
                    "val_metrics": val_metrics,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"  [Train] Best checkpoint saved (AUC={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= model_cfg.get("early_stopping_patience", 8):
                print(f"[Train] Early stopping at epoch {epoch}")
                break

    # Save training history
    import pandas as pd
    hist_df = pd.DataFrame(history)
    hist_path = Path(paths_cfg["logs"]) / "training_history.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_df.to_csv(hist_path, index=False)
    print(f"[Train] Training history saved: {hist_path}")
    print(f"[Train] Best Val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
